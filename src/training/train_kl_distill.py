"""
Condition D: Logit-level KL distillation fine-tuning on GSM8K.

Combines CE task loss with forward KL divergence on output logits at the query position:
  L_total = L_ce  +  λ * T² * KL(teacher_logits/T || student_logits/T)

where KL is computed using the teacher's top-K logit support for efficiency.

This is the standard Hinton-style knowledge distillation applied to the next-token
distribution at the query alignment position, rather than matching hidden states.

KL vs MSE on hidden states:
  - MSE on hidden states (Cond C): matches geometry of internal representations
  - KL on output logits  (Cond D): matches the predictive distribution directly
    Both are valid; D is more principled for probability-valued outputs.

NOTE on attention-weight KL:
  Teacher and student have different sequence lengths (8-shot vs 0-shot context),
  so attention matrices cannot be directly compared. Hidden-state and logit
  matching at the query position are the principled alternatives.

Prerequisites:
  1. Run scripts/precompute_teacher_activations.py (provides example_idx lookup)
  2. Run scripts/precompute_teacher_logits.py     (provides top-K logit cache)

Run command (tmux: vscode on cn14-dgx):
  CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 --mixed_precision bf16 \\
      src/training/train_kl_distill.py \\
      --config experiments/ablations/configs/kl_distill.yaml \\
      --output_dir experiments/ablations/kl_distill
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from accelerate import Accelerator
from src.data.gsm8k_loader import load_gsm8k, make_dataloader
from src.models.student import StudentModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default="experiments/ablations/configs/kl_distill.yaml")
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


def load_logit_cache(logits_path: str) -> dict:
    """
    Load precomputed teacher top-K logits from disk.

    Returns dict with:
        values:  (N, K) float16 — raw logit values (before softmax)
        indices: (N, K) int32  — vocab indices of top-K logits
        top_k:   int
    """
    path = Path(logits_path)
    assert path.exists(), (
        f"Logit cache not found at {path}. "
        "Run scripts/precompute_teacher_logits.py first."
    )
    print(f"Loading teacher logit cache from {path}...")
    cache = torch.load(str(path), map_location="cpu")
    K = cache["top_k"]
    print(f"  values:  {cache['values'].shape} {cache['values'].dtype}")
    print(f"  indices: {cache['indices'].shape} {cache['indices'].dtype}")
    print(f"  top_k: {K}")
    return cache


def kl_distill_loss(
    student_logits: torch.Tensor,    # (B, vocab_size) float — student full logits
    teacher_values: torch.Tensor,    # (B, K) float16 — teacher top-K raw logits
    teacher_indices: torch.Tensor,   # (B, K) int32   — teacher top-K vocab indices
    temperature: float = 2.0,
) -> torch.Tensor:
    """
    Forward KL divergence: KL(teacher || student) over teacher's top-K support.

    Uses teacher's top-K as the support set. Student log probs are taken from
    the full (vocab-sized) student distribution, so normalization is correct.

    Loss is scaled by T² to compensate for soft probability concentration,
    matching the Hinton et al. convention.
    """
    B, V = student_logits.shape
    K = teacher_values.shape[1]

    # Student: full log-softmax over vocab (normalized correctly)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)  # (B, V)

    # Gather student log probs at teacher's top-K positions
    t_idx = teacher_indices.long().to(student_logits.device)            # (B, K)
    student_lp_topk = student_log_probs.gather(1, t_idx)               # (B, K)

    # Teacher: softmax over top-K only (renormalized over support)
    t_vals = teacher_values.float().to(student_logits.device)          # (B, K)
    teacher_probs = F.softmax(t_vals / temperature, dim=-1)             # (B, K)

    # Forward KL = Σ p_teacher * (log p_teacher - log p_student)
    #            = -Σ p_teacher * log p_student  + H(teacher)  (H(teacher) is constant)
    # Minimizing KL = maximizing Σ p_teacher * log p_student
    kl = (teacher_probs * (teacher_probs.clamp(min=1e-9).log() - student_lp_topk))
    kl = kl.sum(dim=-1).mean()     # mean over batch

    return kl * (temperature ** 2)  # Hinton scaling


def main():
    args = parse_args()

    base_cfg = OmegaConf.load("configs/base.yaml")
    dist_cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(base_cfg, dist_cfg)

    output_dir = Path(args.output_dir or cfg.training.output_dir) / "kl_distill"
    output_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        mixed_precision="bf16" if cfg.training.bf16 else "no",
        log_with=None,
    )

    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir=str(output_dir / "tb_logs"))
        print(f"\nCondition D: Logit-level KL distillation")
        print(f"  λ = {cfg.distillation.lambda_distill}")
        print(f"  T = {cfg.distillation.kl_temperature}")
        print(f"  top_k = {cfg.distillation.top_k_logits}")
        print(f"  Output: {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Logit cache (CPU, moved per-batch)
    logit_cache = load_logit_cache(
        Path(cfg.teacher_activations.cache_dir) / f"logits_top{cfg.distillation.top_k_logits}.pt"
    )
    logit_values = logit_cache["values"]   # (N, K) float16, CPU
    logit_indices = logit_cache["indices"] # (N, K) int32, CPU

    train_data = load_gsm8k(cfg.data.train_split)
    train_loader = make_dataloader(
        train_data, tokenizer,
        batch_size=cfg.training.per_device_train_batch_size,
        num_fewshot=cfg.data.num_fewshot_examples,
        max_seq_len_teacher=cfg.data.max_seq_len_teacher,
        max_seq_len_student=cfg.data.max_seq_len_student,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        seed=cfg.training.seed,
    )

    # Student (no activation capture needed — only output logits used)
    student_wrapper = StudentModel(
        model_name=cfg.model.name,
        lora_config=OmegaConf.to_container(cfg.lora),
        num_layers=cfg.model.num_layers,
        layer_indices=list(range(cfg.model.num_layers)),
        device_map=None,
    )
    model = student_wrapper.get_model()

    if cfg.training.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    optimizer = AdamW(
        student_wrapper.get_trainable_parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.training.warmup_steps,
        num_training_steps=cfg.training.max_steps,
    )

    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    model.train()
    step = 0
    train_iter = iter(train_loader)
    ce_loss_accum = 0.0
    kl_loss_accum = 0.0

    progress = tqdm(total=cfg.training.max_steps, desc="KL distill",
                    disable=not accelerator.is_main_process)

    while step < cfg.training.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        with accelerator.accumulate(model):
            # Student forward — get both CE loss and output logits
            outputs = model(
                input_ids=batch.student_input_ids,
                attention_mask=batch.student_attention_mask,
                labels=batch.labels,
            )
            ce_loss = outputs.loss

            # Extract student logits at the query alignment position
            # student_query_pos: (B,) — last token of question prompt
            student_query_pos = batch.student_query_pos.to(accelerator.device)
            B = student_query_pos.shape[0]
            # outputs.logits: (B, seq_len, vocab_size)
            # gather the logit vector at each example's query position
            pos_expanded = student_query_pos.unsqueeze(-1).unsqueeze(-1).expand(
                B, 1, outputs.logits.shape[-1]
            )  # (B, 1, V)
            student_logits_at_query = outputs.logits.gather(1, pos_expanded).squeeze(1)
            # (B, V) — student's next-token logit distribution at query pos

            # Fetch teacher top-K logits for this batch
            idx = batch.example_idx                              # (B,)
            t_vals = logit_values[idx].to(accelerator.device)   # (B, K) float16
            t_idx = logit_indices[idx].to(accelerator.device)   # (B, K) int32

            # KL divergence loss
            kl_loss = kl_distill_loss(
                student_logits=student_logits_at_query.float(),
                teacher_values=t_vals,
                teacher_indices=t_idx,
                temperature=cfg.distillation.kl_temperature,
            )

            total_loss = ce_loss + cfg.distillation.lambda_distill * kl_loss
            accelerator.backward(total_loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(
                    student_wrapper.get_trainable_parameters(),
                    cfg.training.grad_clip,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                step += 1
                ce_loss_accum += ce_loss.item()
                kl_loss_accum += kl_loss.item()

                if step % cfg.training.logging_steps == 0:
                    avg_ce = ce_loss_accum / cfg.training.logging_steps
                    avg_kl = kl_loss_accum / cfg.training.logging_steps

                    if accelerator.is_main_process:
                        writer.add_scalar("train/ce_loss", avg_ce, step)
                        writer.add_scalar("train/kl_loss", avg_kl, step)
                        writer.add_scalar("train/total_loss",
                                          avg_ce + cfg.distillation.lambda_distill * avg_kl,
                                          step)
                        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)
                        progress.set_postfix(ce=f"{avg_ce:.4f}", kl=f"{avg_kl:.4f}", step=step)

                    ce_loss_accum = 0.0
                    kl_loss_accum = 0.0

                if step % cfg.training.save_steps == 0 and accelerator.is_main_process:
                    ckpt_dir = output_dir / f"checkpoint-{step}"
                    accelerator.unwrap_model(model).save_pretrained(str(ckpt_dir))
                    tokenizer.save_pretrained(str(ckpt_dir))

                progress.update(1)
                if step >= cfg.training.max_steps:
                    break

    if accelerator.is_main_process:
        final_dir = output_dir / "final"
        accelerator.unwrap_model(model).save_pretrained(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        writer.close()
        print(f"\n✓ KL distillation done. Model saved to {final_dir}")
        with open(output_dir / "train_config.json", "w") as f:
            json.dump(OmegaConf.to_container(cfg), f, indent=2)

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
