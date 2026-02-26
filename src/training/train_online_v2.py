"""
Variation 2 (V2): Online all-layer, all-token hidden state distillation on GSM8K.

At every answer token position and at every intermediate transformer layer,
match the student's hidden states to the teacher's hidden states (L2-normalized MSE).
No restriction on the number of tokens — every answer token is used.

Teacher and student share the same base model weights.  Teacher runs online
(frozen, no grad) with 8-shot context; student is fine-tuned with LoRA.

  L_total = L_CE  +  λ * mean_{l,t} || norm(h_l^s(t)) - norm(h_l^t(t)) ||²

where the mean is over all answer token positions t and all layers l.

Note: gradient_checkpointing must be False (incompatible with output_hidden_states=True).

Run command (tmux: vscode on cn14-dgx):
  CUDA_VISIBLE_DEVICES=2,3 accelerate launch \\
      --num_processes 2 --mixed_precision bf16 --main_process_port 29501 \\
      src/training/train_online_v2.py --config configs/online_v2.yaml \\
      --output_dir experiments/online_v2
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
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from accelerate import Accelerator
from src.data.gsm8k_loader import load_gsm8k, make_dataloader
from src.models.student import StudentModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/online_v2.yaml")
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


def answer_alignment(t_lens, labels, device):
    """
    Compute index tensors that align teacher and student answer token positions.

    Both sequences end with the same ground-truth answer tokens.
    Teacher's last n_ans content tokens = student's answer tokens (labels != -100).

    Returns:
        b_idx:       (B, K)  batch indices for advanced indexing
        t_ans_idx:   (B, K)  teacher token indices per answer position
        s_ans_idx:   (B, K)  student token indices per answer position
        ans_valid:   (B, K)  bool mask — True where position j < n_ans[b]
        n_ans:       (B,)    number of answer tokens per example
    """
    B = labels.shape[0]
    n_ans = (labels != -100).sum(dim=1)                            # (B,)
    s_ans_start = (labels != -100).float().argmax(dim=1)           # (B,) first answer pos in student
    t_ans_start = t_lens - n_ans                                   # (B,) first answer pos in teacher

    K = int(n_ans.max().item())
    j = torch.arange(K, device=device).unsqueeze(0).expand(B, -1)  # (B, K)

    t_ans_idx = (t_ans_start.unsqueeze(1) + j).clamp(0, t_lens.max().item() - 1)  # (B, K)
    s_ans_idx = (s_ans_start.unsqueeze(1) + j).clamp(0, labels.shape[1] - 1)      # (B, K)
    ans_valid = j < n_ans.unsqueeze(1)                                              # (B, K)
    b_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, K)               # (B, K)

    return b_idx, t_ans_idx, s_ans_idx, ans_valid, n_ans


def main():
    args = parse_args()

    base_cfg = OmegaConf.load("configs/base.yaml")
    v2_cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(base_cfg, v2_cfg)

    if cfg.training.gradient_checkpointing:
        raise ValueError(
            "gradient_checkpointing=True is incompatible with output_hidden_states=True. "
            "Set gradient_checkpointing: false in configs/online_v2.yaml."
        )

    output_dir = Path(args.output_dir or cfg.training.output_dir) / "online_v2"
    output_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        mixed_precision="bf16" if cfg.training.bf16 else "no",
        log_with=None,
    )

    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir=str(output_dir / "tb_logs"))
        print(f"\nVariation 2: Online all-layer all-token hidden state distillation")
        print(f"  λ         = {cfg.distillation.lambda_distill}")
        print(f"  Normalize = {cfg.distillation.normalize_hidden}")
        print(f"  Output    : {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Frozen teacher — same base weights, no LoRA, one copy per DDP process
    teacher = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    teacher = teacher.to(accelerator.device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Dataset — teacher includes answer for suffix alignment
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
        teacher_include_answer=True,
    )

    # Student — LoRA, no hooks needed (use output_hidden_states=True instead)
    student_wrapper = StudentModel(
        model_name=cfg.model.name,
        lora_config=OmegaConf.to_container(cfg.lora),
        num_layers=cfg.model.num_layers,
        device_map=None,
    )
    model = student_wrapper.get_model()
    # gradient_checkpointing already checked above

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

    lam = cfg.distillation.lambda_distill
    normalize = cfg.distillation.normalize_hidden

    model.train()
    step = 0
    train_iter = iter(train_loader)
    ce_loss_accum = 0.0
    dist_loss_accum = 0.0
    alignment_verified = False

    progress = tqdm(total=cfg.training.max_steps, desc="V2 training",
                    disable=not accelerator.is_main_process)

    while step < cfg.training.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        with accelerator.accumulate(model):
            device = accelerator.device
            # Explicitly move teacher-side tensors to device.
            # The DDP-wrapped student model receives batch tensors as-is
            # (accelerate handles that path), but the standalone frozen
            # teacher model needs inputs on the same device as its weights.
            t_input_ids = batch.teacher_input_ids.to(device)
            t_attn_mask = batch.teacher_attention_mask.to(device)
            labels_d    = batch.labels.to(device)
            t_lens = t_attn_mask.sum(dim=1)  # (B,) actual teacher content lengths

            # Teacher forward — all intermediate hidden states, no grad
            with torch.no_grad():
                teacher_out = teacher(
                    input_ids=t_input_ids,
                    attention_mask=t_attn_mask,
                    output_hidden_states=True,
                )
            # Stack layers into (B, T_t, L, H), skipping embedding layer (index 0)
            # Each hidden_states[i] has shape (B, T_t, H)
            teacher_all = torch.stack(
                list(teacher_out.hidden_states[1:]), dim=2
            )  # (B, T_t, L, H)
            del teacher_out  # free logits and other outputs immediately

            # Student forward — CE loss + all intermediate hidden states
            student_out = model(
                input_ids=batch.student_input_ids,
                attention_mask=batch.student_attention_mask,
                labels=batch.labels,
                output_hidden_states=True,
            )
            ce_loss = student_out.loss
            # Stack student layers: (B, T_s, L, H)
            student_all = torch.stack(
                list(student_out.hidden_states[1:]), dim=2
            )  # (B, T_s, L, H)

            # Compute answer alignment indices
            b_idx, t_ans_idx, s_ans_idx, ans_valid, n_ans = answer_alignment(
                t_lens, labels_d, device
            )

            # Sanity check on first batch
            if accelerator.is_main_process and not alignment_verified:
                b0_n = n_ans[0].item()
                t_ids = t_input_ids[0, t_ans_idx[0, :b0_n]]
                s_ids = batch.student_input_ids.to(device)[0, s_ans_idx[0, :b0_n]]
                match = (t_ids == s_ids).all().item()
                L = student_all.shape[2]
                H = student_all.shape[3]
                print(f"\n{'✓' if match else '✗'} Token alignment: "
                      f"teacher/student answer tokens {'match' if match else 'MISMATCH'} "
                      f"({b0_n} tokens, {L} layers, H={H})")
                if not match:
                    print(f"  Teacher: {t_ids[:8].tolist()}")
                    print(f"  Student: {s_ids[:8].tolist()}")
                alignment_verified = True

            # Gather hidden states at aligned answer positions
            # teacher_all: (B, T_t, L, H)  → t_states: (B, K, L, H)
            # student_all: (B, T_s, L, H)  → s_states: (B, K, L, H)
            t_states = teacher_all[b_idx, t_ans_idx]   # (B, K, L, H)
            s_states = student_all[b_idx, s_ans_idx]   # (B, K, L, H)
            del teacher_all, student_all               # free peak tensors

            # Normalize along hidden dimension if requested
            if normalize:
                t_states = F.normalize(t_states, dim=-1)
                s_states = F.normalize(s_states, dim=-1)

            # Masked MSE over valid answer positions across all layers
            # ans_valid: (B, K) → expand to (B, K, 1, 1) for broadcasting
            mask = ans_valid.unsqueeze(-1).unsqueeze(-1).float()
            L = s_states.shape[2]
            H = s_states.shape[3]
            n_valid_elements = mask.sum() * L * H
            dist_loss = ((s_states - t_states.detach()).pow(2) * mask).sum() / (n_valid_elements + 1e-8)

            total_loss = ce_loss + lam * dist_loss
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
                dist_loss_accum += dist_loss.item()

                if step % cfg.training.logging_steps == 0:
                    avg_ce = ce_loss_accum / cfg.training.logging_steps
                    avg_dist = dist_loss_accum / cfg.training.logging_steps
                    if accelerator.is_main_process:
                        writer.add_scalar("train/ce_loss", avg_ce, step)
                        writer.add_scalar("train/dist_loss", avg_dist, step)
                        writer.add_scalar("train/total_loss", avg_ce + lam * avg_dist, step)
                        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)
                        progress.set_postfix(
                            ce=f"{avg_ce:.4f}", dist=f"{avg_dist:.4f}", step=step
                        )
                    ce_loss_accum = 0.0
                    dist_loss_accum = 0.0

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
        print(f"\n✓ V2 training done. Model saved to {final_dir}")
        with open(output_dir / "train_config.json", "w") as f:
            json.dump(OmegaConf.to_container(cfg), f, indent=2)

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
