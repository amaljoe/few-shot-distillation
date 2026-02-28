"""
Distillation duration ablation: how many initial steps of logit distillation are needed?

Trains Qwen3-1.7B (LoRA) for 200 steps total on GSM8K.
  Steps 0 .. distill_steps-1  : CE + λ·MSE  (top-K logit distillation)
  Steps distill_steps .. 199  : CE only      (standard SFT)

For distill_steps=0 the entire run is pure SFT with no teacher.

Usage (run from project root inside the apptainer container):
  CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \\
      --num_processes 4 --mixed_precision bf16 --main_process_port 29500 \\
      scripts/distill_steps_experiment.py --distill_steps 16

Run all 5 conditions via:
  bash scripts/run_distill_steps.sh
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent.parent))

from accelerate import Accelerator
from src.data.gsm8k_loader import (
    GSM8KDistillDataset,
    collate_fn,
    load_gsm8k,
)

# ── Hyperparameters ────────────────────────────────────────────────────────────
MODEL_NAME      = "Qwen/Qwen3-1.7B"
NUM_FEWSHOT     = 8
MAX_SEQ_STUDENT = 512
MAX_SEQ_TEACHER = 4096
TRAIN_STEPS     = 200
BATCH_SIZE      = 4         # per device
LR              = 2e-4
WARMUP_STEPS    = 20
WEIGHT_DECAY    = 0.01
GRAD_CLIP       = 1.0
SEED            = 42
K_VOCAB         = 256       # top-K vocab indices for distillation
LAM_DISTILL     = 0.5       # λ weight on MSE loss

LORA_KWARGS = dict(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)


def answer_alignment(t_lens, labels, device):
    """
    Map teacher answer positions → student answer positions.
    Both sequences share the same last n_ans tokens (the ground-truth answer).
    """
    B = labels.shape[0]
    n_ans       = (labels != -100).sum(dim=1)                             # (B,)
    s_ans_start = (labels != -100).float().argmax(dim=1)                  # (B,)
    t_ans_start = t_lens - n_ans                                          # (B,)

    K = int(n_ans.max().item())
    j = torch.arange(K, device=device).unsqueeze(0).expand(B, -1)        # (B, K)

    t_ans_idx = (t_ans_start.unsqueeze(1) + j).clamp(0, t_lens.max() - 1)
    s_ans_idx = (s_ans_start.unsqueeze(1) + j).clamp(0, labels.shape[1] - 1)
    ans_valid  = j < n_ans.unsqueeze(1)                                   # (B, K)
    b_idx      = torch.arange(B, device=device).unsqueeze(1).expand(B, K)

    return b_idx, t_ans_idx, s_ans_idx, ans_valid


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--distill_steps", type=int, required=True,
                   help="Number of initial training steps that use distillation. "
                        "0 = pure SFT throughout.")
    p.add_argument("--output_dir", default="experiments/distill_steps")
    return p.parse_args()


def main():
    args = parse_args()
    assert 0 <= args.distill_steps <= TRAIN_STEPS, \
        f"--distill_steps must be in [0, {TRAIN_STEPS}]"

    out_dir   = Path(args.output_dir) / f"steps_{args.distill_steps}"
    final_dir = out_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device

    if accelerator.is_main_process:
        print(f"\n{'='*60}")
        print(f" Distillation duration ablation")
        print(f"   distill_steps : {args.distill_steps} / {TRAIN_STEPS}")
        print(f"   Model         : {MODEL_NAME}")
        print(f"   Output        : {out_dir}")
        print(f"{'='*60}\n")

    # ── Tokenizer ──────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Frozen teacher (only needed when distill_steps > 0) ───────────────────
    teacher = None
    if args.distill_steps > 0:
        if accelerator.is_main_process:
            print("Loading frozen teacher …")
        teacher = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.bfloat16, trust_remote_code=True,
        ).to(device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        if accelerator.is_main_process:
            print("  Teacher loaded and frozen.\n")

    # ── Student with LoRA ──────────────────────────────────────────────────────
    if accelerator.is_main_process:
        print("Loading student model …")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map=None,
    )
    lora_cfg = LoraConfig(**LORA_KWARGS)
    model = get_peft_model(base_model, lora_cfg)
    if accelerator.is_main_process:
        model.print_trainable_parameters()
        print()

    # ── Dataset ────────────────────────────────────────────────────────────────
    # teacher_include_answer=True is required for suffix alignment in distill phase.
    # When distill_steps=0 we skip teacher entirely so it doesn't matter.
    train_data = load_gsm8k("train")
    train_ds = GSM8KDistillDataset(
        train_data, tokenizer,
        num_fewshot=NUM_FEWSHOT,
        max_seq_len_teacher=MAX_SEQ_TEACHER,
        max_seq_len_student=MAX_SEQ_STUDENT,
        seed=SEED,
        teacher_include_answer=(args.distill_steps > 0),
    )
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id or tokenizer.eos_token_id),
        pin_memory=True,
    )

    # ── Optimiser & scheduler ──────────────────────────────────────────────────
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, WARMUP_STEPS, TRAIN_STEPS)

    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    # ── Training loop ──────────────────────────────────────────────────────────
    model.train()
    train_iter = iter(train_loader)
    step = 0
    switched_logged = False

    progress = tqdm(
        total=TRAIN_STEPS,
        desc=f"distill_steps={args.distill_steps}",
        disable=not accelerator.is_main_process,
    )

    while step < TRAIN_STEPS:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        use_distill = (step < args.distill_steps) and (args.distill_steps > 0)

        # Log once when distillation phase ends
        if (not use_distill) and (args.distill_steps > 0) and (not switched_logged):
            if accelerator.is_main_process:
                tqdm.write(f"\n  [step {step}] Distillation → CE-only SFT")
            switched_logged = True

        if use_distill:
            t_input_ids = batch.teacher_input_ids.to(device)
            t_attn_mask = batch.teacher_attention_mask.to(device)
            labels_d    = batch.labels.to(device)
            t_lens      = t_attn_mask.sum(dim=1)

            with torch.no_grad():
                teacher_out = teacher(input_ids=t_input_ids, attention_mask=t_attn_mask)

            student_out = model(
                input_ids=batch.student_input_ids,
                attention_mask=batch.student_attention_mask,
                labels=batch.labels,
            )
            ce_loss = student_out.loss

            b_idx, t_ans_idx, s_ans_idx, ans_valid = answer_alignment(
                t_lens, labels_d, device
            )
            t_logits = teacher_out.logits[b_idx, t_ans_idx]   # (B, K, V)
            s_logits = student_out.logits[b_idx, s_ans_idx]   # (B, K, V)
            del teacher_out, student_out

            _, top_idx = t_logits.topk(K_VOCAB, dim=-1)
            t_top = t_logits.gather(-1, top_idx)
            s_top = s_logits.gather(-1, top_idx)
            del t_logits, s_logits

            mask = ans_valid.unsqueeze(-1).float()
            n_valid_elems = mask.sum() * K_VOCAB
            dist_loss = ((t_top - s_top).pow(2) * mask).sum() / (n_valid_elems + 1e-8)

            loss = ce_loss + LAM_DISTILL * dist_loss

        else:
            outputs = model(
                input_ids=batch.student_input_ids,
                attention_mask=batch.student_attention_mask,
                labels=batch.labels,
            )
            loss = outputs.loss

        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        step += 1
        progress.update(1)
        if accelerator.is_main_process and step % 20 == 0:
            mode_tag = "distill" if use_distill else "sft"
            progress.set_postfix(loss=f"{loss.item():.4f}", mode=mode_tag)

    progress.close()

    # ── Save final LoRA adapter ────────────────────────────────────────────────
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.unwrap_model(model).save_pretrained(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        meta = {
            "distill_steps": args.distill_steps,
            "total_steps": TRAIN_STEPS,
            "model": MODEL_NAME,
            "K_vocab": K_VOCAB,
            "lambda_distill": LAM_DISTILL,
            "lr": LR,
            "batch_per_device": BATCH_SIZE,
        }
        with open(out_dir / "config.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"\n✓ Adapter saved to {final_dir}")
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
