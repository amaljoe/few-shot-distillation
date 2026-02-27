"""
Diagnostic: plot student CE loss vs teacher CE loss during single-task training.

Hypothesis: at some point student zero-shot CE drops below teacher few-shot CE,
meaning the KD signal starts pushing the student toward a worse distribution.

Logs per-step: student_ce, teacher_ce, distill_mse.
Saves CSV + PNG plot (main process only).

Run (4 GPUs):
  CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 \
      --mixed_precision bf16 --main_process_port 29503 \
      scripts/plot_pos_losses.py --task pos \
      --base_config configs/xtreme_llama3b.yaml \
      --output_dir experiments/xtreme/pos_loss_curves
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from omegaconf import OmegaConf
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.xtreme_loader import make_xtreme_dataloader
from src.models.student import StudentModel


def answer_alignment(t_lens, labels, device):
    B = labels.shape[0]
    n_ans = (labels != -100).sum(dim=1)
    s_ans_start = (labels != -100).float().argmax(dim=1)
    t_ans_start = t_lens - n_ans
    K = int(n_ans.max().item())
    j = torch.arange(K, device=device).unsqueeze(0).expand(B, -1)
    t_ans_idx = (t_ans_start.unsqueeze(1) + j).clamp(0, t_lens.max().item() - 1)
    s_ans_idx = (s_ans_start.unsqueeze(1) + j).clamp(0, labels.shape[1] - 1)
    ans_valid  = j < n_ans.unsqueeze(1)
    b_idx      = torch.arange(B, device=device).unsqueeze(1).expand(B, K)
    return b_idx, t_ans_idx, s_ans_idx, ans_valid, n_ans


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_config",    default="configs/xtreme_llama3b.yaml")
    p.add_argument("--distill_config", default="configs/xtreme_distill.yaml")
    p.add_argument("--task",           default="pos", choices=["nli","pa","qa","ner","pos"])
    p.add_argument("--lambda_distill", type=float, default=0.05)
    p.add_argument("--max_steps",      type=int, default=1000)
    p.add_argument("--output_dir",     default="experiments/xtreme/pos_loss_curves")
    p.add_argument("--smooth_window",  type=int, default=20)
    return p.parse_args()


def smooth(values, window):
    alpha = 2.0 / (window + 1)
    ema, out = values[0], []
    for v in values:
        ema = alpha * v + (1 - alpha) * ema
        out.append(ema)
    return out


def main():
    args = parse_args()
    cfg  = OmegaConf.merge(OmegaConf.load(args.base_config), OmegaConf.load(args.distill_config))

    accelerator = Accelerator(
        mixed_precision="bf16" if cfg.training.bf16 else "no",
        log_with=None,
    )
    device = accelerator.device

    output_dir = Path(args.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Model: {cfg.model.name}  task={args.task}  λ={args.lambda_distill}  steps={args.max_steps}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Frozen teacher (one copy per process)
    teacher = AutoModelForCausalLM.from_pretrained(
        cfg.model.name, dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Single-task dataloader
    train_loader = make_xtreme_dataloader(
        tokenizer=tokenizer,
        tasks=[args.task],
        train_langs=list(cfg.data.get("eval_langs", ["en"])),
        batch_size=cfg.training.per_device_train_batch_size,
        max_samples_per_task_lang=cfg.data.get("max_samples_per_task", 5000),
        max_seq_len_teacher=cfg.data.get("max_seq_len_teacher", 2048),
        max_seq_len_student=cfg.data.get("max_seq_len_student", 512),
        shuffle=True,
        num_workers=2,
        seed=cfg.training.seed,
        teacher_include_answer=True,
        zeroshot_teacher=False,
    )

    # Student (LoRA)
    use_lora = getattr(cfg.training, "use_lora", True)
    lora_cfg = OmegaConf.to_container(cfg.lora) if (use_lora and hasattr(cfg, "lora")) else None
    student_wrapper = StudentModel(
        model_name=cfg.model.name,
        lora_config=lora_cfg,
        use_lora=use_lora,
        num_layers=cfg.model.num_layers,
        device_map=None,
    )
    model = student_wrapper.get_model()

    optimizer = AdamW(
        student_wrapper.get_trainable_parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.training.warmup_steps,
        num_training_steps=args.max_steps,
    )

    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    K_vocab = cfg.distillation.n_top_logits
    lam     = args.lambda_distill

    # ── Training loop ──────────────────────────────────────────────────────────
    rows = []
    train_iter = iter(train_loader)
    alignment_verified = False

    for step in tqdm(range(1, args.max_steps + 1), desc=f"{args.task} distill",
                     disable=not accelerator.is_main_process):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        with accelerator.accumulate(model):
            t_input_ids = batch.teacher_input_ids.to(device)
            t_attn_mask = batch.teacher_attention_mask.to(device)
            labels_d    = batch.labels.to(device)
            t_lens      = t_attn_mask.sum(dim=1)

            with torch.no_grad():
                teacher_out = teacher(input_ids=t_input_ids, attention_mask=t_attn_mask)

            student_out = model(
                input_ids=batch.student_input_ids.to(device),
                attention_mask=batch.student_attention_mask.to(device),
                labels=labels_d,
            )
            ce_loss = student_out.loss

            b_idx, t_ans_idx, s_ans_idx, ans_valid, n_ans = answer_alignment(
                t_lens, labels_d, device
            )

            if accelerator.is_main_process and not alignment_verified and n_ans[0].item() > 0:
                b0_n = n_ans[0].item()
                t_ids = t_input_ids[0, t_ans_idx[0, :b0_n]]
                s_ids = batch.student_input_ids.to(device)[0, s_ans_idx[0, :b0_n]]
                print(f"\n{'✓' if (t_ids == s_ids).all() else '✗'} Token alignment")
                alignment_verified = True

            t_logits_ans = teacher_out.logits[b_idx, t_ans_idx]  # for distill MSE
            s_logits_ans = student_out.logits[b_idx, s_ans_idx]  # for distill MSE

            # Teacher CE: logits[i] predicts token[i+1], so shift back by 1
            t_ce_idx        = (t_ans_idx - 1).clamp(0, t_input_ids.shape[1] - 1)
            t_logits_for_ce = teacher_out.logits[b_idx, t_ce_idx]
            gold_ids        = labels_d[b_idx, s_ans_idx]
            valid_flat      = ans_valid.reshape(-1)
            t_logits_flat   = t_logits_for_ce.reshape(-1, t_logits_for_ce.size(-1))[valid_flat]
            gold_flat       = gold_ids.reshape(-1)[valid_flat]
            teacher_ce_val  = F.cross_entropy(t_logits_flat, gold_flat)

            # Distillation MSE
            _, top_idx = t_logits_ans.topk(K_vocab, dim=-1)
            t_top = t_logits_ans.gather(-1, top_idx)
            s_top = s_logits_ans.gather(-1, top_idx)
            mask  = ans_valid.unsqueeze(-1).float()
            distill_mse = ((t_top - s_top).pow(2) * mask).sum() / (mask.sum() * K_vocab + 1e-8)

            total_loss = ce_loss + lam * distill_mse
            accelerator.backward(total_loss)
            accelerator.clip_grad_norm_(student_wrapper.get_trainable_parameters(),
                                        cfg.training.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Average losses across all processes for logging
        ce_reduced  = accelerator.reduce(ce_loss.detach(),        reduction="mean")
        tce_reduced = accelerator.reduce(teacher_ce_val.detach(), reduction="mean")
        mse_reduced = accelerator.reduce(distill_mse.detach(),    reduction="mean")

        if accelerator.is_main_process:
            rows.append({
                "step":        step,
                "student_ce":  ce_reduced.item(),
                "teacher_ce":  tce_reduced.item(),
                "distill_mse": mse_reduced.item(),
                "total_loss":  (ce_reduced + lam * mse_reduced).item(),
            })

        del teacher_out, student_out, t_logits_ans, s_logits_ans

    # ── Save CSV + Plot (main process only) ───────────────────────────────────
    if not accelerator.is_main_process:
        return

    csv_path = output_dir / f"{args.task}_loss_curves.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved CSV: {csv_path}")

    steps       = [r["step"]        for r in rows]
    student_ce  = [r["student_ce"]  for r in rows]
    teacher_ce  = [r["teacher_ce"]  for r in rows]
    distill_mse = [r["distill_mse"] for r in rows]

    w = args.smooth_window
    s_ce_s = smooth(student_ce,  w)
    t_ce_s = smooth(teacher_ce,  w)
    mse_s  = smooth(distill_mse, w)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(steps, s_ce_s, color="steelblue",  lw=2, label="Student CE (zero-shot)")
    ax1.plot(steps, t_ce_s, color="darkorange",  lw=2, label="Teacher CE (few-shot)")
    for i in range(1, len(steps)):
        if s_ce_s[i-1] >= t_ce_s[i-1] and s_ce_s[i] < t_ce_s[i]:
            ax1.axvline(x=steps[i], color="red", lw=1.5, ls="--", alpha=0.7,
                        label=f"Crossover @ step {steps[i]}")
    ax1.set_ylabel("Cross-Entropy Loss (nats)")
    ax1.set_title(f"{args.task.upper()} — Student vs Teacher CE Loss  (λ={lam}, EMA-{w})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps, mse_s, color="mediumseagreen", lw=2, label="Distill MSE (top-256)")
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("MSE Loss")
    ax2.set_title("Distillation MSE")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / f"{args.task}_loss_curves.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Saved plot: {plot_path}")

    cross = next((r["step"] for i, r in enumerate(rows)
                  if i > 0 and s_ce_s[i] < t_ce_s[i] and s_ce_s[i-1] >= t_ce_s[i-1]), None)
    print(f"\nStudent CE:  start={student_ce[0]:.3f}  end={student_ce[-1]:.3f}  min={min(student_ce):.3f}")
    print(f"Teacher CE:  start={teacher_ce[0]:.3f}  end={teacher_ce[-1]:.3f}  avg={sum(teacher_ce)/len(teacher_ce):.3f}")
    print(f"Crossover step: {cross if cross else 'none detected'}")


if __name__ == "__main__":
    main()
