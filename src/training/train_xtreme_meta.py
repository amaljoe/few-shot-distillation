"""
XTREME Meta-Learning: Bilevel optimisation for per-task distillation weight λ.

Learns one scalar λ_t per task via a first-order bilevel approximation:

    Inner  (every step)  : θ ← θ - α · ∇_θ [L_CE + λ_t · L_dist]
    Outer  (every K steps): update logit_λ_t using approximate meta-gradient

Meta-gradient approximation (first-order, avoids Hessian):
    d L_val / d λ_t  ≈  -α  ·  g_val^T · g_dist_t
    where:
      g_val    = ∇_θ L_CE_val   (CE gradient on held-out val batch, current θ)
      g_dist_t = ∇_θ L_dist_t  (distillation gradient for task-t examples)
    → if KD and CE gradients align, λ increases; if they conflict, λ decreases.

λ_t is stored in log-odds (logit) space: λ_t = sigmoid(logit_t) * lambda_max
  – naturally bounded to (0, lambda_max), no clamping needed
  – initialised to lambda_init (sigmoid(0) = 0.5 → default 0.5 * lambda_max)
  – Adam outer optimiser adapts per-task step sizes

Val set: dev split of each task (multilingual dev, all available languages).
This gives a cross-lingual outer objective: λ is optimised to maximise
English-trained model's cross-lingual zero-shot CE loss on the dev set.

Run command (Llama-3.2-3B, GPUs 0,1):
  CUDA_VISIBLE_DEVICES=0,1 accelerate launch \\
      --num_processes 2 --mixed_precision bf16 --main_process_port 29500 \\
      src/training/train_xtreme_meta.py \\
      --base_config configs/xtreme_llama3b.yaml \\
      --config configs/xtreme_meta.yaml \\
      --output_dir experiments/xtreme/llama3b
"""

import argparse
import json
import math
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
from src.data.xtreme_loader import TASKS, TASK2ID, make_xtreme_dataloader
from src.models.student import StudentModel


# ============================================================================
# Helpers
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",              type=str, default="configs/xtreme_meta.yaml")
    parser.add_argument("--base_config",         type=str, default="configs/xtreme_llama3b.yaml")
    parser.add_argument("--output_dir",          type=str, default=None)
    parser.add_argument("--tasks",               type=str, nargs="+", default=None)
    parser.add_argument("--max_steps",           type=int, default=None)
    parser.add_argument("--per_device_batch_size", type=int, default=None)
    parser.add_argument("--condition_name",      type=str, default=None)
    return parser.parse_args()


def answer_alignment(t_lens, labels, device):
    """Identical to train_xtreme_distill.py."""
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


def get_lambda(logit_lambda: torch.Tensor, lambda_max: float) -> torch.Tensor:
    """Convert logit parameterisation → λ ∈ (0, lambda_max)."""
    return torch.sigmoid(logit_lambda) * lambda_max


# ============================================================================
# Meta update
# ============================================================================

def meta_update(
    model,
    val_batch,
    saved_batch,
    saved_t_top,       # (B_save, K, K_vocab) teacher top-K logits, detached
    saved_top_idx,     # (B_save, K, K_vocab) top-K vocab indices, detached
    saved_b_idx,       # (B_save, K) batch indices, detached
    saved_s_ans_idx,   # (B_save, K) student answer positions, detached
    saved_ans_valid,   # (B_save, K) answer validity mask, detached
    logit_lambda,
    meta_optimizer,
    trainable_params,  # list of tensors (LoRA params)
    K_vocab,
    n_tasks,
    device,
    accelerator,
    inner_lr,
):
    """
    Compute first-order bilevel gradient and update logit_lambda via Adam.

    Returns dict of per-task dot products for logging.
    """
    # -------------------------------------------------------------------
    # Step 1: val CE gradient  g_val = ∇_θ L_CE(val_batch, θ)
    # -------------------------------------------------------------------
    model.eval()
    val_ce = None
    with torch.enable_grad():
        val_out = model(
            input_ids=val_batch.student_input_ids.to(device),
            attention_mask=val_batch.student_attention_mask.to(device),
            labels=val_batch.labels.to(device),
        )
        val_ce = val_out.loss
        g_val = torch.autograd.grad(
            val_ce, trainable_params, allow_unused=True, retain_graph=False,
        )
    g_val = [g.detach() if g is not None else None for g in g_val]
    model.train()

    # -------------------------------------------------------------------
    # Step 2: recompute student logits on saved train batch
    # (teacher logits are already saved; only student needs grad)
    # -------------------------------------------------------------------
    task_ids_saved = saved_batch.task_ids.to(device)
    dot_by_task = {}

    with torch.enable_grad():
        s_out_saved = model(
            input_ids=saved_batch.student_input_ids.to(device),
            attention_mask=saved_batch.student_attention_mask.to(device),
        )
        # s_logits_ans: (B_save, K, vocab)
        s_logits_ans = s_out_saved.logits[saved_b_idx, saved_s_ans_idx]
        # s_top: (B_save, K, K_vocab)
        s_top = s_logits_ans.gather(-1, saved_top_idx)

        tasks_in_batch = task_ids_saved.unique().tolist()
        for t_int in range(n_tasks):
            task_mask = (task_ids_saved == t_int)  # (B_save,)
            if task_mask.sum() == 0:
                continue

            t_top_t     = saved_t_top[task_mask]     # (B_t, K, K_vocab)
            s_top_t     = s_top[task_mask]            # (B_t, K, K_vocab)
            ans_valid_t = saved_ans_valid[task_mask]  # (B_t, K)

            mask = ans_valid_t.unsqueeze(-1).float()
            n_valid = mask.sum() * K_vocab
            dist_loss_t = ((t_top_t - s_top_t).pow(2) * mask).sum() / (n_valid + 1e-8)

            is_last = (t_int == max(tasks_in_batch))
            g_dist_t = torch.autograd.grad(
                dist_loss_t,
                trainable_params,
                allow_unused=True,
                retain_graph=not is_last,
            )

            # dot product: g_val · g_dist_t
            dot_t = sum(
                (a * b).sum().item()
                for a, b in zip(g_val, g_dist_t)
                if a is not None and b is not None
            )
            dot_by_task[t_int] = dot_t

    # -------------------------------------------------------------------
    # Step 3: set meta-gradient and update logit_lambda with Adam
    # Gradient descent on L_val w.r.t. logit_λ_t:
    #   d L_val / d logit_λ_t  ≈  (d L_val / d λ_t) · sigmoid' · lambda_max
    #                           = -inner_lr · dot_t · sigmoid' · lambda_max
    # In practice we absorb constants into meta_lr and set .grad directly.
    # -------------------------------------------------------------------
    meta_optimizer.zero_grad()
    if logit_lambda.grad is None:
        logit_lambda.grad = torch.zeros_like(logit_lambda)
    else:
        logit_lambda.grad.zero_()

    for t_int, dot_t in dot_by_task.items():
        # Gradient of L_val w.r.t. logit_λ_t (negate → Adam minimises):
        # we want to DECREASE L_val, so gradient descent → subtract meta_lr * grad
        # grad = d L_val / d logit_λ_t = -inner_lr * dot_t * sigmoid' * lambda_max
        # We absorb constants and set: .grad[t] = -dot_t
        # (meta_lr and Adam then scale this appropriately)
        logit_lambda.grad[t_int] = -dot_t

    # Sync λ-gradient across DDP processes (each process may have different data)
    # logit_lambda lives on CPU but NCCL requires CUDA tensors → move, reduce, move back
    if accelerator.num_processes > 1:
        dist_grad = logit_lambda.grad.clone().to(device)
        torch.distributed.all_reduce(dist_grad, op=torch.distributed.ReduceOp.AVG)
        logit_lambda.grad.copy_(dist_grad.cpu())

    meta_optimizer.step()

    return dot_by_task, val_ce.item() if val_ce is not None else 0.0


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()

    base_cfg = OmegaConf.load(args.base_config)
    meta_cfg  = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(base_cfg, meta_cfg)

    if args.tasks is not None:
        cfg.data.tasks = args.tasks
    if args.max_steps is not None:
        cfg.training.max_steps = args.max_steps
    if args.per_device_batch_size is not None:
        cfg.training.per_device_train_batch_size = args.per_device_batch_size

    condition_name = args.condition_name or "xtreme_meta"
    output_dir = Path(args.output_dir or cfg.training.output_dir) / condition_name
    output_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        mixed_precision="bf16" if cfg.training.bf16 else "no",
        log_with=None,
    )

    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir=str(output_dir / "tb_logs"))
        print(f"\nXTREME meta-distillation (bilevel λ per task)")
        print(f"  Model:    {cfg.model.name}")
        print(f"  K logits= {cfg.distillation.n_top_logits}  λ_max= {cfg.distillation.lambda_max}")
        print(f"  meta_lr=  {cfg.distillation.meta_lr}  update_every= {cfg.distillation.meta_update_every}")
        print(f"  Output:   {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------
    # Frozen teacher
    # ------------------------------------------------------------------
    teacher = AutoModelForCausalLM.from_pretrained(
        cfg.model.name, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    teacher = teacher.to(accelerator.device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # ------------------------------------------------------------------
    # Dataloaders
    # ------------------------------------------------------------------
    tasks = list(cfg.data.tasks)
    train_langs = list(cfg.data.get("eval_langs", cfg.data.get("train_langs", ["en"])))

    train_loader = make_xtreme_dataloader(
        tokenizer=tokenizer,
        tasks=tasks,
        train_langs=train_langs,
        batch_size=cfg.training.per_device_train_batch_size,
        max_samples_per_task_lang=cfg.data.get("max_samples_per_task", 5000),
        max_seq_len_teacher=cfg.data.get("max_seq_len_teacher", 2048),
        max_seq_len_student=cfg.data.get("max_seq_len_student", 512),
        shuffle=True,
        num_workers=cfg.data.get("num_workers", 4),
        seed=cfg.training.seed,
        teacher_include_answer=True,
        zeroshot_teacher=False,
        split="train",
    )

    # Val loader: dev split, smaller batch, all available languages
    # Used for the outer (meta) objective — CE only, no teacher needed
    val_langs = list(cfg.data.get("eval_langs", ["en", "hi", "es", "de", "fr", "zh"]))
    val_loader = make_xtreme_dataloader(
        tokenizer=tokenizer,
        tasks=tasks,
        train_langs=val_langs,
        batch_size=cfg.distillation.get("meta_val_batch_size", 16),
        max_samples_per_task_lang=cfg.distillation.get("meta_val_samples", 500),
        max_seq_len_teacher=cfg.data.get("max_seq_len_teacher", 2048),
        max_seq_len_student=cfg.data.get("max_seq_len_student", 512),
        shuffle=True,
        num_workers=2,
        seed=cfg.training.seed + 1000,
        teacher_include_answer=True,   # format is same; we only use student-side
        zeroshot_teacher=False,
        split="dev",
    )

    # ------------------------------------------------------------------
    # Student model
    # ------------------------------------------------------------------
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

    if cfg.training.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    # ------------------------------------------------------------------
    # Per-task λ (learnable, logit-space)
    # ------------------------------------------------------------------
    n_tasks = len(tasks)
    task_name_to_id = {t: i for i, t in enumerate(tasks)}

    # logit_lambda[t]: sigmoid(logit_lambda[t]) * lambda_max = λ_t
    # init: sigmoid(0) = 0.5  → λ_t = 0.5 * lambda_max
    logit_lambda = torch.nn.Parameter(
        torch.zeros(n_tasks, dtype=torch.float32)
    )
    lambda_max = cfg.distillation.get("lambda_max", 1.0)
    meta_lr = cfg.distillation.meta_lr
    meta_update_every = cfg.distillation.meta_update_every

    meta_optimizer = AdamW([logit_lambda], lr=meta_lr, weight_decay=0.0)

    # ------------------------------------------------------------------
    # Inner optimiser & scheduler
    # ------------------------------------------------------------------
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

    # trainable_params: references to LoRA leaf tensors (used for autograd.grad)
    trainable_params = [
        p for p in accelerator.unwrap_model(model).parameters() if p.requires_grad
    ]

    K_vocab = cfg.distillation.n_top_logits

    # ------------------------------------------------------------------
    # Training state
    # ------------------------------------------------------------------
    model.train()
    step = 0
    train_iter = iter(train_loader)
    val_iter   = iter(val_loader)
    ce_loss_accum   = 0.0
    dist_loss_accum = 0.0
    alignment_verified = False

    # Cache for meta update (populated in inner loop, consumed in meta step)
    saved_batch     = None
    saved_t_top     = None
    saved_top_idx   = None
    saved_b_idx     = None
    saved_s_ans_idx = None
    saved_ans_valid = None

    progress = tqdm(
        total=cfg.training.max_steps, desc=condition_name,
        disable=not accelerator.is_main_process,
    )

    if accelerator.is_main_process:
        task_names = tasks
        lam_init = get_lambda(logit_lambda, lambda_max)
        print(f"\n  Initial λ: " + "  ".join(
            f"{t}={lam_init[i].item():.3f}" for i, t in enumerate(task_names)
        ))

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    while step < cfg.training.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        with accelerator.accumulate(model):
            device = accelerator.device
            t_input_ids  = batch.teacher_input_ids.to(device)
            t_attn_mask  = batch.teacher_attention_mask.to(device)
            s_input_ids  = batch.student_input_ids.to(device)
            s_attn_mask  = batch.student_attention_mask.to(device)
            labels_d     = batch.labels.to(device)
            t_lens = t_attn_mask.sum(dim=1)

            with torch.no_grad():
                teacher_out = teacher(input_ids=t_input_ids, attention_mask=t_attn_mask)

            student_out = model(
                input_ids=s_input_ids,
                attention_mask=s_attn_mask,
                labels=labels_d,
            )
            ce_loss = student_out.loss

            b_idx, t_ans_idx, s_ans_idx, ans_valid, n_ans = answer_alignment(
                t_lens, labels_d, device
            )

            # Alignment check on first batch
            if accelerator.is_main_process and not alignment_verified:
                b0_n = n_ans[0].item()
                if b0_n > 0:
                    t_ids = t_input_ids[0, t_ans_idx[0, :b0_n]]
                    s_ids = s_input_ids[0, s_ans_idx[0, :b0_n]]
                    match = (t_ids == s_ids).all().item()
                    print(f"\n{'✓' if match else '✗'} Token alignment: "
                          f"{'match' if match else 'MISMATCH'} ({b0_n} tokens)")
                alignment_verified = True

            t_logits_ans = teacher_out.logits[b_idx, t_ans_idx]
            s_logits_ans = student_out.logits[b_idx, s_ans_idx]
            del teacher_out

            _, top_idx = t_logits_ans.topk(K_vocab, dim=-1)
            t_top = t_logits_ans.gather(-1, top_idx)
            s_top = s_logits_ans.gather(-1, top_idx)

            # Per-task λ weighting
            task_ids_d = batch.task_ids.to(device)
            lam_vals = get_lambda(logit_lambda.to(device), lambda_max)
            lam_batch = lam_vals[task_ids_d]          # (B,)

            # Per-example dist loss, weighted by λ_t
            mask = ans_valid.unsqueeze(-1).float()
            dist_per_ex = (
                ((t_top - s_top).pow(2) * mask).sum(dim=(1, 2))
                / (mask.sum(dim=(1, 2)) * K_vocab + 1e-8)
            )                                           # (B,)
            weighted_dist = (lam_batch * dist_per_ex).mean()
            avg_dist = dist_per_ex.mean()

            total_loss = ce_loss + weighted_dist

            # Save tensors for meta update (every meta_update_every steps)
            if accelerator.sync_gradients and (step + 1) % meta_update_every == 0:
                saved_batch     = batch
                saved_t_top     = t_top.detach().cpu()
                saved_top_idx   = top_idx.detach().cpu()
                saved_b_idx     = b_idx.detach().cpu()
                saved_s_ans_idx = s_ans_idx.detach().cpu()
                saved_ans_valid = ans_valid.detach().cpu()

            del student_out, t_logits_ans, s_logits_ans, s_top

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
                ce_loss_accum   += ce_loss.item()
                dist_loss_accum += avg_dist.item()

                # ----------------------------------------------------------
                # Meta (outer) update
                # ----------------------------------------------------------
                if step % meta_update_every == 0 and saved_batch is not None:
                    # Get val batch (cycle if exhausted)
                    try:
                        val_batch = next(val_iter)
                    except StopIteration:
                        val_iter = iter(val_loader)
                        val_batch = next(val_iter)

                    dot_by_task, val_ce = meta_update(
                        model=model,
                        val_batch=val_batch,
                        saved_batch=saved_batch,
                        saved_t_top=saved_t_top.to(device),
                        saved_top_idx=saved_top_idx.to(device),
                        saved_b_idx=saved_b_idx.to(device),
                        saved_s_ans_idx=saved_s_ans_idx.to(device),
                        saved_ans_valid=saved_ans_valid.to(device),
                        logit_lambda=logit_lambda,
                        meta_optimizer=meta_optimizer,
                        trainable_params=trainable_params,
                        K_vocab=K_vocab,
                        n_tasks=n_tasks,
                        device=device,
                        accelerator=accelerator,
                        inner_lr=cfg.training.lr,
                    )

                    if accelerator.is_main_process:
                        lam_now = get_lambda(logit_lambda.detach(), lambda_max)
                        for i, t in enumerate(tasks):
                            writer.add_scalar(f"meta/lambda_{t}", lam_now[i].item(), step)
                            writer.add_scalar(f"meta/dot_{t}",
                                              dot_by_task.get(i, 0.0), step)
                        writer.add_scalar("meta/val_ce", val_ce, step)

                # ----------------------------------------------------------
                # Logging
                # ----------------------------------------------------------
                if step % cfg.training.logging_steps == 0:
                    avg_ce   = ce_loss_accum   / cfg.training.logging_steps
                    avg_d    = dist_loss_accum / cfg.training.logging_steps
                    if accelerator.is_main_process:
                        lam_now = get_lambda(logit_lambda.detach(), lambda_max)
                        lam_str = " ".join(
                            f"{t}={lam_now[i].item():.3f}" for i, t in enumerate(tasks)
                        )
                        writer.add_scalar("train/ce_loss",    avg_ce, step)
                        writer.add_scalar("train/dist_loss",  avg_d,  step)
                        writer.add_scalar("train/lr",         scheduler.get_last_lr()[0], step)
                        progress.set_postfix(
                            ce=f"{avg_ce:.4f}", dist=f"{avg_d:.4f}",
                            lam=lam_str, step=step,
                        )
                    ce_loss_accum   = 0.0
                    dist_loss_accum = 0.0

                progress.update(1)
                if step >= cfg.training.max_steps:
                    break

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    if accelerator.is_main_process:
        final_dir = output_dir / "final"
        accelerator.unwrap_model(model).save_pretrained(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        writer.close()

        # Save learned λ values
        lam_final = get_lambda(logit_lambda.detach(), lambda_max)
        lambda_record = {
            t: lam_final[i].item() for i, t in enumerate(tasks)
        }
        print(f"\n✓ Meta-distillation done. Final λ per task:")
        for t, lv in lambda_record.items():
            print(f"    {t}: {lv:.4f}")

        with open(output_dir / "train_config.json", "w") as f:
            json.dump(OmegaConf.to_container(cfg), f, indent=2)
        with open(output_dir / "lambda_final.json", "w") as f:
            json.dump(lambda_record, f, indent=2)

        print(f"✓ Model saved to {final_dir}")

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
