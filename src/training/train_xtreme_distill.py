"""
Multi-task few-shot logit distillation for XTREME benchmark.

Same framework as train_online_v1.py, extended to 5 XTREME tasks.

Teacher: frozen base model, sees few-shot English context + target input.
Student: LoRA adapter, sees zero-shot target input.
Loss: L_CE  +  λ * MSE(teacher top-K logits, student logits at answer positions)

With --zeroshot_teacher: teacher also sees zero-shot input (control condition).

Run command (distillation, Qwen3-1.7B, GPUs 0,1):
  CUDA_VISIBLE_DEVICES=0,1 accelerate launch \\
      --num_processes 2 --mixed_precision bf16 --main_process_port 29500 \\
      src/training/train_xtreme_distill.py \\
      --base_config configs/xtreme_qwen1b7.yaml \\
      --config configs/xtreme_distill.yaml \\
      --output_dir experiments/xtreme/qwen1b7

Run command (control / zero-shot teacher):
  ... same but add --zeroshot_teacher
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
from src.data.xtreme_loader import make_xtreme_dataloader, TASKS
from src.models.student import StudentModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/xtreme_distill.yaml")
    parser.add_argument("--base_config", type=str, default="configs/xtreme_qwen1b7.yaml")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--zeroshot_teacher", action="store_true",
                        help="Control condition: teacher uses zero-shot context")
    # Lambda sweep overrides
    parser.add_argument("--lambda_distill", type=float, default=None,
                        help="Override distillation.lambda_distill from config")
    parser.add_argument("--tasks", type=str, nargs="+", default=None,
                        help="Override data.tasks (e.g. --tasks pos ner)")
    parser.add_argument("--condition_name", type=str, default=None,
                        help="Override condition name used for output subdirectory")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Override training.max_steps from config")
    parser.add_argument("--per_device_batch_size", type=int, default=None,
                        help="Override training.per_device_train_batch_size from config")
    parser.add_argument("--distill_warmup_steps", type=int, default=None,
                        help="Only distill for this many steps, then switch to CE-only SFT")
    return parser.parse_args()


def answer_alignment(t_lens, labels, device):
    """
    Align answer token positions between teacher and student sequences.
    Identical logic to train_online_v1.py.
    """
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


def main():
    args = parse_args()

    base_cfg = OmegaConf.load(args.base_config)
    dist_cfg  = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(base_cfg, dist_cfg)

    # CLI overrides for sweep
    if args.lambda_distill is not None:
        cfg.distillation.lambda_distill = args.lambda_distill
    if args.tasks is not None:
        cfg.data.tasks = args.tasks
    if args.max_steps is not None:
        cfg.training.max_steps = args.max_steps
    if args.per_device_batch_size is not None:
        cfg.training.per_device_train_batch_size = args.per_device_batch_size

    if args.condition_name:
        condition_name = args.condition_name
    else:
        condition_name = "xtreme_control" if args.zeroshot_teacher else "xtreme_distill"
    output_dir = Path(args.output_dir or cfg.training.output_dir) / condition_name
    output_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        mixed_precision="bf16" if cfg.training.bf16 else "no",
        log_with=None,
    )

    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir=str(output_dir / "tb_logs"))
        mode = "zero-shot teacher (control)" if args.zeroshot_teacher else "few-shot teacher"
        print(f"\nXTREME distillation — {mode}")
        print(f"  Model:  {cfg.model.name}")
        print(f"  K logits = {cfg.distillation.n_top_logits}  λ = {cfg.distillation.lambda_distill}")
        print(f"  Output: {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Frozen teacher — same base model, loaded on each DDP process's GPU
    teacher = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    teacher = teacher.to(accelerator.device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

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
        zeroshot_teacher=args.zeroshot_teacher,
    )

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

    K_vocab = cfg.distillation.n_top_logits
    lam     = cfg.distillation.lambda_distill

    model.train()
    step = 0
    train_iter = iter(train_loader)
    ce_loss_accum   = 0.0
    dist_loss_accum = 0.0
    alignment_verified = False

    progress = tqdm(total=cfg.training.max_steps, desc=condition_name,
                    disable=not accelerator.is_main_process)

    while step < cfg.training.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        with accelerator.accumulate(model):
            device = accelerator.device
            t_input_ids = batch.teacher_input_ids.to(device)
            t_attn_mask = batch.teacher_attention_mask.to(device)
            labels_d    = batch.labels.to(device)
            t_lens = t_attn_mask.sum(dim=1)

            with torch.no_grad():
                teacher_out = teacher(
                    input_ids=t_input_ids,
                    attention_mask=t_attn_mask,
                )

            student_out = model(
                input_ids=batch.student_input_ids.to(device),
                attention_mask=batch.student_attention_mask.to(device),
                labels=labels_d,
            )
            ce_loss = student_out.loss

            b_idx, t_ans_idx, s_ans_idx, ans_valid, n_ans = answer_alignment(
                t_lens, labels_d, device
            )

            # Alignment sanity check on first batch
            if accelerator.is_main_process and not alignment_verified:
                b0_n = n_ans[0].item()
                if b0_n > 0:
                    t_ids = t_input_ids[0, t_ans_idx[0, :b0_n]]
                    s_ids = batch.student_input_ids.to(device)[0, s_ans_idx[0, :b0_n]]
                    match = (t_ids == s_ids).all().item()
                    print(f"\n{'✓' if match else '✗'} Token alignment: "
                          f"{'match' if match else 'MISMATCH'} ({b0_n} tokens)")
                alignment_verified = True

            t_logits_ans = teacher_out.logits[b_idx, t_ans_idx]
            s_logits_ans = student_out.logits[b_idx, s_ans_idx]
            del teacher_out, student_out

            _, top_idx = t_logits_ans.topk(K_vocab, dim=-1)
            t_top = t_logits_ans.gather(-1, top_idx)
            s_top = s_logits_ans.gather(-1, top_idx)
            del t_logits_ans, s_logits_ans

            mask = ans_valid.unsqueeze(-1).float()
            n_valid_elements = mask.sum() * K_vocab
            dist_loss = ((t_top - s_top).pow(2) * mask).sum() / (n_valid_elements + 1e-8)

            lam_eff = lam if (args.distill_warmup_steps is None or step < args.distill_warmup_steps) else 0.0
            total_loss = ce_loss + lam_eff * dist_loss
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
                dist_loss_accum += dist_loss.item()

                if step % cfg.training.logging_steps == 0:
                    avg_ce   = ce_loss_accum   / cfg.training.logging_steps
                    avg_dist = dist_loss_accum / cfg.training.logging_steps
                    if accelerator.is_main_process:
                        writer.add_scalar("train/ce_loss",    avg_ce,   step)
                        writer.add_scalar("train/dist_loss",  avg_dist, step)
                        writer.add_scalar("train/total_loss", avg_ce + lam * avg_dist, step)
                        writer.add_scalar("train/lr",         scheduler.get_last_lr()[0], step)
                        mode = "distill" if (args.distill_warmup_steps is None or step <= args.distill_warmup_steps) else "sft"
                        progress.set_postfix(ce=f"{avg_ce:.4f}", dist=f"{avg_dist:.4f}", mode=mode, step=step)
                    ce_loss_accum   = 0.0
                    dist_loss_accum = 0.0

                # No intermediate checkpoints — single final save only

                progress.update(1)
                if step >= cfg.training.max_steps:
                    break

    if accelerator.is_main_process:
        final_dir = output_dir / "final"
        accelerator.unwrap_model(model).save_pretrained(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        writer.close()
        print(f"\n✓ XTREME distillation done. Model saved to {final_dir}")
        with open(output_dir / "train_config.json", "w") as f:
            json.dump(OmegaConf.to_container(cfg), f, indent=2)

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
