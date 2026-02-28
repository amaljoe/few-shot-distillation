"""
Ablation training script for causal attribution controls.

Supports three teacher modes (via --condition_name):
  zeroshot_teacher   — same model, 0-shot context (no few-shot examples)
  shuffled_answers   — 8 exemplars but answers randomly mismatched to questions
  fewshot_teacher    — standard 8-shot teacher (same as train_online_v1.py)

Output directory structure matches eval_checkpoints.py expectations:
  {output_dir}/{condition_name}/{condition_name}/checkpoint-{step}

Run examples (inside apptainer, mamba env active):

  # 0-shot teacher (GPUs 0,1):
  CUDA_VISIBLE_DEVICES=0,1 /dev/shm/vllm/bin/accelerate launch \\
      --num_processes 2 --mixed_precision bf16 --main_process_port 29500 \\
      src/training/train_ablation.py \\
      --condition_name zeroshot_teacher \\
      --base_config configs/base.yaml --config configs/ablation_zeroshot_teacher.yaml \\
      --output_dir experiments/ablations

  # Shuffled answers (GPUs 2,3):
  CUDA_VISIBLE_DEVICES=2,3 /dev/shm/vllm/bin/accelerate launch \\
      --num_processes 2 --mixed_precision bf16 --main_process_port 29502 \\
      src/training/train_ablation.py \\
      --condition_name shuffled_answers \\
      --base_config configs/base.yaml --config configs/ablation_shuffled_answers.yaml \\
      --output_dir experiments/ablations
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
from src.data.loader_factory import load_dataset_split, make_dataloader
from src.models.student import StudentModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition_name", type=str, required=True,
                        help="Name for this ablation condition (e.g. zeroshot_teacher)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--base_config", type=str, default="configs/base.yaml")
    parser.add_argument("--output_dir", type=str, default="experiments/ablations")
    return parser.parse_args()


def answer_alignment(t_lens, labels, device):
    B = labels.shape[0]
    n_ans = (labels != -100).sum(dim=1)
    s_ans_start = (labels != -100).float().argmax(dim=1)
    t_ans_start = t_lens - n_ans

    K = int(n_ans.max().item())
    j = torch.arange(K, device=device).unsqueeze(0).expand(B, -1)

    t_ans_idx = (t_ans_start.unsqueeze(1) + j).clamp(0, t_lens.max().item() - 1)
    s_ans_idx = (s_ans_start.unsqueeze(1) + j).clamp(0, labels.shape[1] - 1)
    ans_valid = j < n_ans.unsqueeze(1)
    b_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, K)

    return b_idx, t_ans_idx, s_ans_idx, ans_valid, n_ans


def main():
    args = parse_args()

    base_cfg = OmegaConf.load(args.base_config)
    abl_cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(base_cfg, abl_cfg)

    # Save checkpoints into {output_dir}/{condition_name}/{condition_name}/
    # so eval_checkpoints.py (which uses base_dir/cond/cond/checkpoint-N) finds them.
    cond = args.condition_name
    output_dir = Path(args.output_dir) / cond / cond
    output_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        mixed_precision="bf16" if cfg.training.bf16 else "no",
        log_with=None,
    )

    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir=str(output_dir / "tb_logs"))
        print(f"\nAblation: {cond}")
        print(f"  Teacher fewshot  = {cfg.data.num_fewshot_examples}")
        print(f"  Shuffle answers  = {getattr(cfg.data, 'shuffle_fewshot_answers', False)}")
        print(f"  K (vocab)        = {cfg.distillation.n_top_logits}")
        print(f"  λ                = {cfg.distillation.lambda_distill}")
        print(f"  Output           : {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Frozen teacher
    teacher = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    teacher = teacher.to(accelerator.device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    shuffle_answers = getattr(cfg.data, "shuffle_fewshot_answers", False)

    dataset_name = getattr(cfg.data, "dataset", "gsm8k")
    train_data = load_dataset_split(dataset_name, cfg.data.train_split)
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
        shuffle_fewshot_answers=shuffle_answers,
        dataset_name=dataset_name,
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
        model.enable_input_require_grads()  # required for PEFT+grad-ckpt

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

    # Re-enable input require grads after DDP wrapping (DDP may reset hooks set before prepare).
    # This is required for gradient checkpointing + PEFT LoRA to work correctly.
    if cfg.training.gradient_checkpointing:
        accelerator.unwrap_model(model).enable_input_require_grads()

    K_vocab = cfg.distillation.n_top_logits
    lam = cfg.distillation.lambda_distill

    model.train()
    step = 0
    train_iter = iter(train_loader)
    ce_loss_accum = 0.0
    dist_loss_accum = 0.0
    alignment_verified = False

    progress = tqdm(total=cfg.training.max_steps, desc=f"Ablation [{cond}]",
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
            labels_d = batch.labels.to(device)
            t_lens = t_attn_mask.sum(dim=1)

            with torch.no_grad():
                teacher_out = teacher(
                    input_ids=t_input_ids,
                    attention_mask=t_attn_mask,
                )

            student_out = model(
                input_ids=batch.student_input_ids,
                attention_mask=batch.student_attention_mask,
                labels=batch.labels,
            )
            ce_loss = student_out.loss
            if ce_loss is None:
                # Fallback: compute CE loss manually (shouldn't happen, but guard against it)
                shift_logits = student_out.logits[..., :-1, :].contiguous()
                shift_labels = batch.labels[..., 1:].contiguous().to(device)
                ce_loss = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )

            b_idx, t_ans_idx, s_ans_idx, ans_valid, n_ans = answer_alignment(
                t_lens, labels_d, device
            )

            if accelerator.is_main_process and not alignment_verified:
                b0_n = n_ans[0].item()
                t_ids = t_input_ids[0, t_ans_idx[0, :b0_n]]
                s_ids = batch.student_input_ids.to(device)[0, s_ans_idx[0, :b0_n]]
                match = (t_ids == s_ids).all().item()
                print(f"\n{'✓' if match else '✗'} Token alignment: "
                      f"teacher/student answer tokens {'match' if match else 'MISMATCH'} "
                      f"({b0_n} tokens checked)")
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
        print(f"\n✓ Ablation [{cond}] done. Model saved to {final_dir}")
        with open(output_dir / "train_config.json", "w") as f:
            json.dump(OmegaConf.to_container(cfg), f, indent=2)

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
