"""
Multi-task SFT baseline for XTREME benchmark (CE loss only).

Trains on all 5 tasks (NLI, PA, QA, NER, POS) simultaneously using
English training data. Evaluates cross-lingual zero-shot transfer.

Run command (Qwen3-1.7B, GPUs 0,1):
  CUDA_VISIBLE_DEVICES=0,1 accelerate launch \\
      --num_processes 2 --mixed_precision bf16 --main_process_port 29500 \\
      src/training/train_xtreme_sft.py \\
      --config configs/xtreme_qwen1b7.yaml \\
      --output_dir experiments/xtreme/qwen1b7

Run command (Llama-3.2-3B, GPUs 2,3):
  CUDA_VISIBLE_DEVICES=2,3 accelerate launch \\
      --num_processes 2 --mixed_precision bf16 --main_process_port 29501 \\
      src/training/train_xtreme_sft.py \\
      --config configs/xtreme_llama3b.yaml \\
      --output_dir experiments/xtreme/llama3b
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from accelerate import Accelerator
from src.data.xtreme_loader import make_xtreme_dataloader, TASKS
from src.models.student import StudentModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/xtreme_qwen1b7.yaml")
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    output_dir = Path(args.output_dir or cfg.training.output_dir) / "xtreme_sft"
    output_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        mixed_precision="bf16" if cfg.training.bf16 else "no",
        log_with=None,
    )

    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir=str(output_dir / "tb_logs"))
        print(f"\nXTREME SFT baseline (CE only)")
        print(f"  Model:  {cfg.model.name}")
        print(f"  Tasks:  {list(cfg.data.tasks)}")
        print(f"  Output: {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tasks = list(cfg.data.tasks)
    # Multilingual training: use all eval langs; tasks without multilingual
    # training data (NLI→MultiNLI, QA→SQuAD) automatically fall back to English.
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
        teacher_include_answer=False,  # SFT: no distillation teacher
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

    model.train()
    step = 0
    train_iter = iter(train_loader)
    loss_accum = 0.0

    progress = tqdm(total=cfg.training.max_steps, desc="XTREME SFT",
                    disable=not accelerator.is_main_process)

    while step < cfg.training.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        with accelerator.accumulate(model):
            outputs = model(
                input_ids=batch.student_input_ids,
                attention_mask=batch.student_attention_mask,
                labels=batch.labels,
            )
            loss = outputs.loss
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(
                    student_wrapper.get_trainable_parameters(),
                    cfg.training.grad_clip,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                step += 1
                loss_accum += loss.item()

                if step % cfg.training.logging_steps == 0:
                    avg_loss = loss_accum / cfg.training.logging_steps
                    if accelerator.is_main_process:
                        writer.add_scalar("train/ce_loss", avg_loss, step)
                        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)
                        progress.set_postfix(loss=f"{avg_loss:.4f}", step=step)
                    loss_accum = 0.0

                # No intermediate checkpoints — single final save only

                progress.update(1)
                if step >= cfg.training.max_steps:
                    break

    if accelerator.is_main_process:
        final_dir = output_dir / "final"
        accelerator.unwrap_model(model).save_pretrained(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        writer.close()
        print(f"\n✓ XTREME SFT done. Model saved to {final_dir}")
        with open(output_dir / "train_config.json", "w") as f:
            json.dump(OmegaConf.to_container(cfg), f, indent=2)

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
