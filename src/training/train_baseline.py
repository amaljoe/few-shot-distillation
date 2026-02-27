"""
LoRA SFT baseline: standard zero-shot fine-tuning on GSM8K.

Pure cross-entropy loss on student inputs. No distillation.

Run command (example — 2 GPUs, Qwen3-1.7B):
  CUDA_VISIBLE_DEVICES=0,1 accelerate launch \\
      --num_processes 2 --mixed_precision bf16 --main_process_port 29500 \\
      src/training/train_baseline.py --config configs/base.yaml \\
      --output_dir experiments/qwen

Run command (example — 2 GPUs, Llama-3.2-3B-Instruct):
  CUDA_VISIBLE_DEVICES=0,1 accelerate launch \\
      --num_processes 2 --mixed_precision bf16 --main_process_port 29500 \\
      src/training/train_baseline.py --config configs/llama3b.yaml \\
      --output_dir experiments/llama3b
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
from src.data.gsm8k_loader import load_gsm8k, make_dataloader
from src.models.student import StudentModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output_dir from config")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    output_dir = Path(args.output_dir or cfg.training.output_dir) / "baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        mixed_precision="bf16" if cfg.training.bf16 else "no",
        log_with=None,
    )

    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir=str(output_dir / "tb_logs"))
        print(f"\nCondition B: Zero-shot baseline fine-tuning")
        print(f"Output: {output_dir}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset
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

    # Student model (no activation capture needed for baseline)
    use_lora = getattr(cfg.training, "use_lora", True)
    lora_cfg = OmegaConf.to_container(cfg.lora) if (use_lora and hasattr(cfg, "lora")) else None
    student_wrapper = StudentModel(
        model_name=cfg.model.name,
        lora_config=lora_cfg,
        use_lora=use_lora,
        num_layers=cfg.model.num_layers,
        device_map=None,  # accelerate handles device placement
    )
    model = student_wrapper.get_model()

    if cfg.training.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()  # required for PEFT+grad-ckpt

    # Optimizer and scheduler
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

    # Accelerate preparation
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    # Training loop
    model.train()
    step = 0
    train_iter = iter(train_loader)
    loss_accum = 0.0

    progress = tqdm(total=cfg.training.max_steps, desc="Baseline training",
                    disable=not accelerator.is_main_process)

    while step < cfg.training.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        with accelerator.accumulate(model):
            # Student uses full sequence (question + answer) as input
            # Labels have -100 on prompt tokens, answer tokens are supervised
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

                if step % cfg.training.save_steps == 0 and accelerator.is_main_process:
                    ckpt_dir = output_dir / f"checkpoint-{step}"
                    accelerator.unwrap_model(model).save_pretrained(str(ckpt_dir))
                    tokenizer.save_pretrained(str(ckpt_dir))

                progress.update(1)
                if step >= cfg.training.max_steps:
                    break

    # Save final checkpoint
    if accelerator.is_main_process:
        final_dir = output_dir / "final"
        accelerator.unwrap_model(model).save_pretrained(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        writer.close()
        print(f"\n✓ Baseline training done. Model saved to {final_dir}")

        # Save training config
        with open(output_dir / "train_config.json", "w") as f:
            json.dump(OmegaConf.to_container(cfg), f, indent=2)

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
