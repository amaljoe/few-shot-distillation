"""
Condition C: Layer-wise distillation fine-tuning on GSM8K.

Combines CE task loss with layer-matching distillation loss:
  L_total = L_ce + λ * Σ_l || h_l(student) - h_l(teacher) ||²

Teacher activations are loaded from the precomputed cache
(run scripts/precompute_teacher_activations.py first).

Run command (tmux: vscode on cn14-dgx, in parallel with baseline):
  accelerate launch --num_processes 4 --mixed_precision bf16 \
      src/training/train_layerwise_distill.py \
      --config configs/distill_layerwise.yaml \
      --output_dir experiments/poc/distill
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
from src.losses.layer_matching import cosine_similarity_per_layer, layer_matching_loss
from src.models.student import StudentModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/distill_layerwise.yaml")
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


def load_teacher_cache(cache_path: str, device: str = "cpu") -> torch.Tensor:
    """
    Load precomputed teacher activations into CPU memory.

    Shape: (N, num_layers, hidden_size) float16
    We keep on CPU and move per-batch to GPU to avoid OOM.
    """
    path = Path(cache_path)
    assert path.exists(), (
        f"Teacher cache not found at {path}. "
        f"Run scripts/precompute_teacher_activations.py first."
    )
    print(f"Loading teacher cache from {path}...")
    cache = torch.load(str(path), map_location=device)
    print(f"  Cache shape: {cache.shape} | dtype: {cache.dtype} | "
          f"size: {cache.nbytes / 1e9:.2f} GB")
    return cache  # stays on CPU


def main():
    args = parse_args()

    # Load base config then overlay distillation config
    base_cfg = OmegaConf.load("configs/base.yaml")
    dist_cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(base_cfg, dist_cfg)

    output_dir = Path(args.output_dir or cfg.training.output_dir) / "distill"
    output_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        mixed_precision="bf16" if cfg.training.bf16 else "no",
        log_with=None,
    )

    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir=str(output_dir / "tb_logs"))
        print(f"\nCondition C: Layer-wise distillation fine-tuning")
        print(f"  λ = {cfg.distillation.lambda_distill}")
        print(f"  Layers: {cfg.distillation.layers_to_match}")
        print(f"  Normalize: {cfg.distillation.normalize_hidden}")
        print(f"  Output: {output_dir}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Teacher cache (CPU, moved to GPU per batch)
    teacher_cache = load_teacher_cache(
        Path(cfg.teacher_activations.cache_dir) / "activations.pt"
    )
    # teacher_cache: (N, L, H) float16

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

    # Determine which layers to match
    if cfg.distillation.layers_to_match == "all":
        layers_to_match = "all"
    else:
        layers_to_match = list(cfg.distillation.layers_to_match)

    # Student with activation capture enabled
    layer_indices = list(range(cfg.model.num_layers))
    student_wrapper = StudentModel(
        model_name=cfg.model.name,
        lora_config=OmegaConf.to_container(cfg.lora),
        num_layers=cfg.model.num_layers,
        layer_indices=layer_indices,
        device_map=None,  # accelerate handles device placement
    )
    student_wrapper.enable_capture()
    model = student_wrapper.get_model()

    if cfg.training.gradient_checkpointing:
        model.gradient_checkpointing_enable()

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

    # Accelerate preparation (teacher cache stays outside accelerate)
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    # Training loop
    model.train()
    step = 0
    train_iter = iter(train_loader)
    ce_loss_accum = 0.0
    dist_loss_accum = 0.0

    progress = tqdm(total=cfg.training.max_steps, desc="Distill training",
                    disable=not accelerator.is_main_process)

    while step < cfg.training.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        with accelerator.accumulate(model):
            student_wrapper.clear_capture()

            # Student forward pass (captures hidden states via hooks)
            outputs = model(
                input_ids=batch.student_input_ids,
                attention_mask=batch.student_attention_mask,
                labels=batch.labels,
            )
            ce_loss = outputs.loss

            # Fetch teacher activations for this batch from cache
            # example_idx: (B,) → index into teacher_cache (N, L, H)
            example_idx = batch.example_idx
            # teacher_states: (B, L, H) float16 → permute to (L, B, H)
            t_states = teacher_cache[example_idx].permute(1, 0, 2)
            t_states = t_states.to(accelerator.device)  # move to GPU

            # Student alignment: last token of question prompt (before answer generation)
            # Uses pre-computed student_query_pos from the batch (same structure as teacher)
            student_query_pos = batch.student_query_pos.to(accelerator.device)

            # Get student hidden states at alignment positions
            s_states = student_wrapper.get_hidden_states(student_query_pos)  # (L, B, H)

            # Distillation loss
            dist_loss = layer_matching_loss(
                student_states=s_states,
                teacher_states=t_states,
                layers_to_match=layers_to_match,
                normalize=cfg.distillation.normalize_hidden,
            )

            total_loss = ce_loss + cfg.distillation.lambda_distill * dist_loss
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
                        writer.add_scalar("train/total_loss",
                                          avg_ce + cfg.distillation.lambda_distill * avg_dist,
                                          step)
                        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)

                        # Log per-layer cosine similarity (every 50 steps to avoid overhead)
                        if step % 50 == 0:
                            with torch.no_grad():
                                cos_sims = cosine_similarity_per_layer(
                                    s_states.detach(), t_states
                                )
                            for li, sim in enumerate(cos_sims.tolist()):
                                writer.add_scalar(f"layer_sim/layer_{layer_indices[li]}", sim, step)
                            writer.add_scalar("layer_sim/mean", cos_sims.mean().item(), step)

                        progress.set_postfix(
                            ce=f"{avg_ce:.4f}",
                            dist=f"{avg_dist:.4f}",
                            step=step,
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

    # Save final checkpoint
    if accelerator.is_main_process:
        final_dir = output_dir / "final"
        accelerator.unwrap_model(model).save_pretrained(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        writer.close()
        print(f"\n✓ Distillation training done. Model saved to {final_dir}")

        with open(output_dir / "train_config.json", "w") as f:
            json.dump(OmegaConf.to_container(cfg), f, indent=2)

    student_wrapper.disable_capture()
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
