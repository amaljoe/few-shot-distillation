#!/bin/bash
# Qwen3-8B training sequence: baseline + distillation (parallel, GPUs 0-3),
# then ablation, then checkpoint eval.
# Run in 'overnight' tmux session after ICL eval finishes.
# Usage: bash scripts/train_qwen8b.sh

cd ~/workspace/icl-distillation
exec > >(tee -a experiments/logs/qwen8b_train_run.log) 2>&1

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*"; }

log "=== Qwen3-8B experiments START ==="
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader

# ── Parallel: baseline (GPU 0,1) + distillation (GPU 2,3) ──────────────────
log "Starting baseline (GPU 0,1) and distillation (GPU 2,3) in parallel"

CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --num_processes 2 --mixed_precision bf16 --main_process_port 29500 \
    src/training/train_baseline.py \
    --config configs/qwen8b.yaml \
    --output_dir experiments/qwen8b/baseline \
    > experiments/logs/qwen8b_baseline.log 2>&1 &
BL=$!

CUDA_VISIBLE_DEVICES=2,3 accelerate launch \
    --num_processes 2 --mixed_precision bf16 --main_process_port 29502 \
    src/training/train_online_v1.py \
    --base_config configs/qwen8b.yaml \
    --config configs/online_v1_qwen8b.yaml \
    --output_dir experiments/qwen8b/online_v1 \
    > experiments/logs/qwen8b_distill.log 2>&1 &
DL=$!

log "Waiting for baseline (pid=$BL) and distillation (pid=$DL)..."
wait $BL && log "baseline done" || log "WARN: baseline failed"
wait $DL && log "distillation done" || log "WARN: distillation failed"

# ── Ablation: 0-shot teacher control (GPU 0,1) ─────────────────────────────
log "Starting 0-shot teacher ablation (GPU 0,1)"
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --num_processes 2 --mixed_precision bf16 --main_process_port 29500 \
    src/training/train_ablation.py \
    --base_config configs/qwen8b.yaml \
    --config configs/ablation_zeroshot_teacher.yaml \
    --condition_name zeroshot_teacher \
    --output_dir experiments/qwen8b \
    > experiments/logs/qwen8b_ablation.log 2>&1 \
    && log "ablation done" || log "WARN: ablation failed"

# ── Checkpoint eval (all 4 GPUs) ────────────────────────────────────────────
log "Starting checkpoint eval (GPU 0,1,2,3, 1319 samples)"
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/eval_checkpoints.py \
    --config configs/qwen8b.yaml \
    --base_dir experiments/qwen8b \
    --conditions baseline online_v1 zeroshot_teacher \
    --n_samples 1319 \
    --checkpoint_steps 200 400 600 800 1000 \
    --output experiments/qwen8b_eval.json \
    --tensor_parallel_size 4 \
    --max_model_len 2048 \
    > experiments/logs/qwen8b_eval.log 2>&1 \
    && log "checkpoint eval done" || log "WARN: checkpoint eval failed"

log "=== Qwen3-8B experiments DONE ==="
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader
