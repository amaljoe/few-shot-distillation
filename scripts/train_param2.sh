#!/bin/bash
# Param2-17B-A2.4B-Thinking training sequence:
#   baseline (GPU 0,1) + distillation (GPU 2,3) in parallel,
#   then ablation (GPU 0,1), then checkpoint eval (HF fallback, all 4 GPUs).
#
# Run AFTER Param2 ICL eval finishes (GPU 1 freed).
# Usage: bash scripts/train_param2.sh

cd ~/workspace/icl-distillation
exec > >(tee -a experiments/logs/param2_train_run.log) 2>&1

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*"; }

log "=== Param2-17B experiments START ==="

# ── Parallel: baseline (GPU 0,1) + distillation (GPU 2,3) ──────────────────
log "Starting baseline (GPU 0,1) and distillation (GPU 2,3) in parallel"

CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --num_processes 2 --mixed_precision bf16 --main_process_port 29500 \
    src/training/train_baseline.py \
    --config configs/param2_17b.yaml \
    --output_dir experiments/param2_17b/baseline \
    > experiments/logs/param2_baseline.log 2>&1 &
BL=$!

CUDA_VISIBLE_DEVICES=2,3 accelerate launch \
    --num_processes 2 --mixed_precision bf16 --main_process_port 29502 \
    src/training/train_online_v1.py \
    --base_config configs/param2_17b.yaml \
    --config configs/online_v1_param2.yaml \
    --output_dir experiments/param2_17b/online_v1 \
    > experiments/logs/param2_distill.log 2>&1 &
DL=$!

log "Waiting for baseline (pid=$BL) and distillation (pid=$DL)..."
wait $BL && log "baseline done" || log "WARN: baseline failed"
wait $DL && log "distillation done" || log "WARN: distillation failed"

# ── Ablation: 0-shot teacher control (GPU 0,1) ─────────────────────────────
log "Starting 0-shot teacher ablation (GPU 0,1)"
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --num_processes 2 --mixed_precision bf16 --main_process_port 29500 \
    src/training/train_ablation.py \
    --base_config configs/param2_17b.yaml \
    --config configs/ablation_zeroshot_teacher.yaml \
    --condition_name zeroshot_teacher \
    --output_dir experiments/param2_17b \
    > experiments/logs/param2_ablation.log 2>&1 \
    && log "ablation done" || log "WARN: ablation failed"

# ── Checkpoint eval (HF fallback — vLLM doesn't support Param2MoE) ──────────
log "Starting checkpoint eval via HF (all 4 GPUs, 1319 samples)"
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/eval_hf_checkpoints.py \
    --config configs/param2_17b.yaml \
    --base_dir experiments/param2_17b \
    --conditions baseline online_v1 zeroshot_teacher \
    --n_samples 1319 \
    --checkpoint_steps 200 400 600 800 1000 \
    --output experiments/param2_17b_eval.json \
    --max_new_tokens 512 --batch_size 4 \
    > experiments/logs/param2_eval.log 2>&1 \
    && log "checkpoint eval done" || log "WARN: checkpoint eval failed"

log "=== Param2-17B experiments DONE ==="
