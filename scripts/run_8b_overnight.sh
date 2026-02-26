#!/usr/bin/env bash
# =============================================================================
# Overnight experiment: Qwen3-8B baseline vs distillation
#
# Runs all steps end-to-end:
#   Step 1: Precompute 8B teacher activations  (all 4 GPUs, ~30 min)
#   Step 2: Train B early + C early in parallel (GPUs 0,1 / 2,3, ~15 min)
#   Step 3: Eval early checkpoints              (all 4 GPUs, ~20 min)
#   Step 4: Train B full + C full in parallel   (GPUs 0,1 / 2,3, ~2 hr)
#   Step 5: Eval full checkpoints               (all 4 GPUs, ~30 min)
#   Step 6: Generate figures
#
# Run from repo root inside apptainer (tmux: claude):
#   bash scripts/run_8b_overnight.sh 2>&1 | tee experiments/8b/overnight.log
# =============================================================================
set -e
cd "$(dirname "$0")/.."

EARLY_BASE_DIR="experiments/8b/early_checkpoints"
FULL_BASE_DIR="experiments/8b"
EVAL_OUT="experiments/ablations_8b/checkpoint_curve"
mkdir -p "$EARLY_BASE_DIR" "$EVAL_OUT" experiments/figures

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Precompute teacher activations
# ─────────────────────────────────────────────────────────────────────────────
CACHE="experiments/8b/teacher_cache/activations.pt"
if [ -f "$CACHE" ]; then
    log "Step 1: Teacher cache found at $CACHE — skipping."
else
    log "Step 1: Precomputing Qwen3-8B teacher activations (all 4 GPUs)..."
    CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/precompute_teacher_activations.py \
        --config configs/base_8b.yaml \
        --batch_size 8
    log "Step 1: Done."
fi

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Train B early (GPUs 0,1) + C early (GPUs 2,3) in parallel
# ─────────────────────────────────────────────────────────────────────────────
B_EARLY_DONE="$EARLY_BASE_DIR/baseline/baseline/final"
C_EARLY_DONE="$EARLY_BASE_DIR/distill/distill/final"

if [ -d "$B_EARLY_DONE" ] && [ -d "$C_EARLY_DONE" ]; then
    log "Step 2: Early checkpoints already exist — skipping."
else
    log "Step 2: Training early checkpoints (0-100 steps, parallel)..."

    CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
        --num_processes 2 --mixed_precision bf16 --main_process_port 29500 \
        src/training/train_baseline.py \
        --config configs/early_ckpt_8b.yaml \
        --output_dir "$EARLY_BASE_DIR/baseline" &
    PID_B=$!

    CUDA_VISIBLE_DEVICES=2,3 accelerate launch \
        --num_processes 2 --mixed_precision bf16 --main_process_port 29501 \
        src/training/train_layerwise_distill.py \
        --config configs/distill_early_ckpt_8b.yaml \
        --output_dir "$EARLY_BASE_DIR/distill" &
    PID_C=$!

    wait $PID_B && log "Step 2a: Condition B early done."
    wait $PID_C && log "Step 2b: Condition C early done."
    log "Step 2: Both early runs complete."
fi

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Evaluate early checkpoints (steps 10-100) on full test set
# ─────────────────────────────────────────────────────────────────────────────
EARLY_RESULTS="$EVAL_OUT/results_early.json"
if [ -f "$EARLY_RESULTS" ]; then
    log "Step 3: Early eval results found — skipping."
else
    log "Step 3: Evaluating early checkpoints on 1319 examples..."
    CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/eval_checkpoints.py \
        --config configs/base_8b.yaml \
        --n_samples 1319 \
        --tensor_parallel_size 4 \
        --max_model_len 4096 \
        --checkpoint_steps 10 20 30 40 50 60 70 80 90 100 \
        --base_dir "$EARLY_BASE_DIR" \
        --output "$EARLY_RESULTS"
    log "Step 3: Done."
fi

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Train B full (GPUs 0,1) + C full (GPUs 2,3) in parallel
# ─────────────────────────────────────────────────────────────────────────────
B_FULL_DONE="$FULL_BASE_DIR/baseline/final"
C_FULL_DONE="$FULL_BASE_DIR/distill/final"

if [ -d "$B_FULL_DONE" ] && [ -d "$C_FULL_DONE" ]; then
    log "Step 4: Full training already done — skipping."
else
    log "Step 4: Full training (1000 steps, parallel)..."

    CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
        --num_processes 2 --mixed_precision bf16 --main_process_port 29500 \
        src/training/train_baseline.py \
        --config configs/base_8b.yaml \
        --output_dir "$FULL_BASE_DIR/baseline" &
    PID_B=$!

    CUDA_VISIBLE_DEVICES=2,3 accelerate launch \
        --num_processes 2 --mixed_precision bf16 --main_process_port 29501 \
        src/training/train_layerwise_distill.py \
        --config configs/distill_8b.yaml \
        --output_dir "$FULL_BASE_DIR/distill" &
    PID_C=$!

    wait $PID_B && log "Step 4a: Condition B full training done."
    wait $PID_C && log "Step 4b: Condition C full training done."
    log "Step 4: Both full runs complete."
fi

# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Evaluate full checkpoints (steps 100-1000) on full test set
# ─────────────────────────────────────────────────────────────────────────────
FULL_RESULTS="$EVAL_OUT/results_full.json"
if [ -f "$FULL_RESULTS" ]; then
    log "Step 5: Full eval results found — skipping."
else
    log "Step 5: Evaluating full checkpoints on 1319 examples..."
    CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/eval_checkpoints.py \
        --config configs/base_8b.yaml \
        --n_samples 1319 \
        --tensor_parallel_size 4 \
        --max_model_len 4096 \
        --checkpoint_steps 100 200 300 400 500 600 700 800 900 1000 \
        --base_dir "$FULL_BASE_DIR" \
        --output "$FULL_RESULTS"
    log "Step 5: Done."
fi

# ─────────────────────────────────────────────────────────────────────────────
# Step 6: Generate figures
# ─────────────────────────────────────────────────────────────────────────────
log "Step 6: Generating figures..."
python scripts/plot_curves.py \
    --model_tag 8b \
    --early_results "$EARLY_RESULTS" \
    --full_results "$FULL_RESULTS" \
    --early_tb_baseline "$EARLY_BASE_DIR/baseline/baseline/tb_logs" \
    --early_tb_distill  "$EARLY_BASE_DIR/distill/distill/tb_logs" \
    --full_tb_baseline  "$FULL_BASE_DIR/baseline/baseline/tb_logs" \
    --full_tb_distill   "$FULL_BASE_DIR/distill/distill/tb_logs" \
    --out_dir experiments/figures/8b
log "Step 6: Done."

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
log "=============================="
log "All overnight experiments done."
log "Results:"
log "  Early: $EARLY_RESULTS"
log "  Full:  $FULL_RESULTS"
log "  Figures: experiments/figures/8b/"
log "=============================="
