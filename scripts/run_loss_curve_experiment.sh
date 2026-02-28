#!/usr/bin/env bash
# Run from project root inside the apptainer container (tmux session: claude).
# Assumes: conda activate /dev/shm/vllm  (accelerate and python in PATH)
#
# Usage:
#   cd /path/to/icl-distillation
#   bash scripts/run_loss_curve_experiment.sh

set -e

echo "================================================================"
echo " Loss curve experiment: zero-shot vs few-shot dev loss"
echo " Model  : Qwen/Qwen3-1.7B"
echo " Steps  : 200  |  eval every 16 steps  |  dev size: 32"
echo "================================================================"
echo ""

# ── Experiment 1: SFT Baseline (CE only) ─────────────────────────────────────
echo ">>> [1/3] Training SFT Baseline (CE only) on GPUs 0-3 …"
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes 4 --mixed_precision bf16 --main_process_port 29500 \
    scripts/loss_curve_experiment.py --mode baseline

echo ""
echo ">>> Baseline done."
echo ""

# ── Experiment 2: Distilled SFT (CE + MSE top-K logits) ──────────────────────
echo ">>> [2/3] Training Distilled SFT (CE + MSE) on GPUs 0-3 …"
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes 4 --mixed_precision bf16 --main_process_port 29500 \
    scripts/loss_curve_experiment.py --mode distill

echo ""
echo ">>> Distillation done."
echo ""

# ── Plotting ──────────────────────────────────────────────────────────────────
echo ">>> [3/3] Generating plots …"
python scripts/plot_loss_curves.py

echo ""
echo "================================================================"
echo " All done!  Plots:"
echo "   experiments/loss_curve/loss_curves_baseline.png"
echo "   experiments/loss_curve/loss_curves_distill.png"
echo "================================================================"
