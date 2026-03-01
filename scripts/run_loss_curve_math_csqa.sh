#!/usr/bin/env bash
# Run loss curve experiments for MATH and CommonsenseQA datasets.
#
# Schedule (4 GPUs):
#   Round 1: math     baseline (GPUs 0,1) + math     distill (GPUs 2,3)  [parallel]
#   Round 2: csqa     baseline (GPUs 0,1) + csqa     distill (GPUs 2,3)  [parallel]
#
# Run from the project root inside the apptainer container:
#   bash scripts/run_loss_curve_math_csqa.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON=/dev/shm/vllm/bin/python
ACCELERATE=/dev/shm/vllm/bin/accelerate
LOG_DIR="$REPO/experiments/loss_curve/logs"
mkdir -p "$LOG_DIR"

run() {
    local dataset="$1" mode="$2" gpus="$3" port="$4"
    local log="$LOG_DIR/${dataset}_${mode}.log"
    echo "[$(date +%H:%M:%S)] Launching $dataset/$mode on GPUs $gpus (port $port) → $log"
    CUDA_VISIBLE_DEVICES=$gpus $ACCELERATE launch \
        --num_processes 2 --mixed_precision bf16 --main_process_port "$port" \
        "$REPO/scripts/loss_curve_experiment.py" \
        --mode "$mode" --dataset "$dataset" \
        --output_dir "$REPO/experiments/loss_curve" \
        > "$log" 2>&1
    echo "[$(date +%H:%M:%S)] Done: $dataset/$mode"
}

# ── Round 1: MATH ─────────────────────────────────────────────────────────────
echo "=== Round 1: MATH ==="
run math baseline 0,1 29500 &
PID_MATH_BASE=$!
run math distill   2,3 29501 &
PID_MATH_DIST=$!

wait $PID_MATH_BASE && echo "math/baseline OK" || echo "math/baseline FAILED"
wait $PID_MATH_DIST && echo "math/distill OK"  || echo "math/distill FAILED"

# ── Round 2: CommonsenseQA ────────────────────────────────────────────────────
echo "=== Round 2: CommonsenseQA ==="
run commonsenseqa baseline 0,1 29500 &
PID_CSQA_BASE=$!
run commonsenseqa distill   2,3 29501 &
PID_CSQA_DIST=$!

wait $PID_CSQA_BASE && echo "csqa/baseline OK" || echo "csqa/baseline FAILED"
wait $PID_CSQA_DIST && echo "csqa/distill OK"  || echo "csqa/distill FAILED"

echo ""
echo "=== All done. Results in experiments/loss_curve/ ==="
echo "Generate plots with:"
echo "  python scripts/plot_loss_curves.py --dataset math"
echo "  python scripts/plot_loss_curves.py --dataset commonsenseqa"
