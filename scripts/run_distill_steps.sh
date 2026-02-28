#!/usr/bin/env bash
# Distillation duration ablation.
# Run from project root inside the apptainer container (tmux session: claude).
# Assumes: conda activate /dev/shm/vllm
#
# Usage:
#   cd /path/to/icl-distillation
#   bash scripts/run_distill_steps.sh

set -e

echo "================================================================"
echo " Distillation duration ablation"
echo " Model      : Qwen/Qwen3-1.7B"
echo " Conditions : distill_steps in {0, 4, 16, 64, 200}"
echo " Total steps: 200 each  |  LoRA r=16"
echo "================================================================"
echo ""

# ── Train all 5 conditions sequentially ───────────────────────────────────────
for STEPS in 0 4 16 64 200; do
    echo ">>> Training distill_steps=${STEPS} on GPUs 0-3 …"
    CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
        --num_processes 4 --mixed_precision bf16 --main_process_port 29500 \
        scripts/distill_steps_experiment.py --distill_steps ${STEPS}
    echo ""
    echo ">>> Done: distill_steps=${STEPS}"
    echo ""
done

echo "================================================================"
echo " All training done.  Running evaluation + plotting …"
echo "================================================================"
echo ""

# ── Evaluate all 5 checkpoints + generate plot ────────────────────────────────
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/eval_plot_distill_steps.py

echo ""
echo "================================================================"
echo " All done!  Outputs:"
echo "   experiments/distill_steps/eval_results.json"
echo "   experiments/distill_steps/distill_steps_accuracy.png"
echo "================================================================"
