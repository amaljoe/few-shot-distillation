#!/usr/bin/env bash
# Distillation timing ablation (END variant) for MATH and CommonsenseQA.
# Runs both datasets sequentially: train 5 conditions each, then eval+plot.
#
# Run from project root inside the apptainer container (tmux session: claude).
# Assumes: conda activate /dev/shm/vllm
#
# Usage:
#   cd /path/to/icl-distillation
#   bash scripts/run_distill_steps_end_math_csqa.sh

set -e

DISTILL_STEPS="0 4 16 64 200"

# ══════════════════════════════════════════════════════════════════════════════
echo "================================================================"
echo " Dataset 1/2: MATH"
echo " Conditions : last N steps distilled, N in {${DISTILL_STEPS}}"
echo " Total steps: 200 each  |  LoRA r=16"
echo "================================================================"
echo ""

for STEPS in $DISTILL_STEPS; do
    DISTILL_START=$((200 - STEPS))
    echo ">>> [MATH] distill steps ${DISTILL_START}–199  (last ${STEPS} steps)  on GPUs 0-3 …"
    CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
        --num_processes 4 --mixed_precision bf16 --main_process_port 29500 \
        scripts/distill_steps_end_experiment.py --distill_steps ${STEPS} --dataset math
    echo ""
    echo ">>> [MATH] Done: distill_steps=${STEPS}"
    echo ""
done

echo "----------------------------------------------------------------"
echo " [MATH] All training done.  Running evaluation + plotting …"
echo "----------------------------------------------------------------"
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/eval_plot_distill_steps_end.py --dataset math
echo ""

# ══════════════════════════════════════════════════════════════════════════════
echo "================================================================"
echo " Dataset 2/2: CommonsenseQA"
echo " Conditions : last N steps distilled, N in {${DISTILL_STEPS}}"
echo " Total steps: 200 each  |  LoRA r=16"
echo "================================================================"
echo ""

for STEPS in $DISTILL_STEPS; do
    DISTILL_START=$((200 - STEPS))
    echo ">>> [CSQA] distill steps ${DISTILL_START}–199  (last ${STEPS} steps)  on GPUs 0-3 …"
    CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
        --num_processes 4 --mixed_precision bf16 --main_process_port 29500 \
        scripts/distill_steps_end_experiment.py --distill_steps ${STEPS} --dataset commonsenseqa
    echo ""
    echo ">>> [CSQA] Done: distill_steps=${STEPS}"
    echo ""
done

echo "----------------------------------------------------------------"
echo " [CSQA] All training done.  Running evaluation + plotting …"
echo "----------------------------------------------------------------"
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/eval_plot_distill_steps_end.py --dataset commonsenseqa
echo ""

# ══════════════════════════════════════════════════════════════════════════════
echo "================================================================"
echo " All done!  Outputs:"
echo "   experiments/distill_steps_end_math/eval_results.json"
echo "   experiments/distill_steps_end_math/distill_steps_end_accuracy.png"
echo "   experiments/distill_steps_end_csqa/eval_results.json"
echo "   experiments/distill_steps_end_csqa/distill_steps_end_accuracy.png"
echo "================================================================"
