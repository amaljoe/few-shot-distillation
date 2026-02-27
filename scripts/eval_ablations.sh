#!/bin/bash
# Evaluate baseline + both ablation conditions.
# Run from inside the apptainer container on cn14-dgx.
# The base model is loaded once; LoRA adapters are swapped per checkpoint.
#
# Usage (inside apptainer, /dev/shm/vllm env):
#   cd ~/workspace/icl-distillation
#   bash scripts/eval_ablations.sh
#
# Output:
#   experiments/qwen1b7/baseline_full_eval.json  (SFT checkpoint curve)
#   experiments/ablations/ablation_eval.json     (0-shot + shuffled eval)

set -e
cd ~/workspace/icl-distillation

PYTHON=/dev/shm/vllm/bin/python

echo "=== Step 1: Evaluate SFT baseline checkpoints ==="
$PYTHON scripts/eval_checkpoints.py \
    --config configs/qwen1b7.yaml \
    --n_samples 1319 \
    --conditions baseline \
    --base_dir experiments/qwen1b7 \
    --checkpoint_steps 200 400 600 800 1000 \
    --output experiments/qwen1b7/baseline_full_eval.json \
    --tensor_parallel_size 4 \
    --max_model_len 2048

echo ""
echo "=== Step 2: Evaluate ablation conditions ==="
$PYTHON scripts/eval_checkpoints.py \
    --config configs/qwen1b7.yaml \
    --n_samples 1319 \
    --conditions zeroshot_teacher shuffled_answers \
    --base_dir experiments/ablations \
    --checkpoint_steps 200 400 600 800 1000 \
    --output experiments/ablations/ablation_eval.json \
    --tensor_parallel_size 4 \
    --max_model_len 2048

echo ""
echo "=== Done. Files written: ==="
echo "  experiments/qwen1b7/baseline_full_eval.json"
echo "  experiments/ablations/ablation_eval.json"
