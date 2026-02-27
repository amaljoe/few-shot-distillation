#!/usr/bin/env bash
# Lambda sweep: 6 conditions sequentially on all 4 GPUs.
# Order: pos_lam005, pos_lam010, pos_lam020, ner_lam005, ner_lam010, ner_lam020
# After each train: immediate eval on same GPUs.
# Final: compile summary table + plot.
#
# Run inside apptainer (vllm env active):
#   bash scripts/run_lambda_sweep_seq.sh

set -euo pipefail

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

ACCEL=/dev/shm/vllm/bin/accelerate
PY=/dev/shm/vllm/bin/python
BASE_CFG=configs/xtreme_llama3b.yaml
DIST_CFG=configs/xtreme_distill.yaml
OUT_BASE=experiments/xtreme/lambda_sweep
LOGDIR=experiments/xtreme/logs
GPUS="0,1,2,3"
PORT=29500
NGPU=4
TP=4
MAX_STEPS=200
PER_DEVICE_BS=2   # 2 × 4 GPUs = 8 effective batch size

mkdir -p "$OUT_BASE" "$LOGDIR"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run_one() {
    local TASK=$1
    local LAM=$2
    local LAM_TAG=$(echo "$LAM" | sed 's/\.//g')
    local COND="${TASK}_lam${LAM_TAG}"
    local OUT_DIR="${OUT_BASE}/${COND}"

    log "=== TRAIN  task=${TASK}  λ=${LAM}  cond=${COND} ==="
    CUDA_VISIBLE_DEVICES=$GPUS $ACCEL launch \
        --num_processes $NGPU --mixed_precision bf16 \
        --main_process_port $PORT \
        src/training/train_xtreme_distill.py \
        --base_config $BASE_CFG \
        --config $DIST_CFG \
        --lambda_distill "$LAM" \
        --tasks "$TASK" \
        --condition_name "$COND" \
        --output_dir "$OUT_DIR" \
        --max_steps $MAX_STEPS \
        --per_device_batch_size $PER_DEVICE_BS
    log "=== TRAIN DONE  ${COND} ==="

    log "=== EVAL  ${COND} ==="
    CUDA_VISIBLE_DEVICES=$GPUS $PY scripts/eval_lambda_sweep.py \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --checkpoint "${OUT_DIR}/${COND}/final" \
        --task "$TASK" \
        --output experiments/xtreme/lambda_sweep_results.json \
        --condition "$COND" \
        --tensor_parallel_size $TP
    log "=== EVAL DONE  ${COND} ==="
}

# POS sweep
run_one pos 0.05
run_one pos 0.1
run_one pos 0.2

# NER sweep
run_one ner 0.05
run_one ner 0.1
run_one ner 0.2

# Compile summary
log "=== Generating summary ==="
$PY scripts/eval_lambda_sweep.py \
    --summarise \
    --trained_json experiments/xtreme/llama3b_trained.json \
    --sweep_json   experiments/xtreme/lambda_sweep_results.json \
    --output       experiments/xtreme/lambda_sweep_summary.md

log ">>> ALL DONE — experiments/xtreme/lambda_sweep_summary.md"
