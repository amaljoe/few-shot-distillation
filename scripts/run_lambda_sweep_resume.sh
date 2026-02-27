#!/usr/bin/env bash
# Resume lambda sweep — skips training if checkpoint already exists.
# Runs eval for all conditions, then generates summary.

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
    local CKPT="${OUT_DIR}/${COND}/final"

    if [ -f "${CKPT}/adapter_config.json" ] || [ -f "${CKPT}/config.json" ]; then
        log "=== SKIP TRAIN  ${COND}  (checkpoint exists) ==="
    else
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
    fi

    log "=== EVAL  ${COND} ==="
    CUDA_VISIBLE_DEVICES=$GPUS $PY scripts/eval_lambda_sweep.py \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --checkpoint "${CKPT}" \
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
