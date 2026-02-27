#!/usr/bin/env bash
# Single-task baselines for the lambda sweep comparison.
#
# Adds 4 conditions to lambda_sweep_results.json:
#   pos_sft   — CE only (λ=0), pos, 200 steps
#   ner_sft   — CE only (λ=0), ner, 200 steps
#   pos_ctrl  — zero-shot teacher, λ=0.5, pos, 200 steps
#   ner_ctrl  — zero-shot teacher, λ=0.5, ner, 200 steps
#
# Run AFTER run_lambda_sweep_resume.sh completes:
#   bash scripts/run_single_task_baselines.sh 2>&1 | tee experiments/xtreme/logs/single_task_baselines.log

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
PER_DEVICE_BS=2   # 2 × 4 GPUs = 8 effective

mkdir -p "$OUT_BASE" "$LOGDIR"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run_one() {
    local TASK=$1
    local COND=$2
    local ZEROSHOT="${3:-}"   # pass "zeroshot" to enable zero-shot teacher
    local LAM="${4:-0.0}"
    local OUT_DIR="${OUT_BASE}/${COND}"
    local CKPT="${OUT_DIR}/${COND}/final"

    if [ -f "${CKPT}/adapter_config.json" ] || [ -f "${CKPT}/config.json" ]; then
        log "=== SKIP TRAIN  ${COND}  (checkpoint exists) ==="
    else
        log "=== TRAIN  cond=${COND}  λ=${LAM}  task=${TASK} ==="
        ZS_FLAG=""
        [ "$ZEROSHOT" = "zeroshot" ] && ZS_FLAG="--zeroshot_teacher"
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
            --per_device_batch_size $PER_DEVICE_BS \
            $ZS_FLAG
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

# Single-task SFT (λ=0 ≡ CE only — equivalent to plain finetuning)
run_one pos pos_sft "" 0.0
run_one ner ner_sft "" 0.0

# Single-task Control (zero-shot teacher, λ=0.5)
run_one pos pos_ctrl "zeroshot" 0.5
run_one ner ner_ctrl "zeroshot" 0.5

# Regenerate summary with new conditions included
log "=== Regenerating summary ==="
$PY scripts/eval_lambda_sweep.py \
    --summarise \
    --trained_json experiments/xtreme/llama3b_trained.json \
    --sweep_json   experiments/xtreme/lambda_sweep_results.json \
    --output       xtreme_lambda_sweep.md

log ">>> ALL DONE — xtreme_lambda_sweep.md"
