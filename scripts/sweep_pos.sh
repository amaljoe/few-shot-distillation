#!/usr/bin/env bash
# POS lambda sweep — GPUs 0,1 — runs λ=0.05, 0.1, 0.2 sequentially
# Run in its own tmux window alongside sweep_ner.sh (GPUs 2,3)

set -euo pipefail

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

ACCEL=/dev/shm/vllm/bin/accelerate
PY=/dev/shm/vllm/bin/python
BASE_CFG=configs/xtreme_llama3b.yaml
DIST_CFG=configs/xtreme_distill.yaml
OUT_BASE=experiments/xtreme/lambda_sweep
GPUS="0,1"
PORT=29500
NGPU=2
MAX_STEPS=400
TASK=pos
TP=2   # eval stays on GPUs 0,1 so NER window can run simultaneously
LOGDIR=experiments/xtreme/logs

mkdir -p "$OUT_BASE" "$LOGDIR"
log() { echo "[$(date '+%H:%M:%S')] $*"; }

for LAM in 0.05 0.1 0.2; do
    LAM_TAG=$(echo "$LAM" | sed 's/\.//g')   # 0.05→005
    COND="${TASK}_lam${LAM_TAG}"
    OUT_DIR="${OUT_BASE}/${COND}"

    log "━━━ START  λ=${LAM}  task=${TASK}  GPUs=${GPUS} ━━━"
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
        --max_steps $MAX_STEPS

    log "━━━ TRAIN DONE  ${COND} ━━━"
    log "━━━ EVAL  ${COND} ━━━"

    CUDA_VISIBLE_DEVICES=$GPUS $PY scripts/eval_lambda_sweep.py \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --checkpoint "${OUT_DIR}/${COND}/final" \
        --task "$TASK" \
        --output experiments/xtreme/lambda_sweep_results.json \
        --condition "$COND" \
        --tensor_parallel_size $TP

    log "━━━ EVAL DONE  ${COND} ━━━"
done

log "=== POS SWEEP COMPLETE — see lambda_sweep_results.json ==="
