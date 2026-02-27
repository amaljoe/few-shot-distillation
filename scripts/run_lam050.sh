#!/usr/bin/env bash
# Four additional isolated single-task experiments (200 steps, batch=8):
#
#   pos_lam050      — few-shot teacher,  λ=0.50, POS  (completes the sweep)
#   ner_lam050      — few-shot teacher,  λ=0.50, NER  (completes the sweep)
#   pos_ctrl_lam005 — zero-shot teacher, λ=0.05, POS  (ablation: ICL signal vs regularisation)
#   ner_ctrl_lam005 — zero-shot teacher, λ=0.05, NER  (ablation: ICL signal vs regularisation)
#
# Hypothesis: if zero-shot ctrl at λ=0.05 matches few-shot distilled at λ=0.05,
# the gain is from regularisation alone. If not, it is specifically the ICL signal.
#
# Summary is NOT generated here — run eval_lambda_sweep.py --summarise manually after.
#
# Usage (on cn14-dgx inside apptainer):
#   bash scripts/run_lam050.sh 2>&1 | tee experiments/xtreme/logs/lam050.log

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

# run_one TASK COND LAM [zeroshot]
run_one() {
    local TASK=$1
    local COND=$2
    local LAM=$3
    local ZEROSHOT="${4:-}"
    local OUT_DIR="${OUT_BASE}/${COND}"
    local CKPT="${OUT_DIR}/${COND}/final"

    if [ -f "${CKPT}/adapter_config.json" ]; then
        log "=== SKIP TRAIN  ${COND}  (checkpoint exists) ==="
    else
        log "=== TRAIN  cond=${COND}  λ=${LAM}  task=${TASK}  teacher=$([ -n "$ZEROSHOT" ] && echo zero-shot || echo few-shot) ==="
        ZS_FLAG=""
        [ -n "$ZEROSHOT" ] && ZS_FLAG="--zeroshot_teacher"
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

# 1. Complete the few-shot sweep at λ=0.50
run_one pos pos_lam050      0.50
run_one ner ner_lam050      0.50

# 2. Zero-shot teacher ablation at λ=0.05 (the best-performing λ)
#    Tests: is the gain from ICL signal or just distillation regularisation?
run_one pos pos_ctrl_lam005 0.05 zeroshot
run_one ner ner_ctrl_lam005 0.05 zeroshot

log ">>> All 4 conditions done. Run the summarise step when ready:"
log "    python scripts/eval_lambda_sweep.py --summarise --output xtreme_lambda_sweep.md"
