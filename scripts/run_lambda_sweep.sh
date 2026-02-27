#!/usr/bin/env bash
# Lambda sweep: test λ = 0.05, 0.1, 0.2 on POS and NER independently.
# Runs 2 conditions in parallel (GPUs 0,1 and 2,3) to save time.
# Existing baselines reused: SFT (finetuned/final) and λ=0.5 (xtreme_distill/final).
#
# Usage (inside apptainer, /dev/shm/vllm env active):
#   bash scripts/run_lambda_sweep.sh 2>&1 | tee experiments/xtreme/logs/lambda_sweep.log

set -euo pipefail

ACCEL=/dev/shm/vllm/bin/accelerate
PY=/dev/shm/vllm/bin/python
BASE_CFG=configs/xtreme_llama3b.yaml
DIST_CFG=configs/xtreme_distill.yaml
OUT_BASE=experiments/xtreme/lambda_sweep
LOGDIR=experiments/xtreme/logs
TP=4
MAX_STEPS=400   # single-task: 400 steps ≈ ~1.5 epochs over POS/NER train data

mkdir -p "$OUT_BASE" "$LOGDIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ─────────────────────────────────────────────────────────────────────────────
# Helper: train one lambda × task, then eval immediately
# ─────────────────────────────────────────────────────────────────────────────
train_and_eval() {
    local TASK=$1      # pos or ner
    local LAM=$2       # e.g. 0.05
    local GPUS=$3      # e.g. "0,1"
    local PORT=$4      # accelerate port
    local NGPU=2

    local LAM_TAG=$(echo "$LAM" | sed 's/\.//g')   # 0.05 → 005
    local COND="${TASK}_lam${LAM_TAG}"
    local OUT_DIR="${OUT_BASE}/${COND}"

    log "=== START  task=${TASK}  λ=${LAM}  GPUs=${GPUS} ==="
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
        2>&1 | tee "${LOGDIR}/${COND}_train.log"
    log "=== TRAIN DONE  ${COND} ==="

    # Eval: load the LoRA checkpoint via vLLM, evaluate on the specific task
    log "=== EVAL  ${COND} ==="
    CUDA_VISIBLE_DEVICES=0,1,2,3 $PY scripts/eval_lambda_sweep.py \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --checkpoint "${OUT_DIR}/${COND}/final" \
        --task "$TASK" \
        --output "experiments/xtreme/lambda_sweep_results.json" \
        --condition "$COND" \
        --tensor_parallel_size $TP \
        2>&1 | tee "${LOGDIR}/${COND}_eval.log"
    log "=== EVAL DONE  ${COND} ==="
}

# ─────────────────────────────────────────────────────────────────────────────
# Round 1: λ=0.05 — POS (GPUs 0,1) and NER (GPUs 2,3) in parallel
# ─────────────────────────────────────────────────────────────────────────────
log ">>> Round 1: λ=0.05  (POS + NER in parallel)"
train_and_eval pos 0.05 "0,1" 29500 &
PID1=$!
train_and_eval ner 0.05 "2,3" 29501 &
PID2=$!
wait $PID1 $PID2
log ">>> Round 1 done"

# ─────────────────────────────────────────────────────────────────────────────
# Round 2: λ=0.1
# ─────────────────────────────────────────────────────────────────────────────
log ">>> Round 2: λ=0.1  (POS + NER in parallel)"
train_and_eval pos 0.1 "0,1" 29500 &
PID1=$!
train_and_eval ner 0.1 "2,3" 29501 &
PID2=$!
wait $PID1 $PID2
log ">>> Round 2 done"

# ─────────────────────────────────────────────────────────────────────────────
# Round 3: λ=0.2
# ─────────────────────────────────────────────────────────────────────────────
log ">>> Round 3: λ=0.2  (POS + NER in parallel)"
train_and_eval pos 0.2 "0,1" 29500 &
PID1=$!
train_and_eval ner 0.2 "2,3" 29501 &
PID2=$!
wait $PID1 $PID2
log ">>> Round 3 done"

# ─────────────────────────────────────────────────────────────────────────────
# Final: compile results table (adds existing SFT + λ=0.5 baselines)
# ─────────────────────────────────────────────────────────────────────────────
log ">>> Generating lambda sweep summary"
$PY scripts/eval_lambda_sweep.py \
    --summarise \
    --trained_json experiments/xtreme/llama3b_trained.json \
    --sweep_json experiments/xtreme/lambda_sweep_results.json \
    --output experiments/xtreme/lambda_sweep_summary.md \
    2>&1 | tee "${LOGDIR}/lambda_sweep_summary.log"

log ">>> ALL DONE — see experiments/xtreme/lambda_sweep_summary.md"
