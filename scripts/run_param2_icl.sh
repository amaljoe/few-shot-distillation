#!/bin/bash
# Re-run Param2-17B ICL eval with safe batch sizes.
# 0-shot: bs=128 (short prompts ~100 tokens)
# 8-shot: bs=8 (long prompts ~2077 tokens, KV cache heavy)
# Run AFTER training finishes (all 4 GPUs free).
# Usage: bash scripts/run_param2_icl.sh

cd ~/workspace/icl-distillation

PYTHON=/dev/shm/vllm/bin/python
LOG=experiments/logs/param2_icl_retry.log

exec > >(tee -a "$LOG") 2>&1
timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*"; }

log "=== Param2-17B ICL eval (retry) ==="

CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    $PYTHON scripts/eval_hf_icl.py \
    --model bharatgenai/Param2-17B-A2.4B-Thinking \
    --num_samples 1319 \
    --num_fewshot 0 8 \
    --output experiments/param2_17b/icl_eval.json \
    --max_new_tokens 512 \
    --batch_sizes 128 16 \
    && log "ICL eval DONE" \
    || log "WARN: ICL eval failed"

# Write final param_results.md now that we have ICL results
if [ -f experiments/param2_17b/icl_eval.json ] && [ -f experiments/param2_17b_eval.json ]; then
    log "Writing param_results.md"
    CUDA_VISIBLE_DEVICES="" $PYTHON scripts/write_param2_results.py \
        --icl_eval experiments/param2_17b/icl_eval.json \
        --checkpoint_eval experiments/param2_17b_eval.json \
        --output param_results.md \
        && log "param_results.md written" || log "WARN: write results failed"
fi

log "=== ICL eval complete ==="
