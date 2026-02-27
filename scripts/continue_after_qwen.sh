#!/bin/bash
# Continuation script: waits for Qwen3-8B training to finish,
# then runs Param2-17B ICL eval + full training + write_results.
#
# Run in a new tmux session WHILE train_qwen8b.sh is running:
#   tmux new-session -d -s continue 'bash scripts/continue_after_qwen.sh'

cd ~/workspace/icl-distillation
LOGFILE="experiments/logs/continuation_$(date +%Y%m%d_%H%M).log"
exec > >(tee -a "$LOGFILE") 2>&1

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*"; }

QWEN_PID=${1:-3634226}   # train_qwen8b.sh main PID

log "=== Continuation script started ==="
log "Waiting for Qwen3-8B training (pid=$QWEN_PID) to complete..."
log "Log: $LOGFILE"

# Poll until the Qwen8B training process is gone
while kill -0 "$QWEN_PID" 2>/dev/null; do
    sleep 60
done

log "Qwen3-8B training finished. Checking results..."
if [ -f experiments/qwen8b_eval.json ]; then
    python3 -c "
import json
d = json.load(open('experiments/qwen8b_eval.json'))
conds = d.get('conditions', {})
for cond, steps in conds.items():
    if steps:
        best_step = max(steps, key=lambda k: steps[k].get('accuracy', 0))
        acc = steps[best_step].get('accuracy')
        print(f'  {cond}: best={acc:.2%} at {best_step}' if acc else f'  {cond}: N/A')
    else:
        print(f'  {cond}: no checkpoints evaluated')
" 2>/dev/null || true
fi

log ""
log "=== Starting Param2-17B ICL evaluation (GPU 0, 1319 samples) ==="
CUDA_VISIBLE_DEVICES=0 python scripts/eval_hf_icl.py \
    --model bharatgenai/Param2-17B-A2.4B-Thinking \
    --num_samples 1319 --num_fewshot 0 8 --seed 42 \
    --output experiments/param2_17b/icl_eval.json \
    --max_new_tokens 1024 --batch_size 4 \
    > experiments/logs/param2_icl.log 2>&1 \
    && log "Param2 ICL eval done" \
    || log "WARN: Param2 ICL eval failed — check experiments/logs/param2_icl.log"

log ""
log "=== Starting Param2-17B training sequence ==="
bash scripts/train_param2.sh

log ""
log "=== Writing results.md ==="
python scripts/write_results.py \
    && log "results.md written" \
    || log "WARN: write_results.py failed"

log ""
log "=== ALL DONE ==="
log "Qwen3-8B ICL:  experiments/qwen8b/icl_eval.json"
log "Qwen3-8B eval: experiments/qwen8b_eval.json"
log "Param2 ICL:    experiments/param2_17b/icl_eval.json"
log "Param2 eval:   experiments/param2_17b_eval.json"
log "Results:       results.md"
