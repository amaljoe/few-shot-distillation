#!/bin/bash
# Full Param2-17B experiment pipeline:
#   Phase 1 — ICL eval (0-shot + 8-shot, all 4 GPUs via device_map=auto)
#   Phase 2 — Parallel: baseline (GPU 0,1) + distillation (GPU 2,3)
#   Phase 3 — Ablation: zeroshot_teacher (GPU 0,1)
#   Phase 4 — Checkpoint eval (all 4 GPUs, HF fallback)
#   Phase 5 — Write param_results.md
#
# Run inside the apptainer container (app alias or exp1-4 sessions):
#   bash scripts/run_param2_all.sh

cd ~/workspace/icl-distillation

PYTHON=/dev/shm/vllm/bin/python
ACCELERATE=/dev/shm/vllm/bin/accelerate
LOG=experiments/logs/param2_full_run.log
mkdir -p experiments/logs experiments/param2_17b

exec > >(tee -a "$LOG") 2>&1

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*"; }

log "=== Param2-17B FULL PIPELINE START ==="
log "Python: $($PYTHON --version 2>&1)"

# ─── Phase 1: ICL eval (all 4 GPUs, device_map=auto) ───────────────────────
log "Phase 1: ICL eval (0-shot + 8-shot, 1319 samples)"
CUDA_VISIBLE_DEVICES=0,1,2,3 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    $PYTHON scripts/eval_hf_icl.py \
    --model bharatgenai/Param2-17B-A2.4B-Thinking \
    --num_samples 1319 \
    --num_fewshot 0 8 \
    --output experiments/param2_17b/icl_eval.json \
    --max_new_tokens 512 \
    --batch_sizes 128 16 \
    && log "Phase 1 DONE" \
    || { log "WARN: Phase 1 (ICL eval) failed — continuing with training"; }

# ─── Phase 2: Parallel training ─────────────────────────────────────────────
log "Phase 2: baseline (GPU 0,1) + distillation (GPU 2,3) in parallel"

CUDA_VISIBLE_DEVICES=0,1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    $ACCELERATE launch \
    --num_processes 2 --mixed_precision bf16 --main_process_port 29500 \
    src/training/train_baseline.py \
    --config configs/param2_17b.yaml \
    --output_dir experiments/param2_17b/baseline \
    > experiments/logs/param2_baseline.log 2>&1 &
BL=$!
log "  baseline PID=$BL (GPU 0,1) — logs: experiments/logs/param2_baseline.log"

CUDA_VISIBLE_DEVICES=2,3 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    $ACCELERATE launch \
    --num_processes 2 --mixed_precision bf16 --main_process_port 29502 \
    src/training/train_online_v1.py \
    --base_config configs/param2_17b.yaml \
    --config configs/online_v1_param2.yaml \
    --output_dir experiments/param2_17b/online_v1 \
    > experiments/logs/param2_distill.log 2>&1 &
DL=$!
log "  distillation PID=$DL (GPU 2,3) — logs: experiments/logs/param2_distill.log"

log "Waiting for both training runs to complete..."
wait $BL && log "  baseline finished OK" || log "  WARN: baseline exited non-zero"
wait $DL && log "  distillation finished OK" || log "  WARN: distillation exited non-zero"
log "Phase 2 DONE"

# ─── Phase 3: Ablation — zeroshot_teacher (GPU 0,1) ────────────────────────
log "Phase 3: zeroshot_teacher ablation (GPU 0,1)"
CUDA_VISIBLE_DEVICES=0,1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    $ACCELERATE launch \
    --num_processes 2 --mixed_precision bf16 --main_process_port 29500 \
    src/training/train_ablation.py \
    --base_config configs/param2_17b.yaml \
    --config configs/ablation_zeroshot_teacher.yaml \
    --condition_name zeroshot_teacher \
    --output_dir experiments/param2_17b \
    > experiments/logs/param2_ablation.log 2>&1 \
    && log "Phase 3 DONE" \
    || log "WARN: Phase 3 (ablation) failed"

# ─── Phase 4: Checkpoint eval (all 4 GPUs, HF+PEFT) ────────────────────────
log "Phase 4: checkpoint eval (all 4 GPUs, 1319 samples, steps 200-1000)"
CUDA_VISIBLE_DEVICES=0,1,2,3 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    $PYTHON scripts/eval_hf_checkpoints.py \
    --config configs/param2_17b.yaml \
    --base_dir experiments/param2_17b \
    --conditions baseline online_v1 zeroshot_teacher \
    --n_samples 1319 \
    --checkpoint_steps 200 400 600 800 1000 \
    --output experiments/param2_17b_eval.json \
    --max_new_tokens 512 \
    --batch_size 16 \
    > experiments/logs/param2_eval.log 2>&1 \
    && log "Phase 4 DONE" \
    || log "WARN: Phase 4 (checkpoint eval) failed"

# ─── Phase 5: Write param_results.md ────────────────────────────────────────
log "Phase 5: writing param_results.md"
CUDA_VISIBLE_DEVICES="" \
    $PYTHON scripts/write_param2_results.py \
    --icl_eval experiments/param2_17b/icl_eval.json \
    --checkpoint_eval experiments/param2_17b_eval.json \
    --output param_results.md \
    && log "Phase 5 DONE — see param_results.md" \
    || log "WARN: Phase 5 (write results) failed"

log "=== Param2-17B FULL PIPELINE COMPLETE ==="
