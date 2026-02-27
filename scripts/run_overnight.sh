#!/bin/bash
# ============================================================
# Overnight experiment runner: Qwen3-8B and Param2-17B
#
# Run from: ~/workspace/icl-distillation (inside apptainer + conda env)
# Usage:
#   bash scripts/run_overnight.sh
#
# The script logs to experiments/overnight_YYYYMMDD_HHMM.log
# AND prints everything to the terminal simultaneously.
#
# GPU layout (4 × A100 80GB):
#   ICL eval:   GPU 0 (Qwen8B) and GPU 1 (Param2) in parallel
#   Training:   GPU 0,1 = first condition, GPU 2,3 = second condition (parallel per model)
#   Eval:       GPU 0,1,2,3 (tensor_parallel=4)
# ============================================================

set -uo pipefail

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ_DIR"

# Set up logging — redirect all output to both terminal AND log file
LOGFILE="experiments/overnight_$(date +%Y%m%d_%H%M).log"
mkdir -p experiments/logs experiments/qwen8b experiments/param2_17b
exec > >(tee -a "$LOGFILE") 2>&1

LOG_DIR="experiments/logs"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log()        { echo "[$(timestamp)] $*"; }
step_start() { log ">>> START: $1"; }
step_done()  { log "<<< DONE:  $1"; }
step_warn()  { log "!!! WARN:  $1"; }

run_parallel() {
    # run_parallel LOGA CMDA LOGB CMDB NAMEA NAMEB
    # Launch two commands in parallel on different GPU sets, wait for both.
    local loga=$1; shift
    local cmda=$1; shift
    local logb=$1; shift
    local cmdb=$1; shift
    local namea=$1; shift
    local nameb=$1; shift

    eval "$cmda" > "$loga" 2>&1 &
    local pid_a=$!
    eval "$cmdb" > "$logb" 2>&1 &
    local pid_b=$!

    log "  Launched $namea (pid=$pid_a) and $nameb (pid=$pid_b)"

    wait "$pid_a" && step_done "$namea" || step_warn "$namea failed — check $loga"
    wait "$pid_b" && step_done "$nameb" || step_warn "$nameb failed — check $logb"
}

# ============================================================
log "======================================================"
log " ICL-Distillation Overnight Run"
log " Models: Qwen3-8B  |  Param2-17B-A2.4B-Thinking"
log " Log: $LOGFILE"
log " Start: $(timestamp)"
log "======================================================"
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader
log "======================================================"

# ============================================================
# PHASE 0: ICL Evaluation — both models in parallel (GPU 0 vs GPU 1)
# ============================================================
step_start "Phase 0: ICL evaluation — Qwen3-8B (GPU 0) and Param2-17B (GPU 1)"

run_parallel \
    "$LOG_DIR/qwen8b_icl.log" \
    "CUDA_VISIBLE_DEVICES=0 python scripts/eval_icl.py \
        --model Qwen/Qwen3-8B \
        --num_samples 1319 --num_fewshot 0 8 --seed 42 \
        --output experiments/qwen8b/icl_eval.json \
        --max_model_len 4096 --gpu_memory_utilization 0.85 \
        --max_new_tokens 1024" \
    "$LOG_DIR/param2_icl.log" \
    "bash -c '
        CUDA_VISIBLE_DEVICES=1 python scripts/eval_icl.py \
            --model bharatgenai/Param2-17B-A2.4B-Thinking \
            --num_samples 1319 --num_fewshot 0 8 --seed 42 \
            --output experiments/param2_17b/icl_eval.json \
            --max_model_len 4096 --gpu_memory_utilization 0.85 \
            --max_new_tokens 1024 \
        || (echo FALLBACK_HF && CUDA_VISIBLE_DEVICES=1 python scripts/eval_hf_icl.py \
            --model bharatgenai/Param2-17B-A2.4B-Thinking \
            --num_samples 1319 --num_fewshot 0 8 --seed 42 \
            --output experiments/param2_17b/icl_eval.json \
            --max_new_tokens 1024 --batch_size 2)
    '" \
    "Qwen3-8B ICL eval" \
    "Param2-17B ICL eval"

log "ICL results:"
python3 -c "
import json
for path, name in [
    ('experiments/qwen8b/icl_eval.json', 'Qwen3-8B'),
    ('experiments/param2_17b/icl_eval.json', 'Param2-17B'),
]:
    try:
        d = json.load(open(path))
        evs = d.get('evaluations', {})
        s0 = evs.get('0_shot', {}).get('accuracy')
        s8 = evs.get('8_shot', {}).get('accuracy')
        gap = d.get('icl_gap')
        print(f'  {name}: 0-shot={s0:.2%}  8-shot={s8:.2%}  ICL gap={gap:.2%}')
    except Exception as e:
        print(f'  {name}: parse error ({e})')
" || true

# ============================================================
# QWEN3-8B TRAINING: baseline (GPU 0,1) + distillation (GPU 2,3) in parallel
# ============================================================
step_start "Qwen3-8B: SFT baseline (GPU 0,1) + Distillation (GPU 2,3) — parallel"

run_parallel \
    "$LOG_DIR/qwen8b_baseline.log" \
    "CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
        --num_processes 2 --mixed_precision bf16 --main_process_port 29500 \
        src/training/train_baseline.py \
        --config configs/qwen8b.yaml \
        --output_dir experiments/qwen8b/baseline" \
    "$LOG_DIR/qwen8b_distill.log" \
    "CUDA_VISIBLE_DEVICES=2,3 accelerate launch \
        --num_processes 2 --mixed_precision bf16 --main_process_port 29502 \
        src/training/train_online_v1.py \
        --base_config configs/qwen8b.yaml \
        --config configs/online_v1_qwen8b.yaml \
        --output_dir experiments/qwen8b/online_v1" \
    "Qwen3-8B SFT baseline" \
    "Qwen3-8B distillation"

# ============================================================
# QWEN3-8B ABLATION: 0-shot teacher control (GPU 0,1)
# ============================================================
step_start "Qwen3-8B: 0-shot teacher ablation (GPU 0,1)"
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --num_processes 2 --mixed_precision bf16 --main_process_port 29500 \
    src/training/train_ablation.py \
    --base_config configs/qwen8b.yaml \
    --config configs/ablation_zeroshot_teacher.yaml \
    --condition_name zeroshot_teacher \
    --output_dir experiments/qwen8b \
    > "$LOG_DIR/qwen8b_ablation.log" 2>&1 \
    && step_done "Qwen3-8B ablation" \
    || step_warn "Qwen3-8B ablation — check $LOG_DIR/qwen8b_ablation.log"

# ============================================================
# QWEN3-8B CHECKPOINT EVAL (tensor_parallel=4, 1319 samples)
# ============================================================
step_start "Qwen3-8B: checkpoint eval — baseline, online_v1, zeroshot_teacher"
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/eval_checkpoints.py \
    --config configs/qwen8b.yaml \
    --base_dir experiments/qwen8b \
    --conditions baseline online_v1 zeroshot_teacher \
    --n_samples 1319 \
    --checkpoint_steps 200 400 600 800 1000 \
    --output experiments/qwen8b_eval.json \
    --tensor_parallel_size 4 \
    --max_model_len 2048 \
    > "$LOG_DIR/qwen8b_eval.log" 2>&1 \
    && step_done "Qwen3-8B checkpoint eval" \
    || step_warn "Qwen3-8B checkpoint eval — check $LOG_DIR/qwen8b_eval.log"

# ============================================================
# PARAM2-17B TRAINING: baseline (GPU 0,1) + distillation (GPU 2,3) in parallel
# ============================================================
step_start "Param2-17B: SFT baseline (GPU 0,1) + Distillation (GPU 2,3) — parallel"

run_parallel \
    "$LOG_DIR/param2_baseline.log" \
    "CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
        --num_processes 2 --mixed_precision bf16 --main_process_port 29500 \
        src/training/train_baseline.py \
        --config configs/param2_17b.yaml \
        --output_dir experiments/param2_17b/baseline" \
    "$LOG_DIR/param2_distill.log" \
    "CUDA_VISIBLE_DEVICES=2,3 accelerate launch \
        --num_processes 2 --mixed_precision bf16 --main_process_port 29502 \
        src/training/train_online_v1.py \
        --base_config configs/param2_17b.yaml \
        --config configs/online_v1_param2.yaml \
        --output_dir experiments/param2_17b/online_v1" \
    "Param2-17B SFT baseline" \
    "Param2-17B distillation"

# ============================================================
# PARAM2-17B ABLATION: 0-shot teacher control (GPU 0,1)
# ============================================================
step_start "Param2-17B: 0-shot teacher ablation (GPU 0,1)"
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --num_processes 2 --mixed_precision bf16 --main_process_port 29500 \
    src/training/train_ablation.py \
    --base_config configs/param2_17b.yaml \
    --config configs/ablation_zeroshot_teacher.yaml \
    --condition_name zeroshot_teacher \
    --output_dir experiments/param2_17b \
    > "$LOG_DIR/param2_ablation.log" 2>&1 \
    && step_done "Param2-17B ablation" \
    || step_warn "Param2-17B ablation — check $LOG_DIR/param2_ablation.log"

# ============================================================
# PARAM2-17B CHECKPOINT EVAL — try vLLM, fallback to HF
# ============================================================
step_start "Param2-17B: checkpoint eval (vLLM, fallback HF)"
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/eval_checkpoints.py \
    --config configs/param2_17b.yaml \
    --base_dir experiments/param2_17b \
    --conditions baseline online_v1 zeroshot_teacher \
    --n_samples 1319 \
    --checkpoint_steps 200 400 600 800 1000 \
    --output experiments/param2_17b_eval.json \
    --tensor_parallel_size 4 \
    --max_model_len 2048 \
    > "$LOG_DIR/param2_eval.log" 2>&1 \
    && step_done "Param2-17B checkpoint eval (vLLM)" \
    || {
        step_warn "vLLM eval failed for Param2 — trying HF fallback"
        CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/eval_hf_checkpoints.py \
            --config configs/param2_17b.yaml \
            --base_dir experiments/param2_17b \
            --conditions baseline online_v1 zeroshot_teacher \
            --n_samples 1319 \
            --checkpoint_steps 200 400 600 800 1000 \
            --output experiments/param2_17b_eval.json \
            --max_new_tokens 1024 --batch_size 2 \
            > "$LOG_DIR/param2_eval_hf.log" 2>&1 \
            && step_done "Param2-17B checkpoint eval (HF)" \
            || step_warn "HF eval also failed — check $LOG_DIR/param2_eval_hf.log"
    }

# ============================================================
# WRITE RESULTS
# ============================================================
step_start "Writing results.md"
python scripts/write_results.py \
    && step_done "results.md written" \
    || step_warn "write_results.py failed"

# ============================================================
log "======================================================"
log " ALL DONE: $(timestamp)"
log " Full log: $LOGFILE"
log " Results:  results.md"
log "======================================================"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader

echo ""
echo "Training log sizes:"
for f in qwen8b_baseline qwen8b_distill qwen8b_ablation qwen8b_eval \
          param2_baseline param2_distill param2_ablation param2_eval param2_eval_hf \
          qwen8b_icl param2_icl; do
    logf="$LOG_DIR/${f}.log"
    [[ -f "$logf" ]] && printf "  %-30s %d lines\n" "$logf" "$(wc -l < "$logf")" || true
done
