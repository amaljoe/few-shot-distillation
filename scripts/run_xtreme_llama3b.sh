#!/bin/bash
# =============================================================================
# Llama-3.2-3B-Instruct XTREME: all 5 conditions, all 4 GPUs
# Run inside apptainer + /dev/shm/vllm env on cn14-dgx
# Order: data-check → base eval → fewshot eval → sft → distil → control
#        → eval each trained condition → compile results
# =============================================================================
set -euo pipefail
cd /home/compiling-ganesh/24m0797/workspace/icl-distillation

LOGDIR="experiments/xtreme/logs"
mkdir -p "$LOGDIR"
timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(timestamp)] $*" | tee -a "$LOGDIR/llama3b_run.log"; }

MODEL="meta-llama/Llama-3.2-3B-Instruct"
OUTDIR="experiments/xtreme/llama3b"
ACCELERATE="/dev/shm/vllm/bin/accelerate"
TP=4          # tensor parallel for vLLM (all 4 GPUs)
NGPU=4        # DDP processes for training (all 4 GPUs)
GPUS="0,1,2,3"

pip install seqeval sacrebleu -q 2>/dev/null && log "deps OK"

# ── 0. Verify all 28 (task, lang) pairs load correctly ───────────────────────
log "=== [0] Data check: verifying 28 pairs ==="
python scripts/check_xtreme_data.py 2>&1 | tee "$LOGDIR/data_check.log"
log "=== Data check passed — all 28 pairs OK ==="

# ── 1. Base eval (zero-shot) ──────────────────────────────────────────────────
log "=== [1/5] Base eval (zero-shot) ==="
CUDA_VISIBLE_DEVICES=$GPUS python scripts/eval_xtreme_inference.py \
    --model "$MODEL" \
    --conditions base \
    --n_samples 500 \
    --tensor_parallel_size $TP \
    --max_model_len 8192 \
    --output experiments/xtreme/llama3b_inference.json \
    2>&1 | tee "$LOGDIR/llama3b_base_eval.log"
log "=== Base eval done ==="

# ── 2. Few-shot eval ──────────────────────────────────────────────────────────
log "=== [2/5] Few-shot eval (5-shot ICL) ==="
CUDA_VISIBLE_DEVICES=$GPUS python scripts/eval_xtreme_inference.py \
    --model "$MODEL" \
    --conditions fewshot \
    --n_samples 500 \
    --tensor_parallel_size $TP \
    --max_model_len 8192 \
    --output experiments/xtreme/llama3b_inference.json \
    2>&1 | tee "$LOGDIR/llama3b_fewshot_eval.log"
log "=== Few-shot eval done ==="

# ── 3. SFT training (finetuned) ───────────────────────────────────────────────
log "=== [3/5] SFT training ==="
CUDA_VISIBLE_DEVICES=$GPUS $ACCELERATE launch \
    --num_processes $NGPU --mixed_precision bf16 --main_process_port 29500 \
    src/training/train_xtreme_sft.py \
    --config configs/xtreme_llama3b.yaml \
    --output_dir "$OUTDIR" \
    2>&1 | tee "$LOGDIR/llama3b_sft.log"
log "=== SFT done ==="

# ── 3b. Eval: finetuned checkpoint ───────────────────────────────────────────
log "=== [3b] Eval: finetuned ==="
CUDA_VISIBLE_DEVICES=$GPUS python scripts/eval_xtreme_checkpoints.py \
    --base_model "$MODEL" \
    --checkpoint_dir "$OUTDIR/xtreme_sft/final" \
    --condition finetuned \
    --n_samples 500 \
    --tensor_parallel_size $TP \
    --max_model_len 8192 \
    --output experiments/xtreme/llama3b_trained.json \
    2>&1 | tee "$LOGDIR/llama3b_sft_eval.log"
log "=== Finetuned eval done ==="

# ── 4. Distillation training ──────────────────────────────────────────────────
log "=== [4/5] Distillation training ==="
CUDA_VISIBLE_DEVICES=$GPUS $ACCELERATE launch \
    --num_processes $NGPU --mixed_precision bf16 --main_process_port 29500 \
    src/training/train_xtreme_distill.py \
    --base_config configs/xtreme_llama3b.yaml \
    --config configs/xtreme_distill.yaml \
    --output_dir "$OUTDIR" \
    2>&1 | tee "$LOGDIR/llama3b_distill.log"
log "=== Distillation done ==="

# ── 4b. Eval: distilled checkpoint ───────────────────────────────────────────
log "=== [4b] Eval: distilled ==="
CUDA_VISIBLE_DEVICES=$GPUS python scripts/eval_xtreme_checkpoints.py \
    --base_model "$MODEL" \
    --checkpoint_dir "$OUTDIR/xtreme_distill/final" \
    --condition distilled \
    --n_samples 500 \
    --tensor_parallel_size $TP \
    --max_model_len 8192 \
    --output experiments/xtreme/llama3b_trained.json \
    2>&1 | tee "$LOGDIR/llama3b_distill_eval.log"
log "=== Distilled eval done ==="

# ── 5. Control training (zero-shot teacher) ───────────────────────────────────
log "=== [5/5] Control training ==="
CUDA_VISIBLE_DEVICES=$GPUS $ACCELERATE launch \
    --num_processes $NGPU --mixed_precision bf16 --main_process_port 29500 \
    src/training/train_xtreme_distill.py \
    --base_config configs/xtreme_llama3b.yaml \
    --config configs/xtreme_distill.yaml \
    --zeroshot_teacher \
    --output_dir "$OUTDIR" \
    2>&1 | tee "$LOGDIR/llama3b_control.log"
log "=== Control done ==="

# ── 5b. Eval: control checkpoint ─────────────────────────────────────────────
log "=== [5b] Eval: control ==="
CUDA_VISIBLE_DEVICES=$GPUS python scripts/eval_xtreme_checkpoints.py \
    --base_model "$MODEL" \
    --checkpoint_dir "$OUTDIR/xtreme_control/final" \
    --condition control \
    --n_samples 500 \
    --tensor_parallel_size $TP \
    --max_model_len 8192 \
    --output experiments/xtreme/llama3b_trained.json \
    2>&1 | tee "$LOGDIR/llama3b_control_eval.log"
log "=== Control eval done ==="

# ── Compile results ───────────────────────────────────────────────────────────
log "=== Generating xtreme_results.md ==="
python scripts/gen_xtreme_results.py \
    --inference_dir experiments/xtreme \
    --output xtreme_results.md \
    --assets_dir assets/xtreme \
    2>&1 | tee "$LOGDIR/gen_results.log"
log "=== xtreme_results.md done ==="

log ">>> ALL DONE <<<"
