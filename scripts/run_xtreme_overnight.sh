#!/bin/bash
# =============================================================================
# XTREME overnight experiment runner
# =============================================================================
# Prerequisites:
#   - Running inside apptainer container on cn14-dgx (app alias)
#   - conda activate /dev/shm/vllm
#   - cd ~/workspace/icl-distillation
#
# Two parallel tmux streams:
#   Session `claude`  (GPUs 0,1): Qwen3-1.7B + Gemma-3-270M
#   Session `vscode`  (GPUs 2,3): Llama-3.2-3B + Qwen3-8B
#
# Usage (run this script to queue everything in your current terminal,
#        or paste per-session blocks into each tmux pane):
#   bash scripts/run_xtreme_overnight.sh claude    # enqueue claude stream
#   bash scripts/run_xtreme_overnight.sh vscode    # enqueue vscode stream
#   bash scripts/run_xtreme_overnight.sh eval      # enqueue post-training eval
#   bash scripts/run_xtreme_overnight.sh results   # generate final report
# =============================================================================

set -euo pipefail
STREAM="${1:-help}"
LOGDIR="experiments/xtreme/logs"
mkdir -p "$LOGDIR"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log()       { echo "[$(timestamp)] $*" | tee -a "$LOGDIR/run.log"; }

# ─────────────────────────────────────────────────────────────────────────────
# Dependency check
# ─────────────────────────────────────────────────────────────────────────────
install_deps() {
    log "Checking / installing dependencies ..."
    pip install seqeval sacrebleu -q && log "  seqeval + sacrebleu OK"
}

# ─────────────────────────────────────────────────────────────────────────────
# STREAM: claude  (GPUs 0,1)
# Qwen3-1.7B: inference → SFT → distill → control
# Gemma-270M: inference → SFT → distill → control
# ─────────────────────────────────────────────────────────────────────────────
run_claude() {
    install_deps

    # ── Qwen3-1.7B ──────────────────────────────────────────────────────────
    log "=== Qwen3-1.7B: inference eval (base + fewshot) ==="
    CUDA_VISIBLE_DEVICES=0,1 python scripts/eval_xtreme_inference.py \
        --model Qwen/Qwen3-1.7B \
        --conditions base fewshot \
        --n_samples 500 \
        --batch_size 16 \
        --output experiments/xtreme/qwen1b7_inference.json \
        2>&1 | tee "$LOGDIR/qwen1b7_inference.log"
    log "=== Qwen3-1.7B inference done ==="

    log "=== Qwen3-1.7B: SFT training ==="
    CUDA_VISIBLE_DEVICES=0,1 /dev/shm/vllm/bin/accelerate launch \
        --num_processes 2 --mixed_precision bf16 --main_process_port 29500 \
        src/training/train_xtreme_sft.py \
        --config configs/xtreme_qwen1b7.yaml \
        --output_dir experiments/xtreme/qwen1b7 \
        2>&1 | tee "$LOGDIR/qwen1b7_sft.log"
    log "=== Qwen3-1.7B SFT done ==="

    log "=== Qwen3-1.7B: distillation training ==="
    CUDA_VISIBLE_DEVICES=0,1 /dev/shm/vllm/bin/accelerate launch \
        --num_processes 2 --mixed_precision bf16 --main_process_port 29500 \
        src/training/train_xtreme_distill.py \
        --base_config configs/xtreme_qwen1b7.yaml \
        --config configs/xtreme_distill.yaml \
        --output_dir experiments/xtreme/qwen1b7 \
        2>&1 | tee "$LOGDIR/qwen1b7_distill.log"
    log "=== Qwen3-1.7B distillation done ==="

    log "=== Qwen3-1.7B: control training (zero-shot teacher) ==="
    CUDA_VISIBLE_DEVICES=0,1 /dev/shm/vllm/bin/accelerate launch \
        --num_processes 2 --mixed_precision bf16 --main_process_port 29500 \
        src/training/train_xtreme_distill.py \
        --base_config configs/xtreme_qwen1b7.yaml \
        --config configs/xtreme_distill.yaml \
        --output_dir experiments/xtreme/qwen1b7 \
        --zeroshot_teacher \
        2>&1 | tee "$LOGDIR/qwen1b7_control.log"
    log "=== Qwen3-1.7B control done ==="

    log "=== Qwen3-1.7B: checkpoint eval ==="
    CUDA_VISIBLE_DEVICES=0,1 python scripts/eval_xtreme_checkpoints.py \
        --base_model Qwen/Qwen3-1.7B \
        --base_dir experiments/xtreme/qwen1b7 \
        --n_samples 500 \
        --batch_size 16 \
        --output experiments/xtreme/qwen1b7_trained.json \
        2>&1 | tee "$LOGDIR/qwen1b7_ckpt_eval.log"
    log "=== Qwen3-1.7B checkpoint eval done ==="

    # ── Gemma-3-270M ────────────────────────────────────────────────────────
    log "=== Gemma-270M: inference eval (base + fewshot) ==="
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_xtreme_inference.py \
        --model google/gemma-3-270m \
        --conditions base fewshot \
        --n_samples 500 \
        --batch_size 32 \
        --output experiments/xtreme/gemma270m_inference.json \
        2>&1 | tee "$LOGDIR/gemma270m_inference.log"
    log "=== Gemma-270M inference done ==="

    log "=== Gemma-270M: SFT training ==="
    CUDA_VISIBLE_DEVICES=0,1 /dev/shm/vllm/bin/accelerate launch \
        --num_processes 2 --mixed_precision bf16 --main_process_port 29500 \
        src/training/train_xtreme_sft.py \
        --config configs/xtreme_gemma270m.yaml \
        --output_dir experiments/xtreme/gemma270m \
        2>&1 | tee "$LOGDIR/gemma270m_sft.log"
    log "=== Gemma-270M SFT done ==="

    log "=== Gemma-270M: distillation training ==="
    CUDA_VISIBLE_DEVICES=0,1 /dev/shm/vllm/bin/accelerate launch \
        --num_processes 2 --mixed_precision bf16 --main_process_port 29500 \
        src/training/train_xtreme_distill.py \
        --base_config configs/xtreme_gemma270m.yaml \
        --config configs/xtreme_distill.yaml \
        --output_dir experiments/xtreme/gemma270m \
        2>&1 | tee "$LOGDIR/gemma270m_distill.log"
    log "=== Gemma-270M distillation done ==="

    log "=== Gemma-270M: control training ==="
    CUDA_VISIBLE_DEVICES=0,1 /dev/shm/vllm/bin/accelerate launch \
        --num_processes 2 --mixed_precision bf16 --main_process_port 29500 \
        src/training/train_xtreme_distill.py \
        --base_config configs/xtreme_gemma270m.yaml \
        --config configs/xtreme_distill.yaml \
        --output_dir experiments/xtreme/gemma270m \
        --zeroshot_teacher \
        2>&1 | tee "$LOGDIR/gemma270m_control.log"
    log "=== Gemma-270M control done ==="

    log "=== Gemma-270M: checkpoint eval ==="
    CUDA_VISIBLE_DEVICES=0,1 python scripts/eval_xtreme_checkpoints.py \
        --base_model google/gemma-3-270m \
        --base_dir experiments/xtreme/gemma270m \
        --n_samples 500 \
        --batch_size 16 \
        --output experiments/xtreme/gemma270m_trained.json \
        2>&1 | tee "$LOGDIR/gemma270m_ckpt_eval.log"
    log "=== Gemma-270M checkpoint eval done ==="

    log ">>> claude stream COMPLETE <<<"
}

# ─────────────────────────────────────────────────────────────────────────────
# STREAM: vscode  (GPUs 2,3)
# Llama-3.2-3B: inference → SFT → distill → control
# Qwen3-8B:     inference → SFT → distill → control
# ─────────────────────────────────────────────────────────────────────────────
run_vscode() {
    install_deps

    # ── Llama-3.2-3B ────────────────────────────────────────────────────────
    log "=== Llama-3.2-3B: inference eval (base + fewshot) ==="
    CUDA_VISIBLE_DEVICES=2,3 python scripts/eval_xtreme_inference.py \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --conditions base fewshot \
        --n_samples 500 \
        --batch_size 16 \
        --output experiments/xtreme/llama3b_inference.json \
        2>&1 | tee "$LOGDIR/llama3b_inference.log"
    log "=== Llama-3.2-3B inference done ==="

    log "=== Llama-3.2-3B: SFT training ==="
    CUDA_VISIBLE_DEVICES=2,3 /dev/shm/vllm/bin/accelerate launch \
        --num_processes 2 --mixed_precision bf16 --main_process_port 29501 \
        src/training/train_xtreme_sft.py \
        --config configs/xtreme_llama3b.yaml \
        --output_dir experiments/xtreme/llama3b \
        2>&1 | tee "$LOGDIR/llama3b_sft.log"
    log "=== Llama-3.2-3B SFT done ==="

    log "=== Llama-3.2-3B: distillation training ==="
    CUDA_VISIBLE_DEVICES=2,3 /dev/shm/vllm/bin/accelerate launch \
        --num_processes 2 --mixed_precision bf16 --main_process_port 29501 \
        src/training/train_xtreme_distill.py \
        --base_config configs/xtreme_llama3b.yaml \
        --config configs/xtreme_distill.yaml \
        --output_dir experiments/xtreme/llama3b \
        2>&1 | tee "$LOGDIR/llama3b_distill.log"
    log "=== Llama-3.2-3B distillation done ==="

    log "=== Llama-3.2-3B: control training ==="
    CUDA_VISIBLE_DEVICES=2,3 /dev/shm/vllm/bin/accelerate launch \
        --num_processes 2 --mixed_precision bf16 --main_process_port 29501 \
        src/training/train_xtreme_distill.py \
        --base_config configs/xtreme_llama3b.yaml \
        --config configs/xtreme_distill.yaml \
        --output_dir experiments/xtreme/llama3b \
        --zeroshot_teacher \
        2>&1 | tee "$LOGDIR/llama3b_control.log"
    log "=== Llama-3.2-3B control done ==="

    log "=== Llama-3.2-3B: checkpoint eval ==="
    CUDA_VISIBLE_DEVICES=2,3 python scripts/eval_xtreme_checkpoints.py \
        --base_model meta-llama/Llama-3.2-3B-Instruct \
        --base_dir experiments/xtreme/llama3b \
        --n_samples 500 \
        --batch_size 16 \
        --output experiments/xtreme/llama3b_trained.json \
        2>&1 | tee "$LOGDIR/llama3b_ckpt_eval.log"
    log "=== Llama-3.2-3B checkpoint eval done ==="

    # ── Qwen3-8B  (use all 4 GPUs for speed) ─────────────────────────────────
    # Wait for claude stream to free GPUs 0,1 before using all 4 for Qwen8B
    # Inference on GPUs 2,3 first, training with all 4 after

    log "=== Qwen3-8B: inference eval (base + fewshot) — GPUs 2,3 ==="
    CUDA_VISIBLE_DEVICES=2,3 python scripts/eval_xtreme_inference.py \
        --model Qwen/Qwen3-8B \
        --conditions base fewshot \
        --n_samples 500 \
        --batch_size 8 \
        --output experiments/xtreme/qwen8b_inference.json \
        2>&1 | tee "$LOGDIR/qwen8b_inference.log"
    log "=== Qwen3-8B inference done ==="

    log "=== Qwen3-8B: SFT training — GPUs 2,3 ==="
    CUDA_VISIBLE_DEVICES=2,3 /dev/shm/vllm/bin/accelerate launch \
        --num_processes 2 --mixed_precision bf16 --main_process_port 29501 \
        src/training/train_xtreme_sft.py \
        --config configs/xtreme_qwen8b.yaml \
        --output_dir experiments/xtreme/qwen8b \
        2>&1 | tee "$LOGDIR/qwen8b_sft.log"
    log "=== Qwen3-8B SFT done ==="

    log "=== Qwen3-8B: distillation training — GPUs 2,3 ==="
    CUDA_VISIBLE_DEVICES=2,3 /dev/shm/vllm/bin/accelerate launch \
        --num_processes 2 --mixed_precision bf16 --main_process_port 29501 \
        src/training/train_xtreme_distill.py \
        --base_config configs/xtreme_qwen8b.yaml \
        --config configs/xtreme_distill.yaml \
        --output_dir experiments/xtreme/qwen8b \
        2>&1 | tee "$LOGDIR/qwen8b_distill.log"
    log "=== Qwen3-8B distillation done ==="

    log "=== Qwen3-8B: control training — GPUs 2,3 ==="
    CUDA_VISIBLE_DEVICES=2,3 /dev/shm/vllm/bin/accelerate launch \
        --num_processes 2 --mixed_precision bf16 --main_process_port 29501 \
        src/training/train_xtreme_distill.py \
        --base_config configs/xtreme_qwen8b.yaml \
        --config configs/xtreme_distill.yaml \
        --output_dir experiments/xtreme/qwen8b \
        --zeroshot_teacher \
        2>&1 | tee "$LOGDIR/qwen8b_control.log"
    log "=== Qwen3-8B control done ==="

    log "=== Qwen3-8B: checkpoint eval ==="
    CUDA_VISIBLE_DEVICES=2,3 python scripts/eval_xtreme_checkpoints.py \
        --base_model Qwen/Qwen3-8B \
        --base_dir experiments/xtreme/qwen8b \
        --n_samples 500 \
        --batch_size 8 \
        --output experiments/xtreme/qwen8b_trained.json \
        2>&1 | tee "$LOGDIR/qwen8b_ckpt_eval.log"
    log "=== Qwen3-8B checkpoint eval done ==="

    log ">>> vscode stream COMPLETE <<<"
}

# ─────────────────────────────────────────────────────────────────────────────
# STREAM: results  (no GPU needed)
# ─────────────────────────────────────────────────────────────────────────────
run_results() {
    log "=== Generating xtreme_results.md ==="
    python scripts/gen_xtreme_results.py \
        --inference_dir experiments/xtreme \
        --output xtreme_results.md \
        --assets_dir assets/xtreme \
        2>&1 | tee "$LOGDIR/gen_results.log"
    log "=== Results generated: xtreme_results.md ==="
}

# ─────────────────────────────────────────────────────────────────────────────
# Dispatch
# ─────────────────────────────────────────────────────────────────────────────
case "$STREAM" in
    claude)  run_claude  ;;
    vscode)  run_vscode  ;;
    results) run_results ;;
    help|*)
        echo "Usage: bash scripts/run_xtreme_overnight.sh <stream>"
        echo "  claude   — Qwen3-1.7B + Gemma-270M (GPUs 0,1)"
        echo "  vscode   — Llama-3.2-3B + Qwen3-8B (GPUs 2,3)"
        echo "  results  — generate xtreme_results.md (no GPU)"
        echo ""
        echo "Paste into tmux sessions or run directly in the container."
        ;;
esac
