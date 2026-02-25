#!/usr/bin/env bash
# =============================================================================
# Ablation suite runner
#
# Runs all ablation experiments end-to-end with ingrained evaluation:
#   1. Precompute teacher logits (from hidden-state cache, ~2 min)
#   2. Checkpoint curve eval for B and C (GPU 0,1, ~25 min)  [PARALLEL]
#      Condition D training — logit-level KL (GPU 2,3, ~15 min)  [PARALLEL]
#   3. Lambda sweep: λ=0.1, λ=1.0, λ=2.0 (GPU 0,1 + 2,3, 2-way parallel)
#   4. Eval all new conditions (Condition D + lambda sweep)
#   5. Aggregate results into experiments/ablations/summary.json
#
# Run from repo root inside apptainer (tmux: claude):
#   bash scripts/run_ablations.sh 2>&1 | tee experiments/ablations/run.log
# =============================================================================

set -e
cd "$(dirname "$0")/.."
echo "=== Ablation suite start: $(date) ==="

# ---------------------------------------------------------------------------
# Step 1: Precompute teacher logits (fast, ~2 min)
# ---------------------------------------------------------------------------
LOGIT_CACHE="experiments/poc/teacher_cache/logits_top1024.pt"
if [ ! -f "$LOGIT_CACHE" ]; then
    echo ""
    echo "--- Step 1: Precompute teacher logits ---"
    python scripts/precompute_teacher_logits.py --config configs/base.yaml
else
    echo "--- Step 1: Teacher logit cache found, skipping ---"
fi

# ---------------------------------------------------------------------------
# Step 2 (parallel):
#   GPU 0,1 — checkpoint curve eval for B and C
#   GPU 2,3 — train Condition D (logit-level KL)
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 2: Checkpoint eval (GPU 0,1) + Condition D train (GPU 2,3) ---"

CKPT_RESULTS="experiments/ablations/checkpoint_curve/results.json"
COND_D_FINAL="experiments/ablations/kl_distill/kl_distill/final"

if [ ! -f "$CKPT_RESULTS" ]; then
    CUDA_VISIBLE_DEVICES=0,1 python scripts/eval_checkpoints.py \
        --config configs/base.yaml \
        --n_samples 400 \
        --output "$CKPT_RESULTS" \
        2>&1 | tee experiments/ablations/checkpoint_curve/eval.log &
    CKPT_PID=$!
    echo "  Checkpoint eval PID: $CKPT_PID"
else
    echo "  Checkpoint results found, skipping eval"
    CKPT_PID=""
fi

if [ ! -f "$COND_D_FINAL/adapter_config.json" ]; then
    CUDA_VISIBLE_DEVICES=2,3 accelerate launch \
        --num_processes 2 --mixed_precision bf16 \
        --main_process_port 29501 \
        src/training/train_kl_distill.py \
        --config experiments/ablations/configs/kl_distill.yaml \
        --output_dir experiments/ablations/kl_distill \
        2>&1 | tee experiments/ablations/kl_distill/train.log &
    KL_PID=$!
    echo "  KL distill train PID: $KL_PID"
else
    echo "  Condition D checkpoint found, skipping training"
    KL_PID=""
fi

# Wait for both to complete
[ -n "$CKPT_PID" ] && wait $CKPT_PID && echo "  ✓ Checkpoint eval done"
[ -n "$KL_PID"   ] && wait $KL_PID   && echo "  ✓ Condition D training done"

# ---------------------------------------------------------------------------
# Step 3: Lambda sweep
#   Round A (parallel): λ=0.1 (GPU 0,1) + λ=1.0 (GPU 2,3)
#   Round B:            λ=2.0 (GPU 0,1)
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 3: Lambda sweep ---"

run_lambda() {
    local LAMBDA_TAG=$1        # e.g. "01"
    local GPU_IDS=$2           # e.g. "0,1"
    local PORT=$3              # e.g. 29502
    local OUT_DIR="experiments/ablations/lambda_${LAMBDA_TAG}"
    local FINAL="${OUT_DIR}/distill/final"   # train_layerwise_distill appends /distill
    local LOG="${OUT_DIR}/train.log"

    if [ ! -f "$FINAL/adapter_config.json" ]; then
        mkdir -p "$OUT_DIR"
        CUDA_VISIBLE_DEVICES=$GPU_IDS accelerate launch \
            --num_processes 2 --mixed_precision bf16 \
            --main_process_port $PORT \
            src/training/train_layerwise_distill.py \
            --config experiments/ablations/configs/lambda_${LAMBDA_TAG}.yaml \
            --output_dir "$OUT_DIR" \
            2>&1 | tee "$LOG"
        echo "  ✓ λ=${LAMBDA_TAG} training done"
    else
        echo "  λ=${LAMBDA_TAG} checkpoint found, skipping"
    fi
}

# Round A: λ=0.1 and λ=1.0 in parallel
run_lambda "01" "0,1" 29502 &
LAMBDA01_PID=$!
run_lambda "10" "2,3" 29503 &
LAMBDA10_PID=$!
wait $LAMBDA01_PID && echo "  ✓ λ=0.1 done"
wait $LAMBDA10_PID && echo "  ✓ λ=1.0 done"

# Round B: λ=2.0
run_lambda "20" "0,1" 29502

# ---------------------------------------------------------------------------
# Step 4: Evaluate all new conditions (full test set on 2 GPUs)
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 4: Evaluate new conditions ---"

COMBINED_EVAL="experiments/ablations/ablation_eval.json"
if [ ! -f "$COMBINED_EVAL" ]; then
    CUDA_VISIBLE_DEVICES=0,1 python scripts/eval_adapter.py \
        --config configs/base.yaml \
        --adapter_path \
            experiments/ablations/kl_distill/kl_distill/final \
            experiments/ablations/lambda_01/distill/final \
            experiments/ablations/lambda_10/distill/final \
            experiments/ablations/lambda_20/distill/final \
        --adapter_name cond_d lambda_01 lambda_10 lambda_20 \
        --output "$COMBINED_EVAL" \
        2>&1 | tee experiments/ablations/eval.log
else
    echo "  Ablation eval found, skipping"
fi

# ---------------------------------------------------------------------------
# Step 5: Aggregate all results
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 5: Aggregating results ---"
python - <<'PYEOF'
import json
from pathlib import Path

summary = {
    "poc_results": {},
    "checkpoint_curve": {},
    "ablations": {},
    "lambda_sweep": {},
}

# POC final results
poc_path = Path("experiments/poc/final_results.json")
if poc_path.exists():
    with open(poc_path) as f:
        poc = json.load(f)
    for cond, data in poc["conditions"].items():
        summary["poc_results"][cond] = {
            "accuracy": data["accuracy"],
            "correct": data["correct"],
            "total": data["total"],
        }

# Checkpoint curve
ckpt_path = Path("experiments/ablations/checkpoint_curve/results.json")
if ckpt_path.exists():
    with open(ckpt_path) as f:
        ckpt = json.load(f)
    for cond, steps in ckpt["conditions"].items():
        summary["checkpoint_curve"][cond] = {
            step: data["accuracy"] for step, data in steps.items()
        }

# Ablation eval (Cond D + lambda sweep)
abl_path = Path("experiments/ablations/ablation_eval.json")
if abl_path.exists():
    with open(abl_path) as f:
        abl = json.load(f)
    for name, data in abl["conditions"].items():
        if name.startswith("lambda"):
            lam = name.replace("lambda_", "λ=0.")
            if name == "lambda_10": lam = "λ=1.0"
            elif name == "lambda_20": lam = "λ=2.0"
            elif name == "lambda_01": lam = "λ=0.1"
            summary["lambda_sweep"][lam] = {
                "accuracy": data["accuracy"],
                "correct": data["correct"],
                "total": data["total"],
            }
        else:
            summary["ablations"][name] = {
                "accuracy": data["accuracy"],
                "correct": data["correct"],
                "total": data["total"],
            }

# Add Condition C (λ=0.5) to lambda sweep for completeness
if "distill" in summary["poc_results"]:
    summary["lambda_sweep"]["λ=0.5 (C)"] = summary["poc_results"]["distill"]

out = Path("experiments/ablations/summary.json")
with open(out, "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*60)
print("ABLATION SUMMARY")
print("="*60)

print("\n--- POC Results (full test set, 1319 examples) ---")
for cond, d in summary["poc_results"].items():
    print(f"  {cond:20s}: {d['accuracy']:.2%}  ({d['correct']}/{d['total']})")

print("\n--- Checkpoint Curve ---")
conds = list(summary["checkpoint_curve"].keys())
if conds:
    print(f"  {'Step':>6}", end="")
    for c in conds:
        print(f"  {c:>10}", end="")
    print()
    steps = sorted(summary["checkpoint_curve"][conds[0]].keys())
    for s in steps:
        print(f"  {s:>6}", end="")
        for c in conds:
            acc = summary["checkpoint_curve"][c].get(s, 0)
            print(f"  {acc:>9.2%}", end="")
        print()

print("\n--- Condition D (logit-level KL) vs Condition C (hidden-state MSE) ---")
print(f"  {'Cond B baseline':20s}: {summary['poc_results'].get('baseline', {}).get('accuracy', 0):.2%}")
print(f"  {'Cond C (hidden MSE)':20s}: {summary['poc_results'].get('distill', {}).get('accuracy', 0):.2%}")
for name, d in summary["ablations"].items():
    print(f"  {name:20s}: {d['accuracy']:.2%}")

print("\n--- Lambda Sweep (Condition C with varying λ) ---")
for lam in sorted(summary["lambda_sweep"].keys()):
    d = summary["lambda_sweep"][lam]
    print(f"  {lam:10s}: {d['accuracy']:.2%}  ({d['correct']}/{d['total']})")

print(f"\nFull summary: experiments/ablations/summary.json")
PYEOF

echo ""
echo "=== Ablation suite complete: $(date) ==="
