# Extension Results: CommonsenseQA + MATH

Model: Qwen/Qwen3-1.7B (auto-escalate to Qwen3-8B if ICL gap < 5pp)

---

## 1. ICL Gap Evaluation

### CommonsenseQA (validation set, n=500)

| Shots | Accuracy |
|-------|----------|
| 0-shot | 43.00% (215/500) |
| 4-shot | 55.00% (275/500) |
| **ICL Gap** | **+12.00%** |

> ICL gap ≥ 5pp → using Qwen3-1.7B (no escalation). Training config uses 5-shot; eval ran 4-shot (closest available).

Eval command:
```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/eval_icl.py \
  --model Qwen/Qwen3-1.7B --dataset commonsenseqa \
  --num_samples 500 --output experiments/csqa_qwen1b7/icl_eval.json
```

### MATH (test set, n=200)

| Shots | Accuracy |
|-------|----------|
| 0-shot | 1.50% (3/200) |
| 4-shot | 11.00% (22/200) |
| **ICL Gap** | **+9.50%** |

> ICL gap ≥ 5pp → using Qwen3-1.7B (no escalation).

Eval command:
```bash
CUDA_VISIBLE_DEVICES=2,3 python scripts/eval_icl.py \
  --model Qwen/Qwen3-1.7B --dataset math \
  --num_samples 200 --max_new_tokens 1024 --max_model_len 8192 \
  --output experiments/math_qwen1b7/icl_eval.json
```

---

## 2. Training Commands

### CommonsenseQA

```bash
# Baseline (GPUs 0,1)
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --mixed_precision bf16 \
  --main_process_port 29500 src/training/train_baseline.py \
  --config configs/csqa_qwen1b7.yaml --output_dir experiments/csqa_qwen1b7/baseline

# Distillation (GPUs 2,3 — parallel with baseline)
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 --mixed_precision bf16 \
  --main_process_port 29501 src/training/train_online_v1.py \
  --base_config configs/csqa_qwen1b7.yaml --config configs/csqa_online_v1.yaml \
  --output_dir experiments/csqa_qwen1b7/online_v1
```

### MATH

```bash
# Baseline
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --mixed_precision bf16 \
  --main_process_port 29500 src/training/train_baseline.py \
  --config configs/math_qwen1b7.yaml --output_dir experiments/math_qwen1b7/baseline

# Distillation
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --mixed_precision bf16 \
  --main_process_port 29500 src/training/train_online_v1.py \
  --base_config configs/math_qwen1b7.yaml --config configs/math_online_v1.yaml \
  --output_dir experiments/math_qwen1b7/online_v1
```

---

## 3. Checkpoint Evaluation

### CommonsenseQA

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/eval_checkpoints.py \
  --config configs/csqa_qwen1b7.yaml --dataset commonsenseqa \
  --conditions baseline online_v1 --base_dir experiments/csqa_qwen1b7 \
  --n_samples 1221 --checkpoint_steps 200 400 600 800 1000 \
  --output experiments/csqa_qwen1b7/results.json --tensor_parallel_size 4
```

Results (n=1221, full validation set):

| Step | baseline | online_v1 |
|------|----------|-----------|
| 0 (ICL 0-shot) | 43.00% | — |
| 0 (ICL 4-shot) | — | 55.00% |
| 200 | 77.97% | 72.89% |
| 400 | 79.52% | 75.10% |
| 600 | 79.28% | 75.51% |
| 800 | 77.48% | 73.55% |
| 1000 | 73.38% | 74.04% |

### MATH

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/eval_checkpoints.py \
  --config configs/math_qwen1b7.yaml --dataset math \
  --conditions baseline online_v1 --base_dir experiments/math_qwen1b7 \
  --n_samples 500 --checkpoint_steps 200 400 600 800 1000 \
  --output experiments/math_qwen1b7/results.json --tensor_parallel_size 4 \
  --max_new_tokens 1024 --max_model_len 4096
```

Results (n=500):

| Step | baseline | online_v1 |
|------|----------|-----------|
| 0 (ICL 0-shot) | 1.50% | — |
| 0 (ICL 4-shot) | — | 11.00% |
| 200 | 32.40% | 43.80% |
| 400 | 29.80% | 43.60% |
| 600 | 31.60% | 44.00% |
| 800 | 31.20% | 41.00% |
| 1000 | 28.60% | 41.60% |

---

## 4. Summary (vs GSM8K baseline)

| Dataset | ICL Gap | SFT (step 1000) | Distill (step 1000) | Distill gain |
|---------|---------|-----------------|----------------------|--------------|
| GSM8K | +21.00% | 63.00% | 71.04% | +8.04% |
| CommonsenseQA | +12.00% | 73.38% | 74.04% | +0.66% |
| MATH | +9.50% | 28.60% | 41.60% | **+13.00%** |
