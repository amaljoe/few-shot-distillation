# IFEval Extension Results

Extension of ICL distillation to IFEval — open-ended instruction following with
programmatically verifiable constraints.

**Training data:** `argilla/ifeval-like-data` (filtered config, pre-verified compliant examples)
**Eval data:** `google/IFEval` (541 prompts with constraint metadata)
**Metric:** prompt-level strict accuracy (primary) + instruction-level accuracy

---

## Prerequisites

```bash
# Install dependencies:
pip install langdetect immutabledict nltk absl-py
```

---

## Section 1 — ICL Gap (Qwen3-1.7B on IFEval)

| Condition | Prompt Acc. | Instruction Acc. |
|---|---|---|
| 0-shot | 7.21% (39/541) | 13.79% |
| 4-shot | 8.50% (46/541) | 15.83% |
| **ICL gap** | **+1.29%** | — |

Note: IFEval ICL gap is inherently small — constraints are stated in the prompt itself,
so few-shot examples add little signal. Training on 56k argilla verified examples is
the primary lever.

---

## Section 2 — Commands

### Step 0: ICL gap evaluation
```bash
# claude session, GPUs 0,1
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/eval_ifeval_icl.py \
  --model Qwen/Qwen3-1.7B --num_fewshot 0 4 \
  --n_samples 541 \
  --output experiments/ifeval_qwen1b7/icl_eval.json \
  --tensor_parallel_size 4 --max_new_tokens 512
```

### Step 1: Training (baseline + distillation in parallel on all 4 GPUs)
```bash
# claude session, GPUs 0,1 — SFT baseline
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --mixed_precision bf16 \
  --main_process_port 29500 src/training/train_baseline.py \
  --config configs/ifeval_qwen1b7.yaml \
  --output_dir experiments/ifeval_qwen1b7/baseline

# vscode session, GPUs 2,3 — logit distillation
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 --mixed_precision bf16 \
  --main_process_port 29501 src/training/train_online_v1.py \
  --base_config configs/ifeval_qwen1b7.yaml --config configs/ifeval_online_v1.yaml \
  --output_dir experiments/ifeval_qwen1b7/online_v1
```

### Step 2: Checkpoint evaluation
```bash
# claude session, all 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/eval_ifeval_checkpoints.py \
  --config configs/ifeval_qwen1b7.yaml \
  --conditions baseline online_v1 \
  --base_dir experiments/ifeval_qwen1b7 \
  --n_samples 541 --checkpoint_steps 200 400 600 800 1000 \
  --output experiments/ifeval_qwen1b7/results.json \
  --tensor_parallel_size 4 --max_new_tokens 512
```

---

## Section 3 — Checkpoint Results (Qwen3-1.7B)

### Prompt-level strict accuracy (primary metric)

| Step | Baseline | Distillation (online_v1) |
|---|---|---|
| 200 | 40.67% | 37.89% |
| 400 | 43.25% | 41.04% |
| 600 | 42.70% | 40.67% |
| 800 | 41.04% | 40.30% |
| 1000 | 42.14% | 38.45% |

### Instruction-level accuracy

| Step | Baseline | Distillation (online_v1) |
|---|---|---|
| 200 | 48.44% | 45.08% |
| 400 | 51.08% | 47.48% |
| 600 | 51.20% | 47.48% |
| 800 | 49.64% | 47.84% |
| 1000 | 50.48% | 45.80% |

---

## Section 4 — Summary: Distillation Gain Across Benchmarks (step 1000, Qwen3-1.7B)

| Dataset | 0-shot | SFT baseline | Distillation | Gain (Distill−SFT) |
|---|---|---|---|---|
| GSM8K | 26.08% | 63.00% | 71.04% | **+8.04%** |
| CSQA | 43.00% | 73.38% | 74.04% | +0.66% |
| MATH | 1.50% | 28.60% | 41.60% | **+13.00%** |
| **IFEval** | **7.21%** | **42.14%** | **38.45%** | **−3.69%** |

---

## Smoke Tests (verify before full run)

### Dataset fields check
```bash
python -c "
from datasets import load_dataset
a = load_dataset('argilla/ifeval-like-data', 'filtered', split='train')
g = load_dataset('google/IFEval', split='train')
print('argilla cols:', a.column_names)
print('argilla[0] split field:', a[0].get('split'))
print('argilla n compliant:', sum(1 for x in a if x['prompt_level_strict_acc']))
print('ifeval cols:', g.column_names)
print('ifeval n:', len(g))
"
```

### Loader smoke test
```bash
python -c "
from src.data.ifeval_loader import load_ifeval, IFEvalDistillDataset
from transformers import AutoTokenizer
exs = load_ifeval('train'); print(len(exs), list(exs[0].keys()))
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-1.7B')
ds = IFEvalDistillDataset(exs, tok, num_fewshot=4)
s = ds[0]; print({k: len(v) if isinstance(v, list) else v for k, v in s.items()})
"
```

### ICL eval smoke test (8 examples)
```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/eval_ifeval_icl.py \
  --model Qwen/Qwen3-1.7B --num_fewshot 0 4 --n_samples 8 \
  --output /tmp/ifeval_test.json --tensor_parallel_size 2 --max_new_tokens 128
```

### Training smoke test (5 steps)
```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --mixed_precision bf16 \
  --main_process_port 29500 src/training/train_baseline.py \
  --config configs/ifeval_qwen1b7.yaml --output_dir /tmp/ifeval_smoke --max_steps 5
```
