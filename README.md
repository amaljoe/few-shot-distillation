# Few-Shot Distillation: Distilling In-Context Learning into Model Weights

Distill few-shot (in-context learning) behavior into model parameters via layer-wise
hidden-state supervision. A teacher model running with 8-shot context provides activation
targets; a student model trained zero-shot learns to match those internal representations
alongside the standard task loss.

---

## Hypothesis

Few-shot context performs an implicit low-rank update through attention. If we supervise a
zero-shot model's internal representations using the corresponding few-shot activations at
each layer, the model can internalize that adaptation signal into its weights — without
needing context at inference time.

---

## Training Objective

```
L_total = L_CE  +  λ * Σ_l || h_l(student) − h_l(teacher) ||²
```

- `L_CE` — cross-entropy on the answer tokens (standard SFT loss)
- `h_l(·)` — hidden state at layer `l`, at the last token of the query prompt
- `λ` — distillation weight (default 0.5)
- Teacher activations precomputed once and cached; student activations computed on-the-fly

---

## Experimental Setup

### Phase 0 — ICL Gap Evaluation

Evaluated three Qwen3 models on 16 GSM8K test samples at 0-shot, 4-shot, and 8-shot:

| Model                    | 0-shot | 8-shot | Gap    |
|--------------------------|--------|--------|--------|
| Qwen/Qwen3-1.7B          | 50.0%  | 62.5%  | +12.5% |
| Qwen/Qwen3-4B-Instruct-2507 | 81.2% | 93.8% | +12.5% |
| Qwen/Qwen3-8B            | 18.8%  | 68.8%  | +50.0% |

**Selected: `Qwen/Qwen3-1.7B`** — smallest model with a clear ICL gap (≥5pp).

### Phase 1 — Distillation POC

Three conditions, all using Qwen3-1.7B + LoRA (r=16, α=32) on GSM8K train split:

| Condition | Description |
|-----------|-------------|
| A | 8-shot inference (Phase 0 baseline, no fine-tuning) |
| B | Zero-shot LoRA SFT — standard fine-tuning, no distillation |
| C | Layer-wise distillation — LoRA SFT + hidden-state matching |

- 1000 steps, effective batch size 64 (4 GPUs × batch 4 × grad accum 4)
- Conditions B and C trained in parallel on 4×A100 80GB (B: GPUs 0,1 · C: GPUs 2,3)

---

## Results

Evaluated on the full GSM8K test set (1319 examples), zero-shot inference for all trained conditions.

| Condition | Description | Accuracy | Correct / Total |
|-----------|-------------|----------|-----------------|
| A — 8-shot teacher | Few-shot inference, no fine-tuning (16-sample estimate) | ~62.5% | — |
| B — Zero-shot LoRA baseline | Standard SFT, no distillation | 62.47% | 824 / 1319 |
| C — Layer-wise distillation | LoRA SFT + hidden-state matching (λ=0.5, all 28 layers) | **63.68%** | **840 / 1319** |

**Distillation (C) outperforms baseline (B) by +1.21pp** (+16 correct answers on 1319 examples).
Both fine-tuned conditions match the Phase 0 8-shot teacher accuracy at zero-shot inference time,
with the distilled model edging ahead — consistent with the hypothesis that layer-wise supervision
encodes the teacher's few-shot adaptation signal into the student's weights.

---

## Repository Structure

```
configs/
├── base.yaml                  # Model, data, training, LoRA hyperparameters
└── distill_layerwise.yaml     # Distillation overrides (λ, layers, normalize)

src/
├── data/
│   └── gsm8k_loader.py        # Dataset, prompt formatting, batching
├── hooks/
│   └── activation_capture.py  # Forward hooks for hidden-state extraction
├── models/
│   ├── teacher_wrapper.py     # Frozen teacher with activation capture
│   └── student.py             # LoRA student with optional capture
├── losses/
│   └── layer_matching.py      # MSE + cosine similarity utilities
└── training/
    ├── train_baseline.py      # Condition B
    └── train_layerwise_distill.py  # Condition C

scripts/
├── eval_icl.py                # Phase 0: few-shot vs zero-shot accuracy (vLLM)
├── precompute_teacher_activations.py  # Cache teacher hidden states (HuggingFace)
└── evaluate.py                # Final eval via vLLM server + LoRA adapter

experiments/poc/
├── teacher_cache/
│   ├── activations.pt         # (7473, 28, 2048) float16 — 0.86 GB
│   └── meta.json
├── baseline/                  # Condition B checkpoints + TensorBoard logs
└── distill/                   # Condition C checkpoints + TensorBoard logs
```

---

## Reproducing the Experiment

### Prerequisites

```bash
pip install torch transformers datasets peft accelerate omegaconf \
            vllm tensorboard openai tqdm
```

Compute node: 4×A100 80GB. Start via `app` alias → apptainer → activate `/dev/shm/vllm` env.

### Step 1 — Phase 0: measure ICL gap

```bash
# Run in parallel on separate GPUs (tmux sessions)
CUDA_VISIBLE_DEVICES=0 python scripts/eval_icl.py \
    --model Qwen/Qwen3-1.7B --num_samples 16 --num_fewshot 0 4 8 \
    --output experiments/poc/icl_eval_1b7.json

CUDA_VISIBLE_DEVICES=1 python scripts/eval_icl.py \
    --model Qwen/Qwen3-4B-Instruct-2507 --num_samples 16 --num_fewshot 0 4 8 \
    --output experiments/poc/icl_eval_4b.json

CUDA_VISIBLE_DEVICES=2 python scripts/eval_icl.py \
    --model Qwen/Qwen3-8B --num_samples 16 --num_fewshot 0 4 8 \
    --output experiments/poc/icl_eval_8b.json
```

Pick the smallest model with gap ≥ 5pp. Update `configs/base.yaml` with correct
`model.name`, `model.num_layers`, and `model.hidden_size`.

### Step 2 — Precompute teacher activations (run once)

```bash
# Verify token alignment before full precompute:
python scripts/precompute_teacher_activations.py --config configs/base.yaml --verify

# Full precompute (~7 min on 2×A100):
CUDA_VISIBLE_DEVICES=0,1 python scripts/precompute_teacher_activations.py \
    --config configs/base.yaml
# Output: experiments/poc/teacher_cache/activations.pt
```

Uses HuggingFace (not vLLM) — vLLM does not expose internal hidden states.

### Step 3 — Train Condition B and C in parallel

```bash
# Condition B — zero-shot baseline (GPUs 0,1, tmux: claude)
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --mixed_precision bf16 \
    src/training/train_baseline.py \
    --config configs/base.yaml \
    --output_dir experiments/poc/baseline

# Condition C — layer-wise distillation (GPUs 2,3, tmux: vscode)
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 --mixed_precision bf16 \
    src/training/train_layerwise_distill.py \
    --config configs/distill_layerwise.yaml \
    --output_dir experiments/poc/distill
```

### Step 4 — Final evaluation

```bash
# Serve both LoRA adapters together (all 4 GPUs):
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Qwen/Qwen3-1.7B \
    --enable-lora \
    --lora-modules \
        baseline=experiments/poc/baseline/baseline/final \
        distill=experiments/poc/distill/distill/final \
    --port 8000 --tensor-parallel-size 4 \
    --max-lora-rank 16 --max-model-len 2048

# Evaluate both conditions (run from inside apptainer env):
python scripts/evaluate.py \
    --config configs/base.yaml \
    --api_base http://localhost:8000/v1 \
    --lora_names baseline distill \
    --output experiments/poc/final_results.json
```

Note: run `scripts/evaluate.py` from inside the apptainer environment (tmux session).
vLLM binds to the container's localhost and is not reachable from the host shell directly.

### TensorBoard

```bash
tensorboard --logdir experiments/poc --port 6006 --bind_all
```

---

## Key Implementation Details

**Token alignment.** The distillation loss is computed at a single token per example: the
last token of the target question, just before the `<|im_start|>assistant` generation
prompt. This position is structurally identical in both the teacher (8-shot sequence) and
the student (0-shot sequence), making hidden-state matching meaningful.

**Teacher cache.** Precomputed as `(N, num_layers, hidden_size)` float16 on disk. Loaded
into CPU RAM at training start (~0.86 GB for 1.7B). Per batch, the relevant rows are
indexed by `example_idx` and moved to GPU. This avoids running the teacher at every
training step.

**LoRA hook path.** Hooks attach to `peft_model.base_model.model.model.layers[i]`, not the
PEFT wrapper. `output[0]` is the post-residual hidden state for Qwen3.

**Qwen3 thinking mode.** Always disabled (`enable_thinking=False`). Thinking tokens break
the `#### <number>` answer extraction pattern.

---

## Model Specs (Qwen3 family)

| Model | Layers | Hidden size | Cache size (7473 examples) |
|-------|--------|-------------|---------------------------|
| Qwen3-1.7B | 28 | 2048 | ~0.86 GB |
| Qwen3-4B | 36 | 2560 | ~1.4 GB |
| Qwen3-8B | 36 | 4096 | ~2.2 GB |

---

## Configuration

`configs/base.yaml` — all hyperparameters. Key fields:

```yaml
model:
  name: "Qwen/Qwen3-1.7B"
  num_layers: 28
  hidden_size: 2048

training:
  max_steps: 1000
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  lr: 2.0e-4

lora:
  r: 16
  alpha: 32
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
```

`configs/distill_layerwise.yaml` — distillation overrides:

```yaml
distillation:
  lambda_distill: 0.5
  layers_to_match: "all"
  normalize_hidden: true
```

---

## Contributors

- Amal Joe (IIT Bombay)
