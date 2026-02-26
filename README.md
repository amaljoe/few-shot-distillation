# Few-Shot Distillation: Baking In-Context Learning into Model Weights

> **Train a model once. Get few-shot accuracy at zero-shot inference cost.**

We distill the behavior of few-shot in-context learning (ICL) into model parameters.
A teacher — the *same model* running with 8-shot context — produces a richer output
distribution at every answer token. A LoRA student, trained on zero-shot inputs, is
supervised to match those distributions alongside the standard task loss.
At inference, no context is needed.

---

## Result

**Qwen3-1.7B on GSM8K** (full test set, 1319 examples):

| Method | Accuracy | vs Base |
|--------|----------|---------|
| Base model, 0-shot | 26.08% | — |
| 8-shot in-context learning | 47.08% | +21.0pp |
| LoRA SFT (zero-shot fine-tuning) | 64.29% | +38.2pp |
| **LoRA SFT + Few-Shot Distillation (ours)** | **72.71%** | **+46.6pp** |

Our method improves over standard LoRA fine-tuning by **+8.4pp** while retaining
zero-shot inference — **no context window, no retrieval, no prompts**.

![Main comparison](assets/main_comparison.png)

---

## Why Not Standard Fine-Tuning or Knowledge Distillation?

### The Hard Label Problem

Standard supervised fine-tuning (SFT) trains on **one-hot targets** — for each token
position the model is pushed toward a single correct token and away from everything else.
This ignores the rich uncertainty structure in language: at any given position many tokens
are plausible, near-miss tokens carry useful signal, and the relative probabilities
between likely continuations encode meaning that a hard label discards entirely.

### Why Standard Knowledge Distillation Doesn't Apply Here

Classical knowledge distillation sidesteps the hard-label problem by training a small
student to match the **soft output distribution** of a larger teacher. But this requires
a larger, stronger teacher model — which is expensive to train and serve, and often
unavailable for the exact task you care about.

### Our Key Insight: Few-Shot Context as a Free Teacher

We show that **you don't need a bigger model**. The same model, given 8 in-context
examples, produces a fundamentally different — and richer — output distribution at every
answer token than it would without context. This few-shot model is a free teacher:

- **Same architecture and parameters** — no extra training or storage
- **Soft labels for free** — the teacher's top-256 vocabulary logits encode a full
  probability distribution over plausible next tokens at each answer position
- **Position-specific signal** — each of ~160 answer tokens gets its own soft target
  conditioned on the few-shot context, instead of the one hard label from the gold answer

The student (LoRA fine-tuned, zero-shot) learns to match these soft distributions
alongside the standard CE loss. The result: the few-shot reasoning behavior is
internalized into the student's weights. At inference, no context is needed.

---

## Method

```
L_total = L_CE  +  λ · MSE( top-K teacher logits, student logits at same vocab indices )
```

**Token alignment.** Both teacher and student sequences end with the identical answer
token IDs. The teacher processes `[8-shot context] + [question] + [answer]`; the student
processes `[question] + [answer]`. For each answer token position `t`:

```
Teacher:  [shot₁]...[shot₈][question][answer_t₀][answer_t₁]...
Student:                    [question][answer_t₀][answer_t₁]...
                                       ↑↑↑ identical suffix ↑↑↑
```

**At every answer token**, we take the teacher's top-256 vocabulary logits and supervise
the student to match them. This provides dense, position-wise signals across the full
answer — average ~160 positions per GSM8K example.

**Online teacher.** No precomputed cache. The teacher is the same base model (frozen,
no LoRA), run live under `torch.no_grad()` during each training step. Both models share
base weights; only LoRA parameters in the student are updated.

---

## Checkpoint Accuracy Curve

GSM8K accuracy evaluated at each checkpoint on the full test set (1319 examples):

| Step | LoRA SFT | + Distillation | Δ |
|------|----------|----------------|---|
| 200 | 61.79% | **72.40%** | **+10.61pp** |
| 400 | 64.29% | 71.49% | +7.20pp |
| 600 | 63.68% | **72.71%** | +9.02pp |
| 800 | 63.46% | 71.72% | +8.26pp |
| 1000 | 62.17% | 71.04% | +8.87pp |

Distillation leads at **every** checkpoint by **+7–11pp**. The gap is consistent across
training — not an artifact of early-step noise or a single lucky checkpoint.

### Training Loss

![Loss comparison](assets/loss_comparison.png)

The distillation model has **higher** CE loss throughout training (~0.60 at step 1000
vs ~0.27 for SFT). This is expected: the distillation term pulls the student's logits
toward the teacher's few-shot distribution, which differs from the hard ground-truth
labels the CE loss is measured against. The model is trading off label memorisation for
a richer, teacher-informed signal.

The result is classic soft-label distillation behaviour: **worse fit to training labels,
much better generalisation**. SFT overfits to one-hot targets and plateaus at 64%;
the distillation model, despite (because of) its higher CE loss, reaches 72.71% on the
test set. The teacher's soft distribution acts as a regulariser that prevents over-fitting
to the specific gold tokens and instead captures the broader structure of the problem.

---

## Setup

**Model:** Qwen/Qwen3-1.7B · **Dataset:** GSM8K · **Adapter:** LoRA r=16 α=32
**Training:** 1000 steps · effective batch 32 · lr 2×10⁻⁴ · bf16 · 4×A100 80GB
**Distillation:** λ=0.5 · top-256 vocab logits · online teacher (same model, frozen)

---

## Reproduce

### Prerequisites

```bash
pip install torch transformers datasets peft accelerate omegaconf \
            vllm tensorboard tqdm
```

4×A100 80GB. Start via `app` alias → apptainer → activate `/dev/shm/vllm` env.
See `compute.md` for full environment notes.

### Step 1 — Train LoRA SFT baseline

```bash
# GPUs 0,1
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --num_processes 2 --mixed_precision bf16 --main_process_port 29500 \
    src/training/train_baseline.py --config configs/base.yaml \
    --output_dir experiments/poc/baseline
```

### Step 2 — Train with Few-Shot Distillation

```bash
# GPUs 0,1  (online teacher runs on same GPUs, frozen)
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --num_processes 2 --mixed_precision bf16 --main_process_port 29500 \
    src/training/train_online_v1.py --config configs/online_v1.yaml \
    --output_dir experiments/online_v1
```

### Step 3 — Evaluate checkpoints

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/eval_checkpoints.py \
    --config configs/base.yaml \
    --n_samples 1319 \
    --conditions online_v1 \
    --base_dir experiments \
    --checkpoint_steps 200 400 600 800 1000 \
    --output experiments/online_v1_full_eval.json \
    --tensor_parallel_size 4
```

---

## Repository Structure

```
configs/
├── base.yaml           # model, data, training, LoRA hyperparameters
└── online_v1.yaml      # few-shot distillation overrides (λ, top-K vocab)

src/
├── data/
│   └── gsm8k_loader.py     # dataset, prompt formatting, batching
│                             # teacher_include_answer=True for online distillation
├── models/
│   └── student.py           # LoRA student
└── training/
    ├── train_baseline.py    # LoRA SFT only
    └── train_online_v1.py   # LoRA SFT + online few-shot logit distillation

scripts/
├── eval_checkpoints.py      # checkpoint accuracy curve (vLLM offline + LoRA)
├── gen_main_fig.py          # generate main comparison figure
└── plot_curves.py           # generate loss + accuracy figures

assets/
├── main_comparison.png      # accuracy: base / ICL / SFT / distillation
└── loss_comparison.png      # CE loss convergence: SFT vs distillation
```

---

## Key Implementation Notes

**Token alignment.** Uses `labels != -100` mask to identify exact answer positions in
the student. Teacher answer start = `t_lens - n_ans` (teacher sequence length minus
number of answer tokens). This guarantees both sequences are aligned on the same token
IDs regardless of context length differences.

**No precomputed cache.** Each training step runs two forward passes: teacher (frozen,
`torch.no_grad()`, with 8-shot context) and student (LoRA, with gradients, zero-shot).
This roughly doubles training time (~1.9s/it vs ~1.1s/it for baseline) but eliminates
the 0.86 GB cache and enables dynamic alignment.

**Top-K logit MSE.** We use the teacher's top-256 vocabulary positions (by logit value)
as the distillation target. This focuses supervision on the meaningful part of the
distribution (>99% probability mass) and avoids gradient noise from near-zero logits.

**Qwen3 specifics.** Always run with `enable_thinking=False`. Thinking tokens break the
`#### <number>` answer extraction pattern used by GSM8K eval.

---

## Contributors

- Amal Joe (IIT Bombay)
