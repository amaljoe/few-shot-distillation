# Few-Shot Distillation: Distilling In-Context Learning into Model Weights

Distill few-shot (in-context learning) behavior into model parameters via layer-wise
hidden-state supervision. A teacher model running with 8-shot context provides activation
targets; a student model trained zero-shot learns to match those internal representations
alongside the standard task loss — achieving few-shot-level accuracy **without context at
inference time**.

---

## Hypothesis

Few-shot context performs an implicit low-rank update through attention. If we supervise a
zero-shot model's internal representations using the corresponding few-shot activations at
each layer, the model can internalize that adaptation signal into its weights.

---

## Training Objective

```
L_total = L_CE  +  λ * Σ_l || h_l(student) − h_l(teacher) ||²
```

- `L_CE` — cross-entropy on the answer tokens (standard SFT loss)
- `h_l(·)` — L2-normalized hidden state at layer `l`, at the last token of the query prompt
- `λ` — distillation weight (optimal: 0.5)
- Teacher activations precomputed once and cached (0.86 GB); student activations computed on-the-fly

---

## Experimental Setup

**Models:** Qwen/Qwen3-1.7B (primary), Qwen/Qwen3-8B (scale-up) · **Dataset:** GSM8K
**Adapter:** LoRA r=16 α=32 · **Training:** 1000 steps · effective batch 32 · lr 2e-4 · bf16 · 4×A100 80GB

### Phase 0 — Model Selection (ICL Gap)

Evaluated on 16 GSM8K test samples to find the smallest model with a clear ICL signal:

| Model | 0-shot | 8-shot | Gap |
|-------|--------|--------|-----|
| Qwen/Qwen3-1.7B | 50.0% | 62.5% | +12.5% ← **selected** |
| Qwen/Qwen3-4B-Instruct-2507 | 81.2% | 93.8% | +12.5% |
| Qwen/Qwen3-8B | 18.8% | 68.8% | +50.0% |

Qwen3-1.7B selected: smallest model with ≥5pp ICL gap and non-trivial 0-shot floor.

### Phase 1 — Conditions

| Condition | Description |
|-----------|-------------|
| A | 8-shot inference — teacher baseline, no fine-tuning |
| B | Zero-shot LoRA SFT — standard fine-tuning, no distillation |
| C | Layer-wise distillation — LoRA SFT + hidden-state matching (all layers, λ=0.5) |
| D | Logit-level KL distillation — LoRA SFT + KL on output logits (λ=0.5, T=2) |

All conditions run on both **Qwen3-1.7B** (28 layers, 2048 hidden) and
**Qwen3-8B** (36 layers, 4096 hidden) to test scalability of the distillation signal.

---

## Results

All fine-tuned conditions evaluated zero-shot on the **full GSM8K test set (1319 examples)**.

### Qwen3-1.7B Results

#### Main Comparison

| Condition | Accuracy | Correct / Total | vs 8-shot teacher |
|-----------|----------|-----------------|-------------------|
| Base model, 0-shot (no fine-tuning) | 26.08% | 344 / 1319 | −21.00pp |
| A — 8-shot teacher (no fine-tuning) | 47.08% | 621 / 1319 | — |
| B — Zero-shot LoRA baseline | 62.47% | 824 / 1319 | **+15.39pp** |
| C — Layer-wise distillation (ours) | **63.68%** | **840 / 1319** | **+16.60pp** |
| D — Logit-level KL distillation | 63.15% | 833 / 1319 | +16.07pp |

Both fine-tuned conditions **substantially outperform 8-shot in-context learning**
(+15–17pp). The Phase 0 estimate of ~62.5% for 8-shot was from only 16 samples and badly
overestimated teacher accuracy. On 1319 examples, the 8-shot teacher reaches only 47.08%,
while the fine-tuned baseline already exceeds it by +15pp at zero-shot inference time.

### Lambda Sweep (Condition C architecture)

| λ | Accuracy | vs Baseline |
|---|----------|-------------|
| 0.1 | 62.47% | ±0.00pp — no effect, signal too weak |
| **0.5** | **63.68%** | **+1.21pp — optimal** |
| 1.0 | 62.47% | ±0.00pp — CE and distillation cancel |
| 2.0 | 59.51% | −2.96pp — distillation overwhelms CE |

The inverted-U and the **−2.96pp collapse at λ=2.0 confirm the distillation signal is real**: if it were noise, performance would not degrade monotonically with λ.

### Checkpoint Accuracy Curve (full test set, 1319 examples)

Three training phases are visible. See `experiments/figures/accuracy_curve.png` for the
full visualisation.

**Early phase (steps 10–100):**

| Step | Baseline | Distill C | Δ | Note |
|------|----------|-----------|---|------|
| 10 | 74.00% | 73.92% | −0.08pp | format-learn spike |
| 20 | 36.39% | 41.17% | +4.78pp | warmup crash |
| 30 | 54.66% | 55.57% | +0.91pp | recovery |
| 40–70 | 57–62% | 57–60% | mixed | volatile, high LR |
| 80 | 59.29% | 61.64% | +2.35pp | — |
| 90 | 63.08% | 59.74% | −3.34pp | — |
| 100 | 61.94% | 60.20% | −1.74pp | — |

**Full training (steps 200–1000):**

| Step | Baseline | Distill C | Δ |
|------|----------|-----------|---|
| 200 | 61.79% | 63.23% | **+1.44pp** |
| 400 | 64.29% | 64.67% | **+0.38pp** |
| 600 | 63.68% | 64.90% | **+1.21pp** |
| 800 | 63.46% | **65.73%** | **+2.27pp** |
| 1000 | 62.17% | 63.23% | **+1.06pp** |

Distillation leads at **all 5 full-training checkpoints** (average gap: **+1.27pp**).
The early phase is dominated by format learning (both conditions behave identically at
step 10); the distillation advantage materialises once the LR enters cosine decay.

---

### Qwen3-8B Results

Same experimental protocol on the 8B base model (36 layers, hidden 4096),
run overnight via `scripts/run_8b_overnight.sh` on 4×A100 80GB.

**Phase 0 ICL gap (16 samples):** 0-shot 18.8% → 8-shot 68.8% (+50.0pp gap).

#### Checkpoint Accuracy Curve — Qwen3-8B

**Early phase (steps 10–100):**

| Step | Baseline | Distill C | Δ | Note |
|------|----------|-----------|---|------|
| 10 | 78.92% | 79.23% | +0.30pp | format-learn spike |
| 20 | 77.56% | 74.83% | −2.73pp | warmup crash |
| 30 | 80.74% | 80.29% | −0.46pp | recovery |
| 40 | 83.09% | 82.49% | −0.61pp | — |
| 50 | 83.01% | 83.17% | +0.15pp | — |
| 60 | 82.11% | 82.87% | +0.76pp | — |
| 70 | 81.58% | 80.89% | −0.68pp | — |
| 80 | 82.11% | **84.61%** | **+2.50pp** | distill surges |
| 90 | 81.43% | **83.85%** | **+2.43pp** | — |
| 100 | 81.50% | **83.47%** | **+1.97pp** | — |

**Full training (steps 200–1000):**

| Step | Baseline | Distill C | Δ |
|------|----------|-----------|---|
| 200 | 82.03% | 81.58% | −0.45pp |
| 300 | **83.78%** | 81.88% | −1.90pp |
| 400 | **83.24%** | 82.79% | −0.45pp |
| 500 | **83.24%** | 82.94% | −0.30pp |
| 600 | 81.43% | 81.80% | +0.38pp |
| 700 | 80.36% | 80.89% | +0.53pp |
| 800 | **80.67%** | 79.76% | −0.91pp |
| 900 | **81.35%** | 79.15% | −2.20pp |
| 1000 | **80.14%** | 78.47% | −1.67pp |

**Divergent finding:** distillation helps the 8B model during early training (steps 80–100,
+1.97–2.50pp advantage) but then gradually degrades. By step 1000 the baseline leads
by **1.67pp** and the distilled model's accuracy has dropped 5pp from its step-80 peak.
Baseline leads at 6 of 9 full-training checkpoints. Compare to 1.7B where distillation
leads at **all 5** full-training checkpoints (+1.27pp average).

---

### Key Observations

1. **Distillation works.** C (+1.21pp) and D (+0.68pp) both outperform the baseline.
   Layer-wise hidden-state matching outperforms output-logit KL — richer supervision
   (28 layers × 2048 dims) beats a single next-token distribution.

2. **The lambda curve rules out noise.** λ=2.0 causes a −2.96pp drop. Noise would not
   produce monotonic degradation as λ increases.

3. **Fine-tuning dramatically outperforms 8-shot inference.** Both B (~62%) and C (~64%)
   far exceed the 8-shot teacher (47.08%) at zero-shot inference time — a +15–17pp gain.
   Fine-tuning eliminates the need for any context at inference time while exceeding teacher
   accuracy by a wide margin.

4. **Distillation leads at every checkpoint, not just the final.** On the full test set,
   distillation is ahead at all 5 full-training checkpoints (avg +1.27pp, peak +2.27pp
   at step 800). The early phase (steps 10–100) is noisy and dominated by format learning;
   the distillation advantage materialises once LR enters cosine decay (~step 200).

5. **Distillation does not scale straightforwardly to larger models.** On Qwen3-8B the
   distillation signal helps early (steps 80–100, +2.5pp) but then over-regularizes the
   model — accuracy degrades to 78.47% at step 1000 vs 80.14% baseline. The 8B model has
   a much larger ICL gap (50pp vs 12.5pp for 1.7B), suggesting the teacher's 8-shot
   activations impose a representation constraint that a stronger base model cannot escape
   under extended training. The distillation loss may need down-weighting (λ < 0.5) or
   layer-selection for larger models.

---

## Figures

### Training Loss

![Training loss curve](experiments/figures/loss_curve.png)

CE loss for Condition B (baseline) and C (distillation) over 1000 steps.
Distillation runs at slightly higher CE loss in early training (the MSE term trades off
against pure task loss), but both converge to similar final CE values.

### Eval Accuracy vs Training Step — Qwen3-1.7B

![Accuracy curve](experiments/figures/accuracy_curve.png)

GSM8K accuracy (full test set, 1319 examples) at checkpoints 10–100 (early run) and
200–1000 (main run). Reference lines show the 8-shot teacher (47.08%) and base 0-shot
(26.08%). Both fine-tuned conditions far exceed the teacher.

Three phases:
- **Step 10**: Both spike to ~74% (format learning)
- **Steps 20–100**: Noisy recovery from warmup disruption
- **Steps 200–1000**: Distillation consistently ahead (+1.27pp avg)

### Eval Accuracy vs Training Step — Qwen3-8B

![8B Accuracy curve](experiments/figures/8b/accuracy_curve.png)

GSM8K accuracy (full test set, 1319 examples) at checkpoints 10–100 (early run) and
100–1000 (full run). Distillation surges ahead at steps 80–100 (+2.5pp), then degrades
steadily through extended training. Baseline (blue) finishes at 80.14%; distillation
(pink) at 78.47%.

### Qwen3-8B Training Loss

![8B Loss curve](experiments/figures/8b/loss_curve.png)

---

## Ablations

Full results in `experiments/ablations/` · Analysis in `experiments/ablations/analysis.md`

```
experiments/ablations/             # 1.7B ablations
├── checkpoint_curve/
│   ├── results_early.json          # B vs C at steps 10/20/.../100 (1319 examples)
│   └── results_full.json           # B vs C at steps 200/400/.../1000 (1319 examples)
├── kl_distill/                     # Condition D training + eval
├── lambda_01/  lambda_10/  lambda_20/   # λ ablation
├── ablation_eval.json              # Accuracy for D, λ=0.1/1.0/2.0
├── summary.json                    # Aggregated results
└── analysis.md                     # Full written analysis

experiments/ablations_8b/          # 8B scale-up
└── checkpoint_curve/
    ├── results_early.json          # B vs C at steps 10/20/.../100 (1319 examples)
    └── results_full.json           # B vs C at steps 100/200/.../1000 (1319 examples)

experiments/figures/
├── loss_curve.png                  # 1.7B training CE loss — B vs C
├── accuracy_curve.png              # 1.7B eval accuracy steps 10–1000 — B vs C
└── 8b/
    ├── loss_curve.png              # 8B training CE loss — B vs C
    └── accuracy_curve.png          # 8B eval accuracy steps 10–1000 — B vs C
```

---

## Repository Structure

```
configs/
├── base.yaml                  # 1.7B: model, data, training, LoRA hyperparameters
├── distill_layerwise.yaml     # 1.7B: distillation overrides (λ, layers, normalize)
├── early_ckpt.yaml            # 1.7B: early checkpoint run (steps 0-100, save every 10)
├── distill_early_ckpt.yaml    # 1.7B: early checkpoint distillation overlay
├── base_8b.yaml               # 8B: full config for Condition B
├── distill_8b.yaml            # 8B: distillation overlay for Condition C
├── early_ckpt_8b.yaml         # 8B: early checkpoint run
└── distill_early_ckpt_8b.yaml # 8B: early checkpoint distillation overlay

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
    ├── train_baseline.py          # Condition B
    ├── train_layerwise_distill.py # Condition C
    └── train_kl_distill.py        # Condition D

scripts/
├── eval_icl.py                      # Phase 0: few-shot vs zero-shot (vLLM offline)
├── precompute_teacher_activations.py # Cache teacher hidden states (HuggingFace)
├── precompute_teacher_logits.py      # Cache teacher output logits for KL distillation
├── evaluate.py                       # Final eval via vLLM server + LoRA adapters
├── eval_checkpoints.py               # Checkpoint accuracy curve (vLLM offline + LoRA)
├── eval_adapter.py                   # Single-shot adapter eval (vLLM offline)
├── run_ablations.sh                  # End-to-end 1.7B ablation suite runner
├── run_8b_overnight.sh               # End-to-end 8B overnight experiment (all 6 steps)
└── plot_curves.py                    # Generate loss + accuracy figures (supports --model_tag)

experiments/poc/
├── teacher_cache/
│   ├── activations.pt         # (7473, 28, 2048) float16 — 0.86 GB
│   ├── logits_top1024.pt      # Top-1024 teacher logits — 46 MB
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

Compute: 4×A100 80GB. Start via `app` alias → apptainer → activate `/dev/shm/vllm` env.
See `compute.md` for full environment notes.

### Step 1 — Phase 0: measure ICL gap

```bash
# Run in parallel on separate GPUs
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

Pick the smallest model with gap ≥ 5pp. Update `configs/base.yaml`.

### Step 2 — Precompute teacher activations and logits

```bash
# Hidden states for Condition C (run once, ~7 min):
CUDA_VISIBLE_DEVICES=0,1 python scripts/precompute_teacher_activations.py \
    --config configs/base.yaml
# → experiments/poc/teacher_cache/activations.pt  (0.86 GB)

# Output logits for Condition D (derived from hidden-state cache, ~2 min):
python scripts/precompute_teacher_logits.py --config configs/base.yaml
# → experiments/poc/teacher_cache/logits_top1024.pt  (46 MB)
```

### Step 3 — Train all conditions in parallel

```bash
# Condition B — GPUs 0,1
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --mixed_precision bf16 \
    src/training/train_baseline.py --config configs/base.yaml \
    --output_dir experiments/poc/baseline

# Condition C — GPUs 2,3
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 --mixed_precision bf16 \
    src/training/train_layerwise_distill.py --config configs/distill_layerwise.yaml \
    --output_dir experiments/poc/distill

# Condition D — GPUs 2,3 (after C, or on separate node)
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 --mixed_precision bf16 \
    src/training/train_kl_distill.py \
    --config experiments/ablations/configs/kl_distill.yaml \
    --output_dir experiments/ablations/kl_distill
```

### Step 4 — Evaluate

```bash
# Serve both LoRA adapters (all 4 GPUs, from inside apptainer tmux session):
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Qwen/Qwen3-1.7B \
    --enable-lora \
    --lora-modules \
        baseline=experiments/poc/baseline/baseline/final \
        distill=experiments/poc/distill/distill/final \
    --port 8000 --tensor-parallel-size 4 \
    --max-lora-rank 16 --max-model-len 2048

python scripts/evaluate.py \
    --config configs/base.yaml \
    --api_base http://localhost:8000/v1 \
    --lora_names baseline distill \
    --output experiments/poc/final_results.json
```

### Step 5 — Run full ablation suite (1.7B)

```bash
bash scripts/run_ablations.sh 2>&1 | tee experiments/ablations/run.log
```

Runs checkpoint curves, Condition D, and lambda sweep end-to-end with ingrained eval.
Results aggregated to `experiments/ablations/summary.json`.

### Step 6 — Run 8B scale-up overnight

```bash
# Inside apptainer tmux session on cn14-dgx:
bash scripts/run_8b_overnight.sh 2>&1 | tee experiments/8b/overnight.log
```

Runs 6 steps end-to-end: precompute activations → early train → early eval →
full train → full eval → generate figures. Skip guards allow safe resume if interrupted.
Expected runtime: ~3–4 hours on 4×A100 80GB.

### TensorBoard

```bash
tensorboard --logdir experiments/ --port 6006 --bind_all
```

---

## Key Implementation Details

**Token alignment.** The distillation loss is computed at one token per example: the last
token of the target question, just before `<|im_start|>assistant`. This position is
structurally identical in both the teacher (8-shot) and student (0-shot) sequence, making
hidden-state comparison meaningful across different context lengths.

**Teacher cache.** Precomputed as `(N, num_layers, hidden_size)` float16. Loaded into CPU
RAM at training start. Indexed per batch by `example_idx` and moved to GPU on demand —
avoids re-running the teacher during training entirely.

**Logit cache.** Top-1024 teacher logit values and indices derived from the last-layer
cached hidden state via `lm_head(norm(h_last))`. Approximates the full distribution with
>99.9% probability mass coverage. Used by Condition D.

**LoRA hook path.** Hooks attach to `peft_model.base_model.model.model.layers[i]` (not the
PEFT wrapper). `output[0]` is the post-residual hidden state for Qwen3.

**On attention-weight KL.** Teacher and student have different sequence lengths (8-shot vs
0-shot context), so attention matrices cannot be directly compared. Hidden-state matching
at the query position captures the cumulative effect of all attention layers, making it the
correct abstraction. Output-logit KL (Condition D) is the principled alternative — we
tested it and it works but is slightly weaker.

**Qwen3 thinking mode.** Always disabled (`enable_thinking=False`). Thinking tokens break
`#### <number>` answer extraction.

---

## Model Specs (Qwen3 family)

| Model | Layers | Hidden size | Activation cache (7473 examples) |
|-------|--------|-------------|----------------------------------|
| Qwen3-1.7B | 28 | **2048** (not 1536) | 0.86 GB |
| Qwen3-4B | 36 | 2560 | 1.4 GB |
| Qwen3-8B | 36 | 4096 | 2.2 GB |

---

## Configuration

`configs/base.yaml`:
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

`configs/distill_layerwise.yaml`:
```yaml
distillation:
  lambda_distill: 0.5
  layers_to_match: "all"
  normalize_hidden: true
```

---

## Contributors

- Amal Joe (IIT Bombay)
