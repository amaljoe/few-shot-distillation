# XTREME Isolated Lambda Sweep Results

**Setup**: Qwen3-1.7B, 200 training steps, English-only, single-task isolation.
**Lambda sweep**: λ ∈ {0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0}

## Summary Table

| Task | SFT (λ=0) | Best λ | Best Score | Gain | Metric |
|------|-----------|--------|------------|------|--------|
| NLI | 88.8% | 0.0 | 88.8% | +0.0pp | ACCURACY |
| PA | 56.6% | 0.0 | 56.6% | +0.0pp | ACCURACY |
| QA | 77.3% | 0.01 | 78.2% | +0.9pp | F1 |
| NER | 50.4% | 0.0 | 50.4% | +0.0pp | F1 |
| POS | 18.0% | 0.1 | 23.6% | +5.7pp | ACCURACY |

## Per-Task Lambda Sensitivity

### NLI (XNLI-en)

![nli lambda sweep](figures/nli_lambda_sweep.png)

| λ | Score |
|---|---|
| 0.0 | 88.8% ← best |
| 0.01 | 84.4% |
| 0.05 | 86.4% |
| 0.1 | 86.2% |
| 0.25 | 85.8% |
| 0.5 | 86.8% |
| 0.75 | 82.4% |
| 1.0 | 82.2% |

### Paraphrase (PAWS-X en)

![pa lambda sweep](figures/pa_lambda_sweep.png)

| λ | Score |
|---|---|
| 0.0 | 56.6% ← best |
| 0.01 | 56.6% |
| 0.05 | 56.6% |
| 0.1 | 56.6% |
| 0.25 | 56.6% |
| 0.5 | 56.6% |
| 0.75 | 56.6% |
| 1.0 | 56.6% |

### QA (MLQA en)

![qa lambda sweep](figures/qa_lambda_sweep.png)

| λ | Score |
|---|---|
| 0.0 | 77.3% |
| 0.01 | 78.2% ← best |
| 0.05 | 77.9% |
| 0.1 | 77.6% |
| 0.25 | 76.2% |
| 0.5 | 76.5% |
| 0.75 | 72.5% |
| 1.0 | 70.8% |

### NER (WikiANN en)

![ner lambda sweep](figures/ner_lambda_sweep.png)

| λ | Score |
|---|---|
| 0.0 | 50.4% ← best |
| 0.01 | 44.9% |
| 0.05 | 40.5% |
| 0.1 | 35.9% |
| 0.25 | 31.3% |
| 0.5 | 30.9% |
| 0.75 | 28.0% |
| 1.0 | 26.9% |

### POS (UDPOS en)

![pos lambda sweep](figures/pos_lambda_sweep.png)

| λ | Score |
|---|---|
| 0.0 | 18.0% |
| 0.01 | 20.0% |
| 0.05 | 19.7% |
| 0.1 | 23.6% ← best |
| 0.25 | 19.2% |
| 0.5 | 18.5% |
| 0.75 | 17.0% |
| 1.0 | 17.1% |

## Analysis & Ideal Lambdas

*(Generated automatically — see `scripts/gen_xtreme_isolated_fig.py`)*

- **NLI**: Distillation did not help (best=SFT). Consider task-specific factors.
- **PA**: Distillation did not help (best=SFT). Consider task-specific factors.
- **QA**: Best λ=0.01 (+0.9pp over SFT). Sensitivity: high — needs careful tuning.
- **NER**: Distillation did not help (best=SFT). Consider task-specific factors.
- **POS**: Best λ=0.1 (+5.7pp over SFT). Sensitivity: moderate — prefers mild distillation.
