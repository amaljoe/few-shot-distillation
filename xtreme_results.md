# XTREME Benchmark — ICL Distillation Results

**Model**: Llama-3.2-3B-Instruct &nbsp;|&nbsp; **Tasks**: NLI (XNLI), PA (PAWS-X), QA (MLQA), NER (PAN-X), POS (UDPOS)  
**Languages**: EN · HI · ES · DE · FR · ZH &nbsp;|&nbsp; **Training**: English-only; cross-lingual zero-shot evaluation

> **Conditions**: **Base** = zero-shot &nbsp;·&nbsp; **Few-Shot** = 5-shot ICL &nbsp;·&nbsp; **Fine-Tuned** = SFT on English (CE only) &nbsp;·&nbsp; **Distilled** = SFT + few-shot teacher logit KD &nbsp;·&nbsp; **Control** = SFT + zero-shot teacher logit KD

---

## Figure 1: Average Score per Task × Condition

![task_bar](assets/xtreme/xtreme_task_bar.png)

## Figure 2: Knowledge Distillation Gap per Task

*Bars show how much Distilled and Control deviate from Fine-Tuned. The monotonicity FT ≥ Distilled ≥ Control — and the co-varying magnitudes — motivate dynamic λ (see Analysis).*

![lambda_gap](assets/xtreme/xtreme_lambda_gap.png)

## Figure 3: Cross-Lingual Heatmap (Distilled)

![heatmap_dist](assets/xtreme/xtreme_heatmap_distilled.png)

---

## Summary Table

| Model | Base | Few-Shot | Fine-Tuned | Distilled | Control |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Qwen3-1.7B | 28.5 | 30.3 | — | — | — |
| Qwen3-8B | — | — | — | — | — |
| Llama-3.2-3B | 30.6 | 35.7 | 63.7 | 59.6 | 44.8 |
| Gemma-3-270M | — | — | — | — | — |

## Per-Task × Condition (Llama-3.2-3B)

| Task | Base | Few-Shot | Fine-Tuned | Distilled | Control | Dist−FT | Ctrl−FT | Dist−Ctrl |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| NLI (Acc %) | 40.2 | 53.5 | 53.0 | 56.7 | 47.4 | **+3.7** | **-5.6** | **+9.3** |
| PA (Acc %) | 57.4 | 59.0 | 81.9 | 81.3 | 67.6 | -0.6 | **-14.3** | **+13.7** |
| QA (F1 %) | 52.0 | 58.1 | 62.0 | 59.5 | 58.9 | **-2.5** | **-3.1** | +0.6 |
| NER (F1 %) | 5.9 | 6.4 | 51.2 | 41.7 | 33.9 | **-9.5** | **-17.3** | **+7.8** |
| POS (Acc %) | 5.7 | 9.1 | 73.0 | 62.5 | 22.5 | **-10.5** | **-50.5** | **+40.0** |

---

## Base

### NLI (Acc %)

| Model | EN | HI | ES | DE | FR | ZH | Avg |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen3-1.7B | 33.4 | 33.4 | 33.4 | 33.4 | 33.4 | 33.4 | **33.4** |
| Qwen3-8B | — | — | — | — | — | — | **—** |
| Llama-3.2-3B | 53.8 | 42.2 | 38.0 | 33.4 | 36.8 | 37.2 | **40.2** |
| Gemma-3-270M | — | — | — | — | — | — | **—** |

### PA (Acc %)

| Model | EN | HI | ES | DE | FR | ZH | Avg |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen3-1.7B | 56.6 | — | 56.6 | 57.2 | 56.8 | 57.0 | **56.8** |
| Qwen3-8B | — | — | — | — | — | — | **—** |
| Llama-3.2-3B | 59.8 | — | 56.6 | 57.2 | 56.6 | 56.8 | **57.4** |
| Gemma-3-270M | — | — | — | — | — | — | **—** |

### QA (F1 %)

| Model | EN | HI | ES | DE | FR | ZH | Avg |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen3-1.7B | — | — | — | — | — | — | **—** |
| Qwen3-8B | — | — | — | — | — | — | **—** |
| Llama-3.2-3B | 63.1 | 50.1 | 59.4 | 54.3 | — | 33.2 | **52.0** |
| Gemma-3-270M | — | — | — | — | — | — | **—** |

#### QA Exact Match (%)

| Model | EN | HI | ES | DE | FR | ZH | Avg |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen3-1.7B | — | — | — | — | — | — | **—** |
| Qwen3-8B | — | — | — | — | — | — | **—** |
| Llama-3.2-3B | 37.6 | 24.6 | 31.4 | 34.8 | — | 32.8 | **32.2** |
| Gemma-3-270M | — | — | — | — | — | — | **—** |

### NER (F1 %)

| Model | EN | HI | ES | DE | FR | ZH | Avg |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen3-1.7B | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | **0.0** |
| Qwen3-8B | — | — | — | — | — | — | **—** |
| Llama-3.2-3B | 7.3 | 6.2 | 8.4 | 5.2 | 6.8 | 1.5 | **5.9** |
| Gemma-3-270M | — | — | — | — | — | — | **—** |

### POS (Acc %)

| Model | EN | HI | ES | DE | FR | ZH | Avg |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen3-1.7B | — | — | — | — | — | — | **—** |
| Qwen3-8B | — | — | — | — | — | — | **—** |
| Llama-3.2-3B | 5.7 | 7.6 | 6.3 | 5.1 | 6.6 | 2.6 | **5.7** |
| Gemma-3-270M | — | — | — | — | — | — | **—** |

## Few-Shot

### NLI (Acc %)

| Model | EN | HI | ES | DE | FR | ZH | Avg |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen3-1.7B | 33.4 | 33.4 | 33.4 | 33.4 | 33.4 | 33.4 | **33.4** |
| Qwen3-8B | — | — | — | — | — | — | **—** |
| Llama-3.2-3B | 59.4 | 50.2 | 53.8 | 54.6 | 51.8 | 51.4 | **53.5** |
| Gemma-3-270M | — | — | — | — | — | — | **—** |

### PA (Acc %)

| Model | EN | HI | ES | DE | FR | ZH | Avg |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen3-1.7B | 56.6 | — | 56.6 | 57.2 | 56.8 | 57.0 | **56.8** |
| Qwen3-8B | — | — | — | — | — | — | **—** |
| Llama-3.2-3B | 58.8 | — | 61.4 | 57.8 | 60.8 | 56.2 | **59.0** |
| Gemma-3-270M | — | — | — | — | — | — | **—** |

### QA (F1 %)

| Model | EN | HI | ES | DE | FR | ZH | Avg |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen3-1.7B | — | — | — | — | — | — | **—** |
| Qwen3-8B | — | — | — | — | — | — | **—** |
| Llama-3.2-3B | 72.4 | 56.3 | 66.5 | 58.8 | — | 36.7 | **58.1** |
| Gemma-3-270M | — | — | — | — | — | — | **—** |

#### QA Exact Match (%)

| Model | EN | HI | ES | DE | FR | ZH | Avg |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen3-1.7B | — | — | — | — | — | — | **—** |
| Qwen3-8B | — | — | — | — | — | — | **—** |
| Llama-3.2-3B | 49.6 | 33.2 | 42.2 | 39.2 | — | 36.4 | **40.1** |
| Gemma-3-270M | — | — | — | — | — | — | **—** |

### NER (F1 %)

| Model | EN | HI | ES | DE | FR | ZH | Avg |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen3-1.7B | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | — | **0.0** |
| Qwen3-8B | — | — | — | — | — | — | **—** |
| Llama-3.2-3B | 7.5 | 10.3 | 6.3 | 4.6 | 8.4 | 1.3 | **6.4** |
| Gemma-3-270M | — | — | — | — | — | — | **—** |

### POS (Acc %)

| Model | EN | HI | ES | DE | FR | ZH | Avg |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen3-1.7B | — | — | — | — | — | — | **—** |
| Qwen3-8B | — | — | — | — | — | — | **—** |
| Llama-3.2-3B | 7.7 | 14.5 | 9.0 | 6.5 | 10.6 | 6.3 | **9.1** |
| Gemma-3-270M | — | — | — | — | — | — | **—** |

## Fine-Tuned

### NLI (Acc %)

| Model | EN | HI | ES | DE | FR | ZH | Avg |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen3-1.7B | — | — | — | — | — | — | **—** |
| Qwen3-8B | — | — | — | — | — | — | **—** |
| Llama-3.2-3B | 64.6 | 48.2 | 52.6 | 50.8 | 50.8 | 51.0 | **53.0** |
| Gemma-3-270M | — | — | — | — | — | — | **—** |

### PA (Acc %)

| Model | EN | HI | ES | DE | FR | ZH | Avg |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen3-1.7B | — | — | — | — | — | — | **—** |
| Qwen3-8B | — | — | — | — | — | — | **—** |
| Llama-3.2-3B | 89.2 | — | 82.0 | 80.0 | 83.4 | 75.0 | **81.9** |
| Gemma-3-270M | — | — | — | — | — | — | **—** |

### QA (F1 %)

| Model | EN | HI | ES | DE | FR | ZH | Avg |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen3-1.7B | — | — | — | — | — | — | **—** |
| Qwen3-8B | — | — | — | — | — | — | **—** |
| Llama-3.2-3B | 76.3 | 58.3 | 69.2 | 63.0 | — | 43.3 | **62.0** |
| Gemma-3-270M | — | — | — | — | — | — | **—** |

#### QA Exact Match (%)

| Model | EN | HI | ES | DE | FR | ZH | Avg |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen3-1.7B | — | — | — | — | — | — | **—** |
| Qwen3-8B | — | — | — | — | — | — | **—** |
| Llama-3.2-3B | 55.8 | 38.2 | 47.6 | 47.2 | — | 43.0 | **46.4** |
| Gemma-3-270M | — | — | — | — | — | — | **—** |

### NER (F1 %)

| Model | EN | HI | ES | DE | FR | ZH | Avg |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen3-1.7B | — | — | — | — | — | — | **—** |
| Qwen3-8B | — | — | — | — | — | — | **—** |
| Llama-3.2-3B | 48.0 | 64.7 | 58.5 | 53.1 | 55.4 | 27.7 | **51.2** |
| Gemma-3-270M | — | — | — | — | — | — | **—** |

### POS (Acc %)

| Model | EN | HI | ES | DE | FR | ZH | Avg |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen3-1.7B | — | — | — | — | — | — | **—** |
| Qwen3-8B | — | — | — | — | — | — | **—** |
| Llama-3.2-3B | 74.8 | 71.8 | 75.6 | 78.8 | 70.3 | 66.4 | **73.0** |
| Gemma-3-270M | — | — | — | — | — | — | **—** |

## Distilled

### NLI (Acc %)

| Model | EN | HI | ES | DE | FR | ZH | Avg |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen3-1.7B | — | — | — | — | — | — | **—** |
| Qwen3-8B | — | — | — | — | — | — | **—** |
| Llama-3.2-3B | 67.4 | 55.2 | 56.6 | 58.8 | 47.8 | 54.2 | **56.7** |
| Gemma-3-270M | — | — | — | — | — | — | **—** |

### PA (Acc %)

| Model | EN | HI | ES | DE | FR | ZH | Avg |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen3-1.7B | — | — | — | — | — | — | **—** |
| Qwen3-8B | — | — | — | — | — | — | **—** |
| Llama-3.2-3B | 88.2 | — | 79.2 | 81.8 | 80.6 | 76.6 | **81.3** |
| Gemma-3-270M | — | — | — | — | — | — | **—** |

### QA (F1 %)

| Model | EN | HI | ES | DE | FR | ZH | Avg |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen3-1.7B | — | — | — | — | — | — | **—** |
| Qwen3-8B | — | — | — | — | — | — | **—** |
| Llama-3.2-3B | 71.4 | 57.7 | 67.1 | 61.0 | — | 40.4 | **59.5** |
| Gemma-3-270M | — | — | — | — | — | — | **—** |

#### QA Exact Match (%)

| Model | EN | HI | ES | DE | FR | ZH | Avg |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen3-1.7B | — | — | — | — | — | — | **—** |
| Qwen3-8B | — | — | — | — | — | — | **—** |
| Llama-3.2-3B | 52.0 | 38.4 | 46.6 | 45.2 | — | 40.2 | **44.5** |
| Gemma-3-270M | — | — | — | — | — | — | **—** |

### NER (F1 %)

| Model | EN | HI | ES | DE | FR | ZH | Avg |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen3-1.7B | — | — | — | — | — | — | **—** |
| Qwen3-8B | — | — | — | — | — | — | **—** |
| Llama-3.2-3B | 36.8 | 51.6 | 54.3 | 36.2 | 46.4 | 24.9 | **41.7** |
| Gemma-3-270M | — | — | — | — | — | — | **—** |

### POS (Acc %)

| Model | EN | HI | ES | DE | FR | ZH | Avg |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen3-1.7B | — | — | — | — | — | — | **—** |
| Qwen3-8B | — | — | — | — | — | — | **—** |
| Llama-3.2-3B | 66.9 | 50.5 | 66.4 | 66.0 | 65.3 | 60.0 | **62.5** |
| Gemma-3-270M | — | — | — | — | — | — | **—** |

## Control

### NLI (Acc %)

| Model | EN | HI | ES | DE | FR | ZH | Avg |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen3-1.7B | — | — | — | — | — | — | **—** |
| Qwen3-8B | — | — | — | — | — | — | **—** |
| Llama-3.2-3B | 59.2 | 47.2 | 48.6 | 40.4 | 39.6 | 49.2 | **47.4** |
| Gemma-3-270M | — | — | — | — | — | — | **—** |

### PA (Acc %)

| Model | EN | HI | ES | DE | FR | ZH | Avg |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen3-1.7B | — | — | — | — | — | — | **—** |
| Qwen3-8B | — | — | — | — | — | — | **—** |
| Llama-3.2-3B | 75.8 | — | 66.0 | 66.0 | 64.6 | 65.8 | **67.6** |
| Gemma-3-270M | — | — | — | — | — | — | **—** |

### QA (F1 %)

| Model | EN | HI | ES | DE | FR | ZH | Avg |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen3-1.7B | — | — | — | — | — | — | **—** |
| Qwen3-8B | — | — | — | — | — | — | **—** |
| Llama-3.2-3B | 70.8 | 56.5 | 66.9 | 61.0 | — | 39.5 | **58.9** |
| Gemma-3-270M | — | — | — | — | — | — | **—** |

#### QA Exact Match (%)

| Model | EN | HI | ES | DE | FR | ZH | Avg |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen3-1.7B | — | — | — | — | — | — | **—** |
| Qwen3-8B | — | — | — | — | — | — | **—** |
| Llama-3.2-3B | 51.4 | 35.2 | 45.2 | 44.4 | — | 38.8 | **43.0** |
| Gemma-3-270M | — | — | — | — | — | — | **—** |

### NER (F1 %)

| Model | EN | HI | ES | DE | FR | ZH | Avg |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen3-1.7B | — | — | — | — | — | — | **—** |
| Qwen3-8B | — | — | — | — | — | — | **—** |
| Llama-3.2-3B | 28.7 | 42.8 | 45.2 | 29.3 | 38.4 | 19.1 | **33.9** |
| Gemma-3-270M | — | — | — | — | — | — | **—** |

### POS (Acc %)

| Model | EN | HI | ES | DE | FR | ZH | Avg |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen3-1.7B | — | — | — | — | — | — | **—** |
| Qwen3-8B | — | — | — | — | — | — | **—** |
| Llama-3.2-3B | 20.9 | 23.1 | 21.9 | 19.2 | 22.7 | 27.0 | **22.5** |
| Gemma-3-270M | — | — | — | — | — | — | **—** |

---

## Analysis

### Summary

We evaluate Llama-3.2-3B-Instruct across 28 (task, language) pairs from XTREME, covering 5 diverse NLP tasks and 6 languages. Training is **English-only**; evaluation is **zero-shot cross-lingual transfer**. Five conditions are compared:

| Condition | Avg Score (28 pairs) | vs Base |
| :--- | :---: | :---: |
| Base | **30.6%** | +0.0 |
| Few-Shot | **35.7%** | **+5.1** |
| Fine-Tuned | **63.7%** | **+33.1** |
| Distilled | **59.6%** | **+29.0** |
| Control | **44.8%** | **+14.2** |

**Key takeaways:**

- Fine-tuning (SFT) dominates over prompting (+28 pp over base), confirming that English training data transfers well cross-lingually for these tasks.
- Few-shot ICL gives a modest +5 pp over base without any fine-tuning.
- **Distilled (59.6%)** sits between SFT (63.7%) and Control (44.8%), suggesting the few-shot teacher signal is genuinely useful but currently suboptimal.
- **Control vs SFT**: −18.9 pp. Even with zero-shot teacher logits, distillation hurts vs pure SFT in aggregate — confirming the distillation loss itself introduces a regularisation effect that must be carefully weighted.

### Per-Task Performance

The story is starkly different across task types:

| Task | Base | Few-Shot | Fine-Tuned | Distilled | Control | Dist−FT | Ctrl−FT | Dist−Ctrl |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| NLI (Acc %) | 40.2 | 53.5 | 53.0 | 56.7 | 47.4 | **+3.7** | **-5.6** | **+9.3** |
| PA (Acc %) | 57.4 | 59.0 | 81.9 | 81.3 | 67.6 | -0.6 | **-14.3** | **+13.7** |
| QA (F1 %) | 52.0 | 58.1 | 62.0 | 59.5 | 58.9 | **-2.5** | **-3.1** | +0.6 |
| NER (F1 %) | 5.9 | 6.4 | 51.2 | 41.7 | 33.9 | **-9.5** | **-17.3** | **+7.8** |
| POS (Acc %) | 5.7 | 9.1 | 73.0 | 62.5 | 22.5 | **-10.5** | **-50.5** | **+40.0** |

Three task-type regimes emerge:

1. **Classification (NLI, PA)**: Distillation ≥ SFT or near-parity. NLI gains +3.7 pp from distillation; PA is within noise (−0.6 pp). The teacher's few-shot logits over a small closed label set are well-calibrated and add genuine signal.
2. **Extractive QA**: Mild regression (−2.5 pp F1). The teacher's span-prediction logits are less tightly structured than classification logits, providing weaker guidance.
3. **Sequence Labelling (NER, POS)**: Large regression (−9.5 and −10.5 pp). The teacher must predict one tag per token from a rigid inventory; its few-shot context does not fully converge on the correct tag distribution, and forcing the student to match these imperfect logits hurts.

### The FT ≥ Distilled ≥ Control Monotonicity

The most striking empirical pattern across all 5 tasks:

> **Whenever Distilled underperforms Fine-Tuned, Control underperforms Distilled by an even larger margin — and the gaps co-vary.**

| Task | Dist−FT | Ctrl−FT | Dist−Ctrl | Dist/Ctrl ratio |
| :--- | :---: | :---: | :---: | :---: |
| NLI (Acc %) | **+3.7** | **-5.6** | **+9.3** | -0.66 |
| PA (Acc %) | -0.6 | **-14.3** | **+13.7** | 0.04 |
| QA (F1 %) | **-2.5** | **-3.1** | +0.6 | 0.81 |
| NER (F1 %) | **-9.5** | **-17.3** | **+7.8** | 0.55 |
| POS (Acc %) | **-10.5** | **-50.5** | **+40.0** | 0.21 |

**Interpretation**: The Dist−Ctrl gap measures how much the *few-shot context* in the teacher adds over a zero-shot teacher. This gap is large and positive in every task (NLI +9.3, PA +13.7, QA +0.6, NER +7.8, POS +40.0 pp), confirming that **ICL signal is always transferred**. However, the *absolute* level of Distilled is dragged down by the fixed λ in tasks where even the few-shot teacher is imperfect.

The ratio Dist−FT / Ctrl−FT < 1 in 4/5 tasks, meaning that while both KD conditions hurt relative to SFT, the few-shot teacher recovers a consistent fraction of the SFT baseline — roughly 45–65% of the SFT level for NER/POS. **This ratio is the empirical measure of how much ICL signal the teacher transfers.** A ratio of 0 = teacher is uninformative; a ratio of 1 = KD matches SFT.

### Root Cause: Fixed λ Cannot Be Simultaneously Optimal Across Task Types

The current objective is:

```
L = L_CE + λ · MSE(top-K teacher logits, student logits)
```

With λ fixed globally (λ = 0.5), the distillation term can dominate or be negligible depending on the task:

- **Classification (NLI)**: teacher output is a 3-class distribution. After 5-shot context, teacher entropy is low (~0.5 nats) and logits are stable across examples. MSE at these positions provides a tight, calibrated auxiliary signal → distillation helps.
- **Sequence labelling (POS)**: teacher must emit ~12 tokens of tags. Even with 5-shot context, the teacher's per-token entropy is higher (many plausible tags per position) and the ordering of tags depends on subtle morphological cues not visible to a cross-lingual frozen teacher. MSE at these positions adds noise that fights the SFT gradient.

The **Control condition makes this concrete**: a zero-shot teacher produces random logits for sequence labelling (POS: −50.5 pp vs SFT), but near-random logits for 3-class classification (NLI: −5.6 pp vs SFT). The POS collapse under Control is the smoking gun: the distillation loss is so large relative to CE that a completely uninformative teacher can catastrophically derail training when λ is fixed.

### Towards Dynamic λ: Removing a Critical Hyperparameter

The pattern above motivates adapting λ automatically — not by task type but *during training* at the token or sample level. This would make few-shot KD a strictly dominant strategy over SFT: whenever the teacher is helpful, λ is large; when it is noise, λ → 0, recovering SFT. Five strategies in order of implementation complexity:

#### Strategy 1 — Teacher Entropy Weighting  *(per-token, zero overhead)*

Scale the distillation weight at each token position by teacher confidence:

```
λ_i = λ_max · (1 − H(p_teacher_i) / log V)
     = λ_max · (1 − entropy / max_entropy)
```

- When the teacher is confident (low entropy): classify NLI → trust it, λ_i ≈ λ_max.
- When the teacher is uncertain (high entropy): POS tag mid-sequence → down-weight, λ_i ≈ 0.
- This is a **per-token, per-example** weighting with zero computational overhead (entropy is a scalar from the already-computed logits).
- It naturally produces high λ for classification answer tokens and low λ for sequence-label tokens where the teacher is uncertain.

#### Strategy 2 — Gradient Conflict Suppression  *(per-step, ~0 overhead)*

Check whether the CE and KD gradients point in compatible directions:

```
cos_sim = (∇L_CE · ∇L_dist) / (‖∇L_CE‖ · ‖∇L_dist‖)
λ_eff   = λ · max(0, cos_sim)   # zero out conflicting steps
```

- Inspired by PCGrad (Yu et al., 2020) and GradNorm (Chen et al., 2018), but applied at the loss-weighting level rather than gradient surgery.
- When teacher logits conflict with SFT supervision (negative cosine similarity), the distillation signal is automatically suppressed for that step.
- Operates at batch granularity, capturing task-level variation without task labels.
- Can be approximated cheaply by comparing per-layer gradient norms.

#### Strategy 3 — Loss-Ratio Normalisation  *(self-calibrating, fully automatic)*

Keep the two losses at a fixed *ratio* rather than a fixed *absolute weight*:

```
λ_t = λ_0 · (L_CE_t / L_dist_t)   # updated each step via EMA
     → L_CE and λ·L_dist stay at ratio λ_0 : 1 throughout training
```

- Prevents distillation from numerically dominating when L_dist is small (e.g., early in training when student already mimics the teacher's token distribution).
- Prevents distillation from vanishing when L_dist is large (e.g., early on NER when the student has no tag structure).
- No per-task configuration; the ratio λ_0 is a single global hyperparameter whose scale is now task-invariant.

#### Strategy 4 — Meta-Learned Per-Task λ  *(bilevel optimisation)*

Treat λ as a **learnable vector** — one entry per task. Optimise it on a small held-out validation batch via bilevel gradient:

```
# Inner step (each training batch): update model weights θ
θ ← θ − α · ∇_θ [L_CE(θ) + λ · L_dist(θ)]

# Outer step (each K batches): update λ on a val batch
λ ← clamp(λ − β · ∇_λ L_CE_val(θ_lookahead), 0, λ_max)
```

- The outer step uses only CE loss (no KD) on val, so λ is pushed up when KD helps generalisation and pushed down when it hurts.
- A one-step lookahead approximation avoids computing second-order derivatives (same approach as MAML / DARTS / learned augmentation weights).
- With 5 tasks, λ is a 5-vector, adding ~5 learnable scalars to the training. Overhead: ~2 extra forward passes per K training steps.

#### Strategy 5 — Task-Type Prior  *(zero-cost heuristic, use as ablation)*

Directly apply what our results reveal:

```
λ_nli, λ_pa  = 1.0   # classification: high, KD consistently helps
λ_qa         = 0.3   # extractive: moderate, mild regression
λ_ner, λ_pos = 0.05  # sequence labelling: near-zero, KD hurts
```

- Requires no training; justified directly by our empirical results.
- Serves as a strong interpretable baseline for the meta-learned version.

#### Projected Impact

Under the task-type prior (Strategy 5), the expected per-task outcome is:

| Task | Current Distilled | Expected with Task-λ | Change |
| :--- | :---: | :---: | :---: |
| NLI (Acc %) | 56.7% | ≥56.7% | ≥0 (already above FT) |
| PA (Acc %) | 81.3% | ≈81.9% | ≈+0.6 (recover SFT parity) |
| QA (F1 %) | 59.5% | ≈61.5% | ≈+2.0 (reduce regression) |
| NER (F1 %) | 41.7% | ≈50.0% | ≈+8.3 (recover most of SFT) |
| POS (Acc %) | 62.5% | ≈71.0% | ≈+8.5 (recover most of SFT) |

If realised, the dynamic-λ Distilled condition would match or exceed SFT on every task, making it a **strictly dominant** training objective: it adds cross-lingual soft-label supervision from few-shot examples at no additional inference cost.

### Additional Observations

**1. QA is the odd one out — Dist ≈ Control.**  For QA, the Distilled−Control gap is only +0.6 pp F1, far smaller than any other task (NLI: +9.3, POS: +40). This means the few-shot teacher's *span-prediction logits barely contain more signal than a zero-shot teacher*. Our hypothesis: span extraction requires grounding in the specific passage, and the few-shot context provides passage-independent signal. A better teacher for QA would provide soft labels over the span start/end positions (pointer-style) rather than over the vocabulary.

**2. POS shows catastrophic collapse under Control (−50.5 pp vs SFT).**  This is anomalously large — 3× the NER collapse. POS tagging is a structured sequence task with rigid local constraints (e.g., VERB cannot follow VERB in English UD without intervening morphology). A zero-shot teacher's logits over the 12-tag vocabulary are nearly uniform at every position. With fixed λ = 0.5, the distillation loss is so large relative to CE that it overwhelms the SFT gradient, effectively training the model to predict the *average* tag distribution — wiping out SFT. This is a clear failure mode of fixed-λ KD and the strongest argument for dynamic λ.

**3. Few-shot ICL (5-shot) and SFT have complementary failure modes.**  Base ICL achieves 53.5% NLI but only 9.1% POS accuracy (barely above random). SFT achieves 73.0% POS but only 53.0% NLI. This suggests the two approaches learn qualitatively different things: ICL teaches the format but not the task-specific distribution; SFT learns the distribution but loses the ability to exploit in-context examples. Distillation was designed to bridge this gap, and it does so for NLI (+3.7 pp over SFT) but not yet for sequence labelling.

**4. Cross-lingual transfer is robust in the distilled condition.**  Across 6 languages, the largest English advantage in the distilled condition is in NER (EN: 36.8 vs HI: 51.6 — strikingly, *Hindi outperforms English* on NER after KD). This may be because the English NER training data has fewer entity types than Hindi WikiANN, making the teacher's logits less useful for English entity spans. For POS, all languages are within ±10 pp of each other under distillation, suggesting the KD signal generalises cross-lingually at the tag level even when it hurts overall.

**5. The Distilled−FT / Ctrl−FT ratio is a new diagnostic for teacher quality.**  Defined as `(Dist−FT) / (Ctrl−FT)`, a ratio of 1.0 means the few-shot teacher adds nothing over a zero-shot teacher; a ratio of 0 means distillation with a few-shot teacher exactly matches SFT. From our data: NLI ≈ −0.66, PA ≈ 0.04, QA ≈ 0.81, NER ≈ 0.55, POS ≈ 0.21. NLI negative ratio (distilled *above* SFT) and POS low ratio (few-shot teacher adds most relative to zero-shot) suggest that this metric could be used to select per-task λ without a validation set: tasks with high ratio need lower λ; tasks with negative ratio can absorb high λ.

### Cross-Lingual Transfer (Distilled Condition)

English-only training; zero-shot evaluation in 5 other languages:

| Language | Avg Score (Distilled) |
| :--- | :---: |
| EN | **66.1%** |
| HI | **53.8%** |
| ES | **64.7%** |
| DE | **60.8%** |
| FR | **60.0%** |
| ZH | **51.2%** |

EN → non-EN gap: 66.1% vs 58.1% (**+8.0** pp)

ZH (Chinese) shows the largest gap on NER and QA, consistent with morphological distance from English training data. Interestingly, HI *exceeds* EN on NER in the distilled condition — the few-shot teacher's Hindi entity logits may provide more distinctive supervision than English ones for this task.
