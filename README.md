## 🧠 Few-Shot Distillation for Stable Fine-Tuning

**Distilling In-Context Learning into Model Weights via Layer-wise Attention Supervision**

---

## 📌 Motivation

Large Language Models often show:

* lower loss in **few-shot (in-context learning)** compared to
* **zero-shot fine-tuned models** during early training.

This suggests that:

> Context provides useful adaptation signals that weights do not initially encode.

This project investigates whether **few-shot behavior can be distilled into model parameters** to:

* reduce early representation drift
* accelerate fine-tuning
* stabilize optimization.

---

## 🎯 Core Hypothesis

Few-shot context performs an implicit low-rank weight update through attention.

If we supervise a zero-shot model using internal representations of a few-shot run:

* the model can learn to produce similar behavior **without context**.

---

## 🧩 Key Idea

### Teacher (Few-shot model)

```
Input = [Few-shot context + query]
Output = layer-wise attention activations
```

### Student (Zero-shot model)

```
Input = [query only]
Goal = match teacher internal representations
```

Training objective:

```
L = L_task + λ * Σ || h_l(student) − h_l(teacher) ||²
```

Optional stronger formulation:

```
Δh = h_fewshot − h_zeroshot
```

Train student to predict adaptation signal Δh.

---

# 🧠 Models

### **Qwen3-8B-Instruct**
### **Qwen3-4B-Instruct**
### **Qwen3-2B-Instruct**


# 🧩 Dataset Selection

The goal is NOT pure accuracy benchmarking.

We need datasets where:

```
few-shot performance >> zero-shot performance
```

so that adaptation signals are strong.

---

## ⭐ Primary Dataset — GSM8K

Math reasoning dataset chosen because:

* strong few-shot improvements
* structured reasoning signals
* attention layers carry meaningful computation
* widely accepted in ICL research.

This will be the main experimental environment.

---

## ⭐ Secondary Dataset — MMLU (subset)

Recommended subsets:

* logical reasoning
* abstract algebra
* professional law

Purpose:

* test generalization of learned adaptation.

---

## ⭐ Additional Dataset — BIG-Bench Hard (BBH)

Recommended tasks:

* causal reasoning
* logical deduction

Used for robustness validation.

---

## 🚀 Optional Extension

Multilingual reasoning tasks (future work):

* distilling few-shot adaptation across languages.

---

# 📂 Project Structure

```
project/
│
├── configs/
│   ├── base.yaml
│   ├── distill_layerwise.yaml
│
├── src/
│   ├── models/
│   │   ├── student.py
│   │   ├── teacher_wrapper.py
│   │
│   ├── training/
│   │   ├── train_baseline.py
│   │   ├── train_layerwise_distill.py
│   │
│   ├── losses/
│   │   ├── layer_matching.py
│   │
│   ├── hooks/
│   │   ├── activation_capture.py
│
├── experiments/
│   ├── baseline_ft/
│   ├── fewshot_teacher/
│   ├── layerwise_distill/
│
└── README.md
```

---

# 🧪 Experimental Plan

---

## Experiment 1 — Baseline Comparison

Compare:

| Method             | Description             |
| ------------------ | ----------------------- |
| Few-shot inference | Teacher baseline        |
| Zero-shot FT       | Standard fine-tuning    |
| Proposed           | Layer-wise distillation |

Key metric:

```
Iteration where FT loss < few-shot loss
```

---

## Experiment 2 — Layer Localization

Supervise:

* early layers
* middle layers
* late layers
* all layers

Goal:

> Identify where few-shot adaptation occurs.

---

## Experiment 3 — Attention vs FFN Matching

Variants:

* attention output matching
* FFN output matching
* full block matching

---

## Experiment 4 — Representation Drift Analysis

Track:

```
cosine(pretrained, current representations)
```

Questions:

* Does distillation reduce drift?
* Which layers drift most?

---

# 📊 Evaluation Metrics

### Optimization

* loss crossover iteration
* training stability

### Representation

* cosine similarity
* optional CKA / SVCCA

### Task

* task accuracy
* few-shot gap reduction

---

# ⚙️ Recommended Training Setup

Hardware:

```
4× A100 GPUs
```

Suggested settings:

* LR warmup
* gradient clipping
* layer-wise LR decay (optional)

---

# 🔬 Data Formatting

Teacher input:

```
[Example 1 Q+A]
[Example 2 Q+A]
[Example 3 Q+A]
Target Question
```

Student input:

```
Target Question
```

Teacher/Student output (final loss calculated only on this):
```
Target Answer
```

---

# 🧪 Vision

Transform:

```
Few-shot runtime adaptation
        ↓
Layer-wise context distillation
        ↓
Improved zero-shot model
```

Goal:

Reduce dependence on prompts while preserving adaptation capability.

---

# 🤝 Contributors

* Amal Joe (IIT Bombay)