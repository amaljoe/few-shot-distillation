# Method & Code Critique: Few-Shot Distillation

## Executive take
The observed gain is **promising but not yet causally isolated** to the proposed "few-shot behavior baking" mechanism. The implementation is technically coherent, but current controls are insufficient to rule out alternative explanations (regularization from soft targets, teacher-forcing effects, longer-context teacher conditioning artifacts, or compute/method mismatch).

---

## What looks solid
- **Clear teacher/student setup**: frozen teacher, LoRA-only student updates, and combined CE + distillation objective are implemented consistently.
- **Answer-position masking**: distillation is only applied where labels are valid answer tokens.
- **No obvious direct test leakage in training code**: training loader uses GSM8K train split.

---

## Main validity risks / possible "cheating" vectors

### 1) Teacher is conditioned on ground-truth answer tokens during distillation
In online mode, teacher sequences explicitly include the gold assistant answer (`teacher_include_answer=True`), and then distillation matches teacher/student logits at answer positions.

Why this matters:
- This is standard teacher-forced token KD behavior, but it means teacher logits are computed with access to prior gold answer prefix.
- It blurs whether gains come from **few-shot context transfer** vs **soft-label next-token smoothing under teacher forcing**.

This is not "cheating" in a data-leak sense, but it is a **causal attribution confound** for the paper claim.

### 2) Missing key ablation: no-fewshot teacher distillation
Current code compares:
- baseline SFT (CE only)
- few-shot teacher distillation (CE + KD)

But it does **not** include:
- CE + KD from a **0-shot teacher** (same model, no exemplars)

Without this, you cannot claim the gain specifically comes from few-shot information rather than generic KD regularization.

### 3) Distillation objective is raw-logit MSE on top-K vocab only
Using unnormalized top-K logit MSE can improve optimization but introduces another confound:
- gains may come from a particular surrogate loss choice (logit calibration/regularization), not from few-shot behavior transfer.

Need KL/probability-temperature controls and random-topK controls.

### 4) Potential truncation/alignment sensitivity
Teacher context is much longer than student and uses truncation. If truncation clips earlier few-shot demonstrations unevenly, the effective teacher signal varies by sample.

This does not invalidate results but can produce hidden variance and overstate mechanism clarity.

### 5) Evaluation framing could over-credit one factor
If model selection/hyperparameters (e.g., λ, top-K, steps) were tuned around this setup only, some uplift may be from tuning budget rather than method-specific mechanism.

---

## Can the current results be fully credited to the method?
**Not yet.** You can claim an empirical improvement for this training recipe, but not a clean mechanistic claim that the gain is uniquely due to baking in few-shot reasoning.

What is currently defensible:
- "This CE+online-teacher KD recipe improves GSM8K accuracy over our LoRA SFT baseline under these settings."

What is not yet defensible:
- "The gain is specifically from transferring few-shot ICL behavior," without additional controls.

---

## High-priority experiments to make claims foolproof

### A. Causal ablation matrix (must-have)
Run identical training budgets with:
1. **CE only** (baseline)
2. **CE + KD (0-shot teacher)**
3. **CE + KD (few-shot teacher, your method)**
4. **CE + KD (few-shot teacher but shuffled/irrelevant exemplars)**

Interpretation:
- (3) > (2) isolates few-shot contribution.
- (3) > (4) shows exemplar content matters (not just extra tokens).

### B. Counterfactual teacher quality ablations
- Few-shot teacher with **incorrect assistant exemplars**.
- Few-shot teacher with **random question-answer pairs**.
- Few-shot teacher with **question-only exemplars** (no answers).

If performance survives these, your current mechanism claim is weak.

### C. Loss-form ablations
- Replace raw-logit MSE with KL at temperatures T∈{1,2,4}.
- Distill top-K vs full-vocab (if feasible on small subset).
- Randomly selected K vocab indices (control).

This separates "good distillation loss" from "few-shot transfer".

### D. Compute-fairness controls
- Match wall-clock/FLOPs between baseline and distill (e.g., longer baseline or stronger regularization).
- Compare to stronger non-distill regularizers (label smoothing, R-Drop, entropy bonus).

If matched-baseline closes gap, claimed novelty weakens.

### E. Attribution/diagnostic probes
- Measure KL(student, teacher-fewshot) and KL(student, teacher-zeroshot) during training.
- Evaluate student on **teacher-preference agreement** tasks for ambiguous steps.
- Check if gains concentrate on examples where few-shot teacher differs most from zero-shot teacher.

### F. Generalization checks (anti-overfitting)
- Evaluate on SVAMP, MultiArith, ASDiv, GSM-hard variants.
- Report with and without CoT extraction sensitivity.
- Bootstrap confidence intervals and paired significance tests.

---

## Code-level improvements that increase trust
- Log teacher effective context length and truncation rate per batch.
- Save per-step distillation statistics (mean |t-s| logit gap, entropy).
- Add explicit experiment tags in output dirs so baseline/distill are guaranteed config-matched.
- Add a script that auto-runs the ablation grid and generates a single comparison table.

---

## Bottom line
There is no obvious "test leakage cheat" in the visible training/eval pipeline. But **causal over-attribution risk is high**: current evidence supports "this recipe works" more than "few-shot behavior transfer is the unique reason it works." Run the ablations above to make the claim robust and publication-grade.
