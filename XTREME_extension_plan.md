# XTREME Benchmark Extension Plan

## Context
Extend the existing ICL distillation project (currently GSM8K only) to the XTREME cross-lingual benchmark. The goal is to show that few-shot distillation generalises beyond math reasoning to 5 diverse NLP tasks across 6 languages, and produce a comprehensive result report with tables and plots.

---

## Scope

### Tasks (XTREME mapping)
| Code | Task | Dataset | Metric | Train data |
|------|------|---------|--------|-----------|
| `nli` | NLI | XNLI | Accuracy | MultiNLI/XNLI English (393k) |
| `pa` | Paraphrase | PAWS-X | Accuracy | PAWS-X English (49k) |
| `qa` | Q&A | MLQA | F1 + EM | SQuAD v1.1 English (87k) |
| `ner` | NER | WikiANN | F1 (span) | WikiANN English (20k) |
| `pos` | POS | UDPOS | Accuracy | UD English EWT (12k) |

### Languages: `en`, `hi`, `es`, `de`, `fr`, `zh`

### Language availability per task
| Task | en | hi | es | de | fr | zh |
|------|----|----|----|----|----|----|
| NLI  | ✓  | ✓  | ✓  | ✓  | ✓  | ✓  |
| PA   | ✓  | ✗  | ✓  | ✓  | ✓  | ✓  |
| QA   | ✓  | ✓  | ✓  | ✓  | ✗  | ✓  |
| NER  | ✓  | ✓  | ✓  | ✓  | ✓  | ✓  |
| POS  | ✓  | ✓  | ✓  | ✓  | ✓  | ✓  |

### Models
- `Qwen/Qwen3-1.7B`
- `Qwen/Qwen3-8B`
- `meta-llama/Llama-3.2-3B-Instruct`
- `google/gemma-3-270m`

### Conditions (5 tables in results)
1. **base** – Zero-shot, no fine-tuning
2. **fewshot** – 5-shot ICL, no fine-tuning
3. **finetuned** – SFT on English training data (CE only)
4. **distilled** – SFT + few-shot logit distillation (same as online_v1 on GSM8K)
5. **control** – SFT + zero-shot teacher distillation (ablation)

---

## Architecture Design

### Prompt format (all tasks use chat template when available)

**NLI** (single-token label):
```
System: You are an NLI classifier. Output one of: entailment, neutral, contradiction.
User: Premise: {premise}\nHypothesis: {hypothesis}
Assistant: entailment  ← teacher includes answer; student generates
```

**PA** (single-token label):
```
System: Classify if the sentences are paraphrases. Output yes or no.
User: Sentence 1: {s1}\nSentence 2: {s2}
Assistant: yes/no
```

**QA** (short span):
```
System: Answer the question based on the context. Be concise.
User: Context: {context}\nQuestion: {question}
Assistant: {answer_span}
```

**NER** (tag sequence):
```
System: Tag each token: O B-PER I-PER B-ORG I-ORG B-LOC I-LOC.
User: Tokens: {space_separated_tokens}
Assistant: {space_separated_tags}
```

**POS** (tag sequence):
```
System: Tag each token with its POS: NOUN VERB ADJ ADV PRON DET ADP NUM CONJ PART PUNCT X.
User: Tokens: {space_separated_tokens}
Assistant: {space_separated_tags}
```

### Distillation framework (same as GSM8K)
- Teacher: frozen base model, sees few-shot context (SAME-language examples) + target input
- Student: LoRA adapter, sees zero-shot target input
- Loss: L_CE + λ * MSE(top-K teacher logits, student logits at answer token positions)
- Token alignment: both sequences end with identical answer token IDs

For zero-shot teacher control: teacher also sees zero-shot input (no few-shot examples).

### Training strategy: multilingual where available, English fallback
- For tasks with multilingual training data (WikiANN/NER, UDPOS/POS, PAWS-X/PA):
  - Use same-language training data and same-language few-shot examples
  - This tests in-language distillation
- For tasks with English-only training (XNLI/NLI via MultiNLI, SQuAD/QA):
  - Train on English, evaluate cross-lingually
  - English few-shot in teacher context during training; target-language few-shot from val set at eval time
- **Few-shot examples always from the SAME task AND language as the query**

### Multi-task sampling
- Cap each task training set at 5,000 examples
- Sample tasks uniformly (round-robin or random)
- Single training run per model per condition (not per-task runs)
- Max steps: 1000 (same as GSM8K)

---

## File Structure

### New files to create

```
src/data/
  xtreme_loader.py          # Data loader for all 5 tasks, 6 languages

src/training/
  train_xtreme_sft.py       # Multi-task SFT (CE only), adapted from train_baseline.py
  train_xtreme_distill.py   # Multi-task distillation, adapted from train_online_v1.py

scripts/
  eval_xtreme_inference.py  # Base + fewshot eval (HF generation, no training)
  eval_xtreme_checkpoints.py # Eval trained checkpoints (all conditions)
  gen_xtreme_results.py     # Generate xtreme_results.md with tables + matplotlib plots
  run_xtreme_overnight.sh   # Master orchestration script

configs/
  xtreme_qwen1b7.yaml       # Qwen3-1.7B XTREME config
  xtreme_qwen8b.yaml        # Qwen3-8B XTREME config
  xtreme_llama3b.yaml       # Llama-3.2-3B XTREME config
  xtreme_gemma270m.yaml     # Gemma-3-270M XTREME config
  xtreme_distill.yaml       # Distillation override (n_top_logits, lambda)

xtreme_results.md           # Generated output (tables + analysis)
XTREME_extension_plan.md    # This plan (checked into project)
```

---

## Detailed File Specs

### `src/data/xtreme_loader.py`

Key components:
```python
TASK_LANGUAGES = {
    "nli": ["en", "hi", "es", "de", "fr", "zh"],
    "pa":  ["en", "es", "de", "fr", "zh"],       # no Hindi
    "qa":  ["en", "hi", "es", "de", "zh"],        # no French
    "ner": ["en", "hi", "es", "de", "fr", "zh"],
    "pos": ["en", "hi", "es", "de", "fr", "zh"],
}

UDPOS_CONFIGS = {
    "en": "en_ewt", "hi": "hi_hdtb", "es": "es_ancora",
    "de": "de_gsd", "fr": "fr_gsd", "zh": "zh_gsd",
}

NLI_LABEL_MAP  = {0: "entailment", 1: "neutral", 2: "contradiction"}
PA_LABEL_MAP   = {1: "yes", 0: "no"}
NER_TAG_MAP    = {0: "O", 1: "B-PER", 2: "I-PER", 3: "B-ORG",
                  4: "I-ORG", 5: "B-LOC", 6: "I-LOC"}

def load_task_data(task, split, lang, max_samples=None) -> list[dict]
def build_system_prompt(task) -> str
def build_example_messages(task, example) -> (user_msg, assistant_msg)
def build_teacher_messages(task, fewshot_examples, query) -> list[dict]
def build_student_messages(task, query) -> list[dict]
def get_answer_text(task, example) -> str

@dataclass
class XTREMEBatch:
    teacher_input_ids, teacher_attention_mask
    student_input_ids, student_attention_mask
    labels                  # -100 on prompt, answer tokens otherwise
    example_idx
    task_ids                # (B,) task index for logging

class XTREMEDistillDataset(Dataset):
    # Multi-task: cycles through all 5 tasks
    # For teacher_include_answer=True: teacher ends with gold answer tokens
    # Same token alignment logic as GSM8KDistillDataset

def collate_fn_xtreme(batch, pad_token_id) -> XTREMEBatch
def make_xtreme_dataloader(tasks, lang, tokenizer, ...) -> DataLoader

# Evaluation helpers
def parse_nli_output(text) -> str          # → "entailment"/"neutral"/"contradiction"
def parse_pa_output(text) -> str           # → "yes"/"no"
def parse_qa_output(text) -> str           # → raw generated text (for F1/EM)
def parse_ner_output(text, n_tokens) -> list[str]   # → BIO tag list
def parse_pos_output(text, n_tokens) -> list[str]   # → POS tag list

def compute_accuracy(preds, golds) -> float
def compute_qa_f1_em(preds, golds) -> (float, float)    # SQuAD-style
def compute_ner_f1(preds, golds) -> float               # span-level with seqeval
def compute_pos_accuracy(preds, golds) -> float         # token accuracy
```

**Key design choices:**
- Teacher uses English few-shot examples for ALL target languages (cross-lingual)
- `num_fewshot`: 3 for QA (contexts are long), 5 for NLI/PA/NER/POS
- Student sequences include the gold answer (same as GSM8K online distillation)
- Token alignment: `n_ans = (labels != -100).sum()`, same `answer_alignment()` function

### `src/training/train_xtreme_sft.py`

Near-copy of `train_baseline.py`:
- Import `XTREMEDistillDataset`, `make_xtreme_dataloader` instead of GSM8K equivalents
- Multi-task: instantiate one `XTREMEDistillDataset` per task (all 5) with English train data
- ConcatDataset or interleaved sampling
- Same CE loss, optimizer, cosine LR, checkpoint saving
- Output: `{output_dir}/xtreme_sft/{model}/checkpoint-{step}`

### `src/training/train_xtreme_distill.py`

Near-copy of `train_online_v1.py`:
- Frozen teacher (same base model, no LoRA) processes teacher sequences
- Student (LoRA) processes student sequences
- Same `answer_alignment()` function for token alignment
- Loss: L_CE + λ * MSE(top-K teacher logits, student logits)
- `--zeroshot_teacher` flag controls control condition (teacher uses 0-shot)

### `scripts/eval_xtreme_inference.py`

HF-based inference for base/fewshot conditions:
```
python scripts/eval_xtreme_inference.py \
    --model Qwen/Qwen3-1.7B \
    --conditions base fewshot \
    --tasks nli pa qa ner pos \
    --languages en hi es de fr zh \
    --n_samples 500 \
    --output experiments/xtreme/qwen1b7_inference.json \
    --device cuda:0
```
- Uses `transformers.pipeline` or `AutoModelForCausalLM.generate` with greedy decoding
- max_new_tokens: 5 (NLI/PA), 60 (QA), 150 (NER/POS)
- Saves per-(condition, task, lang) metrics to JSON

### `scripts/eval_xtreme_checkpoints.py`

Evaluates trained checkpoints:
```
python scripts/eval_xtreme_checkpoints.py \
    --base_model Qwen/Qwen3-1.7B \
    --checkpoint_dir experiments/xtreme/qwen1b7/xtreme_sft/final \
    --condition finetuned \
    --output experiments/xtreme/qwen1b7_trained.json \
    --tasks all --languages all --n_samples 500
```
- Loads PEFT adapter if `adapter_config.json` exists
- Same generation as inference script
- Saves results in same JSON format

### `scripts/gen_xtreme_results.py`

Reads all JSON files, produces `xtreme_results.md`:
- 5 tables (one per condition): rows = tasks×metrics, cols = 4 models × 6 languages
- Actually: per-condition table = tasks (rows) × languages (cols) with model sections
- Actually clearest format: one table per model per condition = 4×5 = 20 tables (too many)
- **Best format**: per-condition, one row per task, grouped by model, with "avg" column
- Radar/bar charts with matplotlib → saved as `assets/xtreme_*.png`
- Analysis section comparing conditions

### `scripts/run_xtreme_overnight.sh`

```bash
#!/bin/bash
# Overnight XTREME experiment runner
# Assumes: inside apptainer container on cn14-dgx, conda env active

# Phase 1: Inference eval (base + fewshot) — GPUs 0,1 and 2,3 in parallel
# Session claude: small models
# Session vscode: large models

# Phase 2: SFT training (multi-task)
# Session claude (GPUs 0,1): Qwen3-1.7B, then Gemma-270M
# Session vscode (GPUs 2,3): Llama-3.2-3B, then Qwen3-8B

# Phase 3: Distillation training (parallel with SFT eval)
# Phase 4: Control training
# Phase 5: Eval all checkpoints
# Phase 6: gen_xtreme_results.py → xtreme_results.md
```

---

## Configs

### `configs/xtreme_qwen1b7.yaml`
```yaml
model:
  name: "Qwen/Qwen3-1.7B"
  num_layers: 28
  hidden_size: 2048

data:
  tasks: [nli, pa, qa, ner, pos]
  train_lang: en
  eval_langs: [en, hi, es, de, fr, zh]
  num_fewshot_default: 5
  num_fewshot_qa: 3      # shorter due to long contexts
  max_samples_per_task: 5000
  max_seq_len_teacher: 2048
  max_seq_len_student: 512
  num_workers: 4

training:
  seed: 42
  output_dir: "experiments/xtreme/qwen1b7"
  logging_steps: 10
  save_steps: 200
  max_steps: 1000
  warmup_steps: 50
  lr: 2.0e-4
  weight_decay: 0.01
  grad_clip: 1.0
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  bf16: true
  gradient_checkpointing: false

lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
  bias: "none"
```

### `configs/xtreme_distill.yaml` (override)
```yaml
distillation:
  n_top_logits: 256
  lambda_distill: 0.5
  normalize_logits: false
```

---

## Overnight Schedule (cn14-dgx)

### Setup (5 min)
```bash
ssh -p 4422 cn14-dgx
app  # → apptainer
conda activate /dev/shm/vllm
pip install seqeval -q  # needed for NER F1
cd ~/workspace/icl-distillation
```

### Session `claude` (GPUs 0,1) — run in sequence
```bash
# Phase 1a: Inference eval (Qwen3-1.7B + Gemma)
CUDA_VISIBLE_DEVICES=0,1 python scripts/eval_xtreme_inference.py \
    --model Qwen/Qwen3-1.7B --conditions base fewshot --n_samples 500 \
    --output experiments/xtreme/qwen1b7_inference.json

CUDA_VISIBLE_DEVICES=0 python scripts/eval_xtreme_inference.py \
    --model google/gemma-3-270m --conditions base fewshot --n_samples 500 \
    --output experiments/xtreme/gemma270m_inference.json

# Phase 2a: SFT training
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 \
    --mixed_precision bf16 --main_process_port 29500 \
    src/training/train_xtreme_sft.py \
    --config configs/xtreme_qwen1b7.yaml \
    --output_dir experiments/xtreme/qwen1b7

# Phase 3a: Distillation
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 \
    --mixed_precision bf16 --main_process_port 29500 \
    src/training/train_xtreme_distill.py \
    --base_config configs/xtreme_qwen1b7.yaml \
    --config configs/xtreme_distill.yaml \
    --output_dir experiments/xtreme/qwen1b7

# Phase 4a: Control
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 \
    --mixed_precision bf16 --main_process_port 29500 \
    src/training/train_xtreme_distill.py \
    --base_config configs/xtreme_qwen1b7.yaml \
    --config configs/xtreme_distill.yaml \
    --zeroshot_teacher \
    --output_dir experiments/xtreme/qwen1b7

# Then repeat for Gemma-270M
# Phase 5a: Eval checkpoints (both models)
```

### Session `vscode` (GPUs 2,3) — parallel with `claude`
```bash
# Same sequence but for Llama-3.2-3B, then Qwen3-8B
```

### Time estimate
| Phase | Duration |
|-------|----------|
| Inference eval (2 small models) | ~60 min |
| SFT + distill + control × 2 small models | ~3 hours |
| Inference eval (Llama + Qwen8B) | ~90 min |
| SFT + distill + control × Llama | ~3 hours |
| SFT + distill + control × Qwen8B | ~4 hours (larger model) |
| Checkpoint eval (all models) | ~60 min |
| gen_xtreme_results.py | ~5 min |
| **Total** | **~7-8 hours** (overnight) |

---

## Results Format (`xtreme_results.md`)

### Structure
```markdown
# XTREME Benchmark Results

## Table 1: Condition = Base (Zero-Shot)
### Task: NLI (Accuracy %)
| Model | en | hi | es | de | fr | zh | avg |
|-------|----|----|----|----|----|----|-----|
| Qwen3-1.7B | ... |
| Qwen3-8B   | ... |
| Llama-3.2-3B | ... |
| Gemma-270M | ... |

### Task: PA (Accuracy %)
... [similar table]

### Task: QA (F1 / EM)
... [separate rows for F1 and EM]

... [NER, POS tables]

## Table 2: Condition = Few-Shot (5-shot ICL)
... [same structure]

## Table 3: Condition = Fine-Tuned (SFT)
...

## Table 4: Condition = Distilled (SFT + Few-Shot KD)
...

## Table 5: Condition = Control (SFT + Zero-Shot KD)
...

## Summary: Distill vs SFT improvement by task and language
...

## Plots
![xtreme_bar_by_task.png]
![xtreme_lang_heatmap.png]

## Analysis
...
```

---

## Dependencies to Install
```bash
pip install seqeval -q           # NER F1 evaluation
pip install sacrebleu -q         # Optional: for QA normalization
# datasets, transformers, peft, accelerate already available
```

---

## Critical Implementation Details

1. **Token alignment** (NER/POS): For sequence labeling, teacher and student both end with the same tag sequence tokens. The `answer_alignment()` function from train_online_v1.py works unchanged.

2. **Multi-task sampling**: Use `ConcatDataset` from PyTorch. Each task contributes max 5000 training examples. Shuffle at dataloader level.

3. **Qwen3 thinking mode**: Always `enable_thinking=False` — applies to all tasks, not just GSM8K.

4. **Same-language few-shot**: For each (task, language) pair, k examples are drawn from the SAME language and task. For eval (few-shot condition): use val/dev set examples from the target language. For distillation training (teacher context): use same-language training examples where available (WikiANN/NER has all 6 languages; UDPOS has all 6; PAWS-X has en/de/es/fr/zh), fall back to English for tasks with English-only training data (XNLI→MultiNLI, QA→SQuAD).

5. **max_seq_len_teacher=2048**: Shorter than GSM8K (4096) since XTREME inputs are shorter than 8-shot math problems.

6. **NER evaluation**: Use seqeval for span-level F1 (standard). Parse model output by extracting recognized BIO tags.

7. **UDPOS loading**: `load_dataset("universal_dependencies", config, trust_remote_code=True)` with language-specific config names. Fallback: load from `xtreme` dataset.

8. **Gemma-3-270M**: Full fine-tuning (no LoRA) — `use_lora: false` in config, same as existing gemma270m.yaml.

9. **Qwen3-8B**: Use gradient checkpointing — same as existing qwen8b.yaml.

10. **QA training data**: Load SQuAD v1.1 (not MLQA) for English training. Eval on MLQA.
