"""
XTREME benchmark data loader for ICL distillation experiments.

Tasks:
  nli  — Natural Language Inference    (XNLI, 3-way classification)
  pa   — Paraphrase Identification     (PAWS-X, binary)
  qa   — Question Answering            (MLQA; training via SQuAD)
  ner  — Named Entity Recognition      (WikiANN / PAN-X, BIO tagging)
  pos  — Part-of-Speech Tagging        (Universal Dependencies / UDPOS)

Languages: en, hi, es, de, fr, zh

Few-shot protocol:
  Evaluation (fewshot condition): k examples from the SAME (task, language)
    drawn from the validation/dev set.
  Training teacher context: English few-shot from English training split
    (cross-lingual zero-shot transfer is the standard XTREME protocol).

Teacher sequences include the gold answer so both teacher and student
end with the same answer token IDs — identical to the GSM8K setup.
"""

import random
import re
import string
from dataclasses import dataclass
from typing import Optional

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

# ============================================================================
# Task / language configuration
# ============================================================================

TASKS = ["nli", "pa", "qa", "ner", "pos"]
LANGUAGES = ["en", "hi", "es", "de", "fr", "zh"]

# Which languages have data for each task
TASK_LANGUAGES = {
    "nli": ["en", "hi", "es", "de", "fr", "zh"],   # XNLI
    "pa":  ["en", "es", "de", "fr", "zh"],           # PAWS-X (no Hindi)
    "qa":  ["en", "hi", "es", "de", "zh"],            # MLQA (no French)
    "ner": ["en", "hi", "es", "de", "fr", "zh"],     # WikiANN
    "pos": ["en", "hi", "es", "de", "fr", "zh"],     # UDPOS
}

# xtreme/udpos configs use full English language names, not ISO codes
UDPOS_LANG_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "es": "Spanish",
    "de": "German",
    "fr": "French",
    "zh": "Chinese",
}

# Number of few-shot examples per task (QA has fewer due to long contexts)
NUM_FEWSHOT = {"nli": 5, "pa": 5, "qa": 3, "ner": 5, "pos": 5}

# Max new tokens to generate at evaluation
TASK_MAX_NEW_TOKENS = {"nli": 8, "pa": 4, "qa": 80, "ner": 150, "pos": 150}

# Label maps
NLI_ID2LABEL = {0: "entailment", 1: "neutral", 2: "contradiction"}
PA_ID2LABEL  = {0: "no", 1: "yes"}
NER_ID2LABEL = {
    0: "O",
    1: "B-PER", 2: "I-PER",
    3: "B-ORG", 4: "I-ORG",
    5: "B-LOC", 6: "I-LOC",
}
UPOS_NAMES = [
    "X", "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ",
    "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB",
]

# ============================================================================
# Dataset loading
# ============================================================================

def load_task_data(
    task: str,
    split: str,
    lang: str,
    max_samples: Optional[int] = None,
) -> list[dict]:
    """
    Load XTREME-style data for a given task / split / language.

    split: "train", "dev", or "test"

    Returns a list of normalized dicts:
      nli:  {premise, hypothesis, label_text}
      pa:   {sentence1, sentence2, label_text}
      qa:   {context, question, answers}  (answers is list[str])
      ner:  {tokens, ner_tags}            (string BIO tags)
      pos:  {tokens, pos_tags}            (string POS tags)

    Returns [] if the (task, lang) pair is unavailable.
    """
    if lang not in TASK_LANGUAGES.get(task, []):
        return []

    loaders = {"nli": _load_xnli, "pa": _load_pawsx, "qa": _load_mlqa,
               "ner": _load_wikiann, "pos": _load_udpos}
    return loaders[task](split, lang, max_samples)


def _load_xnli(split: str, lang: str, max_samples: Optional[int]) -> list[dict]:
    # XNLI in the xtreme dataset has only validation+test (no multilingual train split).
    # For training we load English-only; for all other languages at train time return [].
    if split == "train":
        if lang != "en":
            return []
        # Use XNLI English train split (= MultiNLI) for NLI training data
        _STR2ID = {"entailment": 0, "neutral": 1, "contradiction": 2}
        ds = None
        for ds_args in [
            ("xnli", "en", "train"),
            ("multi_nli", None, "train_matched"),
        ]:
            try:
                name, cfg, sp = ds_args
                ds = load_dataset(name, cfg, split=sp) if cfg else load_dataset(name, split=sp)
                break
            except Exception:
                continue
        if ds is None:
            return []
        out = []
        for ex in ds:
            raw = ex.get("label", ex.get("gold_label", -1))
            if isinstance(raw, str):
                label_id = _STR2ID.get(raw.lower(), -1)
            else:
                label_id = int(raw)
            if label_id not in NLI_ID2LABEL:
                continue
            out.append({
                "premise": str(ex.get("premise", ex.get("sentence1", ""))),
                "hypothesis": str(ex.get("hypothesis", ex.get("sentence2", ""))),
                "label_text": NLI_ID2LABEL[label_id],
            })
            if max_samples and len(out) >= max_samples:
                break
        return out

    # Eval: use xtreme/XNLI (all languages in one dataset, filter by language field)
    # split mapping: dev → validation, test → test
    hf_split = "validation" if split == "dev" else "test"
    try:
        ds = load_dataset("xtreme", "XNLI", split=hf_split, trust_remote_code=False)
        ds = ds.filter(lambda x: x["language"] == lang)
    except Exception:
        return []

    out = []
    for ex in ds:
        label_id = ex.get("gold_label", -1)
        if isinstance(label_id, str):
            label_id = {"entailment": 0, "neutral": 1, "contradiction": 2}.get(label_id, -1)
        if label_id not in NLI_ID2LABEL:
            continue
        out.append({
            "premise": str(ex["sentence1"]),
            "hypothesis": str(ex["sentence2"]),
            "label_text": NLI_ID2LABEL[label_id],
        })
        if max_samples and len(out) >= max_samples:
            break
    return out


def _load_pawsx(split: str, lang: str, max_samples: Optional[int]) -> list[dict]:
    hf_split = "train" if split == "train" else ("validation" if split == "dev" else "test")
    try:
        ds = load_dataset("xtreme", f"PAWS-X.{lang}", split=hf_split, trust_remote_code=False)
    except Exception:
        return []

    out = []
    for ex in ds:
        lbl = int(ex.get("label", 0))
        out.append({
            "sentence1": str(ex["sentence1"]),
            "sentence2": str(ex["sentence2"]),
            "label_text": PA_ID2LABEL.get(lbl, "no"),
        })
        if max_samples and len(out) >= max_samples:
            break
    return out


def _load_mlqa(split: str, lang: str, max_samples: Optional[int]) -> list[dict]:
    if split == "train":
        # SQuAD only exists in English; other languages fall back to nothing
        if lang != "en":
            return []
        # Use English SQuAD for training
        try:
            ds = load_dataset("squad", split="train", trust_remote_code=False)
        except Exception:
            return []
        out = []
        for ex in ds:
            answers = ex["answers"]["text"]
            if not answers:
                continue
            out.append({"context": ex["context"], "question": ex["question"],
                        "answers": list(answers)})
            if max_samples and len(out) >= max_samples:
                break
        return out

    hf_split = "validation" if split == "dev" else "test"
    try:
        ds = load_dataset("xtreme", f"MLQA.{lang}.{lang}", split=hf_split, trust_remote_code=False)
    except Exception:
        return []

    out = []
    for ex in ds:
        raw_ans = ex.get("answers", {})
        if isinstance(raw_ans, dict):
            answers = raw_ans.get("text", [])
        else:
            answers = list(raw_ans)
        if not answers:
            continue
        out.append({"context": ex["context"], "question": ex["question"],
                    "answers": list(answers)})
        if max_samples and len(out) >= max_samples:
            break
    return out


def _load_wikiann(split: str, lang: str, max_samples: Optional[int]) -> list[dict]:
    hf_split = "train" if split == "train" else ("validation" if split == "dev" else "test")
    try:
        ds = load_dataset("xtreme", f"PAN-X.{lang}", split=hf_split, trust_remote_code=False)
    except Exception:
        return []

    out = []
    for ex in ds:
        tokens = list(ex["tokens"])
        ner_tags = [NER_ID2LABEL.get(int(t), "O") for t in ex["ner_tags"]]
        if not tokens:
            continue
        out.append({"tokens": tokens, "ner_tags": ner_tags})
        if max_samples and len(out) >= max_samples:
            break
    return out


def _load_udpos(split: str, lang: str, max_samples: Optional[int]) -> list[dict]:
    hf_split = "train" if split == "train" else ("validation" if split == "dev" else "test")
    lang_name = UDPOS_LANG_NAMES.get(lang, lang)
    try:
        ds = load_dataset("xtreme", f"udpos.{lang_name}", split=hf_split, trust_remote_code=False)
    except Exception:
        return []

    out = []
    for ex in ds:
        tokens = list(ex.get("tokens", []))
        raw_pos = ex.get("pos_tags", [])
        if not tokens or not raw_pos:
            continue
        if raw_pos and isinstance(raw_pos[0], int):
            pos_tags = [UPOS_NAMES[t] if 0 <= t < len(UPOS_NAMES) else "X" for t in raw_pos]
        else:
            pos_tags = [str(t) for t in raw_pos]
        out.append({"tokens": tokens, "pos_tags": pos_tags})
        if max_samples and len(out) >= max_samples:
            break
    return out


# ============================================================================
# Prompt / chat message builders
# ============================================================================

TASK_SYSTEM_PROMPTS = {
    "nli": (
        "You are a natural language inference classifier. "
        "Given a premise and a hypothesis, output exactly one of: entailment, neutral, contradiction."
    ),
    "pa": (
        "You are a paraphrase detector. "
        "Given two sentences, output yes if they are paraphrases of each other, or no otherwise."
    ),
    "qa": (
        "You are a reading comprehension assistant. "
        "Answer the question based on the given context. Be concise and output only the answer."
    ),
    "ner": (
        "You are a named entity recognition tagger. "
        "Label each token with one of: O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC. "
        "Output labels as space-separated tokens in the same order as the input."
    ),
    "pos": (
        "You are a part-of-speech tagger. "
        "Label each token with one of: NOUN, VERB, ADJ, ADV, PRON, DET, ADP, NUM, "
        "CCONJ, SCONJ, PART, AUX, PUNCT, PROPN, SYM, INTJ, X. "
        "Output tags as space-separated tokens in the same order as the input."
    ),
}


def _user_content(task: str, example: dict) -> str:
    """Build the user message content for one example (input only, no answer)."""
    if task == "nli":
        return f"Premise: {example['premise']}\nHypothesis: {example['hypothesis']}"
    elif task == "pa":
        return f"Sentence 1: {example['sentence1']}\nSentence 2: {example['sentence2']}"
    elif task == "qa":
        return f"Context: {example['context']}\nQuestion: {example['question']}"
    elif task == "ner":
        return f"Tokens: {' '.join(example['tokens'])}"
    elif task == "pos":
        return f"Tokens: {' '.join(example['tokens'])}"
    raise ValueError(f"Unknown task: {task}")


def get_answer_text(task: str, example: dict) -> str:
    """Return the gold answer string for training/teacher."""
    if task == "nli":
        return example["label_text"]
    elif task == "pa":
        return example["label_text"]
    elif task == "qa":
        answers = example.get("answers", [""])
        return answers[0] if answers else ""
    elif task == "ner":
        return " ".join(example["ner_tags"])
    elif task == "pos":
        return " ".join(example["pos_tags"])
    raise ValueError(f"Unknown task: {task}")


def build_teacher_messages(task: str, fewshot_examples: list[dict], query: dict) -> list[dict]:
    """
    Build chat messages for the teacher (few-shot) input.
    Format: [system] [user/assistant pairs for few-shot] [user target]
    """
    messages = [{"role": "system", "content": TASK_SYSTEM_PROMPTS[task]}]
    for ex in fewshot_examples:
        messages.append({"role": "user", "content": _user_content(task, ex)})
        messages.append({"role": "assistant", "content": get_answer_text(task, ex)})
    messages.append({"role": "user", "content": _user_content(task, query)})
    return messages


def build_student_messages(task: str, query: dict) -> list[dict]:
    """Build chat messages for the student (zero-shot) input."""
    return [
        {"role": "system", "content": TASK_SYSTEM_PROMPTS[task]},
        {"role": "user", "content": _user_content(task, query)},
    ]


# ============================================================================
# Chat template helpers (same pattern as gsm8k_loader)
# ============================================================================

def _has_chat_template(tokenizer: PreTrainedTokenizer) -> bool:
    return getattr(tokenizer, "chat_template", None) is not None


def _format_as_plain_text(messages: list[dict], add_generation_prompt: bool = True) -> str:
    """Plain-text fallback for base models without a chat template."""
    parts = []
    for msg in messages:
        parts.append(msg["content"])
    text = "\n".join(parts)
    if add_generation_prompt:
        text += "\n"
    return text


def apply_xtreme_template(
    tokenizer: PreTrainedTokenizer,
    messages: list[dict],
    add_generation_prompt: bool = True,
) -> str:
    """Apply chat template (thinking disabled for Qwen3); fallback to plain text."""
    if not _has_chat_template(tokenizer):
        return _format_as_plain_text(messages, add_generation_prompt)
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            chat_template_kwargs={"enable_thinking": False},
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


def _get_gen_prompt_len(tokenizer: PreTrainedTokenizer) -> int:
    """Compute the number of tokens added by add_generation_prompt=True."""
    dummy = [{"role": "system", "content": "S"}, {"role": "user", "content": "Q"}]
    with_gen = apply_xtreme_template(tokenizer, dummy, add_generation_prompt=True)
    without_gen = apply_xtreme_template(tokenizer, dummy, add_generation_prompt=False)
    gen_str = with_gen[len(without_gen):]
    return len(tokenizer.encode(gen_str, add_special_tokens=False))


# ============================================================================
# Batch dataclass
# ============================================================================

@dataclass
class XTREMEBatch:
    teacher_input_ids:       torch.Tensor   # (B, T_teach)
    teacher_attention_mask:  torch.Tensor   # (B, T_teach)
    student_input_ids:       torch.Tensor   # (B, T_stud)  — full seq incl. answer
    student_attention_mask:  torch.Tensor   # (B, T_stud)
    labels:                  torch.Tensor   # (B, T_stud)  — -100 on prompt
    example_idx:             torch.Tensor   # (B,)
    task_ids:                torch.Tensor   # (B,)  0..4


# ============================================================================
# Multi-task dataset
# ============================================================================

TASK2ID = {t: i for i, t in enumerate(TASKS)}


class XTREMEDistillDataset(Dataset):
    """
    Multi-task, multi-lingual XTREME dataset for ICL distillation.

    Loads training data for every (task, language) pair where training data
    exists.  For tasks/languages with no training data the pair is skipped
    gracefully (e.g. XNLI only has English MultiNLI training, SQuAD for QA).

    Availability:
      NLI  (XNLI)    — English train only  (MultiNLI)
      PA   (PAWS-X)  — en, es, de, fr, zh train
      QA   (MLQA)    — English train only  (SQuAD)
      NER  (WikiANN) — all 6 languages train
      POS  (UDPOS)   — all 6 languages train

    For each training example the teacher few-shot pool is drawn from the
    SAME (task, language) — true same-language, same-task few-shot signal.

    teacher_include_answer=True: teacher sequence ends with the gold answer so
    both teacher and student end with identical answer token IDs (required for
    online distillation token alignment).
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        tasks: list[str] = None,
        train_langs: list[str] = None,      # languages to include in training
        max_samples_per_task_lang: int = 5000,
        max_seq_len_teacher: int = 2048,
        max_seq_len_student: int = 512,
        seed: int = 42,
        teacher_include_answer: bool = False,
        zeroshot_teacher: bool = False,
        split: str = "train",               # "train" or "dev" (for meta outer objective)
    ):
        self.tokenizer = tokenizer
        self.tasks = tasks or TASKS
        self.train_langs = train_langs or LANGUAGES
        self.max_seq_len_teacher = max_seq_len_teacher
        self.max_seq_len_student = max_seq_len_student
        self.seed = seed
        self.teacher_include_answer = teacher_include_answer
        self.zeroshot_teacher = zeroshot_teacher
        self.split = split

        self.gen_prompt_len = _get_gen_prompt_len(tokenizer)

        # task_data[(task, lang)] = list[dict]
        self.task_data: dict[tuple[str, str], list[dict]] = {}

        for task in self.tasks:
            for lang in self.train_langs:
                data = load_task_data(task, self.split, lang, max_samples_per_task_lang)
                if data:
                    self.task_data[(task, lang)] = data

        if not self.task_data:
            raise RuntimeError("No training data loaded — check tasks/train_langs config.")

        # Build flat index: list of (task, lang, local_idx)
        self.index: list[tuple[str, str, int]] = []
        for (task, lang), data in self.task_data.items():
            for i in range(len(data)):
                self.index.append((task, lang, i))

        print(f"[XTREMEDistillDataset] {len(self.index)} total training examples across "
              f"{len(self.task_data)} (task, lang) pairs:")
        for (task, lang), data in sorted(self.task_data.items()):
            print(f"  {task}/{lang}: {len(data)} examples")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        task, lang, local_idx = self.index[idx]
        example = self.task_data[(task, lang)][local_idx]
        n_fewshot = NUM_FEWSHOT[task]

        # Deterministic same-language few-shot sampling
        rng = random.Random(self.seed + idx)
        pool = self.task_data[(task, lang)]
        candidates = [i for i in range(len(pool)) if i != local_idx]
        fewshot_indices = rng.sample(candidates, min(n_fewshot, len(candidates)))
        fewshot_examples = [pool[i] for i in fewshot_indices]

        # Zero-shot teacher control: no few-shot examples
        if self.zeroshot_teacher:
            fewshot_examples = []

        # ---- Teacher input ----
        teacher_messages = build_teacher_messages(task, fewshot_examples, example)

        if self.teacher_include_answer:
            # Online mode: append gold answer so teacher ends with same tokens as student
            teacher_messages_with_ans = teacher_messages + [
                {"role": "assistant", "content": get_answer_text(task, example)}
            ]
            teacher_prompt = apply_xtreme_template(
                self.tokenizer, teacher_messages_with_ans, add_generation_prompt=False
            )
        else:
            teacher_prompt = apply_xtreme_template(
                self.tokenizer, teacher_messages, add_generation_prompt=True
            )

        teacher_enc = self.tokenizer(
            teacher_prompt,
            truncation=True,
            max_length=self.max_seq_len_teacher,
            return_tensors=None,
            add_special_tokens=False,
        )

        # ---- Student input (zero-shot prompt only) ----
        student_messages = build_student_messages(task, example)
        student_prompt = apply_xtreme_template(
            self.tokenizer, student_messages, add_generation_prompt=True
        )
        student_enc = self.tokenizer(
            student_prompt,
            truncation=True,
            max_length=self.max_seq_len_student,
            return_tensors=None,
            add_special_tokens=False,
        )
        student_ids = student_enc["input_ids"]

        # ---- Full student sequence with answer (for CE loss) ----
        student_with_ans_messages = student_messages + [
            {"role": "assistant", "content": get_answer_text(task, example)}
        ]
        full_prompt = apply_xtreme_template(
            self.tokenizer, student_with_ans_messages, add_generation_prompt=False
        )
        full_enc = self.tokenizer(
            full_prompt,
            truncation=True,
            max_length=self.max_seq_len_student,
            return_tensors=None,
            add_special_tokens=False,
        )
        full_ids = full_enc["input_ids"]

        # Labels: -100 on prompt tokens, answer token IDs otherwise
        prompt_len = len(student_ids)
        labels = [-100] * prompt_len + full_ids[prompt_len:]
        labels = labels[:len(full_ids)]

        return {
            "teacher_input_ids":      teacher_enc["input_ids"],
            "teacher_attention_mask": teacher_enc["attention_mask"],
            "student_input_ids":      full_ids,
            "student_attention_mask": full_enc["attention_mask"],
            "labels":                 labels,
            "example_idx":            idx,
            "task_id":                TASK2ID.get(task, 0),
        }


# ============================================================================
# Collation
# ============================================================================

def collate_fn_xtreme(batch: list[dict], pad_token_id: int) -> XTREMEBatch:
    def pad_seq(seqs: list[list[int]], pad_val: int) -> torch.Tensor:
        max_len = max(len(s) for s in seqs)
        padded = [s + [pad_val] * (max_len - len(s)) for s in seqs]
        return torch.tensor(padded, dtype=torch.long)

    return XTREMEBatch(
        teacher_input_ids      = pad_seq([b["teacher_input_ids"] for b in batch], pad_token_id),
        teacher_attention_mask = pad_seq([b["teacher_attention_mask"] for b in batch], 0),
        student_input_ids      = pad_seq([b["student_input_ids"] for b in batch], pad_token_id),
        student_attention_mask = pad_seq([b["student_attention_mask"] for b in batch], 0),
        labels                 = pad_seq([b["labels"] for b in batch], -100),
        example_idx            = torch.tensor([b["example_idx"] for b in batch], dtype=torch.long),
        task_ids               = torch.tensor([b["task_id"] for b in batch], dtype=torch.long),
    )


def make_xtreme_dataloader(
    tokenizer: PreTrainedTokenizer,
    tasks: list[str] = None,
    train_langs: list[str] = None,   # all languages with training data by default
    batch_size: int = 4,
    max_samples_per_task_lang: int = 5000,
    max_seq_len_teacher: int = 2048,
    max_seq_len_student: int = 512,
    shuffle: bool = True,
    num_workers: int = 4,
    seed: int = 42,
    teacher_include_answer: bool = False,
    zeroshot_teacher: bool = False,
    split: str = "train",
) -> torch.utils.data.DataLoader:
    ds = XTREMEDistillDataset(
        tokenizer=tokenizer,
        tasks=tasks or TASKS,
        train_langs=train_langs or LANGUAGES,
        max_samples_per_task_lang=max_samples_per_task_lang,
        max_seq_len_teacher=max_seq_len_teacher,
        max_seq_len_student=max_seq_len_student,
        seed=seed,
        teacher_include_answer=teacher_include_answer,
        zeroshot_teacher=zeroshot_teacher,
        split=split,
    )
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda b: collate_fn_xtreme(b, pad_id),
        pin_memory=True,
    )


# ============================================================================
# Evaluation dataset (for inference scripts)
# ============================================================================

class XTREMEEvalDataset:
    """
    Lightweight eval dataset for a single (task, language) pair.

    Returns formatted prompts ready for model.generate().
    For fewshot condition, draws k examples from a separate pool
    (val set of the same language).
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        task: str,
        lang: str,
        condition: str = "base",     # "base" or "fewshot"
        n_samples: int = 500,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.task = task
        self.lang = lang
        self.condition = condition
        self.seed = seed

        # Test examples
        self.examples = load_task_data(task, "test", lang, n_samples)

        # Few-shot pool from same-language val/dev set
        self.fewshot_pool = []
        if condition == "fewshot":
            self.fewshot_pool = load_task_data(task, "dev", lang, 200)
            if not self.fewshot_pool:
                # Fallback: use a slice of the test set (excluding examples being tested)
                # This is acceptable for a benchmark like XTREME
                all_test = load_task_data(task, "test", lang, n_samples + 50)
                self.fewshot_pool = all_test[n_samples:]

    def __len__(self) -> int:
        return len(self.examples)

    def get_prompt(self, idx: int) -> str:
        example = self.examples[idx]
        n_fewshot = NUM_FEWSHOT[self.task]

        if self.condition == "fewshot" and self.fewshot_pool:
            rng = random.Random(self.seed + idx)
            fewshot = rng.sample(
                self.fewshot_pool, min(n_fewshot, len(self.fewshot_pool))
            )
            messages = build_teacher_messages(self.task, fewshot, example)
        else:
            messages = build_student_messages(self.task, example)

        return apply_xtreme_template(self.tokenizer, messages, add_generation_prompt=True)

    def get_gold(self, idx: int):
        """Return the gold answer for evaluation."""
        example = self.examples[idx]
        if self.task == "qa":
            return example["answers"]   # list of acceptable answers
        return get_answer_text(self.task, example)


# ============================================================================
# Output parsing
# ============================================================================

def parse_output(task: str, text: str, n_tokens: int = 0) -> str:
    """Parse model-generated text into a canonical answer for evaluation."""
    text = text.strip()
    if task == "nli":
        lower = text.lower()
        for label in ["entailment", "neutral", "contradiction"]:
            if lower.startswith(label[:3]):
                return label
        for label in ["entailment", "neutral", "contradiction"]:
            if label in lower:
                return label
        return "neutral"
    elif task == "pa":
        lower = text.lower()
        if lower.startswith("yes") or lower.startswith("1"):
            return "yes"
        return "no"
    elif task == "qa":
        # Return raw text (caller computes F1/EM against gold answers)
        return text.split("\n")[0].strip()
    elif task in ("ner", "pos"):
        valid_ner  = set(NER_ID2LABEL.values())
        valid_pos  = set(UPOS_NAMES)
        valid = valid_ner if task == "ner" else valid_pos
        words = text.strip().split()
        tags = [w.upper() if w.upper() in valid else ("O" if task == "ner" else "X")
                for w in words]
        # Pad or truncate to expected length
        if n_tokens > 0:
            if len(tags) < n_tokens:
                pad = "O" if task == "ner" else "X"
                tags.extend([pad] * (n_tokens - len(tags)))
            tags = tags[:n_tokens]
        return tags   # returns list[str]
    return text


# ============================================================================
# Evaluation metrics
# ============================================================================

def _normalize_answer(s: str) -> str:
    """Lower-case, strip punctuation and extra whitespace (SQuAD-style)."""
    s = s.lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())


def compute_accuracy(preds: list[str], golds: list[str]) -> float:
    if not preds:
        return 0.0
    return sum(p == g for p, g in zip(preds, golds)) / len(preds)


def compute_qa_f1_em(preds: list[str], golds: list[list[str]]) -> tuple[float, float]:
    """
    Compute F1 and Exact Match for QA (SQuAD-style).
    golds is a list of lists (multiple acceptable answers per question).
    """
    if not preds:
        return 0.0, 0.0
    f1_sum, em_sum = 0.0, 0.0
    for pred, gold_list in zip(preds, golds):
        pred_norm = _normalize_answer(pred)
        best_f1, best_em = 0.0, 0.0
        for gold in gold_list:
            gold_norm = _normalize_answer(gold)
            # EM
            em = float(pred_norm == gold_norm)
            # Token F1
            pred_toks  = pred_norm.split()
            gold_toks  = gold_norm.split()
            common     = set(pred_toks) & set(gold_toks)
            n_common   = sum(min(pred_toks.count(t), gold_toks.count(t)) for t in common)
            prec = n_common / len(pred_toks) if pred_toks else 0.0
            rec  = n_common / len(gold_toks) if gold_toks else 0.0
            f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            best_f1 = max(best_f1, f1)
            best_em = max(best_em, em)
        f1_sum += best_f1
        em_sum += best_em
    n = len(preds)
    return f1_sum / n, em_sum / n


def compute_ner_f1(preds: list[list[str]], golds: list[list[str]]) -> float:
    """Span-level NER F1 using seqeval if available, else token-level F1."""
    try:
        from seqeval.metrics import f1_score
        return f1_score(golds, preds, average="micro", zero_division=0)
    except ImportError:
        # Fallback: token-level F1 (ignoring O)
        tp = fp = fn = 0
        for pred_seq, gold_seq in zip(preds, golds):
            for p, g in zip(pred_seq, gold_seq):
                if g != "O":
                    if p == g:
                        tp += 1
                    else:
                        fn += 1
                        if p != "O":
                            fp += 1
                elif p != "O":
                    fp += 1
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0


def compute_pos_accuracy(preds: list[list[str]], golds: list[list[str]]) -> float:
    """Token-level POS accuracy."""
    correct = total = 0
    for pred_seq, gold_seq in zip(preds, golds):
        for p, g in zip(pred_seq, gold_seq):
            if p == g:
                correct += 1
            total += 1
    return correct / total if total > 0 else 0.0


def compute_task_metric(
    task: str,
    preds,
    golds,
) -> dict:
    """Unified metric computation for all tasks. Returns dict of metric_name → float."""
    if task == "nli":
        return {"accuracy": compute_accuracy(preds, golds)}
    elif task == "pa":
        return {"accuracy": compute_accuracy(preds, golds)}
    elif task == "qa":
        f1, em = compute_qa_f1_em(preds, golds)
        return {"f1": f1, "em": em}
    elif task == "ner":
        return {"f1": compute_ner_f1(preds, golds)}
    elif task == "pos":
        return {"accuracy": compute_pos_accuracy(preds, golds)}
    raise ValueError(f"Unknown task: {task}")
