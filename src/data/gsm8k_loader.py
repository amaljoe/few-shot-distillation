"""
GSM8K dataset loader and prompt formatter.

Teacher prompt: few-shot examples + target question (chat format)
Student prompt: target question only (chat format)

Key output: teacher_query_pos — the index of the last token of the target
question in the teacher's tokenized sequence, just before the model generates
the answer. This is the alignment position for distillation.
"""

import random
import re
from dataclasses import dataclass
from typing import Any

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


def load_gsm8k(split: str = "train"):
    """Load GSM8K from HuggingFace datasets."""
    return load_dataset("gsm8k", "main", split=split)


def get_ground_truth(answer_text: str) -> str:
    """Extract numeric answer from GSM8K answer field (#### <number>)."""
    match = re.search(r"####\s*([\d,]+)", answer_text)
    if match:
        return match.group(1).replace(",", "").strip()
    return ""


def build_fewshot_messages(fewshot_examples: list[dict], query: dict) -> list[dict]:
    """
    Build chat messages for teacher (few-shot) input.

    Format:
      user: Question: ...
      assistant: [CoT solution] #### answer
      ...repeated for each few-shot example...
      user: Question: <target>   ← alignment point (last token here)
    """
    messages = []
    for ex in fewshot_examples:
        messages.append({"role": "user", "content": f"Question: {ex['question']}"})
        messages.append({"role": "assistant", "content": ex["answer"]})
    messages.append({"role": "user", "content": f"Question: {query['question']}"})
    return messages


def build_student_messages(query: dict) -> list[dict]:
    """
    Build chat messages for student (zero-shot) input.

    Format:
      user: Question: <target>   ← last token is alignment point
    """
    return [{"role": "user", "content": f"Question: {query['question']}"}]


def apply_chat_template_no_think(tokenizer: PreTrainedTokenizer, messages: list[dict]) -> str:
    """Apply Qwen3 chat template with thinking mode disabled."""
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": False},
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def find_last_user_token_pos(
    input_ids: list[int],
    tokenizer: PreTrainedTokenizer,
    generation_prompt_len: int,
) -> int:
    """
    Find the position of the last token of the user's target question,
    which is just before the assistant generation prompt tokens.

    The teacher tokenized sequence ends with:
      ... [last token of target question] [generation_prompt tokens]

    So the alignment position = len(input_ids) - generation_prompt_len - 1
    """
    pos = len(input_ids) - generation_prompt_len - 1
    assert pos >= 0, "generation_prompt_len is too large relative to input_ids"
    return pos


@dataclass
class GSM8KBatch:
    teacher_input_ids: torch.Tensor       # (B, T_teach)
    teacher_attention_mask: torch.Tensor  # (B, T_teach)
    teacher_query_pos: torch.Tensor       # (B,) — alignment token index per example
    student_input_ids: torch.Tensor       # (B, T_stud)
    student_attention_mask: torch.Tensor  # (B, T_stud)
    student_query_pos: torch.Tensor       # (B,) — alignment token index in student sequence
    labels: torch.Tensor                  # (B, T_stud) — -100 on prompt, answer tokens otherwise
    example_idx: torch.Tensor             # (B,) — original dataset index for cache lookup


class GSM8KDistillDataset(Dataset):
    """
    Dataset that returns teacher + student inputs for each GSM8K example.

    For each item, the few-shot examples are sampled deterministically from
    the training pool using a fixed seed + example index (reproducible).

    teacher_include_answer (bool): when True the teacher sequence includes the
        ground-truth answer, so it ends with the same token IDs as the student
        sequence.  Required for online distillation (V1 / V2) where alignment
        is done on answer-token positions.  Old precompute-based scripts leave
        this False (default) and the teacher ends with the generation prompt.
    """

    def __init__(
        self,
        dataset,
        tokenizer: PreTrainedTokenizer,
        num_fewshot: int = 8,
        max_seq_len_teacher: int = 1536,
        max_seq_len_student: int = 512,
        seed: int = 42,
        teacher_include_answer: bool = False,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.num_fewshot = num_fewshot
        self.max_seq_len_teacher = max_seq_len_teacher
        self.max_seq_len_student = max_seq_len_student
        self.seed = seed
        self.teacher_include_answer = teacher_include_answer

        # Pre-compute generation prompt length for alignment
        # The generation prompt is the part added by add_generation_prompt=True
        # We compute it as: template_with_gen - template_without_gen
        dummy_msgs = [{"role": "user", "content": "Q"}]
        prompt_with = apply_chat_template_no_think(tokenizer, dummy_msgs)
        # Without generation prompt (add_generation_prompt=False)
        try:
            prompt_without = tokenizer.apply_chat_template(
                dummy_msgs,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            prompt_without = prompt_with
        gen_prompt_str = prompt_with[len(prompt_without):]
        self.gen_prompt_len = len(tokenizer.encode(gen_prompt_str, add_special_tokens=False))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        example = self.dataset[idx]

        # Deterministic few-shot sampling per example
        rng = random.Random(self.seed + idx)
        candidates = [i for i in range(len(self.dataset)) if i != idx]
        fewshot_indices = rng.sample(candidates, min(self.num_fewshot, len(candidates)))
        fewshot_examples = [self.dataset[i] for i in fewshot_indices]

        # --- Teacher input ---
        teacher_messages = build_fewshot_messages(fewshot_examples, example)

        if self.teacher_include_answer:
            # Online distillation mode: include ground-truth answer in teacher sequence.
            # Teacher ends with the same token IDs as student so that the last
            # n_ans tokens of teacher content align with student answer tokens.
            teacher_messages_with_ans = teacher_messages + [
                {"role": "assistant", "content": example["answer"]}
            ]
            try:
                teacher_prompt = self.tokenizer.apply_chat_template(
                    teacher_messages_with_ans,
                    tokenize=False,
                    add_generation_prompt=False,
                    chat_template_kwargs={"enable_thinking": False},
                )
            except TypeError:
                teacher_prompt = self.tokenizer.apply_chat_template(
                    teacher_messages_with_ans,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            teacher_query_pos = 0  # not used by online training scripts
        else:
            teacher_prompt = apply_chat_template_no_think(self.tokenizer, teacher_messages)
            teacher_enc_tmp = self.tokenizer(
                teacher_prompt,
                truncation=True,
                max_length=self.max_seq_len_teacher,
                return_tensors=None,
                add_special_tokens=False,
            )
            teacher_query_pos = find_last_user_token_pos(
                teacher_enc_tmp["input_ids"], self.tokenizer, self.gen_prompt_len
            )

        teacher_enc = self.tokenizer(
            teacher_prompt,
            truncation=True,
            max_length=self.max_seq_len_teacher,
            return_tensors=None,
            add_special_tokens=False,
        )
        teacher_ids = teacher_enc["input_ids"]
        teacher_mask = teacher_enc["attention_mask"]

        # --- Student input ---
        student_messages = build_student_messages(example)
        student_prompt = apply_chat_template_no_think(self.tokenizer, student_messages)
        student_enc = self.tokenizer(
            student_prompt,
            truncation=True,
            max_length=self.max_seq_len_student,
            return_tensors=None,
            add_special_tokens=False,
        )
        student_ids = student_enc["input_ids"]
        student_mask = student_enc["attention_mask"]

        # --- Labels: full sequence as target, mask prompt tokens with -100 ---
        # Student generates from the prompt; we want CE loss on the answer portion.
        # For SFT on GSM8K: we train to predict the full answer including CoT.
        # However at precompute time we only have the question → we need answer tokens too.
        # Build student input+answer sequence for label computation:
        student_with_answer_msgs = [
            {"role": "user", "content": f"Question: {example['question']}"},
            {"role": "assistant", "content": example["answer"]},
        ]
        try:
            full_seq = self.tokenizer.apply_chat_template(
                student_with_answer_msgs,
                tokenize=False,
                add_generation_prompt=False,
                chat_template_kwargs={"enable_thinking": False},
            )
        except TypeError:
            full_seq = self.tokenizer.apply_chat_template(
                student_with_answer_msgs,
                tokenize=False,
                add_generation_prompt=False,
            )
        full_enc = self.tokenizer(
            full_seq,
            truncation=True,
            max_length=self.max_seq_len_student,
            return_tensors=None,
            add_special_tokens=False,
        )
        full_ids = full_enc["input_ids"]
        # Mask prompt tokens (everything up to the student prompt length)
        prompt_len = len(student_ids)
        labels = [-100] * prompt_len + full_ids[prompt_len:]
        # Pad/truncate labels to full_ids length
        labels = labels[:len(full_ids)]

        # Student alignment position: last token of the question prompt (before answer).
        # Same structural position as teacher_query_pos — just before gen_prompt tokens.
        # prompt_len - gen_prompt_len - 1 = last token of student question
        student_query_pos = find_last_user_token_pos(
            student_ids, self.tokenizer, self.gen_prompt_len
        )

        return {
            "teacher_input_ids": teacher_ids,
            "teacher_attention_mask": teacher_mask,
            "teacher_query_pos": teacher_query_pos,
            "student_input_ids": full_ids,       # full sequence for CE loss
            "student_attention_mask": full_enc["attention_mask"],
            "student_query_pos": student_query_pos,  # alignment position in student sequence
            "labels": labels,
            "example_idx": idx,
        }


def collate_fn(batch: list[dict], pad_token_id: int) -> GSM8KBatch:
    """Pad batch items to the same length within each field."""

    def pad_seq(seqs: list[list[int]], pad_val: int) -> torch.Tensor:
        max_len = max(len(s) for s in seqs)
        padded = [s + [pad_val] * (max_len - len(s)) for s in seqs]
        return torch.tensor(padded, dtype=torch.long)

    teacher_ids = pad_seq([b["teacher_input_ids"] for b in batch], pad_token_id)
    teacher_mask = pad_seq([b["teacher_attention_mask"] for b in batch], 0)
    student_ids = pad_seq([b["student_input_ids"] for b in batch], pad_token_id)
    student_mask = pad_seq([b["student_attention_mask"] for b in batch], 0)
    labels = pad_seq([b["labels"] for b in batch], -100)

    return GSM8KBatch(
        teacher_input_ids=teacher_ids,
        teacher_attention_mask=teacher_mask,
        teacher_query_pos=torch.tensor([b["teacher_query_pos"] for b in batch], dtype=torch.long),
        student_input_ids=student_ids,
        student_attention_mask=student_mask,
        student_query_pos=torch.tensor([b["student_query_pos"] for b in batch], dtype=torch.long),
        labels=labels,
        example_idx=torch.tensor([b["example_idx"] for b in batch], dtype=torch.long),
    )


def make_dataloader(
    dataset,
    tokenizer: PreTrainedTokenizer,
    batch_size: int,
    num_fewshot: int = 8,
    max_seq_len_teacher: int = 1536,
    max_seq_len_student: int = 512,
    shuffle: bool = True,
    num_workers: int = 4,
    seed: int = 42,
    teacher_include_answer: bool = False,
) -> torch.utils.data.DataLoader:
    ds = GSM8KDistillDataset(
        dataset=dataset,
        tokenizer=tokenizer,
        num_fewshot=num_fewshot,
        max_seq_len_teacher=max_seq_len_teacher,
        max_seq_len_student=max_seq_len_student,
        seed=seed,
        teacher_include_answer=teacher_include_answer,
    )
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id or tokenizer.eos_token_id),
        pin_memory=True,
    )
