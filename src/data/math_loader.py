"""
MATH dataset loader and prompt formatter.

Dataset: DigitalLearningGmbH/MATH-lighteval
Fields: problem (str), level (str), type (str), solution (str with \\boxed{answer})

No pre-extracted answer field — get_ground_truth() calls extract_answer() on solution.
Answer format: example["solution"] (full solution including \\boxed{} expression).
Token alignment holds: both teacher and student sequences end with example["solution"].
"""

import re
import random

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from src.data.gsm8k_loader import (
    GSM8KBatch,
    apply_chat_template_no_think,
    _has_chat_template,
    _format_as_plain_text,
    find_last_user_token_pos,
    collate_fn,
)


def load_math(split: str = "train"):
    """Load MATH dataset from HuggingFace datasets."""
    return load_dataset("DigitalLearningGmbH/MATH-lighteval", split=split)


def get_ground_truth(example: dict) -> str:
    """Extract the boxed answer from the solution field."""
    ans = extract_answer(example["solution"])
    return ans if ans is not None else ""


def extract_answer(text: str) -> str | None:
    """Extract content of last \\boxed{} in text, handling nested braces."""
    idx = text.rfind(r'\boxed{')
    if idx == -1:
        return None
    start = idx + len(r'\boxed{')
    depth = 0
    for i, ch in enumerate(text[start:]):
        if ch == '{':
            depth += 1
        elif ch == '}':
            if depth == 0:
                return text[start:start + i].strip()
            depth -= 1
    return None


def build_fewshot_messages(fewshot_examples: list[dict], query: dict) -> list[dict]:
    """Build chat messages for teacher (few-shot) input."""
    messages = []
    for ex in fewshot_examples:
        messages.append({"role": "user", "content": f"Problem: {ex['problem']}"})
        messages.append({"role": "assistant", "content": ex["solution"]})
    messages.append({"role": "user", "content": f"Problem: {query['problem']}"})
    return messages


def build_student_messages(query: dict) -> list[dict]:
    """Build chat messages for student (zero-shot) input."""
    return [{"role": "user", "content": f"Problem: {query['problem']}"}]


class MATHDistillDataset(Dataset):
    """
    Dataset that returns teacher + student inputs for each MATH example.
    Mirrors GSM8KDistillDataset in structure; returns dicts compatible with collate_fn.

    teacher_include_answer (bool): when True the teacher sequence includes the
        full solution, so it ends with the same token IDs as the student sequence.
        Required for online distillation (V1) where alignment is done on answer-token
        positions.
    """

    def __init__(
        self,
        dataset,
        tokenizer: PreTrainedTokenizer,
        num_fewshot: int = 4,
        max_seq_len_teacher: int = 6144,
        max_seq_len_student: int = 1024,
        seed: int = 42,
        teacher_include_answer: bool = False,
        shuffle_fewshot_answers: bool = False,  # accepted, ignored for MATH
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.num_fewshot = num_fewshot
        self.max_seq_len_teacher = max_seq_len_teacher
        self.max_seq_len_student = max_seq_len_student
        self.seed = seed
        self.teacher_include_answer = teacher_include_answer

        # Pre-compute generation prompt length for alignment
        dummy_msgs = [{"role": "user", "content": "Q"}]
        prompt_with = apply_chat_template_no_think(tokenizer, dummy_msgs)
        try:
            prompt_without = tokenizer.apply_chat_template(
                dummy_msgs, tokenize=False, add_generation_prompt=False,
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
            teacher_messages_with_ans = teacher_messages + [
                {"role": "assistant", "content": example["solution"]}
            ]
            if not _has_chat_template(self.tokenizer):
                teacher_prompt = _format_as_plain_text(
                    teacher_messages_with_ans, add_generation_prompt=False
                )
            else:
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

        # --- Labels: mask prompt tokens with -100, supervise on full solution ---
        student_with_answer_msgs = [
            {"role": "user", "content": f"Problem: {example['problem']}"},
            {"role": "assistant", "content": example["solution"]},
        ]
        if not _has_chat_template(self.tokenizer):
            full_seq = _format_as_plain_text(
                student_with_answer_msgs, add_generation_prompt=False
            )
        else:
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
        prompt_len = len(student_ids)
        labels = [-100] * prompt_len + full_ids[prompt_len:]
        labels = labels[:len(full_ids)]

        student_query_pos = find_last_user_token_pos(
            student_ids, self.tokenizer, self.gen_prompt_len
        )

        return {
            "teacher_input_ids": teacher_ids,
            "teacher_attention_mask": teacher_mask,
            "teacher_query_pos": teacher_query_pos,
            "student_input_ids": full_ids,
            "student_attention_mask": full_enc["attention_mask"],
            "student_query_pos": student_query_pos,
            "labels": labels,
            "example_idx": idx,
        }


def make_dataloader(
    dataset,
    tokenizer: PreTrainedTokenizer,
    batch_size: int,
    num_fewshot: int = 4,
    max_seq_len_teacher: int = 6144,
    max_seq_len_student: int = 1024,
    shuffle: bool = True,
    num_workers: int = 4,
    seed: int = 42,
    teacher_include_answer: bool = False,
    shuffle_fewshot_answers: bool = False,  # accepted, not used
) -> torch.utils.data.DataLoader:
    ds = MATHDistillDataset(
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
