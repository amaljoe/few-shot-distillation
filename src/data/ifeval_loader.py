"""
IFEval dataset loader and prompt formatter.

Training / few-shot pool: argilla/ifeval-like-data (filtered config, ~56k examples).
  Fields: prompt, response, instruction_id_list, kwargs, prompt_level_strict_acc, ...
  Examples are pre-verified compliant; we additionally filter on prompt_level_strict_acc.

Evaluation set: google/IFEval (541 examples, only 'train' split exists).
  Fields: prompt, instruction_id_list, kwargs.

Evaluation metric: prompt-level strict accuracy (all constraints pass).
Requires: vendored instruction_following_eval/ at project root (see results_extension_IFEVAL.md).
  Dependencies: langdetect, nltk, immutabledict, absl-py (pip install these).
"""

import random

import torch
from datasets import load_dataset as hf_load
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


def load_ifeval(split: str) -> list[dict]:
    """
    Load IFEval examples.

    split == "train" → argilla/ifeval-like-data (filtered config), training subset.
      Only examples with prompt_level_strict_acc == True are kept.
    split == "test" / anything else → google/IFEval (541 examples, train split only).
    """
    if split == "train":
        ds = hf_load("argilla/ifeval-like-data", "filtered", split="train")
        ds = ds.filter(lambda x: x["prompt_level_strict_acc"] is True)
        return list(ds)
    else:
        return list(hf_load("google/IFEval", split="train"))


def build_fewshot_messages(fewshot_examples: list[dict], query_prompt: str) -> list[dict]:
    """Build chat messages for teacher (few-shot) input.

    fewshot_examples come from argilla (have 'prompt' + 'response').
    query_prompt is the raw instruction string from either dataset.
    """
    msgs = []
    for ex in fewshot_examples:
        msgs.append({"role": "user", "content": ex["prompt"]})
        msgs.append({"role": "assistant", "content": ex["response"]})
    msgs.append({"role": "user", "content": query_prompt})
    return msgs


def build_student_messages(query_prompt: str) -> list[dict]:
    """Build zero-shot chat messages for student input."""
    return [{"role": "user", "content": query_prompt}]


def check_instruction_following(example: dict, response: str) -> tuple[bool, list[bool]]:
    """
    Check whether a model response satisfies all constraints in an IFEval example.

    Works for both google/IFEval and argilla filtered examples — both have
    instruction_id_list and kwargs fields.

    Uses the vendored instruction_following_eval module (project root).
    Correct calling convention (from evaluation_lib.py):
      cls(instruction_id) → build_description(**kwargs) → check_following(response)
    """
    from instruction_following_eval import instructions_registry  # vendored at project root
    results = []
    prompt = example.get("prompt", "")
    for inst_id, kw in zip(example["instruction_id_list"], example["kwargs"]):
        instruction = instructions_registry.INSTRUCTION_DICT[inst_id](inst_id)
        # HuggingFace stores all possible kwargs with None for unused ones; strip them.
        kw_filtered = {k: v for k, v in kw.items() if v is not None}
        instruction.build_description(**kw_filtered)
        # Some instructions (e.g. prompt-referencing) also need the original prompt
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=prompt)
        results.append(bool(response.strip() and instruction.check_following(response)))
    return all(results), results


class IFEvalDistillDataset(Dataset):
    """
    Dataset that returns teacher + student inputs for each IFEval training example.
    Mirrors CSQADistillDataset in structure; returns dicts compatible with collate_fn.

    All examples come from argilla/ifeval-like-data (have 'prompt' + 'response').
    The 'response' field is used directly as the gold label — no answer extraction needed.

    teacher_include_answer (bool): when True the teacher sequence includes the
        gold response, so it ends with the same token IDs as the student sequence.
        Required for online distillation (V1) where alignment is done on answer positions.
    """

    def __init__(
        self,
        dataset,
        tokenizer: PreTrainedTokenizer,
        num_fewshot: int = 4,
        max_seq_len_teacher: int = 2048,
        max_seq_len_student: int = 512,
        seed: int = 42,
        teacher_include_answer: bool = False,
        shuffle_fewshot_answers: bool = False,  # accepted, ignored (no answer choices)
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.num_fewshot = num_fewshot
        self.max_seq_len_teacher = max_seq_len_teacher
        self.max_seq_len_student = max_seq_len_student
        self.seed = seed
        self.teacher_include_answer = teacher_include_answer

        # Pre-compute generation prompt length for token alignment
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

        # Deterministic few-shot sampling per example (exclude idx itself)
        rng = random.Random(self.seed + idx)
        candidates = [i for i in range(len(self.dataset)) if i != idx]
        fewshot_indices = rng.sample(candidates, min(self.num_fewshot, len(candidates)))
        fewshot_examples = [self.dataset[i] for i in fewshot_indices]

        gold_response = example["response"]

        # --- Teacher input ---
        teacher_messages = build_fewshot_messages(fewshot_examples, example["prompt"])

        if self.teacher_include_answer:
            teacher_messages_with_ans = teacher_messages + [
                {"role": "assistant", "content": gold_response}
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

        # --- Student input (zero-shot prompt only, no answer) ---
        student_messages = build_student_messages(example["prompt"])
        student_prompt = apply_chat_template_no_think(self.tokenizer, student_messages)
        student_enc = self.tokenizer(
            student_prompt,
            truncation=True,
            max_length=self.max_seq_len_student,
            return_tensors=None,
            add_special_tokens=False,
        )
        student_ids = student_enc["input_ids"]

        # --- Labels: prompt tokens masked with -100, supervise on response ---
        student_with_answer_msgs = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": gold_response},
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
    max_seq_len_teacher: int = 2048,
    max_seq_len_student: int = 512,
    shuffle: bool = True,
    num_workers: int = 4,
    seed: int = 42,
    teacher_include_answer: bool = False,
    shuffle_fewshot_answers: bool = False,  # accepted, not used
) -> torch.utils.data.DataLoader:
    ds = IFEvalDistillDataset(
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
