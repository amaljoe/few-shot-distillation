"""
Central dataset dispatcher for ICL distillation experiments.

Supports: gsm8k, commonsenseqa, math, ifeval
"""

from src.data.gsm8k_loader import (
    load_gsm8k,
    make_dataloader as _gsm8k_dl,
)
from src.data.commonsenseqa_loader import (
    load_commonsenseqa,
    make_dataloader as _csqa_dl,
)
from src.data.math_loader import (
    load_math,
    make_dataloader as _math_dl,
)
from src.data.ifeval_loader import (
    load_ifeval,
    make_dataloader as _ifeval_dl,
)


def load_dataset_split(name: str, split: str):
    """Load a dataset split by name."""
    if name == "gsm8k":
        return load_gsm8k(split)
    elif name == "commonsenseqa":
        return load_commonsenseqa(split)
    elif name == "math":
        return load_math(split)
    elif name == "ifeval":
        return load_ifeval(split)
    else:
        raise ValueError(f"Unknown dataset: {name!r}. Choose from: gsm8k, commonsenseqa, math, ifeval")


def make_dataloader(dataset, tokenizer, batch_size, dataset_name="gsm8k", **kwargs):
    """
    Create a DataLoader for the given dataset.

    kwargs are forwarded to the dataset-specific make_dataloader:
      num_fewshot, max_seq_len_teacher, max_seq_len_student, shuffle,
      num_workers, seed, teacher_include_answer, shuffle_fewshot_answers (gsm8k only)
    """
    if dataset_name == "gsm8k":
        return _gsm8k_dl(dataset, tokenizer, batch_size, **kwargs)
    elif dataset_name == "commonsenseqa":
        return _csqa_dl(dataset, tokenizer, batch_size, **kwargs)
    elif dataset_name == "math":
        return _math_dl(dataset, tokenizer, batch_size, **kwargs)
    elif dataset_name == "ifeval":
        return _ifeval_dl(dataset, tokenizer, batch_size, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name!r}. Choose from: gsm8k, commonsenseqa, math, ifeval")
