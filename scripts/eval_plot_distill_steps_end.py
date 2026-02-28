"""
Evaluate all 5 distillation-at-end checkpoints on a dataset (zero-shot)
and produce one accuracy-vs-distill-steps plot.

Reads LoRA adapters from:
  experiments/distill_steps_end_{dataset}/steps_{N}/final/
  for N in [0, 4, 16, 64, 200]

Writes:
  experiments/distill_steps_end_{dataset}/eval_results.json
  experiments/distill_steps_end_{dataset}/distill_steps_end_accuracy.png

Usage:
  CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/eval_plot_distill_steps_end.py --dataset commonsenseqa
  CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/eval_plot_distill_steps_end.py --dataset math
  CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/eval_plot_distill_steps_end.py           # gsm8k default
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

sys.path.insert(0, str(Path(__file__).parent.parent))

MODEL_NAME     = "Qwen/Qwen3-1.7B"
DISTILL_STEPS  = [0, 4, 16, 64, 200]
SEED           = 42
MAX_NEW_TOKENS = 512
TP_SIZE        = 4

# Per-dataset configuration
DATASET_CFG = {
    "gsm8k": {
        "base_dir":      Path("experiments/distill_steps_end"),
        "eval_split":    "test",
        "n_samples":     1319,
        "max_model_len": 2048,
        "ylabel":        "GSM8K zero-shot accuracy (%)",
        "title_dataset": "GSM8K",
    },
    "commonsenseqa": {
        "base_dir":      Path("experiments/distill_steps_end_csqa"),
        "eval_split":    "validation",
        "n_samples":     9999,   # use full validation set (~1221 examples)
        "max_model_len": 1024,
        "ylabel":        "CommonsenseQA zero-shot accuracy (%)",
        "title_dataset": "CommonsenseQA",
    },
    "math": {
        "base_dir":      Path("experiments/distill_steps_end_math"),
        "eval_split":    "test",
        "n_samples":     1319,
        "max_model_len": 4096,
        "ylabel":        "MATH zero-shot accuracy (%)",
        "title_dataset": "MATH",
    },
}


# ── Dataset loading ────────────────────────────────────────────────────────────

def load_test_data(dataset: str, split: str, n_samples: int, seed: int):
    if dataset == "gsm8k":
        from datasets import load_dataset
        data = list(load_dataset("gsm8k", "main")[split])
    elif dataset == "commonsenseqa":
        from src.data.commonsenseqa_loader import load_commonsenseqa
        data = list(load_commonsenseqa(split))
    elif dataset == "math":
        from src.data.math_loader import load_math
        data = list(load_math(split))
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if n_samples < len(data):
        data = random.Random(seed).sample(data, n_samples)
    return data


# ── Per-dataset answer extraction and ground truth ────────────────────────────

def get_extract_gt_fns(dataset: str):
    if dataset == "gsm8k":
        def extract(text):
            m = re.search(r"####\s*([\d,]+)", text)
            if m:
                return m.group(1).replace(",", "").strip()
            nums = re.findall(r"\b\d+\b", text)
            return nums[-1] if nums else None

        def gt(ex):
            m = re.search(r"####\s*([\d,]+)", ex["answer"])
            return m.group(1).replace(",", "").strip() if m else ""

    elif dataset == "commonsenseqa":
        from src.data.commonsenseqa_loader import extract_answer, get_ground_truth
        extract = extract_answer
        gt      = get_ground_truth

    elif dataset == "math":
        from src.data.math_loader import extract_answer, get_ground_truth
        extract = extract_answer
        gt      = get_ground_truth

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return extract, gt


# ── Zero-shot prompt builders ─────────────────────────────────────────────────

def build_prompt(tokenizer, example: dict, dataset: str) -> str:
    if dataset == "gsm8k":
        messages = [{"role": "user", "content": f"Question: {example['question']}"}]
    elif dataset == "commonsenseqa":
        from src.data.commonsenseqa_loader import format_choices
        choices_str = format_choices(example["choices"])
        messages = [{"role": "user",
                     "content": f"Question: {example['question']}\n{choices_str}"}]
    elif dataset == "math":
        messages = [{"role": "user", "content": f"Problem: {example['problem']}"}]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": False},
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="gsm8k",
                   choices=["gsm8k", "commonsenseqa", "math"])
    return p.parse_args()


def main():
    args  = parse_args()
    dcfg  = DATASET_CFG[args.dataset]
    BASE_DIR = dcfg["base_dir"]

    print(f"\n{'='*60}")
    print(f" Distillation steps ablation (END) — {dcfg['title_dataset']} zero-shot eval")
    print(f"   Model      : {MODEL_NAME}")
    print(f"   Conditions : {DISTILL_STEPS}")
    print(f"{'='*60}\n")

    # ── Load test data ─────────────────────────────────────────────────────────
    print(f"Loading {args.dataset} {dcfg['eval_split']} set …")
    test_data = load_test_data(args.dataset, dcfg["eval_split"], dcfg["n_samples"], SEED)
    print(f"  {len(test_data)} examples\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts       = [build_prompt(tokenizer, ex, args.dataset) for ex in test_data]
    extract_fn, gt_fn = get_extract_gt_fns(args.dataset)
    ground_truths = [gt_fn(ex) for ex in test_data]

    # ── Discover available adapters ────────────────────────────────────────────
    available = []
    for n in DISTILL_STEPS:
        adapter_path = BASE_DIR / f"steps_{n}" / "final"
        if adapter_path.exists() and (adapter_path / "adapter_config.json").exists():
            available.append(n)
        else:
            print(f"[skip] {adapter_path} not found or missing adapter_config.json")

    if not available:
        print(f"No adapters found under {BASE_DIR}. Run training first.")
        return

    print(f"Evaluating conditions: {available}\n")

    # ── vLLM — load base model once, swap LoRA adapters ───────────────────────
    print("Initializing vLLM …")
    llm = LLM(
        model=MODEL_NAME,
        enable_lora=True,
        max_lora_rank=16,
        tensor_parallel_size=TP_SIZE,
        max_model_len=dcfg["max_model_len"],
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
    )
    sampling_params = SamplingParams(temperature=0, max_tokens=MAX_NEW_TOKENS)

    results = {}
    for i, n_distill in enumerate(available):
        adapter_path = str(BASE_DIR / f"steps_{n_distill}" / "final")
        lora_req = LoRARequest(f"{args.dataset}_end_{n_distill}", i + 1, adapter_path)

        distill_start = 200 - n_distill
        label = "SFT only" if n_distill == 0 else f"distill steps {distill_start}–199"
        print(f"\n--- distill_steps={n_distill}  ({label}) ---")

        outputs = llm.generate(prompts, sampling_params, lora_request=lora_req)

        correct = 0
        for output, gt in zip(outputs, ground_truths):
            pred = extract_fn(output.outputs[0].text)
            correct += int(pred == gt if pred is not None else False)

        acc = correct / len(test_data)
        results[n_distill] = {
            "accuracy": acc,
            "correct": correct,
            "total": len(test_data),
            "distill_start_step": distill_start,
        }
        print(f"  Accuracy: {acc:.2%}  ({correct}/{len(test_data)})")

    # ── Save JSON ──────────────────────────────────────────────────────────────
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    out_json = BASE_DIR / "eval_results.json"
    with open(out_json, "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    print(f"\n✓ Results saved to {out_json}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    x_vals = sorted(results.keys())
    y_vals = [results[n]["accuracy"] * 100 for n in x_vals]

    x_pos    = list(range(len(x_vals)))
    x_labels = [str(n) for n in x_vals]
    if x_labels:
        x_labels[0] = "0\n(SFT only)"
    if len(x_labels) > 1:
        x_labels[-1] = f"{x_vals[-1]}\n(full distill)"

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_pos, y_vals, "o-", color="#F44336", linewidth=2.5, markersize=9,
            markerfacecolor="white", markeredgewidth=2.5, markeredgecolor="#F44336")

    for x, y in zip(x_pos, y_vals):
        ax.annotate(f"{y:.1f}%", (x, y),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=9, color="#333333")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_xlabel("Number of distillation steps applied at end of training", fontsize=11)
    ax.set_ylabel(dcfg["ylabel"], fontsize=11)
    ax.set_title(
        f"Does distillation timing matter?  (applied at END)\n"
        f"({dcfg['title_dataset']}, Qwen3-1.7B, LoRA, 200 total steps)",
        fontsize=13, fontweight="bold", pad=10,
    )
    ax.grid(alpha=0.3, axis="y")

    y_margin = max(3.0, (max(y_vals) - min(y_vals)) * 0.2)
    ax.set_ylim(
        bottom=max(0, min(y_vals) - y_margin),
        top=min(100, max(y_vals) + y_margin + 3),
    )

    plt.tight_layout()
    out_png = BASE_DIR / "distill_steps_end_accuracy.png"
    fig.savefig(str(out_png), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Plot saved to {out_png}")


if __name__ == "__main__":
    main()
