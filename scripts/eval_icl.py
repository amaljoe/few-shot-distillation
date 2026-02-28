"""
Phase 0: ICL Evaluation Script

Tests few-shot vs zero-shot accuracy gap across Qwen3 models to select
the smallest one with a meaningful ICL signal.

Usage (run 3 in parallel on cn14-dgx, one per GPU):
  CUDA_VISIBLE_DEVICES=0 python scripts/eval_icl.py \
      --model Qwen/Qwen3-2B-Instruct \
      --num_samples 16 --num_fewshot 0 4 8 \
      --output experiments/poc/icl_eval_2b.json

  CUDA_VISIBLE_DEVICES=1 python scripts/eval_icl.py \
      --model Qwen/Qwen3-4B-Instruct ...

  CUDA_VISIBLE_DEVICES=2 python scripts/eval_icl.py \
      --model Qwen/Qwen3-8B-Instruct ...
"""

import argparse
import json
import re
import random
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--dataset", type=str, default="gsm8k",
                        choices=["gsm8k", "commonsenseqa", "math"],
                        help="Dataset to evaluate on")
    parser.add_argument("--num_samples", type=int, default=16,
                        help="Number of test examples to evaluate (quick sanity check)")
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Alias for --num_samples")
    parser.add_argument("--num_fewshot", type=int, nargs="+", default=[0, 4, 8],
                        help="List of few-shot counts to evaluate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save JSON results")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_model_len", type=int, default=4096)
    return parser.parse_args()


def extract_answer(text: str) -> str | None:
    """Extract the numeric answer from GSM8K-style response (#### <number>)."""
    match = re.search(r"####\s*([\d,]+)", text)
    if match:
        return match.group(1).replace(",", "").strip()
    # Fallback: look for the last number in the text
    numbers = re.findall(r"\b\d+\b", text)
    return numbers[-1] if numbers else None


def get_ground_truth(example: dict) -> str:
    """Extract ground truth number from GSM8K answer field."""
    match = re.search(r"####\s*([\d,]+)", example["answer"])
    if match:
        return match.group(1).replace(",", "").strip()
    return ""


def build_fewshot_messages(fewshot_examples: list[dict], query: dict, num_fewshot: int) -> list[dict]:
    """
    Build chat messages for Qwen3 instruct format.

    Few-shot format:
      user: Question: ...
      assistant: [solution] #### answer
      ...
      user: Question: ... (target)

    Zero-shot: just the target user message.
    """
    messages = []

    for ex in fewshot_examples[:num_fewshot]:
        messages.append({
            "role": "user",
            "content": f"Question: {ex['question']}"
        })
        messages.append({
            "role": "assistant",
            "content": ex["answer"]
        })

    # Final target question
    messages.append({
        "role": "user",
        "content": f"Question: {query['question']}"
    })

    return messages


def main():
    args = parse_args()
    # --n_samples is an alias for --num_samples
    if args.n_samples is not None:
        args.num_samples = args.n_samples
    random.seed(args.seed)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Num samples: {args.num_samples}")
    print(f"Few-shot configs: {args.num_fewshot}")
    print(f"{'='*60}\n")

    # Dataset-specific loading and helpers
    if args.dataset == "gsm8k":
        print("Loading GSM8K dataset...")
        dataset = load_dataset("gsm8k", "main")
        train_data = list(dataset["train"])
        test_data = list(dataset["test"])
        _build_fewshot = lambda fewshot_examples, example, nshot: build_fewshot_messages(
            fewshot_examples, example, nshot
        )
        _get_gt = get_ground_truth
        _extract = extract_answer
    elif args.dataset == "commonsenseqa":
        from src.data.commonsenseqa_loader import (
            load_commonsenseqa,
            build_fewshot_messages as csqa_build_fewshot,
            get_ground_truth as csqa_gt,
            extract_answer as csqa_extract,
        )
        print("Loading CommonsenseQA dataset...")
        train_data = list(load_commonsenseqa("train"))
        test_data = list(load_commonsenseqa("validation"))
        _build_fewshot = lambda fewshot_examples, example, nshot: csqa_build_fewshot(
            fewshot_examples, example
        )
        _get_gt = csqa_gt
        _extract = csqa_extract
    elif args.dataset == "math":
        from src.data.math_loader import (
            load_math,
            build_fewshot_messages as math_build_fewshot,
            get_ground_truth as math_gt,
            extract_answer as math_extract,
        )
        print("Loading MATH dataset...")
        train_data = list(load_math("train"))
        test_data = list(load_math("test"))
        _build_fewshot = lambda fewshot_examples, example, nshot: math_build_fewshot(
            fewshot_examples, example
        )
        _get_gt = math_gt
        _extract = math_extract

    # Sample test examples (fixed seed for reproducibility)
    random.seed(args.seed)
    test_sample = random.sample(test_data, min(args.num_samples, len(test_data)))

    # Fewshot pool = training data (exclude nothing for this quick eval)
    fewshot_pool = train_data

    # Load tokenizer for apply_chat_template
    print(f"Loading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Initialize vLLM (offline mode, single GPU)
    print(f"Initializing vLLM with {args.model}...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        dtype="bfloat16",
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=args.max_new_tokens,
    )

    results = {
        "model": args.model,
        "dataset": args.dataset,
        "num_samples": args.num_samples,
        "seed": args.seed,
        "evaluations": {},
    }

    for nshot in args.num_fewshot:
        print(f"\n--- Evaluating {nshot}-shot ---")

        # Build prompts
        prompts = []
        ground_truths = []

        for example in test_sample:
            # Sample fewshot examples from train (exclude by question text to avoid leakage)
            q_key = "question" if "question" in example else "problem"
            candidates = [ex for ex in fewshot_pool
                          if ex.get("question", ex.get("problem")) != example.get(q_key)]
            fewshot_examples = random.sample(candidates, min(nshot, len(candidates)))

            messages = _build_fewshot(fewshot_examples, example, nshot)

            # Apply chat template; fall back to plain text for base models
            if getattr(tokenizer, "chat_template", None) is None:
                parts = []
                for i, msg in enumerate(messages):
                    parts.append(msg["content"])
                    if (msg["role"] == "assistant"
                            and i + 1 < len(messages)
                            and messages[i + 1]["role"] == "user"):
                        parts.append("")
                prompt = "\n".join(parts) + "\n"
            else:
                try:
                    prompt = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        chat_template_kwargs={"enable_thinking": False},
                    )
                except TypeError:
                    # Fallback if chat_template_kwargs not supported
                    prompt = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )

            prompts.append(prompt)
            ground_truths.append(_get_gt(example))

        # Generate
        outputs = llm.generate(prompts, sampling_params)

        # Evaluate
        correct = 0
        details = []
        for output, gt, example in zip(outputs, ground_truths, test_sample):
            generated = output.outputs[0].text
            pred = _extract(generated)
            is_correct = (pred == gt) if pred is not None else False
            correct += int(is_correct)
            q_text = example.get("question", example.get("problem", ""))
            details.append({
                "question": q_text[:80] + "...",
                "ground_truth": gt,
                "prediction": pred,
                "correct": is_correct,
            })

        accuracy = correct / len(test_sample)
        results["evaluations"][f"{nshot}_shot"] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": len(test_sample),
            "details": details,
        }

        print(f"  {nshot}-shot accuracy: {accuracy:.2%} ({correct}/{len(test_sample)})")

    # Compute ICL gap (8-shot vs 0-shot, or max vs min)
    shot_accs = {k: v["accuracy"] for k, v in results["evaluations"].items()}
    if "0_shot" in shot_accs:
        best_fewshot = max(v for k, v in shot_accs.items() if k != "0_shot")
        gap = best_fewshot - shot_accs["0_shot"]
        results["icl_gap"] = gap
        print(f"\n{'='*60}")
        print(f"ICL Gap (best few-shot - 0-shot): {gap:.2%}")
        print(f"  Recommendation: {'USE THIS MODEL ✓' if gap >= 0.05 else 'Try larger model'}")
        print(f"{'='*60}")

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
