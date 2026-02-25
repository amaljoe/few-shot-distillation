"""
Final evaluation script for all conditions.

Uses vLLM with LoRA adapter support for fast inference on the full GSM8K test set.

Condition A: Base model (few-shot) — evaluated during Phase 0 (eval_icl.py)
Condition B: Fine-tuned baseline — load from experiments/poc/baseline/checkpoint-N
Condition C: Distilled model    — load from experiments/poc/distill/checkpoint-N

Run commands (tmux: vllm + claude on cn14-dgx):

  # Step 1: Start vLLM server with LoRA support
  vllm serve Qwen/Qwen3-2B-Instruct \
      --enable-lora \
      --lora-modules \
          baseline=experiments/poc/baseline/final \
          distill=experiments/poc/distill/final \
      --port 8000 --tensor-parallel-size 2 \
      --max-lora-rank 16

  # Step 2: Run evaluation (tmux: claude)
  python scripts/evaluate.py \
      --config configs/base.yaml \
      --api_base http://localhost:8000/v1 \
      --lora_names baseline distill \
      --output experiments/poc/final_results.json

  # Or evaluate specific checkpoints:
  python scripts/evaluate.py \
      --config configs/base.yaml \
      --api_base http://localhost:8000/v1 \
      --lora_names baseline distill \
      --checkpoint_steps 100 200 300 400 500 600 700 800 900 1000 \
      --output experiments/poc/checkpoint_results.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

from datasets import load_dataset
from omegaconf import OmegaConf
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--api_base", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--lora_names", nargs="+", default=["baseline", "distill"],
                        help="Names of LoRA adapters loaded in vLLM server")
    parser.add_argument("--num_fewshot", type=int, default=0,
                        help="Few-shot examples (0 for zero-shot eval of fine-tuned models)")
    parser.add_argument("--checkpoint_steps", type=int, nargs="*", default=None,
                        help="If provided, evaluate checkpoints at these steps instead of final")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    return parser.parse_args()


def extract_answer(text: str) -> str | None:
    match = re.search(r"####\s*([\d,]+)", text)
    if match:
        return match.group(1).replace(",", "").strip()
    numbers = re.findall(r"\b\d+\b", text)
    return numbers[-1] if numbers else None


def get_ground_truth(answer_text: str) -> str:
    match = re.search(r"####\s*([\d,]+)", answer_text)
    return match.group(1).replace(",", "").strip() if match else ""


def build_messages(fewshot_examples: list[dict], query: dict, num_fewshot: int) -> list[dict]:
    """Build chat messages for evaluation."""
    messages = []
    for ex in fewshot_examples[:num_fewshot]:
        messages.append({"role": "user", "content": f"Question: {ex['question']}"})
        messages.append({"role": "assistant", "content": ex["answer"]})
    messages.append({"role": "user", "content": f"Question: {query['question']}"})
    return messages


def evaluate_model(
    client: OpenAI,
    model_id: str,
    test_dataset,
    fewshot_pool: list[dict],
    num_fewshot: int,
    max_new_tokens: int,
) -> dict:
    """
    Evaluate a model (base or LoRA) on the full GSM8K test set.

    Args:
        model_id: Either the base model name or LoRA adapter name registered in vLLM
    """
    correct = 0
    details = []

    for example in tqdm(test_dataset, desc=f"Evaluating {model_id}"):
        messages = build_messages(fewshot_pool[:num_fewshot], example, num_fewshot)

        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=0,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            generated = response.choices[0].message.content
        except Exception as e:
            print(f"API error for example: {e}")
            generated = ""

        pred = extract_answer(generated)
        gt = get_ground_truth(example["answer"])
        is_correct = (pred == gt) if pred is not None else False
        correct += int(is_correct)

        details.append({
            "question": example["question"][:100],
            "ground_truth": gt,
            "prediction": pred,
            "correct": is_correct,
        })

    accuracy = correct / len(test_dataset)
    return {
        "model_id": model_id,
        "accuracy": accuracy,
        "correct": correct,
        "total": len(test_dataset),
        "details": details,
    }


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = load_dataset("gsm8k", "main")
    test_data = list(dataset["test"])
    train_data = list(dataset["train"])

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, trust_remote_code=True)

    client = OpenAI(base_url=args.api_base, api_key="EMPTY")

    results = {
        "config": OmegaConf.to_container(cfg),
        "num_fewshot": args.num_fewshot,
        "test_size": len(test_data),
        "conditions": {},
    }

    if args.checkpoint_steps is None:
        # Evaluate final checkpoints
        for lora_name in args.lora_names:
            print(f"\nEvaluating: {lora_name} (final checkpoint)")
            result = evaluate_model(
                client, lora_name, test_data, train_data,
                args.num_fewshot, args.max_new_tokens,
            )
            results["conditions"][lora_name] = result
            print(f"  Accuracy: {result['accuracy']:.2%} ({result['correct']}/{result['total']})")
    else:
        # Evaluate at specific checkpoint steps
        # Requires the vLLM server to be restarted for each checkpoint
        # (or use offline vLLM LLM class instead)
        print("Checkpoint evaluation requires restarting vLLM server for each checkpoint.")
        print("Use the offline evaluation script or restart manually per checkpoint.")
        # Placeholder for checkpoint loop
        for lora_name in args.lora_names:
            results["conditions"][lora_name] = {}
            for step in args.checkpoint_steps:
                print(f"\nEvaluating: {lora_name} @ step {step}")
                result = evaluate_model(
                    client, f"{lora_name}_step{step}", test_data, train_data,
                    args.num_fewshot, args.max_new_tokens,
                )
                results["conditions"][lora_name][f"step_{step}"] = result
                print(f"  Accuracy: {result['accuracy']:.2%}")

    # Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    for cond, data in results["conditions"].items():
        if isinstance(data, dict) and "accuracy" in data:
            print(f"  {cond:20s}: {data['accuracy']:.2%}")
        elif isinstance(data, dict):
            for step, step_data in data.items():
                print(f"  {cond:15s} {step:10s}: {step_data['accuracy']:.2%}")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
