"""
HuggingFace-based ICL evaluation (fallback for models vLLM doesn't support).

Evaluates 0-shot and 8-shot accuracy using HF generate() with device_map="auto".
Drop-in replacement for eval_icl.py when vLLM cannot load the model architecture.

Usage:
  CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/eval_hf_icl.py \\
      --model bharatgenai/Param2-17B-A2.4B-Thinking \\
      --num_samples 1319 --num_fewshot 0 8 \\
      --output experiments/param2_17b/icl_eval.json \\
      --max_new_tokens 1024
"""

import argparse
import json
import random
import re
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--num_fewshot", type=int, nargs="+", default=[0, 8])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)
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


def build_messages(fewshot_examples, query, num_fewshot):
    messages = []
    for ex in fewshot_examples[:num_fewshot]:
        messages.append({"role": "user", "content": f"Question: {ex['question']}"})
        messages.append({"role": "assistant", "content": ex["answer"]})
    messages.append({"role": "user", "content": f"Question: {query['question']}"})
    return messages


def format_prompt(tokenizer, messages):
    if getattr(tokenizer, "chat_template", None) is None:
        parts = []
        for i, msg in enumerate(messages):
            parts.append(msg["content"])
            if (msg["role"] == "assistant"
                    and i + 1 < len(messages)
                    and messages[i + 1]["role"] == "user"):
                parts.append("")
        return "\n".join(parts) + "\n"
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": False},
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )


def main():
    args = parse_args()
    random.seed(args.seed)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Model: {args.model}  (HF backend)")
    print(f"Num samples: {args.num_samples}")
    print(f"Few-shot configs: {args.num_fewshot}")
    print(f"{'='*60}\n")

    dataset = load_dataset("gsm8k", "main")
    train_data = list(dataset["train"])
    test_data = list(dataset["test"])

    random.seed(args.seed)
    test_sample = random.sample(test_data, min(args.num_samples, len(test_data)))

    print(f"Loading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # for batch generation

    print(f"Loading model {args.model} with device_map=auto...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    results = {
        "model": args.model,
        "num_samples": args.num_samples,
        "seed": args.seed,
        "backend": "hf",
        "evaluations": {},
    }

    for nshot in args.num_fewshot:
        print(f"\n--- Evaluating {nshot}-shot ---")
        prompts = []
        ground_truths = []

        random.seed(args.seed)
        for example in test_sample:
            candidates = [ex for ex in train_data if ex["question"] != example["question"]]
            fewshot_examples = random.sample(candidates, min(nshot, len(candidates)))
            messages = build_messages(fewshot_examples, example, nshot)
            prompts.append(format_prompt(tokenizer, messages))
            ground_truths.append(get_ground_truth(example["answer"]))

        correct = 0
        details = []

        for i in tqdm(range(0, len(prompts), args.batch_size), desc=f"{nshot}-shot"):
            batch_prompts = prompts[i:i + args.batch_size]
            batch_gts = ground_truths[i:i + args.batch_size]

            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,
                add_special_tokens=False,
            )
            input_ids = inputs["input_ids"].to(next(model.parameters()).device)
            attention_mask = inputs["attention_mask"].to(input_ids.device)

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )

            for j, (output, gt) in enumerate(zip(outputs, batch_gts)):
                input_len = input_ids.shape[1]
                generated = tokenizer.decode(output[input_len:], skip_special_tokens=True)
                pred = extract_answer(generated)
                is_correct = (pred == gt) if pred is not None else False
                correct += int(is_correct)
                details.append({
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

    shot_accs = {k: v["accuracy"] for k, v in results["evaluations"].items()}
    if "0_shot" in shot_accs:
        best_fewshot = max(v for k, v in shot_accs.items() if k != "0_shot")
        gap = best_fewshot - shot_accs["0_shot"]
        results["icl_gap"] = gap
        print(f"\n{'='*60}")
        print(f"ICL Gap (best few-shot - 0-shot): {gap:.2%}")
        print(f"  Recommendation: {'USE THIS MODEL ✓' if gap >= 0.05 else 'Try larger model'}")
        print(f"{'='*60}")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
