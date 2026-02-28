"""
IFEval ICL gap evaluation.

Measures 0-shot vs k-shot prompt-level strict accuracy on google/IFEval (541 examples).
Few-shot pool: argilla/ifeval-like-data (filtered, pre-verified compliant examples).

Requires: pip install instruction-following-eval

Usage (inside apptainer on cn14-dgx):
  CUDA_VISIBLE_DEVICES=0,1 python scripts/eval_ifeval_icl.py \\
      --model Qwen/Qwen3-1.7B --num_fewshot 0 4 \\
      --n_samples 541 \\
      --output experiments/ifeval_qwen1b7/icl_eval.json \\
      --tensor_parallel_size 2 --max_new_tokens 512

Smoke test (8 examples):
  CUDA_VISIBLE_DEVICES=0,1 python scripts/eval_ifeval_icl.py \\
      --model Qwen/Qwen3-1.7B --num_fewshot 0 4 --n_samples 8 \\
      --output /tmp/ifeval_test.json --tensor_parallel_size 2 --max_new_tokens 128
"""

import argparse
import json
import random
import sys
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.ifeval_loader import load_ifeval, build_fewshot_messages, check_instruction_following


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--num_fewshot", type=int, nargs="+", default=[0, 4],
                        help="List of few-shot counts to evaluate")
    parser.add_argument("--n_samples", type=int, default=541,
                        help="Number of IFEval test examples to evaluate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save JSON results")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Model: {args.model}")
    print(f"Dataset: IFEval (google/IFEval)")
    print(f"Num samples: {args.n_samples}")
    print(f"Few-shot configs: {args.num_fewshot}")
    print(f"{'='*60}\n")

    print("Loading argilla/ifeval-like-data (few-shot pool)...")
    fewshot_pool = load_ifeval("train")
    print(f"  Few-shot pool size: {len(fewshot_pool)}")

    print("Loading google/IFEval (eval set)...")
    eval_data = load_ifeval("test")
    print(f"  Eval set size: {len(eval_data)}")

    # Fixed random sample of eval examples
    random.seed(args.seed)
    eval_sample = random.sample(eval_data, min(args.n_samples, len(eval_data)))
    print(f"  Using {len(eval_sample)} examples\n")

    print(f"Loading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print(f"Initializing vLLM with {args.model}...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
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
        "dataset": "ifeval",
        "n_samples": len(eval_sample),
        "seed": args.seed,
        "results_per_shot": {},
    }

    for nshot in args.num_fewshot:
        print(f"\n--- Evaluating {nshot}-shot ---")

        prompts = []
        for i, example in enumerate(eval_sample):
            if nshot > 0:
                # Deterministic per-example sampling from fewshot pool
                rng = random.Random(args.seed + i)
                # Exclude exact prompt match to avoid leakage
                candidates = [ex for ex in fewshot_pool if ex["prompt"] != example["prompt"]]
                fewshot_examples = rng.sample(candidates, min(nshot, len(candidates)))
                messages = build_fewshot_messages(fewshot_examples, example["prompt"])
            else:
                messages = [{"role": "user", "content": example["prompt"]}]

            # Apply chat template (disable thinking for Qwen3)
            if getattr(tokenizer, "chat_template", None) is None:
                parts = []
                for j, msg in enumerate(messages):
                    parts.append(msg["content"])
                    if (msg["role"] == "assistant"
                            and j + 1 < len(messages)
                            and messages[j + 1]["role"] == "user"):
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
                    prompt = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
            prompts.append(prompt)

        # Generate responses
        outputs = llm.generate(prompts, sampling_params)

        # Evaluate with instruction-following constraints
        n_prompt_pass = 0
        total_inst = 0
        n_inst_pass = 0
        details = []

        for output, example in zip(outputs, eval_sample):
            response = output.outputs[0].text
            try:
                prompt_pass, inst_results = check_instruction_following(example, response)
            except Exception as e:
                # If constraint checking fails (e.g., missing instruction type), skip
                print(f"  Warning: constraint check failed: {e}")
                prompt_pass = False
                inst_results = [False] * len(example.get("instruction_id_list", []))

            n_prompt_pass += int(prompt_pass)
            total_inst += len(inst_results)
            n_inst_pass += sum(inst_results)
            details.append({
                "prompt": example["prompt"][:80] + "...",
                "prompt_pass": prompt_pass,
                "inst_results": inst_results,
            })

        prompt_accuracy = n_prompt_pass / len(eval_sample)
        instruction_accuracy = n_inst_pass / total_inst if total_inst > 0 else 0.0

        results["results_per_shot"][str(nshot)] = {
            "prompt_accuracy": prompt_accuracy,
            "instruction_accuracy": instruction_accuracy,
            "n_prompt_pass": n_prompt_pass,
            "n_total": len(eval_sample),
            "n_inst_pass": n_inst_pass,
            "n_inst_total": total_inst,
            "details": details,
        }

        print(f"  {nshot}-shot prompt accuracy:       {prompt_accuracy:.2%} ({n_prompt_pass}/{len(eval_sample)})")
        print(f"  {nshot}-shot instruction accuracy:  {instruction_accuracy:.2%} ({n_inst_pass}/{total_inst})")

    # ICL gap
    shot_results = results["results_per_shot"]
    if "0" in shot_results:
        best_fewshot_acc = max(
            v["prompt_accuracy"] for k, v in shot_results.items() if k != "0"
        )
        gap = best_fewshot_acc - shot_results["0"]["prompt_accuracy"]
        results["icl_gap"] = gap
        recommendation = "USE THIS MODEL ✓" if gap >= 0.05 else "Consider larger model"
        results["recommendation"] = recommendation
        print(f"\n{'='*60}")
        print(f"ICL Gap (best few-shot - 0-shot, prompt accuracy): {gap:.2%}")
        print(f"  Recommendation: {recommendation}")
        print(f"{'='*60}")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
