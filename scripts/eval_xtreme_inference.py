"""
Evaluate base (zero-shot) and few-shot conditions on XTREME using vLLM.

Loads the base pretrained model and evaluates with zero-shot or in-language
few-shot prompting across all tasks × languages.

Run example:
  CUDA_VISIBLE_DEVICES=0,1 python scripts/eval_xtreme_inference.py \\
      --model meta-llama/Llama-3.2-3B-Instruct \\
      --conditions base fewshot \\
      --n_samples 500 \\
      --output experiments/xtreme/llama3b_inference.json \\
      --tensor_parallel_size 2

Large model (tensor parallel across 4 GPUs):
  CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/eval_xtreme_inference.py \\
      --model Qwen/Qwen3-8B \\
      --tensor_parallel_size 4
"""

import argparse
import json
import sys
from pathlib import Path

from vllm import LLM, SamplingParams

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.xtreme_loader import (
    TASKS, LANGUAGES, TASK_LANGUAGES, TASK_MAX_NEW_TOKENS,
    XTREMEEvalDataset, parse_output, compute_task_metric,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",               type=str, required=True)
    parser.add_argument("--conditions",          type=str, nargs="+",
                        default=["base", "fewshot"], choices=["base", "fewshot"])
    parser.add_argument("--tasks",               type=str, nargs="+", default=TASKS)
    parser.add_argument("--languages",           type=str, nargs="+", default=LANGUAGES)
    parser.add_argument("--n_samples",           type=int, default=500)
    parser.add_argument("--output",              type=str, default=None)
    parser.add_argument("--seed",                type=int, default=42)
    parser.add_argument("--tensor_parallel_size",type=int, default=2)
    parser.add_argument("--max_model_len",       type=int, default=4096)
    return parser.parse_args()


def build_output_path(model_name: str) -> Path:
    slug = model_name.replace("/", "__")
    return Path("experiments/xtreme") / f"{slug}_inference.json"


def evaluate_one(llm: "LLM", eval_ds: XTREMEEvalDataset) -> dict:
    task = eval_ds.task
    n = len(eval_ds)
    if n == 0:
        return {}

    prompts = [eval_ds.get_prompt(i) for i in range(n)]
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=TASK_MAX_NEW_TOKENS[task],
    )
    outputs = llm.generate(prompts, sampling_params)
    preds_raw = [o.outputs[0].text for o in outputs]

    golds = [eval_ds.get_gold(i) for i in range(n)]

    if task in ("ner", "pos"):
        parsed_preds, parsed_golds = [], []
        for i, (raw, gold) in enumerate(zip(preds_raw, golds)):
            n_tokens = len(eval_ds.examples[i].get("tokens", []))
            parsed_preds.append(parse_output(task, raw, n_tokens))
            if isinstance(gold, str):
                gold = gold.split()
            pad = "O" if task == "ner" else "X"
            if n_tokens > 0:
                gold = (gold + [pad] * n_tokens)[:n_tokens]
            parsed_golds.append(gold)
        return compute_task_metric(task, parsed_preds, parsed_golds)
    elif task == "qa":
        parsed = [parse_output(task, r) for r in preds_raw]
        return compute_task_metric(task, parsed, golds)
    else:
        parsed = [parse_output(task, r) for r in preds_raw]
        return compute_task_metric(task, parsed, golds)


def main():
    args = parse_args()

    output_path = Path(args.output) if args.output else build_output_path(args.model)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {}
    if output_path.exists():
        with open(output_path) as f:
            results = json.load(f)

    print(f"\nLoading {args.model} with vLLM (tp={args.tensor_parallel_size}) ...")
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        enable_prefix_caching=True,
    )
    tokenizer = llm.get_tokenizer()

    for condition in args.conditions:
        print(f"\n=== Condition: {condition} ===")
        if condition not in results:
            results[condition] = {}

        for task in args.tasks:
            if task not in results[condition]:
                results[condition][task] = {}

            for lang in args.languages:
                if lang not in TASK_LANGUAGES.get(task, []):
                    results[condition][task][lang] = None
                    continue

                if results[condition][task].get(lang) is not None:
                    print(f"  {task}/{lang}: skip (done) → {results[condition][task][lang]}")
                    continue

                try:
                    eval_ds = XTREMEEvalDataset(
                        tokenizer=tokenizer,
                        task=task,
                        lang=lang,
                        condition=condition,
                        n_samples=args.n_samples,
                        seed=args.seed,
                    )
                    if len(eval_ds) == 0:
                        print(f"  {task}/{lang}: no data")
                        results[condition][task][lang] = None
                        continue

                    metrics = evaluate_one(llm, eval_ds)
                    results[condition][task][lang] = metrics
                    print(f"  {task}/{lang}: {metrics}")

                except Exception as e:
                    print(f"  {task}/{lang}: ERROR — {e}")
                    results[condition][task][lang] = {"error": str(e)}

                with open(output_path, "w") as f:
                    json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {output_path}")


if __name__ == "__main__":
    main()
