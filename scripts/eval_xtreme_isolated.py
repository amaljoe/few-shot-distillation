"""
Efficient eval for XTREME isolated lambda sweep.

Loads base model once with enable_lora=True, evaluates all lambda conditions
for a single task without reloading the model between conditions.

Usage:
  CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/eval_xtreme_isolated.py \\
      --base_model Qwen/Qwen3-1.7B \\
      --task nli \\
      --base_dir experiments/xtreme_isolated/nli \\
      --lambdas 0 0.01 0.05 0.1 0.25 0.5 0.75 1.0 \\
      --languages en \\
      --n_samples 500 \\
      --output experiments/xtreme_isolated/results.json \\
      --tensor_parallel_size 4
"""

import argparse
import json
import sys
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.xtreme_loader import (
    TASK_LANGUAGES, TASK_MAX_NEW_TOKENS,
    XTREMEEvalDataset, parse_output, compute_task_metric,
)


def lam_to_key(lam: float) -> str:
    """Convert lambda float to condition suffix, e.g. 0.05 → '5', 0.1 → '10'."""
    return str(int(round(lam * 100)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model",           type=str, required=True)
    parser.add_argument("--task",                 type=str, required=True)
    parser.add_argument("--base_dir",             type=str, required=True)
    parser.add_argument("--lambdas",              type=float, nargs="+",
                        default=[0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--languages",            type=str, nargs="+", default=["en"])
    parser.add_argument("--n_samples",            type=int, default=500)
    parser.add_argument("--output",               type=str, required=True)
    parser.add_argument("--seed",                 type=int, default=42)
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--max_model_len",        type=int, default=4096)
    return parser.parse_args()


def evaluate_one_condition(llm, task, lang, lora_request, n_samples, seed):
    """Evaluate a single (task, lang, adapter) triple."""
    tokenizer = llm.get_tokenizer()
    eval_ds = XTREMEEvalDataset(
        tokenizer=tokenizer,
        task=task,
        lang=lang,
        condition="base",   # zero-shot prompts for trained models
        n_samples=n_samples,
        seed=seed,
    )
    if len(eval_ds) == 0:
        return {}

    prompts = [eval_ds.get_prompt(i) for i in range(len(eval_ds))]
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=TASK_MAX_NEW_TOKENS[task],
    )
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    # Strip Qwen3 thinking tokens (<think>...</think>) if present
    def strip_thinking(text: str) -> str:
        if "</think>" in text:
            return text.split("</think>", 1)[-1].strip()
        return text
    preds_raw = [strip_thinking(o.outputs[0].text) for o in outputs]
    golds = [eval_ds.get_gold(i) for i in range(len(eval_ds))]

    if task in ("ner", "pos"):
        parsed_preds, parsed_golds = [], []
        for i, (raw, gold) in enumerate(zip(preds_raw, golds)):
            n_tok = len(eval_ds.examples[i].get("tokens", []))
            parsed_preds.append(parse_output(task, raw, n_tok))
            if isinstance(gold, str):
                gold = gold.split()
            pad = "O" if task == "ner" else "X"
            if n_tok > 0:
                gold = (gold + [pad] * n_tok)[:n_tok]
            parsed_golds.append(gold)
        return compute_task_metric(task, parsed_preds, parsed_golds)
    else:
        parsed = [parse_output(task, r) for r in preds_raw]
        return compute_task_metric(task, parsed, golds)


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {}
    if output_path.exists():
        with open(output_path) as f:
            results = json.load(f)

    # Build list of valid (condition_name, checkpoint_path, lora_id)
    conditions = []
    for i, lam in enumerate(args.lambdas):
        key = lam_to_key(lam)
        condition = f"{args.task}_lam{key}"
        checkpoint_dir = Path(args.base_dir) / condition / "final"
        if checkpoint_dir.exists():
            conditions.append((condition, str(checkpoint_dir), i + 1))
        else:
            print(f"[skip] {condition}: {checkpoint_dir} not found")

    if not conditions:
        print("No valid checkpoints found — exiting.")
        return

    print(f"\nLoading base model with {len(conditions)} LoRA adapters...")
    llm = LLM(
        model=args.base_model,
        enable_lora=True,
        max_loras=len(conditions),
        max_lora_rank=64,
        dtype="bfloat16",
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
    )

    for condition, ckpt_dir, lora_id in conditions:
        lora_request = LoRARequest(condition, lora_id, ckpt_dir)

        if condition not in results:
            results[condition] = {}
        if args.task not in results[condition]:
            results[condition][args.task] = {}

        for lang in args.languages:
            if lang not in TASK_LANGUAGES.get(args.task, []):
                print(f"  [{condition}] {lang} not supported for {args.task}, skipping")
                continue
            if results[condition][args.task].get(lang) is not None:
                print(f"  [{condition}/{lang}] already done, skipping")
                continue

            try:
                metrics = evaluate_one_condition(
                    llm, args.task, lang, lora_request,
                    args.n_samples, args.seed
                )
                results[condition][args.task][lang] = metrics
                print(f"  [{condition}/{lang}]: {metrics}")
            except Exception as e:
                print(f"  [{condition}/{lang}]: ERROR — {e}")
                results[condition][args.task][lang] = {"error": str(e)}

            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)

    print(f"\n✓ Done. Results saved to {output_path}")


if __name__ == "__main__":
    main()
