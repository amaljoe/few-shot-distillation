"""
Evaluate trained XTREME checkpoints (finetuned, distilled, control) using vLLM.

For PEFT/LoRA checkpoints: uses vLLM's built-in LoRA engine (fast, no model reload).
For full fine-tuned checkpoints: loads directly as base model.

Run example (single checkpoint):
  CUDA_VISIBLE_DEVICES=0,1 python scripts/eval_xtreme_checkpoints.py \\
      --base_model meta-llama/Llama-3.2-3B-Instruct \\
      --checkpoint_dir experiments/xtreme/llama3b/xtreme_sft/final \\
      --condition finetuned \\
      --output experiments/xtreme/llama3b_trained.json \\
      --n_samples 500

Run example (all conditions for a model):
  CUDA_VISIBLE_DEVICES=0,1 python scripts/eval_xtreme_checkpoints.py \\
      --base_model meta-llama/Llama-3.2-3B-Instruct \\
      --base_dir experiments/xtreme/llama3b \\
      --output experiments/xtreme/llama3b_trained.json \\
      --n_samples 500
"""

import argparse
import json
import sys
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.xtreme_loader import (
    TASKS, LANGUAGES, TASK_LANGUAGES, TASK_MAX_NEW_TOKENS,
    XTREMEEvalDataset, parse_output, compute_task_metric,
)


# Maps output subdir name → condition label used in results JSON
CONDITION_DIRS = {
    "xtreme_sft":     "finetuned",
    "xtreme_distill": "distilled",
    "xtreme_control": "control",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model",          type=str, required=True)
    parser.add_argument("--checkpoint_dir",      type=str, default=None)
    parser.add_argument("--condition",           type=str, default=None)
    parser.add_argument("--base_dir",            type=str, default=None)
    parser.add_argument("--tasks",               type=str, nargs="+", default=TASKS)
    parser.add_argument("--languages",           type=str, nargs="+", default=LANGUAGES)
    parser.add_argument("--n_samples",           type=int, default=500)
    parser.add_argument("--output",              type=str, required=True)
    parser.add_argument("--seed",                type=int, default=42)
    parser.add_argument("--tensor_parallel_size",type=int, default=2)
    parser.add_argument("--max_model_len",       type=int, default=4096)
    return parser.parse_args()


def evaluate_one(
    llm: "LLM",
    tokenizer,
    eval_ds: XTREMEEvalDataset,
    lora_request: "LoRARequest | None",
) -> dict:
    task = eval_ds.task
    n = len(eval_ds)
    if n == 0:
        return {}

    prompts = [eval_ds.get_prompt(i) for i in range(n)]
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=TASK_MAX_NEW_TOKENS[task],
    )
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    preds_raw = [o.outputs[0].text for o in outputs]

    golds = [eval_ds.get_gold(i) for i in range(n)]

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
    elif task == "qa":
        parsed = [parse_output(task, r) for r in preds_raw]
        return compute_task_metric(task, parsed, golds)
    else:
        parsed = [parse_output(task, r) for r in preds_raw]
        return compute_task_metric(task, parsed, golds)


def eval_condition(llm, tokenizer, lora_request, condition, tasks, languages,
                   n_samples, seed, results, output_path):
    if condition not in results:
        results[condition] = {}

    for task in tasks:
        if task not in results[condition]:
            results[condition][task] = {}
        for lang in languages:
            if lang not in TASK_LANGUAGES.get(task, []):
                results[condition][task][lang] = None
                continue
            if results[condition][task].get(lang) is not None:
                print(f"  {task}/{lang}: skip (done)")
                continue
            try:
                eval_ds = XTREMEEvalDataset(
                    tokenizer=tokenizer,
                    task=task,
                    lang=lang,
                    condition="base",   # zero-shot prompts for trained models
                    n_samples=n_samples,
                    seed=seed,
                )
                if len(eval_ds) == 0:
                    results[condition][task][lang] = None
                    continue
                metrics = evaluate_one(llm, tokenizer, eval_ds, lora_request)
                results[condition][task][lang] = metrics
                print(f"  {task}/{lang}: {metrics}")
            except Exception as e:
                print(f"  {task}/{lang}: ERROR — {e}")
                results[condition][task][lang] = {"error": str(e)}

            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {}
    if output_path.exists():
        with open(output_path) as f:
            results = json.load(f)

    # Build list of (condition_label, checkpoint_path)
    ckpt_list = []
    if args.checkpoint_dir:
        cond = args.condition or "finetuned"
        ckpt_list.append((cond, args.checkpoint_dir))
    elif args.base_dir:
        base = Path(args.base_dir)
        for subdir_name, cond_label in CONDITION_DIRS.items():
            final_dir = base / subdir_name / "final"
            if final_dir.exists():
                ckpt_list.append((cond_label, str(final_dir)))
    else:
        raise ValueError("Provide --checkpoint_dir or --base_dir")

    for condition, ckpt_dir in ckpt_list:
        ckpt_path = Path(ckpt_dir)
        use_peft = (ckpt_path / "adapter_config.json").exists()

        print(f"\n=== Condition: {condition}  |  {ckpt_dir} ===")
        print(f"  Mode: {'LoRA (vLLM)' if use_peft else 'full fine-tune'}")

        if use_peft:
            llm = LLM(
                model=args.base_model,
                enable_lora=True,
                max_lora_rank=64,
                dtype="bfloat16",
                trust_remote_code=True,
                tensor_parallel_size=args.tensor_parallel_size,
                max_model_len=args.max_model_len,
            )
            lora_request = LoRARequest("adapter", 1, str(ckpt_path))
        else:
            llm = LLM(
                model=str(ckpt_path),
                dtype="bfloat16",
                trust_remote_code=True,
                tensor_parallel_size=args.tensor_parallel_size,
                max_model_len=args.max_model_len,
            )
            lora_request = None

        tokenizer = llm.get_tokenizer()

        eval_condition(
            llm, tokenizer, lora_request, condition,
            tasks=args.tasks,
            languages=args.languages,
            n_samples=args.n_samples,
            seed=args.seed,
            results=results,
            output_path=output_path,
        )

        del llm

    print(f"\n✓ All conditions done. Results saved to {output_path}")


if __name__ == "__main__":
    main()
