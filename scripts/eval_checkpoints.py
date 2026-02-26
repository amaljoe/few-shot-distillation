"""
Checkpoint accuracy curve evaluation.

Evaluates Conditions B and C at every saved checkpoint (200/400/.../1000)
on a fixed subset of the GSM8K test set. Uses vLLM offline (LLM class) with
LoRA so we never restart the server — just swap the LoRARequest per checkpoint.

Run command (tmux: claude, INSIDE apptainer, after training is done):
  CUDA_VISIBLE_DEVICES=0,1 python scripts/eval_checkpoints.py \\
      --config configs/base.yaml \\
      --n_samples 400 \\
      --output experiments/ablations/checkpoint_curve/results.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--n_samples", type=int, default=400,
                        help="Number of test examples to evaluate per checkpoint")
    parser.add_argument("--conditions", nargs="+",
                        default=["baseline", "distill"],
                        help="Condition names (subdirectory names under base_dir/)")
    parser.add_argument("--checkpoint_steps", nargs="+", type=int,
                        default=[200, 400, 600, 800, 1000])
    parser.add_argument("--base_dir", type=str, default="experiments/poc",
                        help="Root dir containing condition subdirectories")
    parser.add_argument("--output", type=str,
                        default="experiments/ablations/checkpoint_curve/results.json")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_model_len", type=int, default=2048,
                        help="vLLM max_model_len (context window cap)")
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
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


def build_prompts(examples: list[dict], tokenizer) -> list[str]:
    prompts = []
    for ex in examples:
        messages = [{"role": "user", "content": f"Question: {ex['question']}"}]
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        prompts.append(text)
    return prompts


def evaluate_checkpoint(
    llm: LLM,
    sampling_params: SamplingParams,
    lora_request: LoRARequest,
    prompts: list[str],
    ground_truths: list[str],
    desc: str,
) -> dict:
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    correct = 0
    for out, gt in zip(outputs, ground_truths):
        generated = out.outputs[0].text
        pred = extract_answer(generated)
        if pred is not None and pred == gt:
            correct += 1
    accuracy = correct / len(prompts)
    print(f"  {desc}: {accuracy:.2%} ({correct}/{len(prompts)})")
    return {"accuracy": accuracy, "correct": correct, "total": len(prompts)}


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Fixed test subset (same seed for reproducibility)
    dataset = load_dataset("gsm8k", "main")
    test_data = list(dataset["test"])
    torch.manual_seed(42)
    indices = torch.randperm(len(test_data))[:args.n_samples].tolist()
    test_subset = [test_data[i] for i in indices]

    prompts = build_prompts(test_subset, tokenizer)
    ground_truths = [get_ground_truth(ex["answer"]) for ex in test_subset]

    print(f"\nCheckpoint curve evaluation")
    print(f"  Test subset: {len(prompts)} examples")
    print(f"  Conditions: {args.conditions}")
    print(f"  Steps: {args.checkpoint_steps}")

    # Load base model once — swap LoRA adapters per checkpoint
    print(f"\nLoading base model: {cfg.model.name}")
    llm = LLM(
        model=cfg.model.name,
        enable_lora=True,
        max_lora_rank=cfg.lora.r,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=args.max_new_tokens,
    )

    results = {
        "config": OmegaConf.to_container(cfg),
        "n_samples": len(prompts),
        "checkpoint_steps": args.checkpoint_steps,
        "conditions": {},
    }

    lora_id_counter = 1
    for cond in args.conditions:
        print(f"\n=== Condition: {cond} ===")
        results["conditions"][cond] = {}

        for step in tqdm(args.checkpoint_steps, desc=cond):
            ckpt_path = Path(args.base_dir) / cond / cond / f"checkpoint-{step}"
            if not ckpt_path.exists():
                print(f"  Skipping {ckpt_path} (not found)")
                continue

            lora_req = LoRARequest(
                lora_name=f"{cond}_{step}",
                lora_int_id=lora_id_counter,
                lora_path=str(ckpt_path),
            )
            lora_id_counter += 1

            result = evaluate_checkpoint(
                llm, sampling_params, lora_req, prompts, ground_truths,
                desc=f"step {step}",
            )
            results["conditions"][cond][f"step_{step}"] = result

    # Summary
    print("\n" + "=" * 60)
    print("CHECKPOINT CURVE SUMMARY")
    print("=" * 60)
    header = f"{'Step':>6}"
    for cond in args.conditions:
        header += f"  {cond:>12}"
    print(header)
    for step in args.checkpoint_steps:
        row = f"{step:>6}"
        for cond in args.conditions:
            acc = results["conditions"].get(cond, {}).get(f"step_{step}", {}).get("accuracy")
            row += f"  {acc:.2%}" if acc is not None else f"  {'N/A':>7}"
        print(row)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
