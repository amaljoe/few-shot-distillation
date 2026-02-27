"""
HuggingFace-based checkpoint evaluator.

Fallback for models whose architecture vLLM does not support (e.g. param2moe).
Loads LoRA checkpoints with PEFT (merge-and-unload for speed) or full-FT
checkpoints directly, then evaluates with HF generate().

Usage:
  CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/eval_hf_checkpoints.py \\
      --config configs/param2_17b.yaml \\
      --conditions baseline online_v1 zeroshot_teacher \\
      --base_dir experiments/param2_17b \\
      --n_samples 1319 \\
      --checkpoint_steps 200 400 600 800 1000 \\
      --output experiments/param2_17b_eval.json \\
      --max_new_tokens 1024
"""

import argparse
import gc
import json
import re
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=400)
    parser.add_argument("--conditions", nargs="+", default=["baseline", "online_v1"])
    parser.add_argument("--checkpoint_steps", nargs="+", type=int,
                        default=[200, 400, 600, 800, 1000])
    parser.add_argument("--base_dir", type=str, default="experiments")
    parser.add_argument("--output", type=str, default="experiments/hf_eval.json")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=2)
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


def build_prompts(examples, tokenizer):
    prompts = []
    for ex in examples:
        messages = [{"role": "user", "content": f"Question: {ex['question']}"}]
        if getattr(tokenizer, "chat_template", None) is None:
            text = f"Question: {ex['question']}\n"
        else:
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


def is_lora_checkpoint(ckpt_path: Path) -> bool:
    return (ckpt_path / "adapter_config.json").exists()


def evaluate_checkpoint(model, tokenizer, prompts, ground_truths, max_new_tokens, batch_size, desc):
    correct = 0
    # Use left padding for batch generation
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    device = next(model.parameters()).device

    for i in tqdm(range(0, len(prompts), batch_size), desc=desc, leave=False):
        batch_prompts = prompts[i:i + batch_size]
        batch_gts = ground_truths[i:i + batch_size]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            add_special_tokens=False,
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        input_len = input_ids.shape[1]

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        for output, gt in zip(outputs, batch_gts):
            generated = tokenizer.decode(output[input_len:], skip_special_tokens=True)
            pred = extract_answer(generated)
            if pred is not None and pred == gt:
                correct += 1

    tokenizer.padding_side = old_padding_side
    accuracy = correct / len(prompts)
    print(f"  {desc}: {accuracy:.2%} ({correct}/{len(prompts)})")
    return {"accuracy": accuracy, "correct": correct, "total": len(prompts)}


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {cfg.model.name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("gsm8k", "main")
    test_data = list(dataset["test"])
    torch.manual_seed(42)
    indices = torch.randperm(len(test_data))[:args.n_samples].tolist()
    test_subset = [test_data[i] for i in indices]

    prompts = build_prompts(test_subset, tokenizer)
    ground_truths = [get_ground_truth(ex["answer"]) for ex in test_subset]

    print(f"\nHF Checkpoint Evaluation (batch_size={args.batch_size})")
    print(f"  Test subset: {len(prompts)} examples")
    print(f"  Conditions: {args.conditions}")
    print(f"  Steps: {args.checkpoint_steps}")

    results = {
        "config": OmegaConf.to_container(cfg),
        "n_samples": len(prompts),
        "checkpoint_steps": args.checkpoint_steps,
        "conditions": {},
        "backend": "hf",
    }

    for cond in args.conditions:
        print(f"\n=== Condition: {cond} ===")
        results["conditions"][cond] = {}

        for step in args.checkpoint_steps:
            ckpt_path = Path(args.base_dir) / cond / cond / f"checkpoint-{step}"
            if not ckpt_path.exists():
                print(f"  Skipping {ckpt_path} (not found)")
                continue

            is_lora = is_lora_checkpoint(ckpt_path)
            print(f"  Loading checkpoint: {ckpt_path} ({'LoRA' if is_lora else 'full-FT'})")

            base_model = AutoModelForCausalLM.from_pretrained(
                cfg.model.name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )

            if is_lora:
                from peft import PeftModel
                lora_model = PeftModel.from_pretrained(base_model, str(ckpt_path))
                model = lora_model.merge_and_unload()
            else:
                model = base_model

            model.eval()

            result = evaluate_checkpoint(
                model, tokenizer, prompts, ground_truths,
                args.max_new_tokens, args.batch_size,
                desc=f"step {step}",
            )
            results["conditions"][cond][f"step_{step}"] = result

            del model, base_model
            gc.collect()
            torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 60)
    print("CHECKPOINT CURVE SUMMARY (HF backend)")
    print("=" * 60)
    header = f"{'Step':>6}"
    for cond in args.conditions:
        header += f"  {cond:>14}"
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
