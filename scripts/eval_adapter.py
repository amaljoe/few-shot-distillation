"""
Evaluate a single LoRA adapter on the GSM8K test set (or subset).
Used for quick ablation evaluation without a persistent vLLM server.

Run command (tmux: inside apptainer):
  CUDA_VISIBLE_DEVICES=0,1 python scripts/eval_adapter.py \\
      --adapter_path experiments/ablations/kl_distill/kl_distill/final \\
      --adapter_name kl_distill \\
      --output experiments/ablations/kl_distill/eval_result.json

Multiple adapters in one shot (avoids loading base model twice):
  CUDA_VISIBLE_DEVICES=0,1 python scripts/eval_adapter.py \\
      --adapter_path path1 path2 path3 \\
      --adapter_name cond_d lambda_01 lambda_10 \\
      --output experiments/ablations/combined_eval.json
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
    parser.add_argument("--adapter_path", nargs="+", required=True)
    parser.add_argument("--adapter_name", nargs="+", required=True)
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Evaluate on a subset (None = full test set)")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=512)
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


def main():
    args = parse_args()
    assert len(args.adapter_path) == len(args.adapter_name), \
        "--adapter_path and --adapter_name must have the same number of entries"

    cfg = OmegaConf.load(args.config)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("gsm8k", "main")
    test_data = list(dataset["test"])

    if args.n_samples is not None:
        torch.manual_seed(42)
        indices = torch.randperm(len(test_data))[:args.n_samples].tolist()
        test_data = [test_data[i] for i in indices]

    prompts = build_prompts(test_data, tokenizer)
    ground_truths = [get_ground_truth(ex["answer"]) for ex in test_data]

    print(f"\nEvaluating {len(args.adapter_name)} adapter(s) on {len(prompts)} examples")

    llm = LLM(
        model=cfg.model.name,
        enable_lora=True,
        max_lora_rank=cfg.lora.r,
        max_model_len=2048,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(temperature=0, max_tokens=args.max_new_tokens)

    results = {"conditions": {}}

    for lora_id, (name, path) in enumerate(
        zip(args.adapter_name, args.adapter_path), start=1
    ):
        print(f"\nEvaluating: {name}  ({path})")
        lora_req = LoRARequest(lora_name=name, lora_int_id=lora_id, lora_path=path)
        outputs = llm.generate(prompts, sampling_params, lora_request=lora_req)

        correct = 0
        details = []
        for out, gt in tqdm(zip(outputs, ground_truths), total=len(ground_truths),
                            desc=name, leave=False):
            generated = out.outputs[0].text
            pred = extract_answer(generated)
            is_correct = (pred == gt) if pred is not None else False
            correct += int(is_correct)
            details.append({"ground_truth": gt, "prediction": pred, "correct": is_correct})

        accuracy = correct / len(prompts)
        results["conditions"][name] = {
            "adapter_path": path,
            "accuracy": accuracy,
            "correct": correct,
            "total": len(prompts),
            "details": details,
        }
        print(f"  {name}: {accuracy:.2%} ({correct}/{len(prompts)})")

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for name, data in results["conditions"].items():
        print(f"  {name:20s}: {data['accuracy']:.2%}")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
