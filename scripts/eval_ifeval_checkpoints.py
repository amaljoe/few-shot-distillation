"""
IFEval checkpoint accuracy curve evaluation.

Evaluates baseline and distillation conditions at every saved checkpoint
on google/IFEval (541 examples, 0-shot generation).

Metrics: prompt-level strict accuracy (primary) + instruction-level accuracy.
Requires: pip install instruction-following-eval

Auto-detects checkpoint type per condition:
  - LoRA checkpoints (adapter_config.json present): loads base model once, swaps
    LoRARequest per checkpoint (fast, single vLLM init).
  - Full-FT checkpoints (no adapter_config.json): loads each checkpoint as a full
    model, evaluates, then releases (one vLLM init per checkpoint).

Run command (inside apptainer on cn14-dgx, after training is done):
  CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/eval_ifeval_checkpoints.py \\
      --config configs/ifeval_qwen1b7.yaml \\
      --conditions baseline online_v1 \\
      --base_dir experiments/ifeval_qwen1b7 \\
      --n_samples 541 --checkpoint_steps 200 400 600 800 1000 \\
      --output experiments/ifeval_qwen1b7/results.json \\
      --tensor_parallel_size 4 --max_new_tokens 512
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.ifeval_loader import load_ifeval, build_student_messages, check_instruction_following


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ifeval_qwen1b7.yaml")
    parser.add_argument("--n_samples", type=int, default=541,
                        help="Number of IFEval examples to evaluate per checkpoint")
    parser.add_argument("--conditions", nargs="+",
                        default=["baseline", "online_v1"],
                        help="Condition names (subdirectory names under base_dir/)")
    parser.add_argument("--checkpoint_steps", nargs="+", type=int,
                        default=[200, 400, 600, 800, 1000])
    parser.add_argument("--base_dir", type=str, default="experiments/ifeval_qwen1b7",
                        help="Root dir containing condition subdirectories")
    parser.add_argument("--output", type=str,
                        default="experiments/ifeval_qwen1b7/results.json")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_model_len", type=int, default=2048,
                        help="vLLM max_model_len (context window cap)")
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_prompts(examples: list[dict], tokenizer) -> list[str]:
    """Build zero-shot prompts for IFEval examples."""
    prompts = []
    for ex in examples:
        messages = build_student_messages(ex["prompt"])
        if getattr(tokenizer, "chat_template", None) is None:
            prompt = messages[0]["content"] + "\n"
        else:
            try:
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    chat_template_kwargs={"enable_thinking": False},
                )
            except TypeError:
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
        prompts.append(prompt)
    return prompts


def evaluate_checkpoint(
    llm: LLM,
    sampling_params: SamplingParams,
    lora_request,
    prompts: list[str],
    examples: list[dict],
    desc: str,
) -> dict:
    """Generate responses and compute prompt/instruction accuracy."""
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)

    n_prompt_pass = 0
    total_inst = 0
    n_inst_pass = 0

    for output, example in zip(outputs, examples):
        response = output.outputs[0].text
        try:
            prompt_pass, inst_results = check_instruction_following(example, response)
        except Exception:
            prompt_pass = False
            inst_results = [False] * len(example.get("instruction_id_list", []))

        n_prompt_pass += int(prompt_pass)
        total_inst += len(inst_results)
        n_inst_pass += sum(inst_results)

    n_total = len(examples)
    prompt_accuracy = n_prompt_pass / n_total
    instruction_accuracy = n_inst_pass / total_inst if total_inst > 0 else 0.0

    print(f"  {desc}: prompt={prompt_accuracy:.2%} ({n_prompt_pass}/{n_total})  "
          f"inst={instruction_accuracy:.2%} ({n_inst_pass}/{total_inst})")

    return {
        "prompt_accuracy": prompt_accuracy,
        "instruction_accuracy": instruction_accuracy,
        "n_prompt_pass": n_prompt_pass,
        "n_total": n_total,
        "n_inst_pass": n_inst_pass,
        "n_inst_total": total_inst,
    }


def is_lora_checkpoint(ckpt_path: Path) -> bool:
    """Return True if checkpoint contains a LoRA adapter."""
    return (ckpt_path / "adapter_config.json").exists()


def detect_condition_type(cond: str, base_dir: str, checkpoint_steps: list[int]) -> str:
    """Return 'lora', 'full_ft', or 'unknown' based on first found checkpoint."""
    for step in checkpoint_steps:
        ckpt_path = Path(base_dir) / cond / cond / f"checkpoint-{step}"
        if ckpt_path.exists():
            return "lora" if is_lora_checkpoint(ckpt_path) else "full_ft"
    return "unknown"


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load eval set
    print("Loading google/IFEval eval set...")
    eval_data = load_ifeval("test")

    import random
    random.seed(args.seed)
    eval_sample = random.sample(eval_data, min(args.n_samples, len(eval_data)))
    print(f"  Using {len(eval_sample)} examples\n")

    prompts = build_prompts(eval_sample, tokenizer)

    print(f"\nCheckpoint curve evaluation")
    print(f"  Test subset: {len(prompts)} examples")
    print(f"  Conditions: {args.conditions}")
    print(f"  Steps: {args.checkpoint_steps}")

    # Classify conditions
    lora_conditions = []
    fullft_conditions = []
    for cond in args.conditions:
        ctype = detect_condition_type(cond, args.base_dir, args.checkpoint_steps)
        print(f"  {cond}: detected as {ctype}")
        if ctype == "lora":
            lora_conditions.append(cond)
        else:
            fullft_conditions.append(cond)

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=args.max_new_tokens,
    )

    results = {
        "config": OmegaConf.to_container(cfg),
        "n_samples": len(eval_sample),
        "checkpoint_steps": args.checkpoint_steps,
        "conditions": {},
    }

    # --- LoRA conditions: load base model once, swap adapters ---
    if lora_conditions:
        lora_cfg = getattr(cfg, "lora", None)
        max_lora_rank = lora_cfg.r if lora_cfg is not None else 64
        print(f"\nLoading base model for LoRA eval: {cfg.model.name}")
        llm = LLM(
            model=cfg.model.name,
            enable_lora=True,
            max_lora_rank=max_lora_rank,
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
            dtype="bfloat16",
            trust_remote_code=True,
        )

        lora_id_counter = 1
        for cond in lora_conditions:
            print(f"\n=== Condition: {cond} (LoRA) ===")
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
                    llm, sampling_params, lora_req, prompts, eval_sample,
                    desc=f"step {step}",
                )
                results["conditions"][cond][f"step_{step}"] = result

        del llm
        gc.collect()
        torch.cuda.empty_cache()

    # --- Full-FT conditions: load each checkpoint as full model ---
    for cond in fullft_conditions:
        print(f"\n=== Condition: {cond} (full-FT) ===")
        results["conditions"][cond] = {}

        for step in tqdm(args.checkpoint_steps, desc=cond):
            ckpt_path = Path(args.base_dir) / cond / cond / f"checkpoint-{step}"
            if not ckpt_path.exists():
                print(f"  Skipping {ckpt_path} (not found)")
                continue

            print(f"  Loading checkpoint: {ckpt_path}")
            llm = LLM(
                model=str(ckpt_path),
                max_model_len=args.max_model_len,
                tensor_parallel_size=args.tensor_parallel_size,
                dtype="bfloat16",
                trust_remote_code=True,
            )

            result = evaluate_checkpoint(
                llm, sampling_params, None, prompts, eval_sample,
                desc=f"step {step}",
            )
            results["conditions"][cond][f"step_{step}"] = result

            del llm
            gc.collect()
            torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 70)
    print("CHECKPOINT CURVE SUMMARY (prompt accuracy)")
    print("=" * 70)
    header = f"{'Step':>6}"
    for cond in args.conditions:
        header += f"  {cond:>16}"
    print(header)
    for step in args.checkpoint_steps:
        row = f"{step:>6}"
        for cond in args.conditions:
            acc = results["conditions"].get(cond, {}).get(f"step_{step}", {}).get("prompt_accuracy")
            row += f"  {acc:.2%}" if acc is not None else f"  {'N/A':>7}"
        print(row)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
