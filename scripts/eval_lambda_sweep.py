"""
Evaluate a single lambda-sweep checkpoint on one task, OR compile a summary table.

Two modes:

  # Evaluate one checkpoint
  python scripts/eval_lambda_sweep.py \\
      --model meta-llama/Llama-3.2-3B-Instruct \\
      --checkpoint experiments/xtreme/lambda_sweep/pos_lam005/pos_lam005/final \\
      --task pos \\
      --condition pos_lam005 \\
      --output experiments/xtreme/lambda_sweep_results.json \\
      --tensor_parallel_size 4

  # Compile summary table (run after all sweep conditions are done)
  python scripts/eval_lambda_sweep.py \\
      --summarise \\
      --trained_json experiments/xtreme/llama3b_trained.json \\
      --sweep_json   experiments/xtreme/lambda_sweep_results.json \\
      --output       experiments/xtreme/lambda_sweep_summary.md
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ──────────────────────────────────────────────────────────────────────────────
# Single-checkpoint evaluation mode
# ──────────────────────────────────────────────────────────────────────────────

def eval_checkpoint(args):
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer
    from src.data.xtreme_loader import (
        TASK_LANGUAGES, TASK_MAX_NEW_TOKENS,
        XTREMEEvalDataset, parse_output, compute_task_metric,
    )

    ckpt_path = Path(args.checkpoint)
    is_lora = (ckpt_path / "adapter_config.json").exists()

    print(f"\nEvaluating: {args.condition}")
    print(f"  Checkpoint: {ckpt_path}  ({'LoRA' if is_lora else 'full-FT'})")
    print(f"  Task: {args.task}  |  TP: {args.tensor_parallel_size}")

    # Tokenizer for building prompts (always from base model name)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Load vLLM
    if is_lora:
        llm = LLM(
            model=args.model,
            dtype="bfloat16",
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            enable_lora=True,
            max_lora_rank=64,
            enable_prefix_caching=True,
        )
        lora_request = LoRARequest("adapter", 1, str(ckpt_path))
    else:
        llm = LLM(
            model=str(ckpt_path),
            dtype="bfloat16",
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            enable_prefix_caching=True,
        )
        lora_request = None

    sp = SamplingParams(temperature=0.0, max_tokens=TASK_MAX_NEW_TOKENS.get(args.task, 64))
    langs = TASK_LANGUAGES.get(args.task, [])
    n_samples = args.n_samples if args.n_samples > 0 else 500

    task_results = {}
    for lang in langs:
        eval_ds = XTREMEEvalDataset(
            tokenizer=tokenizer,
            task=args.task,
            lang=lang,
            condition="base",
            n_samples=n_samples,
        )
        if len(eval_ds) == 0:
            print(f"  {lang}: no data, skipping")
            continue

        prompts = [eval_ds.get_prompt(i) for i in range(len(eval_ds))]
        outputs = llm.generate(prompts, sp, lora_request=lora_request)
        preds_raw = [o.outputs[0].text for o in outputs]
        golds = [eval_ds.get_gold(i) for i in range(len(eval_ds))]

        # Replicate evaluate_one scoring logic from eval_xtreme_inference.py
        if args.task in ("ner", "pos"):
            parsed_preds, parsed_golds = [], []
            for i, (raw, gold) in enumerate(zip(preds_raw, golds)):
                n_tokens = len(eval_ds.examples[i].get("tokens", []))
                parsed_preds.append(parse_output(args.task, raw, n_tokens))
                if isinstance(gold, str):
                    gold = gold.split()
                pad = "O" if args.task == "ner" else "X"
                if n_tokens > 0:
                    gold = (gold + [pad] * n_tokens)[:n_tokens]
                parsed_golds.append(gold)
            metrics = compute_task_metric(args.task, parsed_preds, parsed_golds)
        else:
            parsed = [parse_output(args.task, r) for r in preds_raw]
            metrics = compute_task_metric(args.task, parsed, golds)

        task_results[lang] = metrics
        key_metric = metrics.get("f1", metrics.get("accuracy", 0))
        print(f"  {lang}: {key_metric*100:.1f}%")

    del llm  # free GPU memory

    # Load or init the output JSON
    out_path = Path(args.output)
    if out_path.exists():
        with open(out_path) as f:
            all_results = json.load(f)
    else:
        all_results = {}

    all_results[args.condition] = {args.task: task_results}

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Saved → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Summary table mode
# ──────────────────────────────────────────────────────────────────────────────

def get_task_lang_score(data: dict, condition: str, task: str, lang: str) -> float | None:
    """Extract a scalar score from any results dict structure."""
    try:
        m = data[condition][task][lang]
        if m is None or (isinstance(m, dict) and "error" in m):
            return None
        if task == "qa":
            return round(m.get("f1", 0) * 100, 1)
        return round(m.get("accuracy", m.get("f1", 0)) * 100, 1)
    except (KeyError, TypeError):
        return None


def task_avg(data: dict, condition: str, task: str, langs: list) -> float | None:
    scores = [get_task_lang_score(data, condition, task, l) for l in langs]
    valid = [s for s in scores if s is not None]
    return round(sum(valid) / len(valid), 1) if valid else None


def summarise(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from src.data.xtreme_loader import TASK_LANGUAGES

    # Load results
    with open(args.trained_json) as f:
        trained = json.load(f)      # has finetuned, distilled (λ=0.5), control
    with open(args.sweep_json) as f:
        sweep = json.load(f)        # has pos_lam005, pos_lam010, etc.

    # Merge into one lookup dict, aliasing trained keys
    all_data: dict = {}
    # Baselines from trained_json
    for cond in ["finetuned", "distilled", "control"]:
        if cond in trained:
            all_data[cond] = trained[cond]
    # Sweep checkpoints
    all_data.update(sweep)

    lambdas = [0.05, 0.10, 0.20, 0.50]
    lambda_tags = ["005", "01", "02", "050"]
    tasks_of_interest = ["pos", "ner"]
    task_langs = {t: TASK_LANGUAGES.get(t, []) for t in tasks_of_interest}

    lines = [
        "# Lambda Sweep Results",
        "",
        "Testing λ ∈ {0.05, 0.10, 0.20} vs existing λ=0.50 (distilled) and SFT/Control baselines.",
        "Single-task training: **all** conditions train on only the target task for 200 steps (batch=8).",
        "All conditions share the same Llama-3.2-3B-Instruct base + LoRA (r=16) setup.",
        "",
        "> **Conditions:** all single-task, 200 steps, batch=8, LoRA r=16.  ",
        "> **Few-shot teacher** = teacher sees 5-shot context (the ICL signal).  ",
        "> **Zero-shot teacher** = teacher sees same zero-shot input as student (null ICL signal, tests regularisation only).",
        "",
    ]

    for task in tasks_of_interest:
        langs = task_langs[task]
        task_label = {"pos": "POS (Acc %)", "ner": "NER (F1 %)"}[task]
        lines.append(f"## {task_label}\n")

        # Header
        lang_cols = " | ".join(l.upper() for l in langs)
        lines.append(f"| Condition | Train | λ | {lang_cols} | Avg | Δ vs {task.upper()}-SFT |")
        lines.append(f"| :--- | :---: | :---: | " + " | ".join([":---:"] * len(langs)) + " | :---: | :---: |")

        # Use single-task SFT as the reference if available, else fall back to multi-task finetuned
        st_sft_cond = f"{task}_sft"
        sft_avg = task_avg(all_data, st_sft_cond, task, langs)
        ref_label = f"{task}_sft"
        if sft_avg is None:
            sft_avg = task_avg(all_data, "finetuned", task, langs)
            ref_label = "finetuned"

        def add_row(label, cond, train_tag, lam_str):
            avg = task_avg(all_data, cond, task, langs)
            scores = [get_task_lang_score(all_data, cond, task, l) for l in langs]
            vals = [f"{s:.1f}" if s is not None else "—" for s in scores]
            delta = ""
            if avg is not None and sft_avg is not None:
                d = round(avg - sft_avg, 1)
                sign = "+" if d >= 0 else ""
                delta = f"**{sign}{d}**" if abs(d) >= 1.0 else f"{sign}{d}"
            avg_str = f"**{avg:.1f}**" if avg is not None else "—"
            lines.append(f"| {label} | {train_tag} | {lam_str} | " + " | ".join(vals) + f" | {avg_str} | {delta} |")

        # SFT baseline (λ=0, reference)
        add_row(f"**SFT** (CE only, λ=0)", f"{task}_sft", "few-shot N/A", "0")

        # Few-shot distilled sweep: λ ∈ {0.05, 0.10, 0.20, 0.50}
        for lam, tag in zip(lambdas, lambda_tags):
            cond = f"{task}_lam{tag}"
            add_row(f"Few-shot teacher λ={lam:.2f}", cond, "few-shot", f"{lam:.2f}")

        # Zero-shot teacher ablations: same λ values, no ICL context
        add_row(f"Zero-shot teacher λ=0.05 *(ablation)*", f"{task}_ctrl_lam005", "zero-shot", "0.05")
        add_row(f"Zero-shot teacher λ=0.50",               f"{task}_ctrl",        "zero-shot", "0.50")

        lines.append("")

    # Analysis
    lines += [
        "## Interpretation\n",
        "The sweep answers two questions:\n",
        "1. **Is there any λ where distillation beats SFT on POS/NER?**  "
        "If yes, the teacher signal is valid but λ=0.5 is too aggressive — "
        "dynamic λ will reliably recover the gain.",
        "2. **What is the optimal λ range per task?**  "
        "The λ at which Distilled − SFT turns positive is the lower bound for "
        "the per-task dynamic λ target.",
        "",
        "Expected pattern based on our hypothesis:  "
        "`Acc(λ=0.05) > Acc(λ=0.10) > Acc(λ=0.20) > Acc(λ=0.50)` for POS/NER,  "
        "with at least λ=0.05 meeting or exceeding the SFT baseline.",
        "",
        "If the ordering holds, it validates the dynamic-λ direction and "
        "provides the empirical target λ values for Strategy 5 (task-type prior).",
    ]

    # Plot
    assets_dir = Path("assets/xtreme")
    assets_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, task in zip(axes, tasks_of_interest):
        langs = task_langs[task]
        task_label = {"pos": "POS Accuracy (%)", "ner": "NER F1 (%)"}[task]
        sft_avg = task_avg(all_data, "finetuned", task, langs)

        xs, ys = [], []
        for lam, tag in zip(lambdas, lambda_tags):
            cond = "distilled" if lam == 0.50 else f"{task}_lam{tag}"
            avg = task_avg(all_data, cond, task, langs)
            if avg is not None:
                xs.append(lam)
                ys.append(avg)

        ax.plot(xs, ys, "o-", color="#4CAF50", linewidth=2, markersize=8,
                label="Few-shot teacher (sweep)", zorder=3)

        # Zero-shot ctrl points at λ=0.05 and λ=0.50 (ablation line)
        zs_xs, zs_ys = [], []
        for lam, tag, cond_suffix in [(0.05, "005", "ctrl_lam005"), (0.50, "050", "ctrl")]:
            zs_cond = f"{task}_{cond_suffix}"
            zs_avg = task_avg(all_data, zs_cond, task, langs)
            if zs_avg is not None:
                zs_xs.append(lam)
                zs_ys.append(zs_avg)
        if zs_xs:
            ax.plot(zs_xs, zs_ys, "s--", color="#9C27B0", linewidth=1.5, markersize=7,
                    label="Zero-shot teacher (ablation)", zorder=2)
            for x, y in zip(zs_xs, zs_ys):
                ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                            xytext=(0, -14), ha="center", fontsize=8, color="#9C27B0")

        # SFT baseline
        st_sft_avg = task_avg(all_data, f"{task}_sft", task, langs)
        if st_sft_avg is not None:
            ax.axhline(st_sft_avg, color="#FF9800", linewidth=2.5, linestyle="--",
                       label=f"SFT ({st_sft_avg:.1f}%)", zorder=4)

        # Label each point
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=9, fontweight="bold")

        ax.set_xlabel("λ (distillation weight)", fontsize=11)
        ax.set_ylabel(task_label, fontsize=11)
        ax.set_title(f"λ Sweep — {task.upper()}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, framealpha=0.9)
        ax.set_xlim(-0.02, 0.58)
        ax.yaxis.grid(True, alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plot_path = assets_dir / "lambda_sweep.png"
    plt.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  Saved {plot_path}")

    lines.insert(4, f"![lambda_sweep](assets/xtreme/lambda_sweep.png)\n")

    out_path = Path(args.output)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n✓ Summary written to {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summarise", action="store_true",
                        help="Compile summary table from all sweep results")

    # Eval mode
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--task", type=str, default=None, choices=["pos", "ner", "nli", "pa", "qa"])
    parser.add_argument("--condition", type=str, default=None)
    parser.add_argument("--output", type=str, default="experiments/xtreme/lambda_sweep_results.json")
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--n_samples", type=int, default=0,
                        help="Max test samples per language (0=all)")

    # Summary mode
    parser.add_argument("--trained_json", type=str,
                        default="experiments/xtreme/llama3b_trained.json")
    parser.add_argument("--sweep_json", type=str,
                        default="experiments/xtreme/lambda_sweep_results.json")

    args = parser.parse_args()

    if args.summarise:
        summarise(args)
    else:
        if not args.checkpoint or not args.task or not args.condition:
            parser.error("--checkpoint, --task, and --condition are required in eval mode")
        eval_checkpoint(args)


if __name__ == "__main__":
    main()
