"""
Generate lambda sweep plots for XTREME isolated experiments.

Creates 5 individual plots (one per task) + 1 combined figure,
each showing lambda on x-axis vs metric on y-axis (English only).
Also writes xtreme_isolated_results.md with a comprehensive report.

Usage:
  python scripts/gen_xtreme_isolated_fig.py \\
      --results experiments/xtreme_isolated/results.json \\
      --output_dir experiments/xtreme_isolated/figures
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TASKS = ["nli", "pa", "qa", "ner", "pos"]

TASK_DISPLAY = {
    "nli": "NLI (XNLI-en)",
    "pa":  "Paraphrase (PAWS-X en)",
    "qa":  "QA (MLQA en)",
    "ner": "NER (WikiANN en)",
    "pos": "POS (UDPOS en)",
}

# Primary metric for each task
TASK_METRIC = {
    "nli": "accuracy",
    "pa":  "accuracy",
    "qa":  "f1",
    "ner": "f1",
    "pos": "accuracy",
}

LAMBDAS = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
LAM_KEYS = [str(int(round(l * 100))) for l in LAMBDAS]  # "0","1","5","10","25","50","75","100"


def extract_metric(results, task, lam_key, lang="en"):
    """Extract the primary metric value for a condition."""
    condition = f"{task}_lam{lam_key}"
    metric_name = TASK_METRIC[task]
    try:
        val = results[condition][task][lang][metric_name]
        # Convert 0-1 float to percentage
        return float(val) * 100 if val is not None else None
    except (KeyError, TypeError):
        return None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results",    type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def plot_task(task, lambdas, values, output_path):
    """Generate a single-task lambda sweep plot."""
    metric = TASK_METRIC[task]
    valid = [(l, v) for l, v in zip(lambdas, values) if v is not None]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.set_facecolor("#f8f8f8")

    if valid:
        x, y = zip(*valid)
        ax.plot(x, y, "o-", color="steelblue", linewidth=2.5,
                markersize=9, zorder=3, label="Distilled")

        # SFT baseline (λ=0)
        sft_val = y[0] if x[0] == 0 else None
        if sft_val is not None:
            ax.axhline(sft_val, color="gray", linestyle="--", linewidth=1.5,
                       alpha=0.7, label=f"SFT (λ=0): {sft_val:.1f}%")

        # Best lambda
        best_i = int(np.argmax(y))
        ax.scatter([x[best_i]], [y[best_i]], s=180, color="red",
                   zorder=4, label=f"Best λ={x[best_i]}: {y[best_i]:.1f}%")

    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xticks(LAMBDAS)
    ax.set_xticklabels([str(l) for l in LAMBDAS], fontsize=9)
    ax.set_xlabel("λ (distillation weight)", fontsize=11)
    ax.set_ylabel(f"{metric.upper()} (%)", fontsize=11)
    ax.set_title(TASK_DISPLAY[task], fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4, color="white")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.results) as f:
        results = json.load(f)

    all_lambdas = {}
    all_values  = {}
    best_info   = {}

    for task in TASKS:
        values = [extract_metric(results, task, key) for key in LAM_KEYS]
        all_lambdas[task] = LAMBDAS
        all_values[task]  = values

        valid = [(l, v) for l, v in zip(LAMBDAS, values) if v is not None]
        if valid:
            x, y = zip(*valid)
            best_i = int(np.argmax(y))
            sft_val = y[0] if x[0] == 0.0 else None
            best_info[task] = {
                "best_lambda":   x[best_i],
                "best_value":    y[best_i],
                "sft_value":     sft_val,
                "gain_over_sft": round(y[best_i] - sft_val, 2) if sft_val else None,
                "metric":        TASK_METRIC[task],
                "all_lambdas":   list(x),
                "all_values":    list(y),
            }

        # Individual plot
        plot_task(task, LAMBDAS, values, output_dir / f"{task}_lambda_sweep.png")
        print(f"Saved {task} plot")

    # Combined 1×5 figure
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    fig.suptitle(
        "XTREME Isolated Lambda Sweep — English, Qwen3-1.7B, 200 steps",
        fontsize=13, fontweight="bold"
    )
    for ax, task in zip(axes, TASKS):
        values = all_values[task]
        valid  = [(l, v) for l, v in zip(LAMBDAS, values) if v is not None]
        if valid:
            x, y = zip(*valid)
            ax.plot(x, y, "o-", color="steelblue", linewidth=2, markersize=7)
            if x[0] == 0.0:
                ax.axhline(y[0], color="gray", linestyle="--", linewidth=1, alpha=0.6)
            best_i = int(np.argmax(y))
            ax.scatter([x[best_i]], [y[best_i]], s=120, color="red", zorder=4)
        ax.set_xscale("symlog", linthresh=0.01)
        ax.set_xticks([0, 0.05, 0.25, 1.0])
        ax.set_xticklabels(["0", "0.05", "0.25", "1"], fontsize=8)
        ax.set_title(TASK_DISPLAY[task], fontsize=9)
        ax.set_xlabel("λ", fontsize=9)
        ax.set_ylabel(TASK_METRIC[task].upper(), fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "lambda_sweep_combined.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved combined plot")

    # Save best lambdas JSON
    with open(output_dir / "best_lambdas.json", "w") as f:
        json.dump(best_info, f, indent=2)

    # Write markdown report
    md_lines = [
        "# XTREME Isolated Lambda Sweep Results",
        "",
        "**Setup**: Qwen3-1.7B, 200 training steps, English-only, single-task isolation.",
        "**Lambda sweep**: λ ∈ {0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0}",
        "",
        "## Summary Table",
        "",
        "| Task | SFT (λ=0) | Best λ | Best Score | Gain | Metric |",
        "|------|-----------|--------|------------|------|--------|",
    ]
    for task in TASKS:
        info = best_info.get(task, {})
        sft  = f"{info.get('sft_value', 'N/A'):.1f}" if isinstance(info.get('sft_value'), float) else "N/A"
        best = f"{info.get('best_value', 'N/A'):.1f}" if isinstance(info.get('best_value'), float) else "N/A"
        gain = f"+{info.get('gain_over_sft', 0):.1f}" if isinstance(info.get('gain_over_sft'), float) else "N/A"
        lam  = info.get('best_lambda', 'N/A')
        metric = info.get('metric', 'N/A').upper()
        md_lines.append(f"| {task.upper()} | {sft}% | {lam} | {best}% | {gain}pp | {metric} |")

    md_lines += [
        "",
        "## Per-Task Lambda Sensitivity",
        "",
    ]
    for task in TASKS:
        info = best_info.get(task, {})
        md_lines += [
            f"### {TASK_DISPLAY[task]}",
            "",
            f"![{task} lambda sweep](figures/{task}_lambda_sweep.png)",
            "",
        ]
        if "all_lambdas" in info:
            md_lines += ["| λ | Score |", "|---|---|"]
            for l, v in zip(info["all_lambdas"], info["all_values"]):
                marker = " ← best" if l == info["best_lambda"] else ""
                md_lines.append(f"| {l} | {v:.1f}%{marker} |")
        md_lines.append("")

    md_lines += [
        "## Analysis & Ideal Lambdas",
        "",
        "*(Generated automatically — see `scripts/gen_xtreme_isolated_fig.py`)*",
        "",
    ]
    for task in TASKS:
        info = best_info.get(task, {})
        if not info:
            continue
        lam  = info["best_lambda"]
        gain = info.get("gain_over_sft", 0)
        if gain and gain > 0:
            sensitivity = "low" if lam >= 0.25 else ("moderate" if lam >= 0.05 else "high")
            md_lines.append(
                f"- **{task.upper()}**: Best λ={lam} (+{gain:.1f}pp over SFT). "
                f"Sensitivity: {sensitivity} — "
                + ("very robust to high λ." if lam >= 0.25 else
                   "prefers mild distillation." if lam >= 0.05 else
                   "needs careful tuning.")
            )
        else:
            md_lines.append(
                f"- **{task.upper()}**: Distillation did not help (best=SFT). "
                f"Consider task-specific factors."
            )
    md_lines.append("")

    report_path = Path("xtreme_isolated_results.md")
    report_path.write_text("\n".join(md_lines))
    print(f"Saved report to {report_path}")

    # Console summary
    print("\n=== Best Lambdas Summary ===")
    for task in TASKS:
        info = best_info.get(task, {})
        if not info:
            print(f"  {task.upper()}: no data")
            continue
        gain = info.get("gain_over_sft", 0) or 0
        print(
            f"  {task.upper():4s}: λ={info['best_lambda']:<5}  "
            f"{info['metric'].upper()}={info['best_value']:.1f}%  "
            f"(+{gain:.1f}pp over SFT)"
        )


if __name__ == "__main__":
    main()
