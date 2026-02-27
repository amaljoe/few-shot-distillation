"""
Generate causal ablation comparison figures for the critique response.

Produces two figures:
  assets/ablation_curve.png  — accuracy vs training step for all 4 conditions
  assets/ablation_bar.png    — bar chart of best accuracy per condition

Usage:
  python scripts/gen_ablation_fig.py

Input files (all expected to exist after eval_ablations.sh runs):
  experiments/qwen1b7/baseline_full_eval.json   -- SFT baseline curve
  experiments/online_v1_full_eval.json          -- 8-shot teacher distill curve
  experiments/ablations/ablation_eval.json      -- 0-shot + shuffled ablations
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STEPS = [200, 400, 600, 800, 1000]

COLORS = {
    "SFT (CE only)":              "#4878d0",
    "KD — 0-shot teacher":        "#ee854a",
    "KD — shuffled answers":      "#6acc65",
    "KD — 8-shot teacher (ours)": "#d65f5f",
}


def load_step_accs(json_path: str, condition_key: str) -> dict:
    with open(json_path) as f:
        data = json.load(f)
    cond = data["conditions"][condition_key]
    return {int(k.replace("step_", "")): v["accuracy"] for k, v in cond.items()}


def main():
    out_dir = Path("assets")
    out_dir.mkdir(exist_ok=True)

    sft_accs      = load_step_accs("experiments/qwen1b7/baseline_full_eval.json", "baseline")
    fewshot_accs  = load_step_accs("experiments/online_v1_full_eval.json",         "online_v1")
    zeroshot_accs = load_step_accs("experiments/ablations/ablation_eval.json",     "zeroshot_teacher")
    shuffled_accs = load_step_accs("experiments/ablations/ablation_eval.json",     "shuffled_answers")

    all_conditions = {
        "SFT (CE only)":              sft_accs,
        "KD — 0-shot teacher":        zeroshot_accs,
        "KD — shuffled answers":      shuffled_accs,
        "KD — 8-shot teacher (ours)": fewshot_accs,
    }

    # ── Figure 1: Checkpoint accuracy curve ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    for label, accs in all_conditions.items():
        xs = sorted(accs.keys())
        ys = [accs[s] * 100 for s in xs]
        bold = "ours" in label
        ax.plot(xs, ys,
                label=label,
                color=COLORS[label],
                linestyle="--" if bold else "-",
                linewidth=2.4 if bold else 1.8,
                marker="o", markersize=5.5,
                zorder=3 if bold else 2)

    ax.set_xlabel("Training step", fontsize=11)
    ax.set_ylabel("GSM8K accuracy (%)", fontsize=11)
    ax.set_title("Causal Ablation: What Drives the Distillation Gain?", fontsize=12, pad=8)
    ax.set_xticks(STEPS)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ymin = min(min(a.values()) for a in all_conditions.values()) * 100 - 3
    ymax = max(max(a.values()) for a in all_conditions.values()) * 100 + 3
    ax.set_ylim(ymin, ymax)
    plt.tight_layout()
    curve_path = out_dir / "ablation_curve.png"
    fig.savefig(curve_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {curve_path}")

    # ── Figure 2: Bar chart of best accuracy ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    labels = list(all_conditions.keys())
    best_accs = [max(accs.values()) * 100 for accs in all_conditions.values()]
    colors = [COLORS[l] for l in labels]
    x = np.arange(len(labels))

    bars = ax.bar(x, best_accs, color=colors, width=0.55, edgecolor="white", linewidth=1.2)
    for bar, acc in zip(bars, best_accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.25,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=10.5, fontweight="bold")

    ax.set_ylabel("Best GSM8K accuracy (%)", fontsize=11)
    ax.set_title("Ablation: Best Checkpoint Accuracy by Teacher Condition", fontsize=12, pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["SFT\n(CE only)", "KD\n0-shot teacher", "KD\nshuffled\nanswers",
         "KD\n8-shot teacher\n(ours)"],
        fontsize=9,
    )
    ymin2 = min(best_accs) - 5
    ymax2 = max(best_accs) + 5
    ax.set_ylim(ymin2, ymax2)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    bar_path = out_dir / "ablation_bar.png"
    fig.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {bar_path}")

    # ── Print summary table ───────────────────────────────────────────────────
    print()
    print("Ablation Summary (Qwen3-1.7B, full GSM8K test set, 1319 examples)")
    print("=" * 65)
    print(f"{'Condition':<35} {'Best':>7}  {'@200':>7}  {'@1000':>7}")
    print("-" * 65)
    for label, accs in all_conditions.items():
        best = max(accs.values()) * 100
        at200  = accs.get(200, 0) * 100
        at1000 = accs.get(1000, 0) * 100
        print(f"{label:<35} {best:>6.2f}%  {at200:>6.2f}%  {at1000:>6.2f}%")


if __name__ == "__main__":
    main()
