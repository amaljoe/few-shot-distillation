"""
Generate cross-model summary comparison figure.

Grouped bar chart: 4 models × 4 methods.
- Groups (x-axis): Qwen3-1.7B, Qwen3-8B, Llama-3.2-3B-Instruct, Gemma-3-270M
- Bars per group: Base 0-shot, 8-shot ICL, LoRA SFT (best ckpt), LoRA+Distillation (best ckpt)

Data: hardcoded from confirmed eval results on full GSM8K test set (1319 examples).

Output: assets/summary_comparison.png

Usage:
  python scripts/gen_summary_fig.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ── Confirmed results (full GSM8K test set, 1319 examples) ───────────────────
# Columns: Base 0-shot, 8-shot ICL, LoRA SFT (best), LoRA+Distillation (best)
DATA = {
    "Qwen3-1.7B":            [26.08, 47.08, 64.29, 72.71],
    "Qwen3-8B":              [34.72, 80.14, 82.49, 90.37],
    "Llama-3.2-3B-Instruct": [62.93, 71.42, 61.03, 66.79],
    "Gemma-3-270M":          [ 1.59,  1.59,  4.47,  2.43],
}

METHOD_LABELS = ["Base 0-shot", "8-shot ICL", "LoRA SFT", "LoRA + Distillation"]

# Colors matching per-model figures (gen_main_fig.py, gen_llama_fig.py, gen_gemma_fig.py)
COLORS = ["#9E9E9E", "#FF9800", "#2196F3", "#E91E63"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def main():
    models = list(DATA.keys())
    n_models = len(models)
    n_methods = len(METHOD_LABELS)

    x = np.arange(n_models)
    total_width = 0.72
    bar_width = total_width / n_methods
    offsets = np.linspace(-(total_width - bar_width) / 2,
                           (total_width - bar_width) / 2,
                           n_methods)

    fig, ax = plt.subplots(figsize=(11, 5.5))

    for i, (method, color) in enumerate(zip(METHOD_LABELS, COLORS)):
        vals = [DATA[m][i] for m in models]
        bars = ax.bar(x + offsets[i], vals, width=bar_width,
                      color=color, label=method,
                      edgecolor="white", linewidth=0.8, zorder=3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.8,
                    f"{val:.1f}",
                    ha="center", va="bottom", fontsize=7.5, color="#333")

    ax.set_ylabel("GSM8K accuracy (%)", fontsize=12)
    ax.set_title("Few-Shot Distillation — GSM8K Results Across All Models\n"
                 "(full test set, 1319 examples, zero-shot inference)",
                 fontsize=13, fontweight="bold", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.legend(fontsize=10, loc="upper left", framealpha=0.9,
              ncol=2, columnspacing=1.0)
    ax.grid(True, axis="y", alpha=0.3, zorder=0)

    fig.tight_layout()

    out = Path("assets/summary_comparison.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
