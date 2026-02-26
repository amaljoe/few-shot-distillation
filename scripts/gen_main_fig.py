"""
Generate the main comparison figure: SFT vs. Few-Shot Distillation (V1).
All results on full GSM8K test set (1319 examples).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

steps = [200, 400, 600, 800, 1000]
sft   = [61.79, 64.29, 63.68, 63.46, 62.17]
v1    = [72.40, 71.49, 72.71, 71.72, 71.04]

fig, ax = plt.subplots(figsize=(9, 5.5))

ax.plot(steps, sft, color="#2196F3", linewidth=2.2, marker="o", markersize=6,
        label="LoRA SFT (zero-shot fine-tuning)")
ax.plot(steps, v1,  color="#E91E63", linewidth=2.2, marker="s", markersize=6,
        label="LoRA SFT + Few-Shot Distillation (ours)")

ax.axhline(47.08, color="#FF9800", linestyle="--", linewidth=1.5,
           label="8-shot in-context learning (47.08%)")
ax.axhline(26.08, color="#9E9E9E", linestyle=":", linewidth=1.5,
           label="Base model, 0-shot (26.08%)")

# Annotate peaks
ax.annotate("72.71%", xy=(600, 72.71), xytext=(660, 75.5),
            arrowprops=dict(arrowstyle="->", color="#E91E63", lw=1.5),
            color="#E91E63", fontsize=10, fontweight="bold")
ax.annotate("64.29%", xy=(400, 64.29), xytext=(460, 67.2),
            arrowprops=dict(arrowstyle="->", color="#2196F3", lw=1.5),
            color="#2196F3", fontsize=10)

ax.set_xlabel("Training step", fontsize=12)
ax.set_ylabel("GSM8K accuracy (%)", fontsize=12)
ax.set_title("GSM8K Accuracy — Qwen3-1.7B  (full test set, 1319 examples)",
             fontsize=13, fontweight="bold")
ax.legend(frameon=False, fontsize=10, loc="lower right")
ax.set_ylim(18, 83)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax.set_xticks(steps)

fig.tight_layout()
out = Path("assets/main_comparison.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(str(out), dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
