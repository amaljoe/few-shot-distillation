"""
Generate the Llama-3.2-3B-Instruct comparison figure.
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
sft   = [60.73, 61.03, 60.96, 57.54, 56.18]
v1    = [66.79, 65.81, 65.35, 65.05, 65.66]

fig, ax = plt.subplots(figsize=(9, 5.5))

ax.plot(steps, sft, color="#2196F3", linewidth=2.2, marker="o", markersize=6,
        label="LoRA SFT (zero-shot fine-tuning)")
ax.plot(steps, v1,  color="#E91E63", linewidth=2.2, marker="s", markersize=6,
        label="LoRA SFT + Few-Shot Distillation (ours)")

ax.axhline(71.42, color="#FF9800", linestyle="--", linewidth=1.5,
           label="8-shot in-context learning (71.42%)")
ax.axhline(62.93, color="#9E9E9E", linestyle=":", linewidth=1.5,
           label="Base model, 0-shot (62.93%)")

ax.annotate("66.79%", xy=(200, 66.79), xytext=(270, 69.0),
            arrowprops=dict(arrowstyle="->", color="#E91E63", lw=1.5),
            color="#E91E63", fontsize=10, fontweight="bold")
ax.annotate("61.03%", xy=(400, 61.03), xytext=(460, 63.5),
            arrowprops=dict(arrowstyle="->", color="#2196F3", lw=1.5),
            color="#2196F3", fontsize=10)

ax.set_xlabel("Training step", fontsize=12)
ax.set_ylabel("GSM8K accuracy (%)", fontsize=12)
ax.set_title("GSM8K Accuracy — Llama-3.2-3B-Instruct  (full test set, 1319 examples)",
             fontsize=13, fontweight="bold")
ax.legend(frameon=False, fontsize=10, loc="upper right")
ax.set_ylim(50, 78)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax.set_xticks(steps)

fig.tight_layout()
out = Path("assets/llama_comparison.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(str(out), dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
