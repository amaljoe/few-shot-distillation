"""Generate ablation figures from the merged all_ablations.json."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

with open("experiments/ablations/all_ablations.json") as f:
    data = json.load(f)

STEPS = [200, 400, 600, 800, 1000]
LABELS = {
    "baseline":         "SFT (CE only)",
    "zeroshot_teacher": "KD — 0-shot teacher",
    "shuffled_answers": "KD — shuffled answers",
    "online_v1":        "KD — 8-shot teacher (ours)",
}
COLORS = {
    "baseline":         "#4878d0",
    "zeroshot_teacher": "#ee854a",
    "shuffled_answers": "#6acc65",
    "online_v1":        "#d65f5f",
}

Path("assets").mkdir(exist_ok=True)

# Figure 1: Checkpoint accuracy curve
fig, ax = plt.subplots(figsize=(7.5, 4.8))
for cond, label in LABELS.items():
    accs = data["conditions"][cond]
    xs = sorted([int(k.replace("step_","")) for k in accs])
    ys = [accs[f"step_{s}"]["accuracy"]*100 for s in xs]
    bold = cond == "online_v1"
    ax.plot(xs, ys, label=label, color=COLORS[cond],
            linestyle="--" if bold else "-",
            linewidth=2.4 if bold else 1.8,
            marker="o", markersize=5.5, zorder=3 if bold else 2)

ax.set_xlabel("Training step", fontsize=11)
ax.set_ylabel("GSM8K accuracy (%)", fontsize=11)
ax.set_title("Causal Ablation: What Drives the Distillation Gain?", fontsize=12, pad=8)
ax.set_xticks(STEPS)
ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_ylim(59, 77)
plt.tight_layout()
fig.savefig("assets/ablation_curve.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: assets/ablation_curve.png")

# Figure 2: Bar chart — best accuracy per condition
fig, ax = plt.subplots(figsize=(7.5, 4.8))
conds = list(LABELS.keys())
best_accs = [max(data["conditions"][c][f"step_{s}"]["accuracy"] for s in STEPS)*100 for c in conds]
colors = [COLORS[c] for c in conds]
x = np.arange(len(conds))
bars = ax.bar(x, best_accs, color=colors, width=0.55, edgecolor="white", linewidth=1.2)
for bar, acc in zip(bars, best_accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f"{acc:.1f}%", ha="center", va="bottom", fontsize=10.5, fontweight="bold")

ax.set_ylabel("Best GSM8K accuracy (%)", fontsize=11)
ax.set_title("Ablation: Best Checkpoint Accuracy by Teacher Condition\n(Qwen3-1.7B, full GSM8K test set, N=1319)", fontsize=11, pad=8)
ax.set_xticks(x)
ax.set_xticklabels(["SFT\n(CE only)", "KD\n0-shot teacher", "KD\nshuffled\nanswers",
                    "KD\n8-shot teacher\n(ours)"], fontsize=9)
ax.set_ylim(59, 79)
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig("assets/ablation_bar.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: assets/ablation_bar.png")

# Summary table
print()
print("Ablation Summary (Qwen3-1.7B, N=1319)")
print("="*72)
baseline_best = max(data["conditions"]["baseline"][f"step_{s}"]["accuracy"] for s in STEPS)*100
header = f"{'Condition':<32} {'@200':>7} {'@400':>7} {'@600':>7} {'@800':>7} {'@1000':>7} {'Best':>7}"
print(header)
print("-"*72)
for cond, label in LABELS.items():
    accs = data["conditions"][cond]
    vals = [accs[f"step_{s}"]["accuracy"]*100 for s in STEPS]
    best = max(vals)
    print(f"{label:<32} " + " ".join(f"{v:>6.2f}%" for v in vals) + f" {best:>6.2f}%")
