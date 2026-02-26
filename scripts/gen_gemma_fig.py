"""
Generate figures for Gemma-3-270M experiments.

Produces:
  assets/gemma_comparison.png   — bar chart: base / ICL / SFT / distill / 0-shot-KD
  assets/gemma_curve.png        — checkpoint accuracy curve for all trained conditions
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Load ICL baseline results ─────────────────────────────────────────────────
with open("experiments/gemma270m/icl_eval.json") as f:
    icl = json.load(f)

acc_0shot = icl["evaluations"]["0_shot"]["accuracy"] * 100
acc_8shot = icl["evaluations"]["8_shot"]["accuracy"] * 100

# ── Load checkpoint curve results ─────────────────────────────────────────────
with open("experiments/gemma270m/all_conditions_eval.json") as f:
    ckpt = json.load(f)

STEPS = [200, 400, 600, 800, 1000]
LABELS = {
    "baseline":         "SFT (CE only)",
    "online_v1":        "SFT + 8-shot Distillation (ours)",
    "zeroshot_teacher": "SFT + 0-shot KD (control)",
}
COLORS = {
    "baseline":         "#4878d0",
    "online_v1":        "#d65f5f",
    "zeroshot_teacher": "#ee854a",
}

Path("assets").mkdir(exist_ok=True)

# ── Figure 1: Bar chart comparison ───────────────────────────────────────────
# Best accuracy per condition
best_accs = {}
for cond in LABELS:
    if cond in ckpt["conditions"]:
        best_accs[cond] = max(
            ckpt["conditions"][cond][f"step_{s}"]["accuracy"] * 100
            for s in STEPS
            if f"step_{s}" in ckpt["conditions"][cond]
        )

fig, ax = plt.subplots(figsize=(9, 5))
bar_labels = ["Base\n0-shot", "8-shot\nICL", "SFT\n(CE only)",
              "SFT +\n8-shot Distill\n(ours)", "SFT +\n0-shot KD\n(control)"]
bar_values = [
    acc_0shot,
    acc_8shot,
    best_accs.get("baseline", 0),
    best_accs.get("online_v1", 0),
    best_accs.get("zeroshot_teacher", 0),
]
bar_colors = ["#888888", "#888888", "#4878d0", "#d65f5f", "#ee854a"]
x = np.arange(len(bar_labels))
bars = ax.bar(x, bar_values, color=bar_colors, width=0.55,
              edgecolor="white", linewidth=1.2)
for bar, acc in zip(bars, bar_values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
            f"{acc:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.set_ylabel("GSM8K accuracy (%)", fontsize=11)
ax.set_title("Gemma-3-270M: Full Fine-Tuning vs Distillation\n"
             "(GSM8K test set, N=1319, full fine-tuning — no LoRA)", fontsize=11, pad=8)
ax.set_xticks(x)
ax.set_xticklabels(bar_labels, fontsize=9)
ax.set_ylim(0, max(bar_values) * 1.2 + 3)
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig("assets/gemma_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: assets/gemma_comparison.png")

# ── Figure 2: Checkpoint accuracy curve ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.5, 4.8))
for cond, label in LABELS.items():
    if cond not in ckpt["conditions"]:
        continue
    accs = ckpt["conditions"][cond]
    xs = sorted([int(k.replace("step_", "")) for k in accs])
    ys = [accs[f"step_{s}"]["accuracy"] * 100 for s in xs]
    bold = cond == "online_v1"
    ax.plot(xs, ys, label=label, color=COLORS[cond],
            linestyle="--" if bold else "-",
            linewidth=2.4 if bold else 1.8,
            marker="o", markersize=5.5, zorder=3 if bold else 2)

ax.set_xlabel("Training step", fontsize=11)
ax.set_ylabel("GSM8K accuracy (%)", fontsize=11)
ax.set_title("Gemma-3-270M: Checkpoint Accuracy Curve", fontsize=12, pad=8)
ax.set_xticks(STEPS)
ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig("assets/gemma_curve.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: assets/gemma_curve.png")

# ── Summary table ─────────────────────────────────────────────────────────────
print()
print("Gemma-3-270M Results (N=1319)")
print("=" * 72)
print(f"{'Method':<36} {'@200':>7} {'@400':>7} {'@600':>7} {'@800':>7} {'@1000':>7} {'Best':>7}")
print("-" * 72)
print(f"{'Base model, 0-shot':<36} {'—':>7} {'—':>7} {'—':>7} {'—':>7} {'—':>7} {acc_0shot:>6.2f}%")
print(f"{'8-shot in-context learning':<36} {'—':>7} {'—':>7} {'—':>7} {'—':>7} {'—':>7} {acc_8shot:>6.2f}%")
for cond, label in LABELS.items():
    if cond not in ckpt["conditions"]:
        continue
    accs = ckpt["conditions"][cond]
    vals = [accs.get(f"step_{s}", {}).get("accuracy", 0) * 100 for s in STEPS]
    best = max(vals)
    print(f"{label:<36} " + " ".join(f"{v:>6.2f}%" for v in vals) + f" {best:>6.2f}%")
