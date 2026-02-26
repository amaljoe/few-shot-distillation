"""
Generate CE loss comparison: SFT baseline vs. Few-Shot Distillation (V1).
Shows whether distillation leads to faster/better convergence.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def read_tb(tb_dir, tag):
    ea = EventAccumulator(tb_dir)
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    if tag not in tags:
        print(f"  [warn] tag '{tag}' not in {tb_dir}. Available: {tags}")
        return []
    return [(e.step, e.value) for e in ea.Scalars(tag)]


def merge_tb(dirs, tag):
    """Merge multiple TB log dirs (e.g. two DDP processes) by taking first occurrence."""
    merged = {}
    for d in dirs:
        for step, val in read_tb(d, tag):
            if step not in merged:
                merged[step] = val
    return sorted(merged.items())


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# ── load CE loss from TensorBoard ────────────────────────────────────────────
baseline_tb = "experiments/poc/baseline/baseline/tb_logs"
v1_tb       = "experiments/online_v1/online_v1/tb_logs"

baseline_loss = merge_tb([baseline_tb], "train/ce_loss")
v1_loss       = merge_tb([v1_tb],       "train/ce_loss")

print(f"Baseline steps: {len(baseline_loss)}, V1 steps: {len(v1_loss)}")
if baseline_loss:
    print(f"  Baseline step range: {baseline_loss[0][0]}–{baseline_loss[-1][0]}")
if v1_loss:
    print(f"  V1 step range: {v1_loss[0][0]}–{v1_loss[-1][0]}")

fig, ax = plt.subplots(figsize=(9, 5))

if baseline_loss:
    bsteps, bvals = zip(*baseline_loss)
    ax.plot(bsteps, bvals, color="#2196F3", linewidth=1.8, alpha=0.9,
            label="LoRA SFT (CE loss only)")

if v1_loss:
    vsteps, vvals = zip(*v1_loss)
    ax.plot(vsteps, vvals, color="#E91E63", linewidth=1.8, alpha=0.9,
            label="LoRA SFT + Few-Shot Distillation (CE loss)")

ax.set_xlabel("Training step", fontsize=12)
ax.set_ylabel("CE loss", fontsize=12)
ax.set_title("Training CE Loss — SFT vs. Few-Shot Distillation\n(Qwen3-1.7B, GSM8K)",
             fontsize=13, fontweight="bold")
ax.legend(frameon=False, fontsize=10)

fig.tight_layout()
out = Path("assets/loss_comparison.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(str(out), dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
