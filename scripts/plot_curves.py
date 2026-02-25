"""
Generate training loss and checkpoint accuracy curve plots.

Reads:
  - experiments/early_checkpoints/{baseline,distill}/*/tb_logs  (TensorBoard)
  - experiments/poc/{baseline,distill}/*/tb_logs               (TensorBoard)
  - experiments/ablations/checkpoint_curve/results_early.json  (early eval)
  - experiments/ablations/checkpoint_curve/results_full.json   (full eval)

Writes:
  - experiments/figures/loss_curve.png
  - experiments/figures/accuracy_curve.png

Usage:
  python scripts/plot_curves.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── styling ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})
COLORS = {"baseline": "#2196F3", "distill": "#E91E63"}
LABELS = {"baseline": "B — Baseline (CE only)", "distill": "C — Distillation (CE + MSE)"}

OUT_DIR = Path("experiments/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── TensorBoard reader ────────────────────────────────────────────────────────
def read_tb_scalars(tb_dir: Path, tag: str) -> list[tuple[int, float]]:
    """Return [(step, value), ...] for `tag` from a TensorBoard event file."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("  [warn] tensorboard package not available; skipping TB read")
        return []
    ea = EventAccumulator(str(tb_dir))
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return []
    return [(e.step, e.value) for e in ea.Scalars(tag)]


def load_tb_loss(condition: str) -> list[tuple[int, float]]:
    """Merge early-run and full-run TensorBoard logs for one condition."""
    points: dict[int, float] = {}

    # early run (steps 0-100)
    early_tb = Path(f"experiments/early_checkpoints/{condition}/{condition}/tb_logs")
    if early_tb.exists():
        for step, val in read_tb_scalars(early_tb, "train/ce_loss"):
            points[step] = val

    # full run (steps 0-1000)
    full_tb = Path(f"experiments/poc/{condition}/{condition}/tb_logs")
    if full_tb.exists():
        for step, val in read_tb_scalars(full_tb, "train/ce_loss"):
            if step not in points:  # prefer early-run values where they exist
                points[step] = val

    return sorted(points.items())


# ── accuracy loader ───────────────────────────────────────────────────────────
def load_accuracy(conditions: list[str]) -> dict[str, list[tuple[int, float]]]:
    """Merge early and full checkpoint eval JSONs into per-condition series."""
    result: dict[str, dict[int, float]] = {c: {} for c in conditions}

    for path in [
        Path("experiments/ablations/checkpoint_curve/results_early.json"),
        Path("experiments/ablations/checkpoint_curve/results_full.json"),
    ]:
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        for cond in conditions:
            cond_data = data.get("conditions", {}).get(cond, {})
            for key, vals in cond_data.items():
                step = int(key.replace("step_", ""))
                result[cond][step] = vals["accuracy"] * 100

    return {c: sorted(v.items()) for c, v in result.items()}


# ── plot 1: training loss ─────────────────────────────────────────────────────
def plot_loss():
    fig, ax = plt.subplots(figsize=(8, 4.5))

    any_data = False
    for cond in ["baseline", "distill"]:
        pts = load_tb_loss(cond)
        if not pts:
            print(f"  [warn] No TB loss data for {cond}")
            continue
        steps, vals = zip(*pts)
        ax.plot(steps, vals, color=COLORS[cond], label=LABELS[cond], linewidth=1.8)
        any_data = True

    if not any_data:
        print("  [skip] No loss data found; skipping loss plot")
        plt.close()
        return

    ax.set_xlabel("Training step")
    ax.set_ylabel("CE loss")
    ax.set_title("Training loss — Baseline vs Distillation")
    ax.legend(frameon=False)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(100))

    fig.tight_layout()
    out = OUT_DIR / "loss_curve.png"
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


# ── plot 2: eval accuracy curve ───────────────────────────────────────────────
def plot_accuracy():
    conditions = ["baseline", "distill"]
    series = load_accuracy(conditions)

    if all(len(v) == 0 for v in series.values()):
        print("  [skip] No accuracy data found; skipping accuracy plot")
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    for cond in conditions:
        pts = series[cond]
        if not pts:
            continue
        steps, accs = zip(*pts)
        ax.plot(steps, accs, color=COLORS[cond], label=LABELS[cond],
                linewidth=2, marker="o", markersize=4)

    # reference lines
    ax.axhline(47.08, color="gray", linestyle="--", linewidth=1, label="A — 8-shot teacher (47.08%)")
    ax.axhline(26.08, color="lightgray", linestyle=":", linewidth=1, label="Base 0-shot (26.08%)")

    ax.set_xlabel("Training step")
    ax.set_ylabel("GSM8K accuracy (%)")
    ax.set_title("Eval accuracy vs training step — full test set (1319 examples)")
    ax.legend(frameon=False, fontsize=9.5)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    # mark warmup boundary
    ax.axvline(50, color="orange", linestyle=":", linewidth=1, alpha=0.6)
    ax.text(52, ax.get_ylim()[0] + 1, "warmup end", fontsize=8, color="orange", alpha=0.8)

    fig.tight_layout()
    out = OUT_DIR / "accuracy_curve.png"
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


# ── main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating plots...")
    plot_loss()
    plot_accuracy()
    print("Done.")
