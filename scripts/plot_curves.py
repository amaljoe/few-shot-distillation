"""
Generate training loss and checkpoint accuracy curve plots.

Usage — default (1.7B):
    python scripts/plot_curves.py

Usage — 8B overnight run:
    python scripts/plot_curves.py \\
        --model_tag 8b \\
        --early_results experiments/ablations_8b/checkpoint_curve/results_early.json \\
        --full_results  experiments/ablations_8b/checkpoint_curve/results_full.json \\
        --early_tb_baseline experiments/8b/early_checkpoints/baseline/baseline/tb_logs \\
        --early_tb_distill  experiments/8b/early_checkpoints/distill/distill/tb_logs \\
        --full_tb_baseline  experiments/8b/baseline/baseline/tb_logs \\
        --full_tb_distill   experiments/8b/distill/distill/tb_logs \\
        --out_dir experiments/figures/8b
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── defaults (1.7B run) ───────────────────────────────────────────────────────
DEFAULTS = {
    "early_results": "experiments/ablations/checkpoint_curve/results_early.json",
    "full_results":  "experiments/ablations/checkpoint_curve/results_full.json",
    "early_tb_baseline": "experiments/early_checkpoints/baseline/baseline/tb_logs",
    "early_tb_distill":  "experiments/early_checkpoints/distill/distill/tb_logs",
    "full_tb_baseline":  "experiments/poc/baseline/baseline/tb_logs",
    "full_tb_distill":   "experiments/poc/distill/distill/tb_logs",
    "out_dir": "experiments/figures",
    "model_tag": "1.7b",
}

# ── styling ───────────────────────────────────────────────────────────────────
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


# ── args ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_tag",           default=DEFAULTS["model_tag"])
    p.add_argument("--early_results",       default=DEFAULTS["early_results"])
    p.add_argument("--full_results",        default=DEFAULTS["full_results"])
    p.add_argument("--early_tb_baseline",   default=DEFAULTS["early_tb_baseline"])
    p.add_argument("--early_tb_distill",    default=DEFAULTS["early_tb_distill"])
    p.add_argument("--full_tb_baseline",    default=DEFAULTS["full_tb_baseline"])
    p.add_argument("--full_tb_distill",     default=DEFAULTS["full_tb_distill"])
    p.add_argument("--out_dir",             default=DEFAULTS["out_dir"])
    return p.parse_args()


# ── TensorBoard reader ────────────────────────────────────────────────────────
def read_tb_scalars(tb_dir: str, tag: str) -> list[tuple[int, float]]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return []
    p = Path(tb_dir)
    if not p.exists():
        return []
    ea = EventAccumulator(str(p))
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return []
    return [(e.step, e.value) for e in ea.Scalars(tag)]


def load_tb_loss(
    early_tb_baseline: str,
    early_tb_distill: str,
    full_tb_baseline: str,
    full_tb_distill: str,
) -> dict[str, list[tuple[int, float]]]:
    """Merge early + full TB logs per condition."""
    result: dict[str, dict[int, float]] = {"baseline": {}, "distill": {}}
    tb_map = {
        "baseline": [(early_tb_baseline, "train/ce_loss"), (full_tb_baseline, "train/ce_loss")],
        "distill":  [(early_tb_distill,  "train/ce_loss"), (full_tb_distill,  "train/ce_loss")],
    }
    for cond, sources in tb_map.items():
        for tb_dir, tag in sources:
            for step, val in read_tb_scalars(tb_dir, tag):
                if step not in result[cond]:
                    result[cond][step] = val
    return {c: sorted(v.items()) for c, v in result.items()}


# ── accuracy loader ───────────────────────────────────────────────────────────
def load_accuracy(
    early_results: str, full_results: str
) -> dict[str, list[tuple[int, float]]]:
    result: dict[str, dict[int, float]] = {"baseline": {}, "distill": {}}
    for path in [early_results, full_results]:
        p = Path(path)
        if not p.exists():
            continue
        data = json.loads(p.read_text())
        for cond in ["baseline", "distill"]:
            for key, vals in data.get("conditions", {}).get(cond, {}).items():
                step = int(key.replace("step_", ""))
                result[cond][step] = vals["accuracy"] * 100
    return {c: sorted(v.items()) for c, v in result.items()}


# ── plot 1: training loss ─────────────────────────────────────────────────────
def plot_loss(loss_data, model_tag, out_dir):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    any_data = False
    for cond in ["baseline", "distill"]:
        pts = loss_data[cond]
        if not pts:
            print(f"  [warn] No TB loss data for {cond}")
            continue
        steps, vals = zip(*pts)
        ax.plot(steps, vals, color=COLORS[cond], label=LABELS[cond], linewidth=1.8)
        any_data = True

    if not any_data:
        print("  [skip] No loss data — skipping loss plot")
        plt.close()
        return

    ax.set_xlabel("Training step")
    ax.set_ylabel("CE loss")
    ax.set_title(f"Training loss — Baseline vs Distillation (Qwen3-{model_tag.upper()})")
    ax.legend(frameon=False)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(100))

    fig.tight_layout()
    out = Path(out_dir) / "loss_curve.png"
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


# ── plot 2: eval accuracy curve ───────────────────────────────────────────────
def plot_accuracy(acc_data, model_tag, out_dir):
    fig, ax = plt.subplots(figsize=(9, 5))

    for cond in ["baseline", "distill"]:
        pts = acc_data[cond]
        if not pts:
            continue
        steps, accs = zip(*pts)
        ax.plot(steps, accs, color=COLORS[cond], label=LABELS[cond],
                linewidth=2, marker="o", markersize=4)

    # Reference lines — use 1.7B values as placeholders; will show if 8B differs
    ax.axhline(47.08, color="gray",      linestyle="--", linewidth=1,
               label="A — 8-shot teacher (47.08%)")
    ax.axhline(26.08, color="lightgray", linestyle=":",  linewidth=1,
               label="Base 0-shot (26.08%)")

    ax.set_xlabel("Training step")
    ax.set_ylabel("GSM8K accuracy (%)")
    ax.set_title(
        f"Eval accuracy vs training step — Qwen3-{model_tag.upper()} "
        f"(full test set, 1319 examples)"
    )
    ax.legend(frameon=False, fontsize=9.5)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    ax.axvline(50, color="orange", linestyle=":", linewidth=1, alpha=0.6)
    ax.text(52, ax.get_ylim()[0] + 1, "warmup end", fontsize=8,
            color="orange", alpha=0.8)

    fig.tight_layout()
    out = Path(out_dir) / "accuracy_curve.png"
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating plots for Qwen3-{args.model_tag.upper()}...")

    loss_data = load_tb_loss(
        args.early_tb_baseline, args.early_tb_distill,
        args.full_tb_baseline,  args.full_tb_distill,
    )
    acc_data = load_accuracy(args.early_results, args.full_results)

    plot_loss(loss_data, args.model_tag, out_dir)
    plot_accuracy(acc_data, args.model_tag, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
