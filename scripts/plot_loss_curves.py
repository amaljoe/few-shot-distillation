"""
Plot zero-shot vs few-shot dev loss curves for baseline and distill experiments.

Reads:  experiments/loss_curve/{baseline,distill}/results.json
Writes: experiments/loss_curve/loss_curves_baseline.png
        experiments/loss_curve/loss_curves_distill.png

Can run locally (no GPU needed).
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path("experiments/loss_curve")

CONDITION_TITLES = {
    "baseline": "SFT Baseline (CE loss only)",
    "distill":  "Distilled SFT (CE + λ·MSE top-K logits)",
}

ZS_COLOR = "#2196F3"   # blue  — zero-shot (what we train on)
FS_COLOR = "#F44336"   # red   — few-shot  (upper bound context)


def plot_condition(results: dict, out_path: Path, title: str):
    steps   = results["steps"]
    zs_loss = results["zs_loss"]
    fs_loss = results["fs_loss"]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(steps, zs_loss, "o-",  color=ZS_COLOR, linewidth=2, markersize=5,
            label="Zero-shot dev loss (model trained on this)")
    ax.plot(steps, fs_loss, "s--", color=FS_COLOR,  linewidth=2, markersize=5,
            label="Few-shot dev loss (8-shot context, same answer tokens)")

    # Shade the region between the two curves to visualise the gap
    ax.fill_between(steps, zs_loss, fs_loss,
                    where=[z >= f for z, f in zip(zs_loss, fs_loss)],
                    alpha=0.12, color=ZS_COLOR, label="_nolegend_")
    ax.fill_between(steps, zs_loss, fs_loss,
                    where=[z < f for z, f in zip(zs_loss, fs_loss)],
                    alpha=0.12, color=FS_COLOR, label="_nolegend_")

    # Crossover annotation if it happens
    for i in range(1, len(steps)):
        if zs_loss[i - 1] >= fs_loss[i - 1] and zs_loss[i] < fs_loss[i]:
            cross_step = steps[i]
            ax.axvline(cross_step, color="gray", linestyle=":", alpha=0.7)
            ax.text(cross_step + 1, ax.get_ylim()[1] * 0.98,
                    f"crossover\n@ step {cross_step}",
                    fontsize=8, va="top", color="gray")
            break

    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Training step", fontsize=11)
    ax.set_ylabel("CE loss (answer tokens only)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓  Saved {out_path}")


def main():
    any_found = False
    for mode, title in CONDITION_TITLES.items():
        res_path = BASE_DIR / mode / "results.json"
        if not res_path.exists():
            print(f"[skip] {res_path} not found")
            continue

        with open(res_path) as f:
            results = json.load(f)

        out_path = BASE_DIR / f"loss_curves_{mode}.png"
        plot_condition(results, out_path, title)
        any_found = True

    if not any_found:
        print("No results found. Run loss_curve_experiment.py first.")
    else:
        print(f"\nPlots saved to {BASE_DIR}/")


if __name__ == "__main__":
    main()
