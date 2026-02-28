"""
Write param_results.md from ICL eval JSON and checkpoint eval JSON.

Usage:
  python scripts/write_param2_results.py \
      --icl_eval experiments/param2_17b/icl_eval.json \
      --checkpoint_eval experiments/param2_17b_eval.json \
      --output param_results.md
"""

import argparse
import json
from pathlib import Path


def fmt(v):
    return f"{v*100:.1f}%" if v is not None else "—"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--icl_eval", required=True)
    p.add_argument("--checkpoint_eval", required=True)
    p.add_argument("--output", default="param_results.md")
    return p.parse_args()


def load_json(path):
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def best_checkpoint(cond_data):
    """Return (step_number, accuracy) for the best checkpoint."""
    best_step, best_acc = None, None
    for step_str, info in cond_data.items():
        if isinstance(info, dict) and "accuracy" in info:
            acc = info["accuracy"]
            if best_acc is None or acc > best_acc:
                best_acc = acc
                # Extract number from "step_200" → 200
                best_step = step_str.replace("step_", "")
    return best_step, best_acc


def checkpoint_row(cond_data, steps):
    """Return list of accuracy strings for each step."""
    row = []
    for s in steps:
        key = f"step_{s}"
        if key in cond_data and isinstance(cond_data[key], dict):
            row.append(fmt(cond_data[key].get("accuracy")))
        else:
            row.append("—")
    return row


def main():
    args = parse_args()
    icl = load_json(args.icl_eval)
    ckpt = load_json(args.checkpoint_eval)

    lines = []
    lines.append("# Param2-17B-A2.4B-Thinking — GSM8K Results\n")
    lines.append("> Model: `bharatgenai/Param2-17B-A2.4B-Thinking`  ")
    lines.append("> Dataset: GSM8K (1319 test samples, seed=42)  ")
    lines.append("> Backend: HuggingFace + PEFT (LoRA)  ")
    lines.append("> Hardware: 4×A100 80GB (cn14-dgx)\n")

    # ── ICL gap table ────────────────────────────────────────────────────────
    lines.append("## ICL Gap (Baseline Capability)\n")
    if icl is not None:
        evals = icl.get("evaluations", {})
        zero = evals.get("0_shot", {}).get("accuracy")
        eight = evals.get("8_shot", {}).get("accuracy")
        gap = icl.get("icl_gap")
        lines.append("| Setting | Accuracy |")
        lines.append("|---------|----------|")
        lines.append(f"| 0-shot (zero-shot) | {fmt(zero)} |")
        lines.append(f"| 8-shot (few-shot ICL) | {fmt(eight)} |")
        lines.append(f"| **ICL Gap** | **{fmt(gap)}** |")
    else:
        lines.append("*ICL eval results not yet available.*")
    lines.append("")

    # ── Checkpoint curves ────────────────────────────────────────────────────
    lines.append("## Checkpoint Accuracy by Training Step\n")
    if ckpt is not None:
        steps = [200, 400, 600, 800, 1000]
        results = ckpt.get("conditions", ckpt)  # top-level key is "conditions"

        condition_labels = {
            "baseline": "SFT Baseline",
            "online_v1": "Distilled (online_v1)",
            "zeroshot_teacher": "Control (0-shot teacher)",
        }

        header = "| Condition | " + " | ".join(f"Step {s}" for s in steps) + " | Best |"
        sep = "|-----------|" + "|".join(["--------"] * len(steps)) + "|------|"
        lines.append(header)
        lines.append(sep)

        for cond_key, cond_label in condition_labels.items():
            cond_data = results.get(cond_key, {})
            row = checkpoint_row(cond_data, steps)
            best_step, best_acc = best_checkpoint(cond_data)
            best_str = f"{fmt(best_acc)} (step {best_step})" if best_acc is not None else "—"
            lines.append(f"| {cond_label} | " + " | ".join(row) + f" | {best_str} |")
    else:
        lines.append("*Checkpoint eval results not yet available.*")
    lines.append("")

    # ── Summary table ────────────────────────────────────────────────────────
    lines.append("## Summary: Best Accuracy per Condition\n")
    if icl is not None and ckpt is not None:
        evals = icl.get("evaluations", {})
        zero_acc = evals.get("0_shot", {}).get("accuracy")
        eight_acc = evals.get("8_shot", {}).get("accuracy")
        results = ckpt.get("conditions", ckpt)

        _, sft_best = best_checkpoint(results.get("baseline", {}))
        _, dist_best = best_checkpoint(results.get("online_v1", {}))
        _, ctrl_best = best_checkpoint(results.get("zeroshot_teacher", {}))

        def delta(val, ref):
            if val is not None and ref is not None:
                d = (val - ref) * 100
                sign = "+" if d >= 0 else ""
                return f"{sign}{d:.1f}pp"
            return "—"

        lines.append("| Condition | Accuracy | Δ vs 0-shot |")
        lines.append("|-----------|----------|-------------|")
        lines.append(f"| 0-shot (no training) | {fmt(zero_acc)} | — |")
        lines.append(f"| 8-shot ICL (no training) | {fmt(eight_acc)} | {delta(eight_acc, zero_acc)} |")
        lines.append(f"| SFT Baseline (best ckpt) | {fmt(sft_best)} | {delta(sft_best, zero_acc)} |")
        lines.append(f"| Distilled / online_v1 (best ckpt) | {fmt(dist_best)} | {delta(dist_best, zero_acc)} |")
        lines.append(f"| Control / 0-shot teacher (best ckpt) | {fmt(ctrl_best)} | {delta(ctrl_best, zero_acc)} |")
        lines.append("")

        if dist_best is not None and sft_best is not None:
            gap_closed = None
            icl_gap = icl.get("icl_gap")
            if icl_gap and icl_gap > 0:
                gain_over_sft = dist_best - sft_best
                gap_closed = gain_over_sft / icl_gap * 100
            lines.append("### Key Finding\n")
            lines.append(f"- **Distillation gain over SFT**: {delta(dist_best, sft_best)}")
            if gap_closed is not None:
                lines.append(f"- **ICL gap**: {fmt(eight_acc)} − {fmt(zero_acc)} = {fmt(icl_gap)}")
                lines.append(f"- **% of ICL gap closed by distillation** (over SFT): {gap_closed:.1f}%")
    else:
        lines.append("*Full results not yet available.*")
    lines.append("")

    out = Path(args.output)
    out.write_text("\n".join(lines) + "\n")
    print(f"Written: {out}")


if __name__ == "__main__":
    main()
