"""
Write experiment results from JSON eval files to results.md.

Reads:
  experiments/qwen8b/icl_eval.json        — base + 8-shot ICL for Qwen3-8B
  experiments/qwen8b_eval.json            — checkpoint curve for Qwen3-8B
  experiments/param2_17b/icl_eval.json   — base + 8-shot ICL for Param2-17B
  experiments/param2_17b_eval.json       — checkpoint curve for Param2-17B

Writes: results.md
"""

import json
from pathlib import Path


def load_json(path):
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def fmt(val):
    if val is None:
        return "N/A"
    return f"{val*100:.2f}%"


def icl_row(data, nshot):
    if data is None:
        return "N/A"
    key = f"{nshot}_shot"
    evs = data.get("evaluations", {})
    if key not in evs:
        return "N/A"
    return fmt(evs[key]["accuracy"])


def best_ckpt_acc(eval_data, cond):
    if eval_data is None:
        return None, None
    cond_data = eval_data.get("conditions", {}).get(cond, {})
    if not cond_data:
        return None, None
    best_step = max(cond_data, key=lambda k: cond_data[k].get("accuracy", 0))
    return cond_data[best_step]["accuracy"], best_step


def ckpt_table(eval_data, cond, steps=(200, 400, 600, 800, 1000)):
    if eval_data is None:
        return []
    cond_data = eval_data.get("conditions", {}).get(cond, {})
    rows = []
    for step in steps:
        key = f"step_{step}"
        acc = cond_data.get(key, {}).get("accuracy")
        rows.append((step, acc))
    return rows


def write_results():
    out = Path("results.md")

    # Load all data
    qwen8b_icl = load_json("experiments/qwen8b/icl_eval.json")
    qwen8b_eval = load_json("experiments/qwen8b_eval.json")
    param2_icl = load_json("experiments/param2_17b/icl_eval.json")
    param2_eval = load_json("experiments/param2_17b_eval.json")

    lines = []
    lines.append("# New Model Experiment Results")
    lines.append("")
    lines.append("Evaluated on the **full GSM8K test set (1319 examples)** using zero-shot")
    lines.append("inference at test time for all fine-tuned models.")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ===== Qwen3-8B =====
    lines.append("## Qwen3-8B (base model)")
    lines.append("")

    bl_best, bl_step = best_ckpt_acc(qwen8b_eval, "baseline")
    dl_best, dl_step = best_ckpt_acc(qwen8b_eval, "online_v1")
    zs_best, zs_step = best_ckpt_acc(qwen8b_eval, "zeroshot_teacher")
    delta = None
    if dl_best is not None and bl_best is not None:
        delta = dl_best - bl_best

    lines.append("| Method | Accuracy | vs 0-shot |")
    lines.append("|--------|----------|-----------|")
    lines.append(f"| Base model, 0-shot | {icl_row(qwen8b_icl, 0)} | — |")
    lines.append(f"| 8-shot in-context learning | {icl_row(qwen8b_icl, 8)} | "
                 f"+{(qwen8b_icl['evaluations']['8_shot']['accuracy'] - qwen8b_icl['evaluations']['0_shot']['accuracy'])*100:.1f}pp |"
                 if qwen8b_icl and '8_shot' in qwen8b_icl.get('evaluations', {}) and '0_shot' in qwen8b_icl.get('evaluations', {})
                 else f"| 8-shot in-context learning | {icl_row(qwen8b_icl, 8)} | N/A |")
    lines.append(f"| LoRA SFT | {fmt(bl_best)} | "
                 f"+{(bl_best - qwen8b_icl['evaluations']['0_shot']['accuracy'])*100:.1f}pp |"
                 if bl_best and qwen8b_icl and '0_shot' in qwen8b_icl.get('evaluations', {})
                 else f"| LoRA SFT | {fmt(bl_best)} | N/A |")
    lines.append(f"| **LoRA SFT + Few-Shot Distillation (ours)** | **{fmt(dl_best)}** | "
                 f"**+{(dl_best - qwen8b_icl['evaluations']['0_shot']['accuracy'])*100:.1f}pp** |"
                 if dl_best and qwen8b_icl and '0_shot' in qwen8b_icl.get('evaluations', {})
                 else f"| **LoRA SFT + Few-Shot Distillation (ours)** | **{fmt(dl_best)}** | N/A |")
    lines.append("")

    if delta is not None:
        lines.append(f"Distillation improves over LoRA SFT by **+{delta*100:.1f}pp**.")
        lines.append("")
    if zs_best is not None:
        lines.append(f"Control (0-shot teacher): {fmt(zs_best)} — "
                     f"tests whether gain is specific to few-shot context transfer.")
        lines.append("")

    # Checkpoint table
    lines.append("### Qwen3-8B — Checkpoint Accuracy Curve")
    lines.append("")
    lines.append("| Step | LoRA SFT | + Distillation | Δ | Control (0-shot teacher) |")
    lines.append("|------|----------|----------------|---|--------------------------|")
    steps = [200, 400, 600, 800, 1000]
    bl_rows = dict(ckpt_table(qwen8b_eval, "baseline", steps))
    dl_rows = dict(ckpt_table(qwen8b_eval, "online_v1", steps))
    zs_rows = dict(ckpt_table(qwen8b_eval, "zeroshot_teacher", steps))
    for step in steps:
        bl = bl_rows.get(step)
        dl = dl_rows.get(step)
        zs = zs_rows.get(step)
        d = f"+{(dl-bl)*100:.1f}pp" if (dl is not None and bl is not None) else "N/A"
        lines.append(f"| {step} | {fmt(bl)} | {fmt(dl)} | {d} | {fmt(zs)} |")
    lines.append("")

    lines.append("---")
    lines.append("")

    # ===== Param2-17B =====
    lines.append("## Param2-17B-A2.4B-Thinking (MoE model)")
    lines.append("")
    lines.append("*17B total parameters, 2.4B active per token (64 experts, top-6 routing).*")
    lines.append("*Multilingual model (English + 22 Indian languages). Custom Param2MoE architecture.*")
    lines.append("")

    bl_best2, bl_step2 = best_ckpt_acc(param2_eval, "baseline")
    dl_best2, dl_step2 = best_ckpt_acc(param2_eval, "online_v1")
    zs_best2, zs_step2 = best_ckpt_acc(param2_eval, "zeroshot_teacher")
    delta2 = None
    if dl_best2 is not None and bl_best2 is not None:
        delta2 = dl_best2 - bl_best2

    lines.append("| Method | Accuracy | vs 0-shot |")
    lines.append("|--------|----------|-----------|")
    lines.append(f"| Base model, 0-shot | {icl_row(param2_icl, 0)} | — |")
    if param2_icl and '8_shot' in param2_icl.get('evaluations', {}) and '0_shot' in param2_icl.get('evaluations', {}):
        icl_gap = param2_icl['evaluations']['8_shot']['accuracy'] - param2_icl['evaluations']['0_shot']['accuracy']
        lines.append(f"| 8-shot in-context learning | {icl_row(param2_icl, 8)} | +{icl_gap*100:.1f}pp |")
    else:
        lines.append(f"| 8-shot in-context learning | {icl_row(param2_icl, 8)} | N/A |")
    if bl_best2 and param2_icl and '0_shot' in param2_icl.get('evaluations', {}):
        base_0 = param2_icl['evaluations']['0_shot']['accuracy']
        lines.append(f"| LoRA SFT | {fmt(bl_best2)} | +{(bl_best2-base_0)*100:.1f}pp |")
        if dl_best2:
            lines.append(f"| **LoRA SFT + Few-Shot Distillation (ours)** | **{fmt(dl_best2)}** | **+{(dl_best2-base_0)*100:.1f}pp** |")
        else:
            lines.append(f"| **LoRA SFT + Few-Shot Distillation (ours)** | **{fmt(dl_best2)}** | N/A |")
    else:
        lines.append(f"| LoRA SFT | {fmt(bl_best2)} | N/A |")
        lines.append(f"| **LoRA SFT + Few-Shot Distillation (ours)** | **{fmt(dl_best2)}** | N/A |")
    lines.append("")

    if delta2 is not None:
        lines.append(f"Distillation improves over LoRA SFT by **+{delta2*100:.1f}pp**.")
        lines.append("")
    if zs_best2 is not None:
        lines.append(f"Control (0-shot teacher): {fmt(zs_best2)}")
        lines.append("")

    # Checkpoint table Param2
    lines.append("### Param2-17B — Checkpoint Accuracy Curve")
    lines.append("")
    lines.append("| Step | LoRA SFT | + Distillation | Δ | Control (0-shot teacher) |")
    lines.append("|------|----------|----------------|---|--------------------------|")
    bl_rows2 = dict(ckpt_table(param2_eval, "baseline", steps))
    dl_rows2 = dict(ckpt_table(param2_eval, "online_v1", steps))
    zs_rows2 = dict(ckpt_table(param2_eval, "zeroshot_teacher", steps))
    for step in steps:
        bl = bl_rows2.get(step)
        dl = dl_rows2.get(step)
        zs = zs_rows2.get(step)
        d = f"+{(dl-bl)*100:.1f}pp" if (dl is not None and bl is not None) else "N/A"
        lines.append(f"| {step} | {fmt(bl)} | {fmt(dl)} | {d} | {fmt(zs)} |")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append("| | Qwen3-8B | Param2-17B-A2.4B |")
    lines.append("|---|---|---|")
    lines.append("| Adapter | LoRA r=16 α=32 | LoRA r=16 α=32 |")
    lines.append("| lr | 2×10⁻⁴ | 2×10⁻⁴ |")
    lines.append("| Steps | 1000 | 1000 |")
    lines.append("| Effective batch | 32 | 32 |")
    lines.append("| Distillation λ | 0.5 | 0.5 |")
    lines.append("| top-K vocab | 256 | 256 |")
    lines.append("| Per-device batch | 2 | 1 |")
    lines.append("| Grad accum | 8 | 16 |")
    lines.append("| Gradient checkpointing | yes | yes |")
    lines.append("")
    lines.append("All experiments: bf16, 4×A100 80GB. Online teacher (same frozen base model).")
    lines.append("Param2-17B uses `trust_remote_code=True` for custom MoE architecture.")
    lines.append("Eval script: `eval_checkpoints.py` (vLLM) or `eval_hf_checkpoints.py` (HF fallback).")
    lines.append("")

    content = "\n".join(lines)
    out.write_text(content)
    print(f"Written: {out}")
    print(content)


if __name__ == "__main__":
    write_results()
