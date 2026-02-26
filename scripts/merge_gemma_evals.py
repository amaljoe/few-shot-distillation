"""Merge Gemma eval JSONs from individual eval runs into one file."""
import json
from pathlib import Path

with open("experiments/gemma270m/baseline_eval.json") as f:
    base = json.load(f)

with open("experiments/gemma270m/distill_eval.json") as f:
    dist = json.load(f)

with open("experiments/gemma270m/ablations/zeroshot_eval.json") as f:
    abl = json.load(f)

# Merge all conditions into one dict
main = base  # start with baseline as the base structure
main["conditions"]["online_v1"] = dist["conditions"]["online_v1"]
main["conditions"]["zeroshot_teacher"] = abl["conditions"]["zeroshot_teacher"]

out = "experiments/gemma270m/all_conditions_eval.json"
with open(out, "w") as f:
    json.dump(main, f, indent=2)
print(f"Merged to: {out}")

# Print summary
STEPS = [200, 400, 600, 800, 1000]
LABELS = {
    "baseline": "SFT (CE only)",
    "online_v1": "SFT + 8-shot Distill (ours)",
    "zeroshot_teacher": "SFT + 0-shot KD (control)",
}
print()
print("Gemma-3-270M Results (N=1319)")
print("=" * 72)
baseline_best = max(
    main["conditions"]["baseline"][f"step_{s}"]["accuracy"]
    for s in STEPS
    if f"step_{s}" in main["conditions"]["baseline"]
) * 100

for cond, label in LABELS.items():
    if cond not in main["conditions"]:
        continue
    vals = [main["conditions"][cond].get(f"step_{s}", {}).get("accuracy", 0) * 100
            for s in STEPS]
    best = max(vals)
    delta = f"+{best - baseline_best:.2f}pp" if cond != "baseline" else "—"
    print(f"{label}: best={best:.2f}% ({delta})")
    print("  Steps:", " ".join(f"step{s}={v:.2f}%" for s, v in zip(STEPS, vals)))
