"""
Sanity check: verify all 28 (task, lang) pairs load test data correctly,
and verify NLI + POS training data loads (the two historically tricky ones).
Exits with code 1 if any expected pair returns empty.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.xtreme_loader import TASK_LANGUAGES, load_task_data

EXPECTED = [
    (task, lang)
    for task, langs in TASK_LANGUAGES.items()
    for lang in langs
]
assert len(EXPECTED) == 28, f"Expected 28 pairs, got {len(EXPECTED)}"

print("--- Test split (28 eval pairs) ---")
ok, fail = [], []
for task, lang in EXPECTED:
    data = load_task_data(task, "test", lang, max_samples=5)
    if data:
        ok.append((task, lang))
        print(f"  OK   {task}/{lang}")
    else:
        fail.append((task, lang))
        print(f"  FAIL {task}/{lang}: no data")

print(f"\n{len(ok)}/28 eval pairs OK")

print("\n--- Train split (NLI/en and POS check) ---")
train_checks = [("nli", "en"), ("pos", "en"), ("pos", "zh")]
for task, lang in train_checks:
    data = load_task_data(task, "train", lang, max_samples=5)
    status = f"OK ({len(data)} ex)" if data else "FAIL — no data"
    print(f"  {task}/{lang} train: {status}")
    if not data:
        fail.append((f"train:{task}", lang))

if fail:
    print(f"\nFAILED: {fail}")
    sys.exit(1)
else:
    print("\nAll checks passed. Proceeding.")
