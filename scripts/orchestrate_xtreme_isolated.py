#!/usr/bin/env python3
"""
Orchestrates XTREME isolated lambda sweep experiments.

Manages 4 tmux sessions (exp1–exp4) on cn14-dgx via SSH.
For each of 5 tasks:
  Round 1: λ ∈ {0, 0.01, 0.05, 0.1}   — 4 GPUs in parallel
  Round 2: λ ∈ {0.25, 0.5, 0.75, 1.0} — 4 GPUs in parallel
  Eval:    all 8 checkpoints at once (tp=4)

Usage (run from local machine, shared filesystem):
  python scripts/orchestrate_xtreme_isolated.py
"""

import subprocess
import time
import sys
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────
REPO         = "/home/compiling-ganesh/24m0797/workspace/icl-distillation"
SSH_HOST     = "cn14-dgx"
SSH_PORT     = "4422"
SESSIONS     = ["exp1", "exp2", "exp3", "exp4"]
GPUS         = [0, 1, 2, 3]
PORTS        = [29500, 29501, 29502, 29503]
TASKS        = ["nli", "pa", "qa", "ner", "pos"]

# (lam_int_key, lam_float)  — lam_int used in condition name & sentinel filename
ROUND1 = [(0,  0.0),  (1,  0.01), (5,  0.05), (10, 0.1)]
ROUND2 = [(25, 0.25), (50, 0.5),  (75, 0.75), (100, 1.0)]
ALL_LAMS = ROUND1 + ROUND2

BASE_CONFIG    = "configs/xtreme_qwen1b7.yaml"
DISTILL_CONFIG = "configs/xtreme_isolated.yaml"
BASE_MODEL     = "Qwen/Qwen3-1.7B"

POLL_INTERVAL  = 30   # seconds between sentinel checks
TRAIN_TIMEOUT  = 7200 # 2 h per round
EVAL_TIMEOUT   = 3600 # 1 h per eval


# ── SSH / tmux helpers ───────────────────────────────────────────────────────
def ssh(cmd: str, capture=False):
    full = f"ssh -p {SSH_PORT} {SSH_HOST} {cmd!r}"
    if capture:
        r = subprocess.run(full, shell=True, capture_output=True, text=True)
        return r.stdout.strip()
    return subprocess.run(full, shell=True)


def tmux_send(session: str, cmd: str):
    """Send a command string to a tmux session on cn14-dgx."""
    # Escape single quotes inside cmd
    safe_cmd = cmd.replace("'", "'\\''")
    ssh(f"tmux send-keys -t {session} '{safe_cmd}' Enter")


def session_exists(session: str) -> bool:
    r = subprocess.run(
        f"ssh -p {SSH_PORT} {SSH_HOST} 'tmux has-session -t {session} 2>/dev/null && echo yes || echo no'",
        shell=True, capture_output=True, text=True
    )
    return r.stdout.strip() == "yes"


# ── Environment setup ────────────────────────────────────────────────────────
def setup_sessions():
    print("=" * 60)
    print("Setting up tmux sessions on cn14-dgx …")

    for sess in SESSIONS:
        if session_exists(sess):
            print(f"  {sess}: already exists, will reuse")
        else:
            ssh(f"tmux new-session -d -s {sess}")
            print(f"  {sess}: created")

    print("\nSending 'app' (start apptainer container) to all sessions …")
    for sess in SESSIONS:
        tmux_send(sess, "app")

    print("Waiting 15 s for containers to start …")
    time.sleep(15)

    print("Activating conda env …")
    for sess in SESSIONS:
        tmux_send(sess, "mamba activate /dev/shm/vllm")

    print("Waiting 10 s for activation …")
    time.sleep(10)

    print("cd into repo …")
    for sess in SESSIONS:
        tmux_send(sess, f"cd {REPO}")

    time.sleep(3)
    print("All sessions ready.\n")


# ── Sentinel helpers ─────────────────────────────────────────────────────────
def sentinel_dir(task: str) -> Path:
    return Path(REPO) / "experiments" / "xtreme_isolated" / task / "sentinels"


def wait_for_file(path: Path, timeout: int, label: str):
    start = time.time()
    while not path.exists():
        elapsed = int(time.time() - start)
        print(f"  [{label}] waiting {elapsed:4d}s …", end="\r", flush=True)
        if time.time() - start > timeout:
            print(f"\n  TIMEOUT waiting for {path}")
            return False
        time.sleep(POLL_INTERVAL)
    print(f"  [{label}] done ({int(time.time()-start)}s)  ")
    return True


def wait_for_n_sentinels(sd: Path, pattern: str, n: int, timeout: int):
    """Wait until at least n files matching sd/pattern exist."""
    start = time.time()
    while True:
        found = list(sd.glob(pattern))
        elapsed = int(time.time() - start)
        print(f"  sentinels: {len(found)}/{n}  elapsed={elapsed}s", end="\r", flush=True)
        if len(found) >= n:
            print()
            return True
        if time.time() - start > timeout:
            print(f"\n  TIMEOUT: only {len(found)}/{n} sentinels after {timeout}s")
            return False
        time.sleep(POLL_INTERVAL)


# ── Training ─────────────────────────────────────────────────────────────────
def send_training(task: str, lam_pairs: list, sd: Path):
    """Send one round of training commands (one per session/GPU)."""
    for i, (lam_int, lam_float) in enumerate(lam_pairs):
        sess    = SESSIONS[i]
        gpu     = GPUS[i]
        port    = PORTS[i]
        cond    = f"{task}_lam{lam_int}"
        log     = f"{REPO}/experiments/xtreme_isolated/logs/{cond}.log"
        done_f  = str(sd / f"train_{lam_int}")

        cmd = (
            f"CUDA_VISIBLE_DEVICES={gpu} "
            f"/dev/shm/vllm/bin/accelerate launch "
            f"--num_processes 1 --mixed_precision bf16 --main_process_port {port} "
            f"src/training/train_xtreme_distill.py "
            f"--base_config {BASE_CONFIG} "
            f"--config {DISTILL_CONFIG} "
            f"--tasks {task} "
            f"--lambda_distill {lam_float} "
            f"--condition_name {cond} "
            f"--output_dir experiments/xtreme_isolated/{task} "
            f"2>&1 | tee {log} && touch {done_f}"
        )
        print(f"    [{sess}|GPU{gpu}] λ={lam_float:5.2f}  cond={cond}")
        tmux_send(sess, cmd)


# ── Evaluation ───────────────────────────────────────────────────────────────
def send_eval(task: str, sd: Path):
    """Send evaluation command (all lambdas, tp=4) to exp1."""
    lam_values = " ".join(str(lf) for _, lf in ALL_LAMS)
    log   = f"{REPO}/experiments/xtreme_isolated/logs/{task}_eval.log"
    done_f = str(sd / "eval_done")

    cmd = (
        f"CUDA_VISIBLE_DEVICES=0,1,2,3 "
        f"/dev/shm/vllm/bin/python scripts/eval_xtreme_isolated.py "
        f"--base_model {BASE_MODEL} "
        f"--task {task} "
        f"--base_dir experiments/xtreme_isolated/{task} "
        f"--lambdas {lam_values} "
        f"--languages en "
        f"--n_samples 500 "
        f"--output experiments/xtreme_isolated/results.json "
        f"--tensor_parallel_size 4 "
        f"2>&1 | tee {log} && touch {done_f}"
    )
    print(f"    [exp1] eval task={task}  tp=4")
    tmux_send("exp1", cmd)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    repo = Path(REPO)
    (repo / "experiments" / "xtreme_isolated" / "logs").mkdir(parents=True, exist_ok=True)

    setup_sessions()

    for task in TASKS:
        print(f"\n{'='*60}")
        print(f"  TASK: {task.upper()}")
        print(f"{'='*60}")

        sd = sentinel_dir(task)
        sd.mkdir(parents=True, exist_ok=True)
        # Clean any leftover sentinels
        for f in sd.glob("*"):
            f.unlink()

        # Round 1
        print(f"\n  Round 1: λ ∈ {{0, 0.01, 0.05, 0.1}}")
        send_training(task, ROUND1, sd)
        ok = wait_for_n_sentinels(sd, "train_*", 4, TRAIN_TIMEOUT)
        if not ok:
            print(f"  WARNING: round 1 timed out for {task}, continuing anyway")
        print("  Round 1 complete!")

        # Round 2
        print(f"\n  Round 2: λ ∈ {{0.25, 0.5, 0.75, 1.0}}")
        send_training(task, ROUND2, sd)
        ok = wait_for_n_sentinels(sd, "train_*", 8, TRAIN_TIMEOUT)
        if not ok:
            print(f"  WARNING: round 2 timed out for {task}, continuing anyway")
        print("  Round 2 complete!")

        # Eval
        print(f"\n  Evaluating {task} (all 8 λ, English, tp=4) …")
        send_eval(task, sd)
        ok = wait_for_file(sd / "eval_done", EVAL_TIMEOUT, f"{task} eval")
        if not ok:
            print(f"  WARNING: eval timed out for {task}")

        print(f"\n  ✓ Task {task} done!\n")

    # Generate figures & report
    print("\n" + "="*60)
    print("Generating figures and report …")
    subprocess.run([
        sys.executable, "scripts/gen_xtreme_isolated_fig.py",
        "--results",    "experiments/xtreme_isolated/results.json",
        "--output_dir", "experiments/xtreme_isolated/figures",
    ], cwd=REPO)

    print("\n=== ALL EXPERIMENTS COMPLETE ===")
    print("Results : experiments/xtreme_isolated/results.json")
    print("Figures : experiments/xtreme_isolated/figures/")
    print("Logs    : experiments/xtreme_isolated/logs/")
    print("Report  : xtreme_isolated_results.md")


if __name__ == "__main__":
    main()
