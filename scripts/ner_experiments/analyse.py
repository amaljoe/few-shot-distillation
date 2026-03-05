"""
NER Analysis: Selective Distillation Signal Study

Studies whether per-token distillation signal from a few-shot teacher is
uniformly beneficial for NER, or whether some tokens are already better
predicted by the zero-shot student.

Runs two conditions sequentially (distill, sft) on 4 GPUs, tracks per-token
loss comparisons during training, evaluates at specific checkpoints, and
generates 4 comparative plots + a written report.

Launch command:
  CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \\
      --num_processes 4 --mixed_precision bf16 --main_process_port 29500 \\
      scripts/ner_experiments/analyse.py \\
      --output_dir experiments/ner_analysis
"""

import argparse
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from accelerate import Accelerator
from src.data.xtreme_loader import (
    NUM_FEWSHOT,
    NER_ID2LABEL,
    XTREMEDistillDataset,
    collate_fn_xtreme,
    compute_ner_f1,
    load_task_data,
    parse_output,
    build_student_messages,
    apply_xtreme_template,
)
from src.models.student import StudentModel

# ============================================================================
# Hardcoded hyperparameters
# ============================================================================

MODEL_NAME        = "Qwen/Qwen3-1.7B"
TASK              = "ner"
LANG              = "en"
LORA_CONFIG       = {"r": 16, "alpha": 32, "dropout": 0.05,
                     "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                                        "gate_proj", "up_proj", "down_proj"]}
PER_DEVICE_BS     = 2
LR                = 2e-4
WARMUP_STEPS      = 50
MAX_STEPS         = 1000
LAMBDA_DISTILL    = 0.5
N_TOP_LOGITS      = 256
MAX_SEQ_LEN_T     = 2048
MAX_SEQ_LEN_S     = 512
DEV_MONITOR_INTERVAL = 32
DEV_MONITOR_SAMPLES  = 8
CHECKPOINT_STEPS           = [0, 4, 16, 32, 64, 128, 256, 512, 1000]
CHECKPOINT_STEPS_SELECTIVE = [0, 4, 16, 32, 64, 128, 256, 512]
MAX_STEPS_SELECTIVE        = 512
CHECKPOINT_STEPS_ONLINE    = [0, 4, 16, 32, 64, 128, 256, 512]
MAX_STEPS_ONLINE           = 512
CHECKPOINT_STEPS_ONLINE_SELECTIVE = [0, 4, 16, 32, 64, 128, 256, 512]
MAX_STEPS_ONLINE_SELECTIVE        = 512
CHECKPOINT_STEPS_ADAPTIVE         = [0, 4, 16, 32, 64, 128, 256, 512]
MAX_STEPS_ADAPTIVE                = 512
ADAPTIVE_LAMBDA_MAX               = 0.35
ADAPTIVE_WARMUP_STEPS             = 64
ADAPTIVE_RAMP_STEPS               = 128
ADAPTIVE_ADV_MARGIN               = 0.15
ADAPTIVE_ADV_TEMP                 = 0.35
EVAL_DEV_SAMPLES  = 200
GRAD_CLIP         = 1.0
WEIGHT_DECAY      = 0.01
NUM_LAYERS        = 28   # Qwen3-1.7B


# ============================================================================
# answer_alignment  (copied from train_online_v1.py)
# ============================================================================

def answer_alignment(t_lens, labels, device):
    """
    Compute index tensors that align teacher and student answer token positions.

    Returns:
        b_idx:       (B, K)  batch indices for advanced indexing
        t_ans_idx:   (B, K)  teacher token indices per answer position
        s_ans_idx:   (B, K)  student token indices per answer position
        ans_valid:   (B, K)  bool mask — True where position j < n_ans[b]
        n_ans:       (B,)    number of answer tokens per example
    """
    B = labels.shape[0]
    n_ans        = (labels != -100).sum(dim=1)                              # (B,)
    s_ans_start  = (labels != -100).float().argmax(dim=1)                   # (B,)
    t_ans_start  = t_lens - n_ans                                           # (B,)

    K = int(n_ans.max().item())
    if K == 0:
        b_idx     = torch.zeros(B, 1, dtype=torch.long, device=device)
        t_ans_idx = torch.zeros(B, 1, dtype=torch.long, device=device)
        s_ans_idx = torch.zeros(B, 1, dtype=torch.long, device=device)
        ans_valid = torch.zeros(B, 1, dtype=torch.bool,  device=device)
        return b_idx, t_ans_idx, s_ans_idx, ans_valid, n_ans

    j = torch.arange(K, device=device).unsqueeze(0).expand(B, -1)          # (B, K)
    t_ans_idx = (t_ans_start.unsqueeze(1) + j).clamp(0, t_lens.max().item() - 1)
    s_ans_idx = (s_ans_start.unsqueeze(1) + j).clamp(0, labels.shape[1] - 1)
    ans_valid  = j < n_ans.unsqueeze(1)
    b_idx      = torch.arange(B, device=device).unsqueeze(1).expand(B, K)

    return b_idx, t_ans_idx, s_ans_idx, ans_valid, n_ans


# ============================================================================
# Dev metrics
# ============================================================================

@torch.no_grad()
def compute_dev_metrics(teacher, unwrapped_student, dev_batch, device, teacher_source="frozen"):
    """
    Compute per-token loss comparison between teacher (few-shot) and
    student (zero-shot) on a fixed dev batch.

    Returns dict with:
      dev_zs_loss       — mean zero-shot CE on labeled tokens
      pct_teacher_worse — % labeled tokens where few-shot CE > zero-shot CE
      mean_loss_diff    — mean(few-shot CE − zero-shot CE) over labeled tokens
    """
    if teacher_source == "frozen":
        teacher.eval()
    unwrapped_student.eval()

    t_input_ids  = dev_batch.teacher_input_ids.to(device)
    t_attn_mask  = dev_batch.teacher_attention_mask.to(device)
    s_input_ids  = dev_batch.student_input_ids.to(device)
    s_attn_mask  = dev_batch.student_attention_mask.to(device)
    labels       = dev_batch.labels.to(device)

    t_lens = t_attn_mask.sum(dim=1)

    # Teacher forward
    if teacher_source == "online":
        t_out = unwrapped_student(input_ids=t_input_ids, attention_mask=t_attn_mask)
    else:
        t_out = teacher(input_ids=t_input_ids, attention_mask=t_attn_mask)

    # Student forward (no labels — we'll compute CE manually per token)
    s_out = unwrapped_student(input_ids=s_input_ids, attention_mask=s_attn_mask)

    # Alignment
    b_idx, t_ans_idx, s_ans_idx, ans_valid, n_ans = answer_alignment(
        t_lens, labels, device
    )

    # Gold token IDs at student answer positions
    gold_ids = s_input_ids[b_idx, s_ans_idx]  # (B, K)

    # logits[i] predicts input_ids[i+1], so to predict answer token j
    # (at position s_ans_idx[b,j]) we need logits at s_ans_idx[b,j] - 1.
    t_logit_idx = (t_ans_idx - 1).clamp(min=0)
    s_logit_idx = (s_ans_idx - 1).clamp(min=0)

    # Teacher per-token CE at answer positions
    t_logits_ans = t_out.logits[b_idx, t_logit_idx]  # (B, K, V)
    t_ce = F.cross_entropy(
        t_logits_ans.reshape(-1, t_logits_ans.size(-1)),
        gold_ids.reshape(-1),
        reduction="none",
    ).reshape(b_idx.shape)  # (B, K)

    # Student per-token CE at answer positions
    s_logits_ans = s_out.logits[b_idx, s_logit_idx]  # (B, K, V)
    s_ce = F.cross_entropy(
        s_logits_ans.reshape(-1, s_logits_ans.size(-1)),
        gold_ids.reshape(-1),
        reduction="none",
    ).reshape(b_idx.shape)  # (B, K)

    # Only consider valid (non-padded) answer positions
    valid_mask = ans_valid  # (B, K)
    n_valid = valid_mask.sum().item()

    if n_valid == 0:
        return {"dev_zs_loss": 0.0, "pct_teacher_worse": 0.0, "mean_loss_diff": 0.0}

    s_ce_valid = s_ce[valid_mask]
    t_ce_valid = t_ce[valid_mask]

    dev_zs_loss      = s_ce_valid.mean().item()
    loss_diff        = t_ce_valid - s_ce_valid           # positive = teacher worse
    pct_teacher_worse = (loss_diff > 0).float().mean().item() * 100.0
    mean_loss_diff   = loss_diff.mean().item()

    unwrapped_student.train()
    return {
        "dev_zs_loss":       dev_zs_loss,
        "pct_teacher_worse": pct_teacher_worse,
        "mean_loss_diff":    mean_loss_diff,
    }


# ============================================================================
# Training
# ============================================================================

def run_training(
    mode,
    accelerator,
    tokenizer,
    teacher,
    student_wrapper,
    train_loader,
    dev_batch,
    checkpoint_steps,
    output_dir,
    max_steps_override=None,
):
    """
    Train student under `mode` ∈ {"distill", "sft", "selective", "online",
    "online_selective", "adaptive"}.

    Saves LoRA adapter checkpoints at each step in checkpoint_steps.
    Runs compute_dev_metrics every DEV_MONITOR_INTERVAL steps and at
    checkpoint_steps.

    Returns metrics dict:
      {"steps": [...], "dev_zs_loss": [...], "pct_teacher_worse": [...],
       "mean_loss_diff": [...]}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_steps = max_steps_override if max_steps_override is not None else MAX_STEPS

    model = student_wrapper.get_model()
    # gradient checkpointing off for 1.7B (fits in memory)
    optimizer = AdamW(
        student_wrapper.get_trainable_parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=n_steps,
    )

    model, optimizer, train_loader_p, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    ckpt_steps_set = set(checkpoint_steps)
    metrics = {"steps": [], "dev_zs_loss": [], "pct_teacher_worse": [], "mean_loss_diff": []}

    # Save step-0 checkpoint before first update
    accelerator.wait_for_everyone()
    if accelerator.is_main_process and 0 in ckpt_steps_set:
        ckpt_dir = output_dir / "checkpoint-0"
        accelerator.unwrap_model(model).save_pretrained(str(ckpt_dir))
        tokenizer.save_pretrained(str(ckpt_dir))
        print(f"[{mode}] Saved checkpoint-0")

    teacher_source = "online" if mode in ("online", "online_selective", "adaptive") else "frozen"

    # Dev metrics at step 0
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        m = compute_dev_metrics(
            teacher, unwrapped, dev_batch, accelerator.device, teacher_source=teacher_source
        )
        metrics["steps"].append(0)
        metrics["dev_zs_loss"].append(m["dev_zs_loss"])
        metrics["pct_teacher_worse"].append(m["pct_teacher_worse"])
        metrics["mean_loss_diff"].append(m["mean_loss_diff"])
        print(f"[{mode}] step=0  zs_loss={m['dev_zs_loss']:.4f}  "
              f"pct_worse={m['pct_teacher_worse']:.1f}%  "
              f"diff={m['mean_loss_diff']:.4f}")
    accelerator.wait_for_everyone()

    model.train()
    step = 0
    train_iter = iter(train_loader_p)
    alignment_verified = False

    progress = tqdm(
        total=n_steps,
        desc=f"{mode} training",
        disable=not accelerator.is_main_process,
    )

    while step < n_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader_p)
            batch = next(train_iter)

        device = accelerator.device
        t_input_ids = batch.teacher_input_ids.to(device)
        t_attn_mask = batch.teacher_attention_mask.to(device)
        labels_d    = batch.labels.to(device)
        t_lens      = t_attn_mask.sum(dim=1)

        # Teacher forward:
        # - distill/selective: frozen teacher model
        # - online: same model's current weights on few-shot prompt (detached target)
        with torch.no_grad():
            if mode in ("online", "online_selective", "adaptive"):
                was_training = model.training
                model.eval()
                teacher_out = model(input_ids=t_input_ids, attention_mask=t_attn_mask)
                if was_training:
                    model.train()
            else:
                teacher_out = teacher(input_ids=t_input_ids, attention_mask=t_attn_mask)

        # Student forward
        student_out = model(
            input_ids=batch.student_input_ids,
            attention_mask=batch.student_attention_mask,
            labels=batch.labels,
        )
        ce_loss = student_out.loss

        if mode in ("distill", "selective", "online", "online_selective", "adaptive"):
            b_idx, t_ans_idx, s_ans_idx, ans_valid, n_ans = answer_alignment(
                t_lens, labels_d, device
            )

            # Token alignment sanity check (first batch, main process)
            if accelerator.is_main_process and not alignment_verified:
                b0_n = n_ans[0].item()
                if b0_n > 0:
                    t_ids = t_input_ids[0, t_ans_idx[0, :b0_n]]
                    s_ids = batch.student_input_ids.to(device)[0, s_ans_idx[0, :b0_n]]
                    match = (t_ids == s_ids).all().item()
                    print(f"\n[{mode}] {'✓' if match else '✗'} Token alignment: "
                          f"{'match' if match else 'MISMATCH'} ({b0_n} tokens checked)")
                alignment_verified = True

            t_logits_ans = teacher_out.logits[b_idx, t_ans_idx]   # (B, K, V)
            s_logits_ans = student_out.logits[b_idx, s_ans_idx]   # (B, K, V)

            if mode in ("selective", "online_selective", "adaptive"):
                # Per-token CE to decide which positions benefit from teacher signal
                V = t_logits_ans.shape[-1]
                gold_ids = batch.student_input_ids.to(device)[b_idx, s_ans_idx]  # (B, K)
                t_logit_idx = (t_ans_idx - 1).clamp(min=0)
                s_logit_idx = (s_ans_idx - 1).clamp(min=0)
                with torch.no_grad():
                    t_ce_tok = F.cross_entropy(
                        teacher_out.logits[b_idx, t_logit_idx].reshape(-1, V),
                        gold_ids.reshape(-1), reduction="none",
                    ).reshape(b_idx.shape)  # (B, K)
                    s_ce_tok = F.cross_entropy(
                        student_out.logits[b_idx, s_logit_idx].reshape(-1, V),
                        gold_ids.reshape(-1), reduction="none",
                    ).reshape(b_idx.shape)  # (B, K)
                if mode == "adaptive":
                    # Hypothesis: only distill where online teacher is meaningfully better,
                    # and ramp distillation strength after CE warmup.
                    adv = (s_ce_tok - t_ce_tok)  # >0 means teacher better
                    tok_weight = ((adv - ADAPTIVE_ADV_MARGIN) / ADAPTIVE_ADV_TEMP).clamp(0.0, 1.0)
                    tok_weight = tok_weight * ans_valid.float()
                    pos_mask = tok_weight > 0
                else:
                    # Only distill where teacher predicts better AND position is valid
                    pos_mask = (t_ce_tok < s_ce_tok) & ans_valid          # (B, K)
                    tok_weight = pos_mask.float()
            else:
                pos_mask = ans_valid                                    # (B, K) all valid
                tok_weight = pos_mask.float()

            del teacher_out, student_out

            _, top_idx = t_logits_ans.topk(N_TOP_LOGITS, dim=-1)      # (B, K, K_v)
            t_top = t_logits_ans.gather(-1, top_idx)
            s_top = s_logits_ans.gather(-1, top_idx)
            del t_logits_ans, s_logits_ans

            mask = tok_weight.unsqueeze(-1)
            n_valid_elements = mask.sum() * N_TOP_LOGITS
            if n_valid_elements > 0:
                dist_loss = ((t_top - s_top).pow(2) * mask).sum() / (n_valid_elements + 1e-8)
                if mode == "adaptive":
                    if step < ADAPTIVE_WARMUP_STEPS:
                        lambda_eff = 0.0
                    else:
                        ramp = min(1.0, (step - ADAPTIVE_WARMUP_STEPS) / max(1, ADAPTIVE_RAMP_STEPS))
                        lambda_eff = ADAPTIVE_LAMBDA_MAX * ramp
                else:
                    lambda_eff = LAMBDA_DISTILL
                total_loss = ce_loss + lambda_eff * dist_loss
            else:
                total_loss = ce_loss
        else:
            del teacher_out, student_out
            total_loss = ce_loss

        accelerator.backward(total_loss)
        accelerator.clip_grad_norm_(student_wrapper.get_trainable_parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        step += 1

        # Dev metrics logging
        do_dev = (step % DEV_MONITOR_INTERVAL == 0) or (step in ckpt_steps_set)
        if do_dev:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped = accelerator.unwrap_model(model)
                m = compute_dev_metrics(
                    teacher, unwrapped, dev_batch, accelerator.device, teacher_source=teacher_source
                )
                metrics["steps"].append(step)
                metrics["dev_zs_loss"].append(m["dev_zs_loss"])
                metrics["pct_teacher_worse"].append(m["pct_teacher_worse"])
                metrics["mean_loss_diff"].append(m["mean_loss_diff"])
                progress.set_postfix(
                    zs_loss=f"{m['dev_zs_loss']:.4f}",
                    pct_worse=f"{m['pct_teacher_worse']:.1f}",
                )
            accelerator.wait_for_everyone()
            model.train()

        # Checkpoint saving
        if step in ckpt_steps_set:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                ckpt_dir = output_dir / f"checkpoint-{step}"
                accelerator.unwrap_model(model).save_pretrained(str(ckpt_dir))
                tokenizer.save_pretrained(str(ckpt_dir))
                print(f"[{mode}] Saved checkpoint-{step}")
            accelerator.wait_for_everyone()

        progress.update(1)

    progress.close()
    return metrics


# ============================================================================
# vLLM server management
# ============================================================================

def restart_vllm_server(base_model_name, adapter_dirs, port=8001, gpu="0,1,2,3"):
    """
    Kill any existing vLLM server on `port`, start a new one with all
    adapter_dirs registered as --lora-modules name=path.
    Polls /health until ready (up to 3 min).
    """
    import requests

    # Kill existing server on this port
    subprocess.run(f"fuser -k {port}/tcp 2>/dev/null || true", shell=True)
    time.sleep(3)

    lora_args = " ".join(f"{name}={path}" for name, path in adapter_dirs.items())
    cmd = (
        f"CUDA_VISIBLE_DEVICES={gpu} /dev/shm/vllm/bin/vllm serve {base_model_name} "
        f"--enable-lora --max-lora-rank 16 --port {port} "
        f"--max-model-len 512 --dtype bfloat16 "
        f"--lora-modules {lora_args}"
    )
    print(f"[vllm] Starting server with {len(adapter_dirs)} adapters...")
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)

    # Poll until ready
    url = f"http://localhost:{port}/health"
    for _ in range(180):
        time.sleep(1)
        try:
            if requests.get(url, timeout=2).status_code == 200:
                print(f"[vllm] Server ready on port {port}")
                return proc
        except Exception:
            pass
    raise RuntimeError(f"vLLM server did not start within 3 minutes")


# ============================================================================
# Checkpoint evaluation via vLLM server with dynamic LoRA swapping
# ============================================================================

def evaluate_checkpoints(base_model_name, checkpoint_steps, output_dir, eval_data,
                         tokenizer, vllm_port=8001, batch_size=32):
    """
    Evaluate each checkpoint via a running vLLM server pre-loaded with
    --lora-modules {cond}_{step}=<path>.

    Uses aiohttp + asyncio with a semaphore to limit concurrency.

    Returns {step: f1_score}.
    """
    import asyncio
    import aiohttp

    output_dir = Path(output_dir)
    cond_name  = output_dir.name
    url        = f"http://localhost:{vllm_port}/v1/chat/completions"
    results    = {}

    async def _run_all():
        timeout = aiohttp.ClientTimeout(total=120)
        connector = aiohttp.TCPConnector(limit=batch_size)
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            for step in checkpoint_steps:
                ckpt_dir = output_dir / f"checkpoint-{step}"
                if not ckpt_dir.exists():
                    print(f"[eval] Checkpoint {ckpt_dir} not found, skipping.")
                    results[step] = None
                    continue

                adapter_name = f"{cond_name}_{step}"
                print(f"[eval] Evaluating {adapter_name} via vLLM ...")
                sem = asyncio.Semaphore(batch_size)

                async def _call_one(example, _adapter=adapter_name):
                    payload = {
                        "model": _adapter,
                        "messages": build_student_messages(TASK, example),
                        "max_tokens": 150,
                        "temperature": 0,
                        "chat_template_kwargs": {"enable_thinking": False},
                    }
                    async with sem:
                        try:
                            async with session.post(url, json=payload) as resp:
                                data = await resp.json()
                            return data["choices"][0]["message"]["content"] or ""
                        except Exception as e:
                            print(f"[eval] Request failed: {e}")
                            return ""

                texts = await asyncio.gather(*[_call_one(ex) for ex in eval_data])

                all_preds = [parse_output(TASK, t, n_tokens=len(eval_data[i]["tokens"]))
                             for i, t in enumerate(texts)]
                all_golds = [ex["ner_tags"] for ex in eval_data]
                f1 = compute_ner_f1(all_preds, all_golds)
                results[step] = f1
                print(f"[eval] {adapter_name}: F1={f1:.4f}")

    asyncio.run(_run_all())
    return results


# ============================================================================
# Plotting
# ============================================================================

def _trim(metrics, max_step=512):
    """Return a copy of metrics dict with all entries trimmed to max_step."""
    steps = metrics["steps"]
    idx   = [i for i, s in enumerate(steps) if s <= max_step]
    return {k: [v[i] for i in idx] if isinstance(v, list) else v
            for k, v in metrics.items()}


def plot_results(
    distill_metrics,
    sft_metrics,
    ckpt_results,
    output_dir,
    selective_metrics=None,
    online_metrics=None,
    online_selective_metrics=None,
    adaptive_metrics=None,
    max_plot_step=512,
):
    """Generate 4-panel comparison figure and individual PNGs."""
    output_dir = Path(output_dir)
    plots_dir  = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Trim all monitoring metrics to max_plot_step
    dm = _trim(distill_metrics, max_plot_step)
    sm = _trim(sft_metrics,     max_plot_step)
    sel = _trim(selective_metrics, max_plot_step) if selective_metrics else None
    onl = _trim(online_metrics, max_plot_step) if online_metrics else None
    osl = _trim(online_selective_metrics, max_plot_step) if online_selective_metrics else None
    adp = _trim(adaptive_metrics, max_plot_step) if adaptive_metrics else None

    def _ckpt_xy(cond):
        d = ckpt_results.get(cond, {})
        steps = sorted(k for k, v in d.items() if v is not None and k <= max_plot_step)
        return steps, [d[s] for s in steps]

    xs_d, ys_d   = _ckpt_xy("distill")
    xs_s, ys_s   = _ckpt_xy("sft")
    xs_sel, ys_sel = _ckpt_xy("selective") if selective_metrics else ([], [])
    xs_onl, ys_onl = _ckpt_xy("online") if online_metrics else ([], [])
    xs_osl, ys_osl = _ckpt_xy("online_selective") if online_selective_metrics else ([], [])
    xs_adp, ys_adp = _ckpt_xy("adaptive") if adaptive_metrics else ([], [])

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("NER Analysis: Selective Distillation Signal Study", fontsize=14)

    def _add_lines(ax, key=None, xs_d=xs_d, ys_d=ys_d, xs_s=xs_s, ys_s=ys_s):
        if key:
            ax.plot(dm["steps"], dm[key],  "b-",   label="distill",   alpha=0.8)
            ax.plot(sm["steps"], sm[key],  "r--",  label="sft",        alpha=0.8)
            if sel:
                ax.plot(sel["steps"], sel[key], "g-.", label="selective", alpha=0.8)
            if onl:
                ax.plot(onl["steps"], onl[key], "m:", label="online", alpha=0.8)
            if osl:
                ax.plot(osl["steps"], osl[key], color="#00a5a5", linestyle="-.", label="online+selective", alpha=0.8)
            if adp:
                ax.plot(adp["steps"], adp[key], color="#5c5c5c", linestyle="--", label="adaptive", alpha=0.8)
        else:
            if xs_d:  ax.plot(xs_d,   ys_d,   "b-o",  label="distill",   markersize=4)
            if xs_s:  ax.plot(xs_s,   ys_s,   "r--s", label="sft",        markersize=4)
            if xs_sel: ax.plot(xs_sel, ys_sel, "g-^",  label="selective",  markersize=4)
            if xs_onl: ax.plot(xs_onl, ys_onl, "m-D", label="online", markersize=4)
            if xs_osl: ax.plot(xs_osl, ys_osl, color="#00a5a5", marker="P", linestyle="-", label="online+selective", markersize=4)
            if xs_adp: ax.plot(xs_adp, ys_adp, color="#5c5c5c", marker="X", linestyle="--", label="adaptive", markersize=4)

    # Panel 1 — Checkpoint F1
    ax = axes[0, 0]
    _add_lines(ax)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("Checkpoint step"); ax.set_ylabel("NER F1")
    ax.set_title("Checkpoint Eval F1 (WikiANN en dev)")
    ax.legend(); ax.grid(True, alpha=0.3)

    # Panel 2 — Dev ZS loss
    ax = axes[0, 1]
    _add_lines(ax, key="dev_zs_loss")
    ax.set_xlabel("Training step"); ax.set_ylabel("Zero-shot CE loss")
    ax.set_title("Dev Set Zero-Shot CE Loss")
    ax.legend(); ax.grid(True, alpha=0.3)

    # Panel 3 — % tokens where teacher worse
    ax = axes[1, 0]
    _add_lines(ax, key="pct_teacher_worse")
    ax.axhline(50, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Training step"); ax.set_ylabel("% tokens")
    ax.set_title("% Tokens Where Few-Shot Loss > Zero-Shot Loss")
    ax.legend(); ax.grid(True, alpha=0.3)

    # Panel 4 — Mean loss diff
    ax = axes[1, 1]
    _add_lines(ax, key="mean_loss_diff")
    ax.axhline(0, color="black", linestyle="-", linewidth=0.8)
    ax.set_xlabel("Training step"); ax.set_ylabel("Mean CE diff (few-shot − zero-shot)")
    ax.set_title("Mean Few-Shot Loss − Zero-Shot Loss")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(str(plots_dir / "ner_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved plots/ner_analysis.png")

    # Individual PNGs
    def _save_individual(fname, ylabel, title, hline=None, xscale=None,
                         key=None, xs_d=xs_d, ys_d=ys_d, xs_s=xs_s, ys_s=ys_s):
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        if key:
            ax2.plot(dm["steps"], dm[key],  "b-",   label="distill",   alpha=0.85)
            ax2.plot(sm["steps"], sm[key],  "r--",  label="sft",        alpha=0.85)
            if sel:
                ax2.plot(sel["steps"], sel[key], "g-.", label="selective", alpha=0.85)
            if onl:
                ax2.plot(onl["steps"], onl[key], "m:", label="online", alpha=0.85)
            if osl:
                ax2.plot(osl["steps"], osl[key], color="#00a5a5", linestyle="-.", label="online+selective", alpha=0.85)
            if adp:
                ax2.plot(adp["steps"], adp[key], color="#5c5c5c", linestyle="--", label="adaptive", alpha=0.85)
        else:
            if xs_d:   ax2.plot(xs_d,   ys_d,   "b-o",  label="distill",   markersize=3)
            if xs_s:   ax2.plot(xs_s,   ys_s,   "r--s", label="sft",        markersize=3)
            if xs_sel: ax2.plot(xs_sel, ys_sel, "g-^",  label="selective",  markersize=3)
            if xs_onl: ax2.plot(xs_onl, ys_onl, "m-D", label="online", markersize=3)
            if xs_osl: ax2.plot(xs_osl, ys_osl, color="#00a5a5", marker="P", linestyle="-", label="online+selective", markersize=3)
            if xs_adp: ax2.plot(xs_adp, ys_adp, color="#5c5c5c", marker="X", linestyle="--", label="adaptive", markersize=3)
        if hline is not None:
            ax2.axhline(hline, color="gray" if hline != 0 else "black",
                        linestyle=":" if hline != 0 else "-", linewidth=0.9)
        if xscale:
            ax2.set_xscale(xscale, linthresh=1)
        ax2.set_xlabel("Step"); ax2.set_ylabel(ylabel); ax2.set_title(title)
        ax2.legend(); ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        fig2.savefig(str(plots_dir / fname), dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"[plot] Saved plots/{fname}")

    _save_individual("checkpoint_eval.png",    "NER F1",   "Checkpoint Eval F1",   xscale="symlog")
    _save_individual("dev_loss.png",           "Zero-shot CE loss", "Dev Set Zero-Shot CE Loss", key="dev_zs_loss")
    _save_individual("pct_teacher_worse.png",  "% tokens", "% Tokens Where Few-Shot Loss > Zero-Shot Loss",
                     hline=50, key="pct_teacher_worse")
    _save_individual("loss_diff.png",          "Mean CE diff (few-shot − zero-shot)",
                     "Mean Few-Shot Loss − Zero-Shot Loss", hline=0, key="mean_loss_diff")


# ============================================================================
# Report
# ============================================================================

def write_report(
    distill_metrics,
    sft_metrics,
    ckpt_results,
    output_dir,
    selective_metrics=None,
    online_metrics=None,
    online_selective_metrics=None,
    adaptive_metrics=None,
):
    """Write ner_report.md with setup summary, F1 table, and observations."""
    output_dir = Path(output_dir)

    # Final F1 values
    def _final_f1(results):
        steps = sorted(k for k, v in results.items() if v is not None)
        if not steps:
            return "N/A"
        return f"{results[steps[-1]]:.4f} (step {steps[-1]})"

    final_d   = _final_f1(ckpt_results.get("distill",   {}))
    final_s   = _final_f1(ckpt_results.get("sft",       {}))
    final_sel = _final_f1(ckpt_results.get("selective", {}))
    final_onl = _final_f1(ckpt_results.get("online",    {}))
    final_osl = _final_f1(ckpt_results.get("online_selective", {}))
    final_adp = _final_f1(ckpt_results.get("adaptive",  {}))

    # Last monitoring values
    def _last(metrics, key):
        vals = metrics.get(key, [])
        return f"{vals[-1]:.4f}" if vals else "N/A"

    pct_d   = _last(distill_metrics,   "pct_teacher_worse")
    pct_s   = _last(sft_metrics,       "pct_teacher_worse")
    pct_sel = _last(selective_metrics, "pct_teacher_worse") if selective_metrics else "N/A"
    pct_onl = _last(online_metrics, "pct_teacher_worse") if online_metrics else "N/A"
    pct_osl = _last(online_selective_metrics, "pct_teacher_worse") if online_selective_metrics else "N/A"
    pct_adp = _last(adaptive_metrics, "pct_teacher_worse") if adaptive_metrics else "N/A"
    diff_d  = _last(distill_metrics,   "mean_loss_diff")
    diff_s  = _last(sft_metrics,       "mean_loss_diff")
    diff_sel= _last(selective_metrics, "mean_loss_diff") if selective_metrics else "N/A"
    diff_onl= _last(online_metrics, "mean_loss_diff") if online_metrics else "N/A"
    diff_osl= _last(online_selective_metrics, "mean_loss_diff") if online_selective_metrics else "N/A"
    diff_adp= _last(adaptive_metrics, "mean_loss_diff") if adaptive_metrics else "N/A"

    report = f"""# NER Selective Distillation Analysis Report

## Experiment Setup

| Parameter         | Value                              |
|-------------------|------------------------------------|
| Model             | {MODEL_NAME}                       |
| Task              | WikiANN NER (English)              |
| Few-shot examples | {NUM_FEWSHOT[TASK]}                |
| LoRA              | r={LORA_CONFIG['r']}, α={LORA_CONFIG['alpha']} |
| Batch size        | {PER_DEVICE_BS * 4} (effective, 4 GPUs) |
| Learning rate     | {LR}                               |
| Max steps         | {MAX_STEPS}                        |
| λ (distill)       | {LAMBDA_DISTILL}                   |
| Top-K logits      | {N_TOP_LOGITS}                     |
| Dev monitor every | {DEV_MONITOR_INTERVAL} steps       |

## Final F1 Scores

| Condition  | Final F1 (≤512 steps) |
|------------|-----------------------|
| Distill    | {final_d}             |
| SFT        | {final_s}             |
| Selective  | {final_sel}           |
| Online     | {final_onl}           |
| Online+Selective | {final_osl}      |
| Adaptive   | {final_adp}           |

## Key Observations

### Plot 3: % Tokens Where Few-Shot Loss > Zero-Shot Loss

- **Distill (final)**:   {pct_d}% of labeled tokens have higher teacher CE than student CE
- **SFT (final)**:      {pct_s}% of labeled tokens have higher teacher CE than student CE
- **Selective (final)**: {pct_sel}% of labeled tokens have higher teacher CE than student CE
- **Online (final)**:   {pct_onl}% of labeled tokens have higher teacher CE than student CE
- **Online+Selective (final)**: {pct_osl}% of labeled tokens have higher teacher CE than student CE
- **Adaptive (final)**: {pct_adp}% of labeled tokens have higher teacher CE than student CE

A value above 50% means the few-shot teacher is *worse* than the zero-shot student on
more than half of the labeled tokens. This suggests the teacher signal is not uniformly
beneficial and that selective distillation (only apply distillation where teacher loss <
student loss) could filter out harmful signal.

### Plot 4: Mean Few-Shot Loss − Zero-Shot Loss

- **Distill (final)**:   {diff_d}
- **SFT (final)**:      {diff_s}
- **Selective (final)**: {diff_sel}
- **Online (final)**:   {diff_onl}
- **Online+Selective (final)**: {diff_osl}
- **Adaptive (final)**: {diff_adp}

A positive value means the few-shot teacher is on average *worse* than the zero-shot
student at predicting the gold tokens — i.e. the teacher introduces noise. A negative
value means the teacher provides useful signal on average.

## Conclusion: Selective Distillation Viability

{"The analysis supports selective distillation" if float(pct_d.replace("N/A","0")) > 30 else "The analysis does not strongly support selective distillation"}.

If a significant fraction (>30–40%) of labeled tokens have worse teacher predictions
than student predictions, applying distillation uniformly adds harmful gradient signal.
A selective strategy — computing per-token loss differences and only distilling where
`loss_teacher < loss_student` — would retain beneficial signal while filtering noise.

The checkpoint F1 curves (Plot 1) show the realized training effect: if SFT matches or
exceeds distill, the unfiltered distillation signal is likely hurting generalization on
NER. Selective distillation is expected to close or reverse this gap.

## Files

- `plots/ner_analysis.png`     — 4-panel combined figure
- `plots/checkpoint_eval.png`  — Checkpoint F1 curves
- `plots/dev_loss.png`         — Dev zero-shot CE loss
- `plots/pct_teacher_worse.png` — % tokens teacher worse
- `plots/loss_diff.png`        — Mean loss difference
- `eval_results.json`          — Numeric F1 results per step
- `distill/metrics.json`       — Per-step monitoring (distill)
- `sft/metrics.json`           — Per-step monitoring (sft)
- `selective/metrics.json`     — Per-step monitoring (selective)
- `online/metrics.json`        — Per-step monitoring (online)
- `online_selective/metrics.json` — Per-step monitoring (online+selective)
- `adaptive/metrics.json`      — Per-step monitoring (adaptive)
"""

    report_path = output_dir / "ner_report.md"
    report_path.write_text(report)
    print(f"[report] Written to {report_path}")


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="NER selective distillation analysis")
    parser.add_argument("--output_dir",  type=str,  default="experiments/ner_analysis")
    parser.add_argument("--model",       type=str,  default=MODEL_NAME)
    parser.add_argument("--eval_only",      action="store_true",
                        help="Skip training; load existing metrics.json and run eval+plot+report")
    parser.add_argument("--selective_only", action="store_true",
                        help="Train selective condition only (reuse existing distill/sft results)")
    parser.add_argument("--online_only", action="store_true",
                        help="Train online-teacher condition only (reuse existing distill/sft results)")
    parser.add_argument("--online_selective_only", action="store_true",
                        help="Train online+selective condition only (reuse existing distill/sft results)")
    parser.add_argument("--adaptive_only", action="store_true",
                        help="Train adaptive hypothesis condition only (reuse existing distill/sft results)")
    parser.add_argument("--vllm_port",      type=int,  default=8001,
                        help="Port of running vLLM server for checkpoint eval")
    args = parser.parse_args()
    one_only_flags = [
        args.selective_only,
        args.online_only,
        args.online_selective_only,
        args.adaptive_only,
    ]
    if sum(int(x) for x in one_only_flags) > 1:
        parser.error("Use at most one of --selective_only, --online_only, --online_selective_only, --adaptive_only")
    return args


def main():
    args       = parse_args()
    output_dir = Path(args.output_dir)
    model_name = args.model

    # ---- Eval-only mode: skip training, load saved metrics, run eval+plot+report ----
    if args.eval_only:
        output_dir.mkdir(parents=True, exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        dev_data_raw = load_task_data(TASK, "dev", LANG, max_samples=200)
        eval_data    = dev_data_raw[:EVAL_DEV_SAMPLES]

        with open(output_dir / "distill" / "metrics.json") as f:
            distill_metrics = json.load(f)
        with open(output_dir / "sft" / "metrics.json") as f:
            sft_metrics = json.load(f)
        selective_metrics = None
        online_metrics = None
        online_selective_metrics = None
        adaptive_metrics = None
        if (output_dir / "selective" / "metrics.json").exists():
            with open(output_dir / "selective" / "metrics.json") as f:
                selective_metrics = json.load(f)
        if (output_dir / "online" / "metrics.json").exists():
            with open(output_dir / "online" / "metrics.json") as f:
                online_metrics = json.load(f)
        if (output_dir / "online_selective" / "metrics.json").exists():
            with open(output_dir / "online_selective" / "metrics.json") as f:
                online_selective_metrics = json.load(f)
        if (output_dir / "adaptive" / "metrics.json").exists():
            with open(output_dir / "adaptive" / "metrics.json") as f:
                adaptive_metrics = json.load(f)

        # Load existing eval results if present
        eval_path = output_dir / "eval_results.json"
        if (args.selective_only or args.online_only or args.online_selective_only or args.adaptive_only) and eval_path.exists():
            with open(eval_path) as f:
                eval_results = {cond: {int(k): v for k, v in res.items()}
                                for cond, res in json.load(f).items()}
        else:
            eval_results = {}
            for cond in ["distill", "sft"]:
                print(f"\n--- Evaluating {cond} checkpoints ---")
                eval_results[cond] = evaluate_checkpoints(
                    base_model_name=model_name,
                    checkpoint_steps=[s for s in CHECKPOINT_STEPS if s <= 512],
                    output_dir=output_dir / cond,
                    eval_data=eval_data, tokenizer=tokenizer, vllm_port=args.vllm_port,
                )

        if selective_metrics is not None and not (args.online_only or args.online_selective_only or args.adaptive_only):
            print(f"\n--- Evaluating selective checkpoints ---")
            eval_results["selective"] = evaluate_checkpoints(
                base_model_name=model_name,
                checkpoint_steps=CHECKPOINT_STEPS_SELECTIVE,
                output_dir=output_dir / "selective",
                eval_data=eval_data, tokenizer=tokenizer, vllm_port=args.vllm_port,
            )
        if online_metrics is not None and not (args.selective_only or args.online_selective_only or args.adaptive_only):
            print(f"\n--- Evaluating online checkpoints ---")
            eval_results["online"] = evaluate_checkpoints(
                base_model_name=model_name,
                checkpoint_steps=CHECKPOINT_STEPS_ONLINE,
                output_dir=output_dir / "online",
                eval_data=eval_data, tokenizer=tokenizer, vllm_port=args.vllm_port,
            )
        if online_selective_metrics is not None and args.online_selective_only:
            print(f"\n--- Evaluating online_selective checkpoints ---")
            eval_results["online_selective"] = evaluate_checkpoints(
                base_model_name=model_name,
                checkpoint_steps=CHECKPOINT_STEPS_ONLINE_SELECTIVE,
                output_dir=output_dir / "online_selective",
                eval_data=eval_data, tokenizer=tokenizer, vllm_port=args.vllm_port,
            )
        if adaptive_metrics is not None and args.adaptive_only:
            print(f"\n--- Evaluating adaptive checkpoints ---")
            eval_results["adaptive"] = evaluate_checkpoints(
                base_model_name=model_name,
                checkpoint_steps=CHECKPOINT_STEPS_ADAPTIVE,
                output_dir=output_dir / "adaptive",
                eval_data=eval_data, tokenizer=tokenizer, vllm_port=args.vllm_port,
            )

        with open(eval_path, "w") as f:
            json.dump(eval_results, f, indent=2)
        plot_results(distill_metrics, sft_metrics, eval_results, output_dir,
                     selective_metrics=selective_metrics,
                     online_metrics=online_metrics,
                     online_selective_metrics=online_selective_metrics,
                     adaptive_metrics=adaptive_metrics,
                     max_plot_step=512)
        write_report(distill_metrics, sft_metrics, eval_results, output_dir,
                     selective_metrics=selective_metrics,
                     online_metrics=online_metrics,
                     online_selective_metrics=online_selective_metrics,
                     adaptive_metrics=adaptive_metrics)
        print(f"\n=== Done! Results in {output_dir} ===")
        return

    # Kill vLLM server before training to free GPU memory (all ranks, idempotent)
    subprocess.run(f"fuser -k {args.vllm_port}/tcp 2>/dev/null || true", shell=True)
    time.sleep(2)

    # ---- Accelerator ----
    accelerator = Accelerator(mixed_precision="bf16")

    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        if args.selective_only:
            mode_str = "Selective only"
        elif args.online_only:
            mode_str = "Online only"
        elif args.online_selective_only:
            mode_str = "Online+Selective only"
        elif args.adaptive_only:
            mode_str = "Adaptive only"
        else:
            mode_str = "Full (distill + sft + selective + online)"
        print(f"\n=== NER Selective Distillation Analysis ===")
        print(f"  Model      : {model_name}")
        print(f"  Output dir : {output_dir}")
        print(f"  Num GPUs   : {accelerator.num_processes}")
        print(f"  Mode       : {mode_str}")

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Teacher ----
    teacher = None
    if not (args.online_only or args.online_selective_only or args.adaptive_only):
        if accelerator.is_main_process:
            print("\nLoading frozen teacher...")
        teacher = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(accelerator.device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
    elif accelerator.is_main_process:
        print("\nOnline-like-only mode: skipping frozen teacher load.")

    # ---- Data ----
    if accelerator.is_main_process:
        print("\nLoading WikiANN NER data...")

    train_dataset = XTREMEDistillDataset(
        tokenizer=tokenizer,
        tasks=[TASK],
        train_langs=[LANG],
        max_samples_per_task_lang=20000,
        max_seq_len_teacher=MAX_SEQ_LEN_T,
        max_seq_len_student=MAX_SEQ_LEN_S,
        seed=42,
        teacher_include_answer=True,
        split="train",
    )

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    train_loader = DataLoader(
        train_dataset,
        batch_size=PER_DEVICE_BS,
        shuffle=True,
        num_workers=4,
        collate_fn=lambda b: collate_fn_xtreme(b, pad_id),
        pin_memory=True,
    )

    # Dev data for monitoring
    dev_data_raw = load_task_data(TASK, "dev", LANG, max_samples=200)

    # Build fixed dev_batch (DEV_MONITOR_SAMPLES examples)
    dev_samples_raw = dev_data_raw[:DEV_MONITOR_SAMPLES]
    dev_dataset = XTREMEDistillDataset(
        tokenizer=tokenizer,
        tasks=[TASK],
        train_langs=[LANG],
        max_samples_per_task_lang=DEV_MONITOR_SAMPLES + 10,
        max_seq_len_teacher=MAX_SEQ_LEN_T,
        max_seq_len_student=MAX_SEQ_LEN_S,
        seed=99,
        teacher_include_answer=True,
        split="dev",
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=DEV_MONITOR_SAMPLES,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: collate_fn_xtreme(b, pad_id),
    )
    dev_batch = next(iter(dev_loader))
    # Move to device
    dev_batch.teacher_input_ids      = dev_batch.teacher_input_ids.to(accelerator.device)
    dev_batch.teacher_attention_mask = dev_batch.teacher_attention_mask.to(accelerator.device)
    dev_batch.student_input_ids      = dev_batch.student_input_ids.to(accelerator.device)
    dev_batch.student_attention_mask = dev_batch.student_attention_mask.to(accelerator.device)
    dev_batch.labels                 = dev_batch.labels.to(accelerator.device)

    if not (args.selective_only or args.online_only or args.online_selective_only or args.adaptive_only):
        # ---- CONDITION 1: DISTILL ----
        if accelerator.is_main_process:
            print("\n" + "="*50)
            print("CONDITION 1: Distillation (CE + λ·MSE)")
            print("="*50)

        student_wrapper = StudentModel(
            model_name=model_name, lora_config=LORA_CONFIG,
            use_lora=True, num_layers=NUM_LAYERS, device_map=None,
        )
        distill_metrics = run_training(
            mode="distill", accelerator=accelerator, tokenizer=tokenizer,
            teacher=teacher, student_wrapper=student_wrapper,
            train_loader=train_loader, dev_batch=dev_batch,
            checkpoint_steps=CHECKPOINT_STEPS, output_dir=output_dir / "distill",
        )
        if accelerator.is_main_process:
            with open(output_dir / "distill" / "metrics.json", "w") as f:
                json.dump(distill_metrics, f, indent=2)
        del student_wrapper; gc.collect(); torch.cuda.empty_cache()
        accelerator.wait_for_everyone()

        # ---- CONDITION 2: SFT ----
        if accelerator.is_main_process:
            print("\n" + "="*50)
            print("CONDITION 2: SFT baseline (CE only)")
            print("="*50)

        student_wrapper = StudentModel(
            model_name=model_name, lora_config=LORA_CONFIG,
            use_lora=True, num_layers=NUM_LAYERS, device_map=None,
        )
        sft_metrics = run_training(
            mode="sft", accelerator=accelerator, tokenizer=tokenizer,
            teacher=teacher, student_wrapper=student_wrapper,
            train_loader=train_loader, dev_batch=dev_batch,
            checkpoint_steps=CHECKPOINT_STEPS, output_dir=output_dir / "sft",
        )
        if accelerator.is_main_process:
            with open(output_dir / "sft" / "metrics.json", "w") as f:
                json.dump(sft_metrics, f, indent=2)
        del student_wrapper; gc.collect(); torch.cuda.empty_cache()
        accelerator.wait_for_everyone()

    else:
        # Load existing distill/sft metrics
        if accelerator.is_main_process:
            with open(output_dir / "distill" / "metrics.json") as f:
                distill_metrics = json.load(f)
            with open(output_dir / "sft" / "metrics.json") as f:
                sft_metrics = json.load(f)
        else:
            distill_metrics = sft_metrics = None

    selective_metrics = None
    if not (args.online_only or args.online_selective_only or args.adaptive_only):
        # ---- CONDITION 3: SELECTIVE DISTILLATION ----
        if accelerator.is_main_process:
            print("\n" + "="*50)
            print("CONDITION 3: Selective Distillation")
            print("="*50)

        student_wrapper = StudentModel(
            model_name=model_name, lora_config=LORA_CONFIG,
            use_lora=True, num_layers=NUM_LAYERS, device_map=None,
        )
        selective_metrics = run_training(
            mode="selective", accelerator=accelerator, tokenizer=tokenizer,
            teacher=teacher, student_wrapper=student_wrapper,
            train_loader=train_loader, dev_batch=dev_batch,
            checkpoint_steps=CHECKPOINT_STEPS_SELECTIVE,
            output_dir=output_dir / "selective",
            max_steps_override=MAX_STEPS_SELECTIVE,
        )
        if accelerator.is_main_process:
            with open(output_dir / "selective" / "metrics.json", "w") as f:
                json.dump(selective_metrics, f, indent=2)
        del student_wrapper; gc.collect(); torch.cuda.empty_cache()
        accelerator.wait_for_everyone()
    elif accelerator.is_main_process and (output_dir / "selective" / "metrics.json").exists():
        with open(output_dir / "selective" / "metrics.json") as f:
            selective_metrics = json.load(f)

    online_metrics = None
    if not (args.selective_only or args.online_selective_only or args.adaptive_only):
        # ---- CONDITION 4: ONLINE TEACHER ----
        if accelerator.is_main_process:
            print("\n" + "="*50)
            print("CONDITION 4: Online Teacher Distillation")
            print("="*50)

        student_wrapper = StudentModel(
            model_name=model_name, lora_config=LORA_CONFIG,
            use_lora=True, num_layers=NUM_LAYERS, device_map=None,
        )
        online_metrics = run_training(
            mode="online", accelerator=accelerator, tokenizer=tokenizer,
            teacher=teacher, student_wrapper=student_wrapper,
            train_loader=train_loader, dev_batch=dev_batch,
            checkpoint_steps=CHECKPOINT_STEPS_ONLINE,
            output_dir=output_dir / "online",
            max_steps_override=MAX_STEPS_ONLINE,
        )
        if accelerator.is_main_process:
            with open(output_dir / "online" / "metrics.json", "w") as f:
                json.dump(online_metrics, f, indent=2)
        del student_wrapper; gc.collect(); torch.cuda.empty_cache()
        accelerator.wait_for_everyone()
    elif accelerator.is_main_process and (output_dir / "online" / "metrics.json").exists():
        with open(output_dir / "online" / "metrics.json") as f:
            online_metrics = json.load(f)

    online_selective_metrics = None
    if args.online_selective_only:
        if accelerator.is_main_process:
            print("\n" + "="*50)
            print("CONDITION 5: Online + Selective Distillation")
            print("="*50)
        student_wrapper = StudentModel(
            model_name=model_name, lora_config=LORA_CONFIG,
            use_lora=True, num_layers=NUM_LAYERS, device_map=None,
        )
        online_selective_metrics = run_training(
            mode="online_selective", accelerator=accelerator, tokenizer=tokenizer,
            teacher=teacher, student_wrapper=student_wrapper,
            train_loader=train_loader, dev_batch=dev_batch,
            checkpoint_steps=CHECKPOINT_STEPS_ONLINE_SELECTIVE,
            output_dir=output_dir / "online_selective",
            max_steps_override=MAX_STEPS_ONLINE_SELECTIVE,
        )
        if accelerator.is_main_process:
            with open(output_dir / "online_selective" / "metrics.json", "w") as f:
                json.dump(online_selective_metrics, f, indent=2)
        del student_wrapper; gc.collect(); torch.cuda.empty_cache()
        accelerator.wait_for_everyone()
    elif accelerator.is_main_process and (output_dir / "online_selective" / "metrics.json").exists():
        with open(output_dir / "online_selective" / "metrics.json") as f:
            online_selective_metrics = json.load(f)

    adaptive_metrics = None
    if args.adaptive_only:
        if accelerator.is_main_process:
            print("\n" + "="*50)
            print("CONDITION 6: Adaptive Online Distillation")
            print("="*50)
        student_wrapper = StudentModel(
            model_name=model_name, lora_config=LORA_CONFIG,
            use_lora=True, num_layers=NUM_LAYERS, device_map=None,
        )
        adaptive_metrics = run_training(
            mode="adaptive", accelerator=accelerator, tokenizer=tokenizer,
            teacher=teacher, student_wrapper=student_wrapper,
            train_loader=train_loader, dev_batch=dev_batch,
            checkpoint_steps=CHECKPOINT_STEPS_ADAPTIVE,
            output_dir=output_dir / "adaptive",
            max_steps_override=MAX_STEPS_ADAPTIVE,
        )
        if accelerator.is_main_process:
            with open(output_dir / "adaptive" / "metrics.json", "w") as f:
                json.dump(adaptive_metrics, f, indent=2)
        del student_wrapper; gc.collect(); torch.cuda.empty_cache()
        accelerator.wait_for_everyone()
    elif accelerator.is_main_process and (output_dir / "adaptive" / "metrics.json").exists():
        with open(output_dir / "adaptive" / "metrics.json") as f:
            adaptive_metrics = json.load(f)

    # ---- Evaluation (main process only) ----
    if accelerator.is_main_process:
        print("\n" + "="*50)
        print("EVALUATING CHECKPOINTS")
        print("="*50)

        eval_data = dev_data_raw[:EVAL_DEV_SAMPLES]

        # Load or init existing eval results
        eval_path = output_dir / "eval_results.json"
        if (args.selective_only or args.online_only or args.online_selective_only or args.adaptive_only) and eval_path.exists():
            with open(eval_path) as f:
                eval_results = {cond: {int(k): v for k, v in res.items()}
                                for cond, res in json.load(f).items()}
        else:
            eval_results = {}
            for cond in ["distill", "sft"]:
                print(f"\n--- Evaluating {cond} checkpoints ---")
                eval_results[cond] = evaluate_checkpoints(
                    base_model_name=model_name,
                    checkpoint_steps=[s for s in CHECKPOINT_STEPS if s <= 512],
                    output_dir=output_dir / cond,
                    eval_data=eval_data, tokenizer=tokenizer, vllm_port=args.vllm_port,
                )

        if not (args.online_only or args.online_selective_only or args.adaptive_only):
            print(f"\n--- Evaluating selective checkpoints ---")
            eval_results["selective"] = evaluate_checkpoints(
                base_model_name=model_name,
                checkpoint_steps=CHECKPOINT_STEPS_SELECTIVE,
                output_dir=output_dir / "selective",
                eval_data=eval_data, tokenizer=tokenizer, vllm_port=args.vllm_port,
            )
        if not (args.selective_only or args.online_selective_only or args.adaptive_only):
            print(f"\n--- Evaluating online checkpoints ---")
            eval_results["online"] = evaluate_checkpoints(
                base_model_name=model_name,
                checkpoint_steps=CHECKPOINT_STEPS_ONLINE,
                output_dir=output_dir / "online",
                eval_data=eval_data, tokenizer=tokenizer, vllm_port=args.vllm_port,
            )
        if args.online_selective_only:
            print(f"\n--- Evaluating online_selective checkpoints ---")
            eval_results["online_selective"] = evaluate_checkpoints(
                base_model_name=model_name,
                checkpoint_steps=CHECKPOINT_STEPS_ONLINE_SELECTIVE,
                output_dir=output_dir / "online_selective",
                eval_data=eval_data, tokenizer=tokenizer, vllm_port=args.vllm_port,
            )
        if args.adaptive_only:
            print(f"\n--- Evaluating adaptive checkpoints ---")
            eval_results["adaptive"] = evaluate_checkpoints(
                base_model_name=model_name,
                checkpoint_steps=CHECKPOINT_STEPS_ADAPTIVE,
                output_dir=output_dir / "adaptive",
                eval_data=eval_data, tokenizer=tokenizer, vllm_port=args.vllm_port,
            )

        with open(eval_path, "w") as f:
            json.dump(eval_results, f, indent=2)

        plot_results(distill_metrics, sft_metrics, eval_results, output_dir,
                     selective_metrics=selective_metrics,
                     online_metrics=online_metrics,
                     online_selective_metrics=online_selective_metrics,
                     adaptive_metrics=adaptive_metrics,
                     max_plot_step=512)
        write_report(distill_metrics, sft_metrics, eval_results, output_dir,
                     selective_metrics=selective_metrics,
                     online_metrics=online_metrics,
                     online_selective_metrics=online_selective_metrics,
                     adaptive_metrics=adaptive_metrics)
        print(f"\n=== Done! Results in {output_dir} ===")

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
