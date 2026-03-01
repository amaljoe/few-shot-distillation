"""
Loss curve experiment: track zero-shot vs few-shot dev loss during SFT/distillation.

Trains Qwen3-1.7B (LoRA) on a chosen dataset for 200 steps.
Every 16 steps evaluates CE loss on 32 dev examples in two formats:
  - zs: zero-shot  — question → answer              (loss on answer tokens only)
  - fs: few-shot   — k-shot context + question → answer (loss on answer tokens only)

Mode:
  baseline  CE loss only (standard SFT)
  distill   CE + λ·MSE on top-K teacher logits (our method)

Dataset:
  gsm8k          8-shot, math reasoning
  math           4-shot, MATH-lighteval
  commonsenseqa  5-shot, multiple-choice

Usage (run from project root inside the apptainer container):
  CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \\
      --num_processes 4 --mixed_precision bf16 --main_process_port 29500 \\
      scripts/loss_curve_experiment.py --mode baseline --dataset gsm8k

  CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \\
      --num_processes 4 --mixed_precision bf16 --main_process_port 29500 \\
      scripts/loss_curve_experiment.py --mode distill --dataset math
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent.parent))

from accelerate import Accelerator
from src.data.gsm8k_loader import (
    apply_chat_template_no_think,
    collate_fn,
)
import src.data.gsm8k_loader as _gsm8k
import src.data.math_loader as _math
import src.data.commonsenseqa_loader as _csqa
from src.data.loader_factory import load_dataset_split, make_dataloader as _factory_dl

# ── Hyperparameters ───────────────────────────────────────────────────────────
MODEL_NAME      = "Qwen/Qwen3-1.7B"
DEV_SIZE        = 32
DEV_BATCH       = 4         # mini-batch size for dev evaluation
TRAIN_STEPS     = 200
EVAL_STEPS      = 16
BATCH_SIZE      = 4         # per device
LR              = 2e-4
WARMUP_STEPS    = 20
WEIGHT_DECAY    = 0.01
GRAD_CLIP       = 1.0
SEED            = 42
K_VOCAB         = 256       # distillation: top-K vocab indices from teacher
LAM_DISTILL     = 0.5       # distillation: loss weight λ

# ── Per-dataset config ────────────────────────────────────────────────────────
# Each entry provides the callables and seq-len defaults needed for dev batch
# construction and training, so the rest of the script stays dataset-agnostic.
DATASET_CONFIGS = {
    "gsm8k": {
        "num_fewshot":            8,
        "max_seq_student":        512,
        "max_seq_teacher":        4096,
        "build_fewshot_messages": _gsm8k.build_fewshot_messages,
        "build_student_messages": _gsm8k.build_student_messages,
        "get_answer":             lambda ex: ex["answer"],
    },
    "math": {
        "num_fewshot":            4,
        "max_seq_student":        1024,
        "max_seq_teacher":        6144,
        "build_fewshot_messages": _math.build_fewshot_messages,
        "build_student_messages": _math.build_student_messages,
        "get_answer":             lambda ex: ex["solution"],
    },
    "commonsenseqa": {
        "num_fewshot":            5,
        "max_seq_student":        256,
        "max_seq_teacher":        1024,
        "build_fewshot_messages": _csqa.build_fewshot_messages,
        "build_student_messages": _csqa.build_student_messages,
        "get_answer":             _csqa.get_answer_text,
    },
}

LORA_KWARGS = dict(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _template_with_answer(tokenizer, messages_ending_with_assistant):
    """Apply chat template to a sequence that ends with an assistant answer (no gen prompt)."""
    try:
        return tokenizer.apply_chat_template(
            messages_ending_with_assistant,
            tokenize=False,
            add_generation_prompt=False,
            chat_template_kwargs={"enable_thinking": False},
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages_ending_with_assistant,
            tokenize=False,
            add_generation_prompt=False,
        )


# ── Dev set builder ───────────────────────────────────────────────────────────

def build_dev_batches(tokenizer, dataset_name):
    """
    Build fixed zero-shot and few-shot dev batches from DEV_SIZE test examples.

    For each example:
      zs: full_ids = [question tokens][answer tokens]
          labels   = [-100 × prompt_len] + [answer tokens]
      fs: full_ids = [k-shot context tokens][question tokens][answer tokens]
          labels   = [-100 × (total - n_ans)] + [last n_ans tokens]

    where n_ans = len(zs_full_ids) - prompt_len  (same answer token count in both formats).

    Returns:
        zs_batches, fs_batches: lists of {input_ids, attention_mask, labels} dicts (tensors)
    """
    cfg        = DATASET_CONFIGS[dataset_name]
    test_data  = load_dataset_split(dataset_name, "test")
    train_data = load_dataset_split(dataset_name, "train")
    pad_id     = tokenizer.pad_token_id or tokenizer.eos_token_id

    max_seq_student = cfg["max_seq_student"]
    max_seq_teacher = cfg["max_seq_teacher"]
    num_fewshot     = cfg["num_fewshot"]

    rng = random.Random(SEED)
    indices = rng.sample(range(len(test_data)), DEV_SIZE)

    zs_items, fs_items = [], []

    for idx in indices:
        example = test_data[idx]
        answer  = cfg["get_answer"](example)

        # Sample few-shot examples from training set (deterministic per example)
        shot_rng = random.Random(SEED * 10000 + idx)
        shots = [train_data[i] for i in shot_rng.sample(range(len(train_data)), num_fewshot)]

        # ── Zero-shot ──────────────────────────────────────────────────────
        zs_msgs       = cfg["build_student_messages"](example)
        zs_prompt_str = apply_chat_template_no_think(tokenizer, zs_msgs)
        zs_prompt_ids = tokenizer(zs_prompt_str, add_special_tokens=False)["input_ids"]
        prompt_len    = len(zs_prompt_ids)

        zs_full_msgs = zs_msgs + [{"role": "assistant", "content": answer}]
        zs_full_str  = _template_with_answer(tokenizer, zs_full_msgs)
        zs_enc = tokenizer(
            zs_full_str, add_special_tokens=False,
            max_length=max_seq_student, truncation=True,
        )
        zs_ids = zs_enc["input_ids"]
        n_ans  = len(zs_ids) - prompt_len
        zs_labels = [-100] * prompt_len + zs_ids[prompt_len:]

        zs_items.append({"ids": zs_ids, "mask": zs_enc["attention_mask"], "labels": zs_labels})

        # ── Few-shot ───────────────────────────────────────────────────────
        fs_msgs = cfg["build_fewshot_messages"](shots, example) + [
            {"role": "assistant", "content": answer}
        ]
        fs_str = _template_with_answer(tokenizer, fs_msgs)
        fs_enc = tokenizer(
            fs_str, add_special_tokens=False,
            max_length=max_seq_teacher, truncation=True,
        )
        fs_ids  = fs_enc["input_ids"]
        n_valid = min(n_ans, len(fs_ids))
        fs_labels = [-100] * (len(fs_ids) - n_valid) + fs_ids[-n_valid:]

        fs_items.append({"ids": fs_ids, "mask": fs_enc["attention_mask"], "labels": fs_labels})

    def to_batches(items, bsz, pad):
        batches = []
        for i in range(0, len(items), bsz):
            chunk = items[i:i + bsz]
            ml = max(len(x["ids"]) for x in chunk)
            ids_t  = [x["ids"]    + [pad]  * (ml - len(x["ids"])) for x in chunk]
            mask_t = [x["mask"]   + [0]    * (ml - len(x["ids"])) for x in chunk]
            lbl_t  = [x["labels"] + [-100] * (ml - len(x["ids"])) for x in chunk]
            batches.append({
                "input_ids":      torch.tensor(ids_t,  dtype=torch.long),
                "attention_mask": torch.tensor(mask_t, dtype=torch.long),
                "labels":         torch.tensor(lbl_t,  dtype=torch.long),
            })
        return batches

    return to_batches(zs_items, DEV_BATCH, pad_id), to_batches(fs_items, DEV_BATCH, pad_id)


@torch.no_grad()
def eval_dev_loss(model, batches, device):
    """Mean per-token CE loss on a list of batches (answer tokens only, -100 elsewhere)."""
    total_loss, total_tok = 0.0, 0
    for b in batches:
        ids  = b["input_ids"].to(device)
        mask = b["attention_mask"].to(device)
        lbl  = b["labels"].to(device)

        logits = model(input_ids=ids, attention_mask=mask).logits  # (B, T, V)

        # Causal-LM shift: logits[i] predicts token i+1
        sl = logits[:, :-1].contiguous()    # (B, T-1, V)
        tl = lbl[:, 1:].contiguous()        # (B, T-1)

        n_valid = (tl != -100).sum().item()
        if n_valid == 0:
            continue
        loss = F.cross_entropy(
            sl.reshape(-1, sl.size(-1)), tl.reshape(-1),
            ignore_index=-100, reduction="sum",
        )
        total_loss += loss.item()
        total_tok  += n_valid

    return total_loss / total_tok if total_tok > 0 else float("nan")


# ── Distillation helpers ──────────────────────────────────────────────────────

def answer_alignment(t_lens, labels, device):
    """
    Map teacher answer positions to student answer positions.
    Both sequences end with the same n_ans answer tokens.
    """
    B = labels.shape[0]
    n_ans       = (labels != -100).sum(dim=1)                            # (B,)
    s_ans_start = (labels != -100).float().argmax(dim=1)                 # (B,)
    t_ans_start = t_lens - n_ans                                         # (B,)

    K = int(n_ans.max().item())
    j = torch.arange(K, device=device).unsqueeze(0).expand(B, -1)       # (B, K)

    t_ans_idx = (t_ans_start.unsqueeze(1) + j).clamp(0, t_lens.max() - 1)
    s_ans_idx = (s_ans_start.unsqueeze(1) + j).clamp(0, labels.shape[1] - 1)
    ans_valid  = j < n_ans.unsqueeze(1)                                  # (B, K)
    b_idx      = torch.arange(B, device=device).unsqueeze(1).expand(B, K)

    return b_idx, t_ans_idx, s_ans_idx, ans_valid, n_ans


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",    choices=["baseline", "distill"], required=True)
    p.add_argument("--dataset", choices=list(DATASET_CONFIGS), default="gsm8k")
    p.add_argument("--output_dir", default="experiments/loss_curve")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = DATASET_CONFIGS[args.dataset]
    out_dir = Path(args.output_dir) / args.dataset / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device

    if accelerator.is_main_process:
        print(f"\n{'='*60}")
        print(f"Loss curve experiment — mode: {args.mode}")
        print(f"Model  : {MODEL_NAME}")
        print(f"Dataset: {args.dataset}  ({cfg['num_fewshot']}-shot)")
        print(f"Steps  : {TRAIN_STEPS}  |  eval every {EVAL_STEPS}")
        print(f"Dev    : {DEV_SIZE} test examples")
        print(f"Output : {out_dir}")
        print(f"{'='*60}\n")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Dev batches (built on every process, identical, only main process uses them) ──
    if accelerator.is_main_process:
        print("Building dev batches …")
    zs_dev, fs_dev = build_dev_batches(tokenizer, args.dataset)
    if accelerator.is_main_process:
        print(f"  zs batches: {len(zs_dev)}, fs batches: {len(fs_dev)}\n")

    # ── Teacher (distill mode only) ───────────────────────────────────────────
    teacher = None
    if args.mode == "distill":
        if accelerator.is_main_process:
            print("Loading frozen teacher …")
        teacher = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.bfloat16, trust_remote_code=True,
        ).to(device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        if accelerator.is_main_process:
            print("  Teacher loaded and frozen.\n")

    # ── Student (LoRA) ────────────────────────────────────────────────────────
    if accelerator.is_main_process:
        print("Loading student model …")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map=None,
    )
    lora_cfg = LoraConfig(**LORA_KWARGS)
    model = get_peft_model(base_model, lora_cfg)
    if accelerator.is_main_process:
        model.print_trainable_parameters()
        print()

    # ── Dataset / DataLoader ──────────────────────────────────────────────────
    train_data   = load_dataset_split(args.dataset, "train")
    train_loader = _factory_dl(
        train_data, tokenizer, BATCH_SIZE,
        dataset_name=args.dataset,
        num_fewshot=cfg["num_fewshot"],
        max_seq_len_teacher=cfg["max_seq_teacher"],
        max_seq_len_student=cfg["max_seq_student"],
        shuffle=True,
        num_workers=4,
        seed=SEED,
        teacher_include_answer=(args.mode == "distill"),
    )

    # ── Optimizer & scheduler ─────────────────────────────────────────────────
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, WARMUP_STEPS, TRAIN_STEPS)

    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train()
    train_iter = iter(train_loader)
    step = 0

    results = {"mode": args.mode, "dataset": args.dataset, "steps": [], "zs_loss": [], "fs_loss": []}

    progress = tqdm(total=TRAIN_STEPS, desc=f"[{args.mode}]",
                    disable=not accelerator.is_main_process)

    while step <= TRAIN_STEPS:

        # ── Dev evaluation every EVAL_STEPS (including step 0 and final step) ──
        if step % EVAL_STEPS == 0 or step == TRAIN_STEPS:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                m = accelerator.unwrap_model(model)
                m.eval()
                zs_loss = eval_dev_loss(m, zs_dev, device)
                fs_loss = eval_dev_loss(m, fs_dev, device)
                m.train()
                results["steps"].append(step)
                results["zs_loss"].append(zs_loss)
                results["fs_loss"].append(fs_loss)
                tqdm.write(f"  step={step:3d}  zs_dev={zs_loss:.4f}  fs_dev={fs_loss:.4f}")
            accelerator.wait_for_everyone()

        if step == TRAIN_STEPS:
            break  # eval at final step done, exit

        # ── Training step ─────────────────────────────────────────────────────
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        if args.mode == "baseline":
            outputs = model(
                input_ids=batch.student_input_ids,
                attention_mask=batch.student_attention_mask,
                labels=batch.labels,
            )
            loss = outputs.loss

        else:  # distill
            t_input_ids = batch.teacher_input_ids.to(device)
            t_attn_mask = batch.teacher_attention_mask.to(device)
            labels_d    = batch.labels.to(device)
            t_lens      = t_attn_mask.sum(dim=1)

            with torch.no_grad():
                teacher_out = teacher(input_ids=t_input_ids, attention_mask=t_attn_mask)

            student_out = model(
                input_ids=batch.student_input_ids,
                attention_mask=batch.student_attention_mask,
                labels=batch.labels,
            )
            ce_loss = student_out.loss

            b_idx, t_ans_idx, s_ans_idx, ans_valid, _ = answer_alignment(
                t_lens, labels_d, device
            )
            t_logits = teacher_out.logits[b_idx, t_ans_idx]   # (B, K, V)
            s_logits = student_out.logits[b_idx, s_ans_idx]   # (B, K, V)
            del teacher_out, student_out

            _, top_idx = t_logits.topk(K_VOCAB, dim=-1)       # (B, K, K_vocab)
            t_top = t_logits.gather(-1, top_idx)
            s_top = s_logits.gather(-1, top_idx)
            del t_logits, s_logits

            mask = ans_valid.unsqueeze(-1).float()
            n_valid_elems = mask.sum() * K_VOCAB
            dist_loss = ((t_top - s_top).pow(2) * mask).sum() / (n_valid_elems + 1e-8)

            loss = ce_loss + LAM_DISTILL * dist_loss

        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        step += 1
        progress.update(1)
        if accelerator.is_main_process and step % 10 == 0:
            progress.set_postfix(loss=f"{loss.item():.4f}")

    progress.close()

    # ── Save results ──────────────────────────────────────────────────────────
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        out_path = out_dir / "results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to {out_path}")

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
