"""
Precompute and cache teacher model hidden states for all GSM8K training examples.

Run this ONCE before distillation training. Uses HuggingFace (not vLLM) because
vLLM does not expose intermediate hidden states.

Output:
  experiments/poc/teacher_cache/
    activations.pt     — tensor of shape (N, num_layers, hidden_size) in float16
    meta.json          — maps dataset index → few-shot config and query position

Storage:
  Qwen3-2B (28 layers, hidden=1536):  ~640MB for 7473 train examples
  Qwen3-4B (36 layers, hidden=2560):  ~1.4GB
  Qwen3-8B (36 layers, hidden=4096):  ~2.2GB

Runtime: ~8-15 minutes on 2×A100

Run command (tmux: claude on cn14-dgx):
  CUDA_VISIBLE_DEVICES=0,1 python scripts/precompute_teacher_activations.py \
      --config configs/base.yaml
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.gsm8k_loader import GSM8KDistillDataset, collate_fn, load_gsm8k
from src.models.teacher_wrapper import TeacherWrapper


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for teacher inference (no grad, can be larger)")
    parser.add_argument("--verify", action="store_true",
                        help="Run sanity check on first example before full precompute")
    return parser.parse_args()


def sanity_check(teacher: TeacherWrapper, tokenizer, dataset, cfg):
    """
    Verify that teacher_query_pos is correctly pointing to the right token.

    Prints the token at the alignment position and surrounding context,
    so you can confirm it's the last token of the target question.
    """
    print("\n--- SANITY CHECK: Token alignment ---")
    example = dataset[0]

    teacher_ids = example["teacher_input_ids"]
    teacher_query_pos = example["teacher_query_pos"]

    # Decode tokens around the alignment position
    context_start = max(0, teacher_query_pos - 5)
    context_end = min(len(teacher_ids), teacher_query_pos + 5)
    context_ids = teacher_ids[context_start:context_end]
    context_tokens = tokenizer.convert_ids_to_tokens(context_ids)

    print(f"Alignment position: {teacher_query_pos} (out of {len(teacher_ids)} tokens)")
    print(f"Context tokens around alignment pos:")
    for i, (tid, tok) in enumerate(zip(context_ids, context_tokens)):
        marker = " ← ALIGNMENT" if (context_start + i) == teacher_query_pos else ""
        print(f"  pos {context_start + i:4d}: id={tid:6d} | '{tok}'{marker}")

    print("\n✓ If the ALIGNMENT token is the last token of the target question, proceed.")
    print("  If not, adjust find_last_user_token_pos in src/data/gsm8k_loader.py\n")


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    cache_dir = Path(cfg.teacher_activations.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    activations_path = cache_dir / "activations.pt"
    meta_path = cache_dir / "meta.json"

    if activations_path.exists():
        print(f"Cache already exists at {activations_path}")
        print("Delete it to recompute. Exiting.")
        return

    # Load tokenizer
    print(f"Loading tokenizer: {cfg.model.name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset (shuffle=False to preserve example_idx mapping)
    print("Loading GSM8K train split...")
    raw_dataset = load_gsm8k(cfg.data.train_split)

    dataset = GSM8KDistillDataset(
        dataset=raw_dataset,
        tokenizer=tokenizer,
        num_fewshot=cfg.data.num_fewshot_examples,
        max_seq_len_teacher=cfg.data.max_seq_len_teacher,
        max_seq_len_student=cfg.data.max_seq_len_student,
        seed=cfg.training.seed,
    )

    # Load teacher
    teacher = TeacherWrapper(
        model_name=cfg.model.name,
        num_layers=cfg.model.num_layers,
        device_map="auto",
    )

    if args.verify:
        sanity_check(teacher, tokenizer, dataset, cfg)
        response = input("Continue with full precompute? [y/N] ")
        if response.lower() != "y":
            print("Aborted.")
            return

    # DataLoader (shuffle=False — index ordering must match for cache lookup during training)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id or tokenizer.eos_token_id),
        pin_memory=True,
    )

    print(f"\nPrecomputing teacher activations for {len(dataset)} examples...")
    print(f"  Layers: {cfg.model.num_layers} | Hidden: {cfg.model.hidden_size}")
    print(f"  Batch size: {args.batch_size} | Batches: {len(loader)}")

    all_activations = []  # list of (B, L, H) float16 tensors
    all_query_positions = []
    all_example_indices = []

    for batch in tqdm(loader, desc="Extracting teacher activations"):
        teacher_ids = batch.teacher_input_ids.cuda()
        teacher_mask = batch.teacher_attention_mask.cuda()
        query_pos = batch.teacher_query_pos.cuda()

        # (L, B, H) float16
        hidden = teacher.get_hidden_states(teacher_ids, teacher_mask, query_pos)

        # Rearrange to (B, L, H) for storage
        hidden = hidden.permute(1, 0, 2).cpu()
        all_activations.append(hidden)
        all_query_positions.extend(batch.teacher_query_pos.tolist())
        all_example_indices.extend(batch.example_idx.tolist())

    print("Concatenating activations...")
    activations = torch.cat(all_activations, dim=0)  # (N, L, H) float16
    print(f"Activations shape: {activations.shape}")
    print(f"Storage: {activations.nbytes / 1e9:.2f} GB")

    print(f"Saving to {activations_path}...")
    torch.save(activations, activations_path)

    meta = {
        "model_name": cfg.model.name,
        "num_examples": len(dataset),
        "num_layers": cfg.model.num_layers,
        "hidden_size": cfg.model.hidden_size,
        "num_fewshot": cfg.data.num_fewshot_examples,
        "seed": cfg.training.seed,
        "example_indices": all_example_indices,  # should be [0, 1, 2, ..., N-1]
        "query_positions": all_query_positions,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    teacher.remove_hooks()
    print(f"\n✓ Done. Cache saved to {cache_dir}/")
    print(f"  activations.pt: {activations.shape} (N, num_layers, hidden_size)")
    print(f"  meta.json: {len(all_example_indices)} examples recorded")


if __name__ == "__main__":
    main()
