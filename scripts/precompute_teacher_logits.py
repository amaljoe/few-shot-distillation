"""
Derive and cache teacher output logits at query positions from the existing
hidden-state cache. No new forward pass through the full teacher is needed —
we apply the teacher's norm + lm_head to the already-cached last-layer
hidden states.

This produces the logit distribution P(next_token | 8-shot context, query)
used by Condition D (logit-level KL distillation).

Only the top-K logit positions are saved to keep disk/RAM footprint small.
Top-1024 captures >99.9% of probability mass for typical LLM distributions.

Output:
  experiments/poc/teacher_cache/
    logits_top1024.pt   — {"values": (N, K), "indices": (N, K)} in float16 / int32

Run command (tmux: claude on cn14-dgx):
  python scripts/precompute_teacher_logits.py --config configs/base.yaml
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--top_k", type=int, default=1024,
                        help="Number of top logit positions to save per example")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for norm+lm_head computation")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    cache_dir = Path(cfg.teacher_activations.cache_dir)
    activations_path = cache_dir / "activations.pt"
    logits_path = cache_dir / f"logits_top{args.top_k}.pt"
    meta_path = cache_dir / "meta.json"

    if logits_path.exists():
        print(f"Logit cache already exists at {logits_path}")
        print("Delete it to recompute. Exiting.")
        return

    assert activations_path.exists(), (
        f"Hidden state cache not found at {activations_path}. "
        "Run scripts/precompute_teacher_activations.py first."
    )

    print(f"Loading hidden state cache: {activations_path}")
    # activations: (N, num_layers, hidden_size) float16
    activations = torch.load(str(activations_path), map_location="cpu")
    print(f"  Shape: {activations.shape} | dtype: {activations.dtype}")

    # We only need the last layer's hidden state
    last_layer_states = activations[:, -1, :].float()  # (N, H) — fp32 for lm_head precision
    del activations  # free memory
    N = last_layer_states.shape[0]

    # Load teacher model to extract norm and lm_head (frozen, no full forward needed)
    print(f"\nLoading teacher model for norm+lm_head: {cfg.model.name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        dtype=torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Extract components: Qwen3 uses model.model.norm + model.lm_head
    norm = model.model.norm
    lm_head = model.lm_head
    # Determine device of lm_head
    lm_head_device = next(lm_head.parameters()).device
    print(f"  lm_head device: {lm_head_device}")

    print(f"\nComputing top-{args.top_k} logits for {N} examples...")
    all_values = []
    all_indices = []

    with torch.no_grad():
        for start in tqdm(range(0, N, args.batch_size), desc="Deriving logits"):
            end = min(start + args.batch_size, N)
            batch = last_layer_states[start:end].to(lm_head_device)  # (B, H)

            # Apply norm then lm_head: logits shape (B, vocab_size)
            normed = norm(batch)
            logits = lm_head(normed)  # (B, vocab_size)

            # Top-K
            top_vals, top_idx = torch.topk(logits, k=args.top_k, dim=-1)
            all_values.append(top_vals.cpu().half())    # float16
            all_indices.append(top_idx.cpu().to(torch.int32))

    values = torch.cat(all_values, dim=0)   # (N, K) float16
    indices = torch.cat(all_indices, dim=0) # (N, K) int32

    print(f"\nLogit cache:")
    print(f"  values:  {values.shape} {values.dtype}  ({values.nbytes / 1e6:.1f} MB)")
    print(f"  indices: {indices.shape} {indices.dtype}  ({indices.nbytes / 1e6:.1f} MB)")

    logit_cache = {"values": values, "indices": indices, "top_k": args.top_k}
    torch.save(logit_cache, str(logits_path))
    print(f"\n✓ Saved to {logits_path}")

    # Sanity check: what fraction of prob mass is captured at temperature=1?
    sample_logits_full = values[:100].float()
    probs_full = torch.softmax(sample_logits_full, dim=-1)
    coverage = probs_full.sum(dim=-1).mean().item()
    print(f"  Top-{args.top_k} prob coverage (T=1, first 100 examples): {coverage:.4f}")

    # Update meta.json
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        meta["logits_top_k"] = args.top_k
        meta["logits_path"] = str(logits_path)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
