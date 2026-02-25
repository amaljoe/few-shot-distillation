"""
Layer-wise hidden state matching loss for few-shot distillation.

Computes the distillation term in:
  L_total = L_task + λ * Σ_l || h_l(student) - h_l(teacher) ||²

With optional L2 normalization to remove scale differences between
teacher (frozen base) and student (LoRA-adapted) representations.
"""

import torch
import torch.nn.functional as F


def layer_matching_loss(
    student_states: torch.Tensor,
    teacher_states: torch.Tensor,
    layers_to_match: list[int] | str = "all",
    normalize: bool = True,
) -> torch.Tensor:
    """
    Compute MSE loss between student and teacher hidden states.

    Args:
        student_states: (num_captured_layers, B, H) — from ActivationCapture, requires_grad=True
        teacher_states: (num_captured_layers, B, H) — from cache, float16, no grad
        layers_to_match: "all" to use all captured layers, or a list of layer indices
                         (indices into the first dimension of the state tensors, not absolute
                          layer numbers — so 0 = first captured layer)
        normalize: If True, L2-normalize along the hidden dim before MSE.
                   Strongly recommended: teacher and student may have different activation scales
                   especially early in training when LoRA weights are random.

    Returns:
        Scalar loss tensor (averaged over layers and batch).
    """
    if layers_to_match == "all":
        s = student_states                        # (L, B, H)
        t = teacher_states.to(student_states)     # cast to student dtype
    else:
        idx = torch.tensor(layers_to_match, device=student_states.device)
        s = student_states[idx]                   # (num_selected, B, H)
        t = teacher_states[idx].to(student_states)

    if normalize:
        s = F.normalize(s, p=2, dim=-1)
        t = F.normalize(t, p=2, dim=-1)

    return F.mse_loss(s, t)


def cosine_similarity_per_layer(
    student_states: torch.Tensor,
    teacher_states: torch.Tensor,
) -> torch.Tensor:
    """
    Compute mean cosine similarity between student and teacher per layer.

    Useful for logging/monitoring during training to track how well
    student representations are converging to teacher representations.

    Returns:
        (num_layers,) tensor of cosine similarities in [-1, 1].
        Values closer to 1.0 indicate better alignment.
    """
    s = F.normalize(student_states.float(), p=2, dim=-1)  # (L, B, H)
    t = F.normalize(teacher_states.float().to(s.device), p=2, dim=-1)
    cos_sim = (s * t).sum(dim=-1).mean(dim=-1)  # (L,)
    return cos_sim


def representation_drift(
    current_states: torch.Tensor,
    pretrained_states: torch.Tensor,
) -> torch.Tensor:
    """
    Compute cosine similarity between current and pretrained representations.

    Used in Experiment 4 (Representation Drift Analysis) to track how much
    the model drifts from its pretrained initialization.

    Args:
        current_states: (L, B, H) — current model hidden states
        pretrained_states: (L, B, H) — pretrained model hidden states (frozen reference)

    Returns:
        (L,) per-layer drift score (1.0 = no drift, 0.0 = fully orthogonal)
    """
    return cosine_similarity_per_layer(current_states, pretrained_states)
