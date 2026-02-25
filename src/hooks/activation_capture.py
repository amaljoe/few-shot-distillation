"""
Activation capture via PyTorch forward hooks.

Registers hooks on transformer block layers to intercept hidden states
after each full transformer block (post-residual, post-layernorm).

Usage:
    # Teacher (no grad)
    capture = ActivationCapture(model, layer_indices=list(range(28)), detach=True)
    with torch.no_grad():
        model(input_ids=..., attention_mask=...)
    states = capture.get_states_at_pos(query_pos)  # (L, B, H)
    capture.clear()

    # Student (with grad — for distillation loss backprop)
    capture = ActivationCapture(model, layer_indices=list(range(28)), detach=False)
    out = model(input_ids=..., attention_mask=..., labels=...)
    states = capture.get_states_at_pos(last_pos)   # (L, B, H), requires_grad=True
    capture.clear()
"""

import torch
import torch.nn as nn


class ActivationCapture:
    """
    Registers forward hooks on specified transformer block layers.

    After a forward pass, hidden_states[layer_idx] contains the full
    (B, T, H) hidden state tensor at that layer.

    Args:
        model: The HuggingFace transformer model (Qwen3 base model, not PEFT wrapper).
                Access via peft_model.base_model.model for LoRA-wrapped models.
        layer_indices: Which layers to capture (0-indexed).
        detach: If True, captured tensors are detached (use for teacher, no grad).
                If False, tensors retain grad (use for student during training).
    """

    def __init__(self, model: nn.Module, layer_indices: list[int], detach: bool = True):
        self.layer_indices = sorted(layer_indices)
        self.detach = detach
        self.hidden_states: dict[int, torch.Tensor] = {}
        self._hooks: list[torch.utils.hooks.RemovableHook] = []

        # Qwen3 layers are at model.model.layers (HuggingFace Qwen2/Qwen3 architecture)
        layers = self._get_layers(model)

        for idx in self.layer_indices:
            assert idx < len(layers), (
                f"Layer index {idx} out of range (model has {len(layers)} layers)"
            )
            hook = layers[idx].register_forward_hook(self._make_hook(idx))
            self._hooks.append(hook)

    def _get_layers(self, model: nn.Module):
        """Navigate to the transformer block list, handling PEFT wrappers."""
        # Direct HF model: model.model.layers
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers
        # Unwrapped decoder: model.layers
        if hasattr(model, "layers"):
            return model.layers
        raise AttributeError(
            "Cannot find transformer layers. Expected model.model.layers or model.layers. "
            "For PEFT models, pass peft_model.base_model.model instead of the PEFT wrapper."
        )

    def _make_hook(self, idx: int):
        def hook(module: nn.Module, input: tuple, output: tuple | torch.Tensor):
            # Qwen3DecoderLayer returns a tuple: (hidden_states, ...)
            # output[0] is the post-residual hidden state (B, T, H)
            if isinstance(output, tuple):
                hs = output[0]
            else:
                hs = output

            if self.detach:
                hs = hs.detach()

            self.hidden_states[idx] = hs

        return hook

    def clear(self):
        """Clear captured states (call between batches)."""
        self.hidden_states.clear()

    def remove(self):
        """Remove all hooks from the model (call when done training)."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def get_states_at_pos(self, pos_per_example: torch.Tensor) -> torch.Tensor:
        """
        Extract hidden states at a specific token position for each batch item.

        Args:
            pos_per_example: (B,) int tensor, token index per batch item.
                             For teacher: teacher_query_pos from the batch.
                             For student: index of last non-padding token.

        Returns:
            Tensor of shape (num_layers, B, H) with states at the alignment token.
        """
        assert len(self.hidden_states) == len(self.layer_indices), (
            f"Expected {len(self.layer_indices)} layers captured, "
            f"got {len(self.hidden_states)}. Did you run a forward pass?"
        )

        out = []
        for idx in self.layer_indices:
            hs = self.hidden_states[idx]  # (B, T, H)
            B = hs.size(0)
            pos = pos_per_example.to(hs.device)  # (B,)

            # Clamp positions to valid range (safety for truncated sequences)
            pos = torch.clamp(pos, 0, hs.size(1) - 1)

            # Gather: hs[b, pos[b], :] for each b
            gathered = hs[torch.arange(B, device=hs.device), pos, :]  # (B, H)
            # For teacher (detach=True, device_map="auto" may split layers across GPUs):
            # move to CPU so all layers can be stacked regardless of device placement.
            # For student (detach=False, single-GPU with accelerate): keep on device
            # so gradients can flow back through the gathered states.
            out.append(gathered.cpu() if self.detach else gathered)

        return torch.stack(out, dim=0)  # (num_layers, B, H)

    def get_last_nonpad_pos(
        self, input_ids: torch.Tensor, pad_token_id: int
    ) -> torch.Tensor:
        """
        Utility: compute the index of the last non-padding token per batch item.
        Use this for student alignment position when student pads on the right.

        Args:
            input_ids: (B, T)
            pad_token_id: the padding token id

        Returns:
            (B,) int tensor of last non-pad positions
        """
        is_not_pad = (input_ids != pad_token_id)  # (B, T)
        lengths = is_not_pad.sum(dim=1)            # (B,)
        return (lengths - 1).clamp(min=0)

    def __del__(self):
        self.remove()
