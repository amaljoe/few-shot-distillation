"""
Teacher model wrapper for activation extraction.

The teacher is the same base model as the student, but:
- Run with few-shot context prepended to the query
- All parameters frozen (no gradient computation)
- Forward hooks capture layer-wise hidden states at the alignment token position

NOTE: vLLM cannot be used here — HuggingFace is required for hook-based
      activation extraction. vLLM is used only for text generation evaluation.
"""

import torch
from transformers import AutoModelForCausalLM

from src.hooks.activation_capture import ActivationCapture


class TeacherWrapper:
    """
    Wraps a frozen HuggingFace model for teacher activation extraction.

    Args:
        model_name: HuggingFace model ID (e.g., "Qwen/Qwen3-2B-Instruct")
        num_layers: Total number of transformer layers in the model
        layer_indices: Which layers to capture; defaults to all layers
        device_map: HuggingFace device map (use "auto" for multi-GPU)
    """

    def __init__(
        self,
        model_name: str,
        num_layers: int,
        layer_indices: list[int] | None = None,
        device_map: str = "auto",
    ):
        if layer_indices is None:
            layer_indices = list(range(num_layers))

        print(f"Loading teacher model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
        )
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        self.layer_indices = layer_indices
        # detach=True: teacher activations are constants, no grad needed
        self.capture = ActivationCapture(self.model, layer_indices, detach=True)

        print(f"Teacher: {model_name} | Layers captured: {len(layer_indices)}")

    @torch.no_grad()
    def get_hidden_states(
        self,
        input_ids: torch.Tensor,         # (B, T)
        attention_mask: torch.Tensor,    # (B, T)
        query_pos: torch.Tensor,         # (B,) — alignment token index per example
    ) -> torch.Tensor:
        """
        Run teacher forward pass and return hidden states at alignment positions.

        Returns:
            Tensor of shape (num_layers, B, H) in float16.
            Detached (no gradient).
        """
        self.capture.clear()
        self.model(input_ids=input_ids, attention_mask=attention_mask)
        states = self.capture.get_states_at_pos(query_pos)  # (L, B, H)
        return states.half()  # save memory when caching to disk

    def remove_hooks(self):
        self.capture.remove()
