"""
Student model wrapper for LoRA fine-tuning with optional activation capture.

The student:
- Uses the same base weights as the teacher initially
- Is adapted via LoRA (efficient, ~1% trainable parameters)
- During distillation training, activation hooks capture hidden states for
  computing the layer-matching loss against the pre-computed teacher cache

IMPORTANT: Hooks must be attached to the unwrapped base model, NOT the PEFT wrapper.
  Correct: peft_model.base_model.model  (HF model)
  Wrong:   peft_model                   (PEFT wrapper, hooks won't fire correctly)
"""

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM

from src.hooks.activation_capture import ActivationCapture


class StudentModel:
    """
    LoRA-wrapped student model with optional activation capture for distillation.

    Args:
        model_name: HuggingFace model ID
        lora_config: Dict with LoRA hyperparameters (r, alpha, dropout, etc.)
        num_layers: Total number of transformer layers
        layer_indices: Layers to capture for distillation (None = all layers)
        device_map: HuggingFace device map
    """

    def __init__(
        self,
        model_name: str,
        lora_config: dict,
        num_layers: int,
        layer_indices: list[int] | None = None,
        device_map: str = "auto",
    ):
        if layer_indices is None:
            layer_indices = list(range(num_layers))
        self.layer_indices = layer_indices

        print(f"Loading student base model: {model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
        )

        peft_cfg = LoraConfig(
            r=lora_config["r"],
            lora_alpha=lora_config["alpha"],
            lora_dropout=lora_config["dropout"],
            target_modules=lora_config["target_modules"],
            bias=lora_config.get("bias", "none"),
            task_type=TaskType.CAUSAL_LM,
        )
        self.peft_model = get_peft_model(base_model, peft_cfg)
        self.peft_model.print_trainable_parameters()

        # Activation capture — attach to unwrapped base model, detach=False for grad flow
        # peft_model.base_model.model is the original HF AutoModelForCausalLM
        self.capture: ActivationCapture | None = None
        self._base_model_for_hooks = self.peft_model.base_model.model

    def enable_capture(self):
        """Enable activation capture for distillation training."""
        if self.capture is not None:
            self.capture.remove()
        # detach=False: student activations need grad for distillation loss backprop
        self.capture = ActivationCapture(
            self._base_model_for_hooks,
            layer_indices=self.layer_indices,
            detach=False,
        )

    def disable_capture(self):
        """Disable activation capture (e.g., during pure CE training)."""
        if self.capture is not None:
            self.capture.remove()
            self.capture = None

    def get_hidden_states(self, last_pos: torch.Tensor) -> torch.Tensor:
        """
        Get captured student hidden states at alignment positions.

        Call this AFTER a forward pass with capture enabled.

        Args:
            last_pos: (B,) int tensor — last non-pad token index per example.
                      For student, this is the last token of the question prompt.

        Returns:
            (num_layers, B, H) tensor, requires_grad=True
        """
        assert self.capture is not None, "Call enable_capture() before training"
        states = self.capture.get_states_at_pos(last_pos)
        return states

    def clear_capture(self):
        if self.capture is not None:
            self.capture.clear()

    def get_model(self):
        """Return the PEFT model for training."""
        return self.peft_model

    def get_trainable_parameters(self):
        return filter(lambda p: p.requires_grad, self.peft_model.parameters())
