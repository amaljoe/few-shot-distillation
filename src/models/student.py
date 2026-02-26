"""
Student model wrapper for LoRA fine-tuning.

Loads the base model, applies a LoRA adapter, and exposes helpers for
retrieving trainable parameters. The online distillation training scripts
use output_hidden_states=True directly rather than activation hooks.
"""

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM


class StudentModel:
    """
    LoRA-wrapped student model.

    Args:
        model_name: HuggingFace model ID
        lora_config: Dict with LoRA hyperparameters (r, alpha, dropout, etc.)
        num_layers: Total number of transformer layers (unused, kept for API compat)
        device_map: HuggingFace device map
    """

    def __init__(
        self,
        model_name: str,
        lora_config: dict,
        num_layers: int,
        device_map: str = "auto",
    ):
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

    def get_model(self):
        """Return the PEFT model for training."""
        return self.peft_model

    def get_trainable_parameters(self):
        return filter(lambda p: p.requires_grad, self.peft_model.parameters())
