"""
Student model wrapper supporting both LoRA and full fine-tuning.

Pass use_lora=True (default) for parameter-efficient LoRA adaptation.
Pass use_lora=False for full fine-tuning (all parameters updated).
The online distillation training scripts use output logits directly
rather than activation hooks.
"""

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM


class StudentModel:
    """
    Student model wrapper for LoRA or full fine-tuning.

    Args:
        model_name: HuggingFace model ID
        lora_config: Dict with LoRA hyperparameters (r, alpha, dropout, etc.).
                     Required when use_lora=True; ignored otherwise.
        num_layers: Total number of transformer layers (unused, kept for API compat)
        device_map: HuggingFace device map
        use_lora: If True (default), apply LoRA adapter. If False, full fine-tuning.
    """

    def __init__(
        self,
        model_name: str,
        lora_config: dict | None,
        num_layers: int,
        device_map: str = "auto",
        use_lora: bool = True,
    ):
        print(f"Loading student base model: {model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
        )

        if use_lora and lora_config is not None:
            peft_cfg = LoraConfig(
                r=lora_config["r"],
                lora_alpha=lora_config["alpha"],
                lora_dropout=lora_config["dropout"],
                target_modules=lora_config["target_modules"],
                bias=lora_config.get("bias", "none"),
                task_type=TaskType.CAUSAL_LM,
            )
            self.model = get_peft_model(base_model, peft_cfg)
            self.model.print_trainable_parameters()
            self.is_lora = True
        else:
            self.model = base_model
            n_trainable = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
            print(f"Full fine-tuning: {n_trainable:,} trainable parameters")
            self.is_lora = False

    def get_model(self):
        """Return the model for training."""
        return self.model

    def get_trainable_parameters(self):
        return filter(lambda p: p.requires_grad, self.model.parameters())
