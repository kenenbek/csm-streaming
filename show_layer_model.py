from transformers import AutoModel
import torch
import torch.nn as nn


def print_lora_trainable_layers(model_name, target_layer_names=None):
    model = AutoModel.from_pretrained(model_name)
    print(f"Model: {model_name}")
    print("LoRA-trainable layers (nn.Linear):")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if target_layer_names is None or any(t in name for t in target_layer_names):
                print(f"  {name}: {module.__class__.__name__} (in_features={module.in_features}, out_features={module.out_features})")

# Example usage:
target_layers = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'down_proj', 'up_proj']
print_lora_trainable_layers("sesame/csm-1b", target_layer_names=target_layers)
