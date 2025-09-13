from models import Model
import torch
import torch.nn as nn


def print_all_layers(model_name):
    model = Model.from_pretrained(model_name)
    print(f"Model: {model_name}")
    print("All layers:")
    for name, module in model.named_modules():
        print(f"  {name}: {module.__class__.__name__} -> {module}")

# Example usage:
print_all_layers("sesame/csm-1b")
