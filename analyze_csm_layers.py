import argparse
import json
import math
import os
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import List, Tuple, Dict

import torch
from transformers import AutoProcessor, CsmForConditionalGeneration

# Heuristic classification rules for LoRA vs full finetune vs freeze.
# Adjust these as you gather empirical results.
LORA_NAME_HINTS = {
    "q_proj", "k_proj", "v_proj", "o_proj", "output_proj",
    "w1", "w2", "w3", "gate_proj", "up_proj", "down_proj"
}
FULL_FINETUNE_HINTS = {
    "projection", "codebook", "codebook0_head"  # newly introduced / alignment specific
}
ALWAYS_FREEZE_SUBSTRINGS = {
    "embed_tokens",  # large embeddings
    "lm_head",       # output head often frozen when using LoRA
    "layernorm", "norm",  # layer norms typically stable
    "codec_model",  # external audio tokenizer kept eval in your scripts
}
# If True, we will skip marking any module inside codec/audio models for LoRA.
SKIP_AUDIO_SUBMODULES = True

@dataclass
class ModuleInfo:
    name: str
    class_name: str
    params: int
    shape: str
    parent_path: str
    classification: str  # lora | full | freeze


def classify_module(name: str, module, full_path: str) -> str:
    lname = name.lower()
    path_l = full_path.lower()

    if sum(p.numel() for p in module.parameters(recurse=False)) == 0:
        return "freeze"

    # Embeddings / Norms / Codec kept frozen
    if any(key in path_l for key in ALWAYS_FREEZE_SUBSTRINGS):
        return "freeze"
    if SKIP_AUDIO_SUBMODULES and ("codec" in path_l or "audio" in path_l):
        return "freeze"

    cls_name = module.__class__.__name__.lower()
    if cls_name in {"layernorm", "rmsnorm"}:
        return "freeze"
    if cls_name in {"embedding"}:
        return "freeze"

    # Linear layers: decide LoRA vs full
    if isinstance(module, torch.nn.Linear):
        # Heuristic: name hints -> LoRA
        if any(h in lname for h in LORA_NAME_HINTS):
            return "lora"
        # If dimension is very small, freezing is fine
        in_f, out_f = module.in_features, module.out_features
        if min(in_f, out_f) < 64:
            return "freeze"
        # Large linear not matched by hints -> LoRA default
        return "lora"

    # Convolution or other projection layers could benefit from LoRA, but start frozen
    if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d)):
        if any(h in lname for h in FULL_FINETUNE_HINTS):
            return "full"
        return "freeze"

    # Modules explicitly hinted for full finetune
    if any(h in lname for h in FULL_FINETUNE_HINTS):
        return "full"

    # Default fallback
    return "freeze"


def analyze_model(model) -> List[ModuleInfo]:
    results: List[ModuleInfo] = []
    for full_path, module in model.named_modules():
        # Skip root container (empty path)
        if full_path == "":
            continue
        # Only consider leaf modules (no children with parameters)
        children_with_params = False
        for _n, child in module.named_children():
            if any(p.requires_grad for p in child.parameters(recurse=False)):
                children_with_params = True
                break
        # We still record non-leaf but classification mostly matters for leaves
        params = sum(p.numel() for p in module.parameters(recurse=False))
        if params == 0:
            # Still record if it conveys structure? Skip to reduce noise
            continue
        shape_repr = ""  # gather first parameter shape for display
        for p in module.parameters(recurse=False):
            shape_repr = str(tuple(p.shape))
            break
        name = full_path.split(".")[-1]
        classification = classify_module(name, module, full_path)
        info = ModuleInfo(
            name=name,
            class_name=module.__class__.__name__,
            params=params,
            shape=shape_repr,
            parent_path=full_path.rsplit(".", 1)[0] if "." in full_path else "<root>",
            classification=classification,
        )
        results.append(info)
    return results


def summarize(results: List[ModuleInfo]) -> Dict:
    total_params = sum(r.params for r in results)
    by_classification = defaultdict(int)
    for r in results:
        by_classification[r.classification] += r.params
    return {
        "total_leaf_params": total_params,
        "param_breakdown": {k: v for k, v in by_classification.items()},
    }


def collect_lora_targets(results: List[ModuleInfo]) -> List[str]:
    # Collect unique module name patterns for LoRA injection (leaf names)
    lora_leaf_names = sorted({r.name for r in results if r.classification == "lora"})
    return lora_leaf_names


def main():
    parser = argparse.ArgumentParser(description="Analyze CSM model layers for LoRA vs full finetune suggestions")
    parser.add_argument("--model-id", default="sesame/csm-1b")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-json", default=None, help="Optional path to save JSON report")
    parser.add_argument("--print-top", type=int, default=100, help="Print only top-N largest param leaf modules")
    parser.add_argument("--show-all", action="store_true", help="Ignore --print-top and show all leaves")
    args = parser.parse_args()

    print(f"Loading model {args.model_id} on {args.device} (weights) ...")
    model = CsmForConditionalGeneration.from_pretrained(args.model_id)
    model.to(args.device)
    model.eval()

    results = analyze_model(model)
    # Sort by parameter count desc
    results.sort(key=lambda r: r.params, reverse=True)

    summary = summarize(results)
    lora_targets = collect_lora_targets(results)

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))
    print("\nSuggested LoRA target module leaf names (use in LoraConfig.target_modules):")
    print(lora_targets)

    print("\n=== Detailed Leaf Modules (sorted by params) ===")
    header = f"{'#':>3}  {'Params(M)':>9}  {'Class':>18}  {'Name':>30}  {'Shape':>20}  {'Classif':>7}  Path"
    print(header)
    print('-' * len(header))

    limit = len(results) if args.show_all else min(args.print_top, len(results))
    for i, r in enumerate(results[:limit]):
        print(f"{i:3d}  {r.params/1e6:9.3f}  {r.class_name:18}  {r.name:30}  {r.shape:20}  {r.classification:7}  {r.parent_path}")

    if args.save_json:
        report = {
            "summary": summary,
            "lora_targets": lora_targets,
            "modules": [r.__dict__ for r in results],
        }
        with open(args.save_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved JSON report to {args.save_json}")

    print("\nNotes:")
    print("- 'lora' modules are high-impact linear layers; attach LoRA adapters there.")
    print("- 'full' modules suggested for full finetuning (new / small alignment heads).")
    print("- 'freeze' modules kept frozen (embeddings, norms, codec/audio, tiny layers).")
    print("Adjust heuristics in classify_module() as needed for your experiments.")

if __name__ == "__main__":
    main()

