"""Per-output-channel symmetric RTN; dequantized weights retain the model dtype."""

import json
import math
from pathlib import Path

import torch
from torch import nn

SUPPORTED_BITS = (0, 4, 8, 16)


def target_modules(model):
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, (nn.Linear, nn.Conv2d)) and name.startswith(("vision_backbone.", "language_model.model."))
    }


@torch.no_grad()
def quantize_rows(weight, bit):
    """Keep BF16 scale arithmetic consistent between scoring and injection."""
    if bit not in SUPPORTED_BITS:
        raise ValueError(f"Unsupported weight bit: {bit}; choose {SUPPORTED_BITS}")
    if bit == 16:
        return weight
    if bit == 0:
        return torch.zeros_like(weight)
    flat = weight.flatten(1)
    qmax = 2 ** (bit - 1) - 1
    scale = flat.abs().amax(1, keepdim=True).clamp_min(1e-8) / qmax
    return (torch.round(flat / scale).clamp(-qmax - 1, qmax) * scale).reshape_as(weight)


def load_gates(path):
    path = Path(path)
    if path.suffix == ".pt":
        raw = torch.load(path, map_location="cpu", weights_only=True)
    else:
        raw = json.loads(path.read_text())
    if not isinstance(raw, dict) or not raw:
        raise ValueError("Empty or invalid gates")
    assignment = raw.get("assign", raw)
    if not isinstance(assignment, dict) or not assignment:
        raise ValueError("Expected a nonempty layer-to-channel-bit assignment")
    result = {}
    for name, values in assignment.items():
        if torch.is_tensor(values):
            values = values.tolist()
        if not isinstance(values, list) or not values:
            raise ValueError(f"Invalid channel assignment: {name}")
        if any(type(bit) is not int or bit not in SUPPORTED_BITS for bit in values):
            raise ValueError(f"Invalid bit assignment: {name}")
        result[name] = values
    return result, raw if "assign" in raw else {}


@torch.no_grad()
def apply_gates(model, path, chunk_size=128):
    """Validate the entire map before mutation; zero-bit weights keep their bias."""
    assignment, metadata = load_gates(path)
    modules = target_modules(model)
    missing = set(assignment) - set(modules)
    if missing:
        raise ValueError(f"Unknown or protected modules in gates: {sorted(missing)}")
    cap = metadata.get("stats", {}).get("max_prune_ratio_per_layer", 0.01)
    if cap is not None and (not math.isfinite(cap) or not 0 <= cap <= 1):
        raise ValueError("Invalid pruning cap")
    for name, values in assignment.items():
        if len(values) != modules[name].weight.shape[0]:
            raise ValueError(f"Gate length mismatch: {name}")
        if cap is not None and values.count(0) > int(cap * len(values)):
            raise ValueError(f"Pruning cap exceeded: {name}")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    for name, values in assignment.items():
        module = modules[name]
        for bit in sorted(set(values) - {16}):
            indices = torch.tensor([i for i, value in enumerate(values) if value == bit], device=module.weight.device)
            for start in range(0, len(indices), chunk_size):
                rows = indices[start : start + chunk_size]
                module.weight.index_copy_(0, rows, quantize_rows(module.weight.index_select(0, rows), bit))
    return len(assignment)
