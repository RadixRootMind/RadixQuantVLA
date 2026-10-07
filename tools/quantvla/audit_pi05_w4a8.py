"""Audit Pi0.5 Linear names against current and conservative W4A8 scopes.

Run with the OpenPI Python and this repository on PYTHONPATH. This only
enumerates targets; it does not replace modules or quantize weights.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

CURRENT_INCLUDE = re.compile(
    r"(.*paligemma_with_expert\.paligemma\.model\.language_model\.layers\.[0-9]+\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)|.*paligemma_with_expert\.gemma_expert\.model\.layers\.[0-9]+\.mlp\.(gate_proj|up_proj|down_proj))$"
)
CONSERVATIVE_INCLUDE = re.compile(
    r".*paligemma_with_expert\.(paligemma\.model\.language_model|gemma_expert\.model)\.layers\.[0-9]+\.mlp\.(gate_proj|up_proj|down_proj)$"
)
PROTECTED_ATTENTION = re.compile(
    r".*paligemma_with_expert\.(paligemma\.model\.language_model|gemma_expert\.model)\.layers\.[0-9]+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$"
)
EXCLUDED = re.compile(
    r"(?:^|\.)(vision|norm|ln|layernorm|embed|lm_head|pos_embed|timestep|state_encoder|action_encoder|action_decoder|state_proj|action_in_proj|action_out_proj|time_mlp)(?:\.|$)"
)


def classify(name: str) -> str:
    if ".paligemma.model.language_model.layers." in name:
        side = "paligemma_language"
    elif ".gemma_expert.model.layers." in name:
        side = "gemma_expert"
    elif "vision" in name:
        side = "vision"
    else:
        side = "other"
    if PROTECTED_ATTENTION.fullmatch(name):
        return side + ":attention"
    if ".mlp." in name:
        return side + ":mlp"
    return side + ":other"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    import torch
    from openpi.policies import policy_config
    from openpi.training import config as config_module

    checkpoint = Path(args.checkpoint)
    cfg = config_module.get_config("pi05_libero")
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, pytorch_compile_mode=None))
    policy = policy_config.create_trained_policy(cfg, checkpoint, pytorch_device=args.device)
    model = policy._model
    from gr00t.quantization.duquant_layers import select_targets

    current_names = {
        n
        for n, _ in select_targets(
            model, include_regex=CURRENT_INCLUDE.pattern, exclude_regex=EXCLUDED.pattern
        )
    }
    conservative_names = {
        n
        for n, _ in select_targets(
            model, include_regex=CONSERVATIVE_INCLUDE.pattern, exclude_regex=EXCLUDED.pattern
        )
    }
    if not conservative_names:
        raise RuntimeError("MLP-only scope selected zero Linear layers; check model naming")
    if any(PROTECTED_ATTENTION.fullmatch(n) for n in conservative_names):
        raise RuntimeError("MLP-only scope selected a protected attention projection")
    linears = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            linears.append(
                {
                    "name": name,
                    "in_features": module.in_features,
                    "weight_dtype": str(module.weight.dtype),
                    "out_features": module.out_features,
                    "category": classify(name),
                    "current_profile": name in current_names,
                    "conservative_mlp_only": name in conservative_names,
                }
            )
    current = [x for x in linears if x["current_profile"]]
    conservative = [x for x in linears if x["conservative_mlp_only"]]
    protected = [x for x in linears if PROTECTED_ATTENTION.fullmatch(x["name"])]
    with checkpoint.joinpath("model.safetensors").open("rb") as stream:
        checkpoint_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {
        "scope": "Pi0.5 PyTorch Linear audit; no module replacement or quantization",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "current_profile": {
            "count": len(current),
            "by_category": dict(Counter(x["category"] for x in current)),
        },
        "conservative_mlp_only": {
            "count": len(conservative),
            "by_category": dict(Counter(x["category"] for x in conservative)),
        },
        "protected_attention": {"count": len(protected), "names": [x["name"] for x in protected]},
        "all_linear_count": len(linears),
        "current_layers": current,
        "conservative_layers": conservative,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "current_profile",
                    "conservative_mlp_only",
                    "protected_attention",
                    "all_linear_count",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
