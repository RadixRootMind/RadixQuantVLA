import argparse
import json
import os
import sys
import shutil
from typing import Dict, Optional

import torch
import torch.nn as nn


def _default_backend_root() -> str:
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(repo_root, "third_party", "openvla")


def _resolve_backend_root(path: Optional[str]) -> str:
    backend_root = path or os.environ.get("QVLA_BACKEND_ROOT") or _default_backend_root()
    backend_root = os.path.abspath(os.path.expanduser(backend_root))
    if not os.path.isdir(backend_root):
        raise FileNotFoundError(f"backend_root not found: {backend_root}")
    if backend_root not in sys.path:
        sys.path.insert(0, backend_root)
    return backend_root


def _is_target_module(module_name: str, module: nn.Module) -> bool:
    if module_name.startswith("projector.") or "action_head" in module_name or module_name.startswith("language_model.lm_head"):
        return False
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        if module_name.startswith("language_model."):
            return True
        if module_name.startswith("vision_backbone."):
            return True
    return False


def _is_excluded_module(module_name: str) -> bool:
    return (
        module_name.startswith("projector.")
        or "action_head" in module_name
        or module_name.startswith("language_model.lm_head")
    )


def _load_gates(gate_path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Load per-layer gate assignments.
    Expected format: {layer_name: LongTensor[num_output_channels]}.
    """
    if gate_path.endswith(".pt"):
        raw = torch.load(gate_path, map_location=device)
    else:
        with open(gate_path, "r") as f:
            raw = json.load(f)

    out: Dict[str, torch.Tensor] = {}
    if isinstance(raw, dict) and "assign" in raw and isinstance(raw["assign"], dict):
        raw = raw["assign"]

    for k, v in raw.items():
        if isinstance(v, list):
            t = torch.tensor(v, device=device, dtype=torch.int64)
        elif isinstance(v, torch.Tensor):
            t = v.to(device=device, dtype=torch.int64)
        else:
            t = torch.tensor(v, device=device, dtype=torch.int64)
        out[k] = t
    return out


@torch.no_grad()
def _fake_quantize_tensor_sym(x: torch.Tensor, num_bits: int) -> torch.Tensor:
    if num_bits >= 16:
        return x
    if num_bits <= 0:
        return torch.zeros_like(x)
    qmax = (1 << (num_bits - 1)) - 1
    xmax = x.abs().max().clamp_min(1e-8)
    scale = xmax / qmax
    x_q = torch.clamp(torch.round(x / scale), min=-(qmax + 1), max=qmax)
    return x_q * scale


@torch.no_grad()
def _apply_weight_only_fake_quant(module: nn.Module, gates: torch.Tensor) -> None:
    if not hasattr(module, "weight"):
        return
    w = module.weight.data
    device = w.device
    g = gates.to(device=device, dtype=torch.int64)

    if isinstance(module, nn.Linear):
        out_channels = w.size(0)
        if g.numel() != out_channels:
            g = torch.full((out_channels,), int(g.median().item()), device=device, dtype=torch.int64)
        for i in range(out_channels):
            bw = int(g[i].item())
            if bw >= 16:
                continue
            if bw <= 0:
                w[i, :].zero_()
                continue
            row = w[i, :]
            w[i, :] = _fake_quantize_tensor_sym(row, bw)
    elif isinstance(module, nn.Conv2d):
        out_channels = w.size(0)
        if g.numel() != out_channels:
            g = torch.full((out_channels,), int(g.median().item()), device=device, dtype=torch.int64)
        for i in range(out_channels):
            bw = int(g[i].item())
            if bw >= 16:
                continue
            if bw <= 0:
                w[i, :, :, :].zero_()
                continue
            kernel = w[i, :, :, :]
            w[i, :, :, :] = _fake_quantize_tensor_sym(kernel, bw)


@torch.no_grad()
def inject_qvla_weight_fake_quant(
    model: nn.Module,
    gates_path: str,
    device: Optional[torch.device] = None,
) -> int:
    if device is None:
        device = next(iter(model.parameters())).device

    gates_map = _load_gates(gates_path, device=device)
    if len(gates_map) == 0:
        print("[qvla][fake-w] warning: empty gates map")
        return 0

    injected = 0
    for name, module in model.named_modules():
        if _is_excluded_module(name):
            continue
        if not _is_target_module(name, module):
            continue

        g_tensor = None
        if name in gates_map:
            g_tensor = gates_map[name]
        else:
            if name.startswith("language_model."):
                short = name[len("language_model.") :]
                if short in gates_map:
                    g_tensor = gates_map[short]
            if g_tensor is None and name.startswith("vision_backbone."):
                short = name[len("vision_backbone.") :]
                if short in gates_map:
                    g_tensor = gates_map[short]

        if g_tensor is None:
            continue

        _apply_weight_only_fake_quant(module, gates=g_tensor)
        injected += 1

    print(f"[qvla][fake-w] applied to {injected} modules")
    return injected


def _parse_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("f16", "fp16", "float16", "half"):
        return torch.float16
    return torch.float32


def main():
    p = argparse.ArgumentParser(description="Inject weight-only fake quant (W8A16/W4A16) and save model.")
    p.add_argument("--pretrained_checkpoint", type=str, required=True, help="OpenVLA checkpoint dir")
    p.add_argument("--backend-root", type=str, default=None, help="OpenVLA/OpenVLA-OFT backend root")
    p.add_argument("--gates_path", type=str, required=True, help="gates .pt/.json")
    p.add_argument("--out_dir", type=str, required=True, help="output dir")
    p.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    p.add_argument("--dtype", type=str, default="bf16", help="bf16|f16|f32; cpu uses f32")
    args = p.parse_args()

    _resolve_backend_root(args.backend_root)

    from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    device = torch.device(args.device)
    torch_dtype = torch.float32 if device.type == "cpu" else _parse_dtype(args.dtype)
    attn_impl = "flash_attention_2" if device.type == "cuda" else "eager"

    os.makedirs(args.out_dir, exist_ok=True)

    model = AutoModelForVision2Seq.from_pretrained(
        args.pretrained_checkpoint,
        attn_implementation=attn_impl,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    stats_path = os.path.join(args.pretrained_checkpoint, "dataset_statistics.json")
    if os.path.isfile(stats_path):
        with open(stats_path, "r") as f:
            model.norm_stats = json.load(f)

    injected = inject_qvla_weight_fake_quant(model, gates_path=args.gates_path, device=device)
    print(f"[qvla][fake-w] applied to {injected} modules")

    model.save_pretrained(args.out_dir)

    try:
        processor = AutoProcessor.from_pretrained(args.pretrained_checkpoint, trust_remote_code=True)
        processor.save_pretrained(args.out_dir)
    except Exception as e:  # pragma: no cover
        print(f"[qvla][fake-w] processor save failed: {e}")

    include_names = {
        "tokenizer_config.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_special_tokens_map.json",
        "special_tokens_map.json",
        "vocab.txt",
        "vocab.json",
        "merges.txt",
        "preprocessor_config.json",
        "processing_prismatic.py",
        "configuration_prismatic.py",
        "configuration.json",
        "config.json",
        "generation_config.json",
        "processor_config.json",
        "processor.json",
        "dataset_statistics.json",
        "added_tokens.json",
        "README.md",
    }
    for name in include_names:
        src = os.path.join(args.pretrained_checkpoint, name)
        if os.path.isfile(src):
            dst = os.path.join(args.out_dir, name)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            try:
                shutil.copy2(src, dst)
            except Exception as e:  # pragma: no cover
                print(f"[qvla][fake-w] copy {name} failed: {e}")

    print(f"[qvla][fake-w] saved model to {args.out_dir}")


if __name__ == "__main__":
    from oft_cli import maybe_dispatch
    if not maybe_dispatch("export"):
        main()
