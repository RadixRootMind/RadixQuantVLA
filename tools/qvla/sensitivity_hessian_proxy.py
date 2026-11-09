"""
Hessian-based proxy sensitivity for QVLA gating.

Core idea:
- Accumulate per-layer input covariance H via batched input statistics.
- Use diag of cholesky(H^{-1}) to weight per-column errors.
- Produce per-output-channel proxy_b for bits in {0,2,4,8,16}.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn

try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(x, *args, **kwargs):  # type: ignore
        return x


def _is_target_module(name: str, m: nn.Module, include_projector: bool) -> bool:
    if name.startswith("projector.") or "action_head" in name or name.startswith("language_model.lm_head"):
        return False
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        if name.startswith("language_model.") or name.startswith("vision_backbone."):
            return True
    return False


def _build_calib_batches(calib_jsonl: str, ckpt: str, max_samples: int) -> List[Dict[str, torch.Tensor]]:
    from PIL import Image
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True, local_files_only=True)
    batches: List[Dict[str, torch.Tensor]] = []
    with open(calib_jsonl, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "text" not in obj or "image" not in obj:
                continue
            img_path = obj["image"]
            if not os.path.isfile(img_path):
                continue
            image = Image.open(img_path).convert("RGB")
            text = obj.get("text", "")
            if not isinstance(text, str):
                text = str(text)
            inputs = processor([text], [image], return_tensors="pt")
            batch: Dict[str, torch.Tensor] = {}
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    batch[k] = v
            batches.append(batch)
            if len(batches) >= max_samples:
                break
    return batches


@dataclass
class _HessianProxy:
    layer: nn.Module
    device: torch.device

    def __post_init__(self) -> None:
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.device)
        self.nsamples = 0

    def add_batch(self, inp: torch.Tensor) -> None:
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride,
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = torch.sqrt(torch.tensor(2.0 / self.nsamples, device=inp.device)) * inp.float()
        self.H += inp.matmul(inp.t())

    def diag_hinv(self, percdamp: float = 0.01) -> torch.Tensor:
        H = self.H
        diag = torch.arange(self.columns, device=self.device)
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        damp = percdamp * torch.mean(torch.diag(H))
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        return torch.diag(H)


def _quantize_row_sym(w: torch.Tensor, bits: int) -> torch.Tensor:
    if bits >= 16:
        return w
    if bits <= 0:
        return torch.zeros_like(w)
    qmax = (1 << (bits - 1)) - 1
    max_abs = w.abs().max().clamp_min(1e-6)
    scale = max_abs / float(qmax)
    q = torch.round(w / scale).clamp_(-qmax - 1, qmax)
    return q * scale


def _compute_proxy_for_bits(
    layer: nn.Module,
    diag_hinv: torch.Tensor,
    bits: Iterable[int],
) -> Dict[int, torch.Tensor]:
    W = layer.weight.data.clone()
    if isinstance(layer, nn.Conv2d):
        W = W.flatten(1)
    W = W.float()
    d2 = (diag_hinv.float() ** 2).clamp_min(1e-12)
    out: Dict[int, torch.Tensor] = {}
    for b in bits:
        if b >= 16:
            out[int(b)] = torch.zeros(W.shape[0], device=W.device)
            continue
        q = torch.stack([_quantize_row_sym(W[i], int(b)) for i in range(W.shape[0])], dim=0)
        loss = ((W - q) ** 2) / d2.unsqueeze(0)
        out[int(b)] = loss.sum(dim=1)
    return out


def _parse_bits(bits: str) -> List[int]:
    return sorted({int(x) for x in bits.split(",") if x.strip()})


def _default_backend_root() -> str:
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(repo_root, "third_party", "openvla")


def _resolve_backend_root(path: str | None) -> str:
    backend_root = path or os.environ.get("QVLA_BACKEND_ROOT") or _default_backend_root()
    backend_root = os.path.abspath(os.path.expanduser(backend_root))
    if not os.path.isdir(backend_root):
        raise FileNotFoundError(f"backend_root not found: {backend_root}")
    if backend_root not in sys.path:
        sys.path.insert(0, backend_root)
    return backend_root


def _load_repo_openvla_classes(backend_root: str):
    import types
    import importlib.util

    repo_root = backend_root
    prismatic_root = os.path.join(repo_root, "prismatic")
    hf_root = os.path.join(prismatic_root, "extern", "hf")

    if "prismatic" not in sys.modules:
        p = types.ModuleType("prismatic")
        p.__path__ = [prismatic_root]
        sys.modules["prismatic"] = p
    if "prismatic.extern" not in sys.modules:
        e = types.ModuleType("prismatic.extern")
        e.__path__ = [os.path.join(prismatic_root, "extern")]
        sys.modules["prismatic.extern"] = e
    if "prismatic.extern.hf" not in sys.modules:
        h = types.ModuleType("prismatic.extern.hf")
        h.__path__ = [hf_root]
        sys.modules["prismatic.extern.hf"] = h

    def _load(fullname: str, filename: str):
        path = os.path.join(hf_root, filename)
        spec = importlib.util.spec_from_file_location(fullname, path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        sys.modules[fullname] = module
        spec.loader.exec_module(module)
        return module

    cfg_mod = _load("prismatic.extern.hf.configuration_prismatic", "configuration_prismatic.py")
    mdl_mod = _load("prismatic.extern.hf.modeling_prismatic", "modeling_prismatic.py")
    return cfg_mod.OpenVLAConfig, mdl_mod.OpenVLAForActionPrediction


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_checkpoint", type=str, required=True)
    p.add_argument("--backend-root", type=str, default=None, help="OpenVLA/OpenVLA-OFT backend root")
    p.add_argument("--calib_jsonl", type=str, required=True)
    p.add_argument("--out_path", type=str, required=True)
    p.add_argument("--bits", type=str, default="0,2,4,8")
    p.add_argument("--max_samples", type=int, default=32)
    p.add_argument("--max_layers", type=int, default=None)
    p.add_argument("--percdamp", type=float, default=0.01)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    ckpt = args.pretrained_checkpoint
    if not os.path.isdir(ckpt):
        raise FileNotFoundError(f"model dir not found: {ckpt}")
    if not os.path.isfile(args.calib_jsonl):
        raise FileNotFoundError(f"calib_jsonl not found: {args.calib_jsonl}")

    device = torch.device(args.device)
    bits = _parse_bits(args.bits)

    backend_root = _resolve_backend_root(args.backend_root)
    OpenVLAConfig, OpenVLAForActionPrediction = _load_repo_openvla_classes(backend_root)
    config = OpenVLAConfig.from_pretrained(ckpt, local_files_only=True)
    model = OpenVLAForActionPrediction.from_pretrained(
        ckpt,
        config=config,
        local_files_only=True,
        attn_implementation="eager",
        low_cpu_mem_usage=False,
        torch_dtype=(torch.bfloat16 if device.type == "cuda" else torch.float32),
    ).to(device)
    model.eval()
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    batches = _build_calib_batches(args.calib_jsonl, ckpt, max_samples=args.max_samples)
    if not batches:
        raise RuntimeError("no calibration batches built")

    model_dtype = next(iter(model.parameters())).dtype

    targets: List[Tuple[str, nn.Module]] = []
    for name, m in model.named_modules():
        if _is_target_module(name, m, include_projector=False):
            targets.append((name, m))

    if args.max_layers is not None:
        targets = targets[: int(args.max_layers)]

    out: Dict[str, Dict[str, torch.Tensor]] = {}
    if args.resume and os.path.isfile(args.out_path):
        try:
            out = torch.load(args.out_path, map_location="cpu")
            print(f"[proxy] resuming from {args.out_path}, layers={len(out)}")
        except Exception:
            out = {}

    processed = 0
    for name, m in tqdm(targets, desc="[proxy] layers", dynamic_ncols=True):
        if name in out:
            continue
        proxy = _HessianProxy(m, device=device)

        def _hook(mod, inp, outp):
            x = inp[0] if isinstance(inp, (tuple, list)) else inp
            if torch.is_tensor(x):
                proxy.add_batch(x)

        handle = m.register_forward_hook(_hook)

        with torch.no_grad():
            for batch in batches:
                batch_dev: Dict[str, torch.Tensor] = {}
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        v = v.to(device)
                        if k == "pixel_values":
                            v = v.to(dtype=model_dtype)
                        batch_dev[k] = v
                if (
                    "labels" not in batch_dev
                    or batch_dev["labels"] is None
                    or not torch.is_tensor(batch_dev["labels"])
                ):
                    if "input_ids" not in batch_dev:
                        raise RuntimeError("cannot synthesize labels: batch has no input_ids")
                    batch_dev["labels"] = batch_dev["input_ids"].clone()

                model(**batch_dev, output_hidden_states=False, return_dict=True, use_cache=False)

        if handle is not None:
            handle.remove()

        diag_hinv = proxy.diag_hinv(percdamp=float(args.percdamp))
        layer_proxy = _compute_proxy_for_bits(m, diag_hinv, bits=bits)
        out[name] = {f"proxy_{b}": v.detach().cpu() for b, v in layer_proxy.items()}
        processed += 1
        if args.save_every > 0 and processed % int(args.save_every) == 0:
            os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
            torch.save(out, args.out_path)
            print(f"[proxy] checkpoint saved ({len(out)} layers) -> {args.out_path}")

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    torch.save(out, args.out_path)
    print(f"[proxy] saved to {args.out_path} (layers={len(out)})")


if __name__ == "__main__":
    from oft_cli import maybe_dispatch
    if not maybe_dispatch("proxy"):
        main()
