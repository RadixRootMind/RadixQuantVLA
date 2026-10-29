# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""The quantized checkpoint: the base checkpoint with its quantized weights replaced.

``quantize`` writes one directory that loads like the base checkpoint and holds,
for every projection FoldQuant quantized, the integer codes and per-row scales
the TensorRT plugins carry, in place of the bf16 weight::

    <projection>.qweight        INT8 codes (N, K) int8, or INT4 codes nibble-packed (N, K/2) uint8
    <projection>.weight_scale   per-output-row scale (N,) float32

Everything FoldQuant does not quantize (the vision tower, encoders, norms,
biases) stays as it was. The activation-side transforms of each site (the
SmoothQuant vector, a learned dense rotation and its permutation, a learned
clip) are stored beside them as ``foldquant.<module>.<group>.<site>`` tensors.
A ModelOpt baseline tower has no packs: it keeps its (smoothed) bf16 weights
and stores its quantizers' ``amax`` / ``pre_quant_scale`` in a ``modelopt``
group, the example inputs its graph is traced from in a ``trace`` group, and
the ModelOpt mode state in ``foldquant_modelopt_<module>.pt``.
and ``foldquant_quant.json`` records the schemes, the fold settings, which
projections make up each site (in row order), the graph digests and the base
checkpoint's identity. ``hf_quant_config.json`` summarises the quantization the
way NVIDIA ModelOpt's Hugging Face export does.

This is the same layout the GPTQ, AWQ and compressed-tensors checkpoints use
(codes plus scales under the projection's own name), so the size is that of a
packed low-bit model rather than the base checkpoint plus a state. A loader
that builds the policy from it sees the quantized projections' ``weight``
missing and FoldQuant installs the fake-quant layers over them from the codes;
``export`` rebuilds the plugin graphs from the same codes, byte for byte.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from . import schemes
from .quant_state import MANIFEST_NAME, ModuleQuantState, QuantState, _unpack_codes

logger = logging.getLogger(__name__)

__all__ = [
    "FORMAT",
    "HF_QUANT_CONFIG_NAME",
    "bits_of",
    "is_quantized_checkpoint",
    "load_quantized_checkpoint",
    "site_tables",
    "write_quantized_checkpoint",
]

FORMAT = "foldquant-quantized-checkpoint"
FORMAT_VERSION = 1
HF_QUANT_CONFIG_NAME = "hf_quant_config.json"
#: ``config["modelopt_state"]`` of a ModelOpt tower, saved with ``torch.save`` under this name.
MODELOPT_STATE_FILE = "foldquant_modelopt_{module}.pt"
INDEX_NAME = "model.safetensors.index.json"
SINGLE_FILE = "model.safetensors"
#: Largest shard written (bytes); the base's own sharding is not kept. The
#: verified Pi0.5 deployment used 100 GiB to emit one safetensors file, while
#: smaller filesystems can retain the upstream 5 GiB default.
MAX_SHARD_BYTES = int(float(os.environ.get("FOLDQUANT_MAX_SHARD_GIB", "5")) * 1024**3)

#: The projections behind each LLM site, in the row order the emitter concatenates them.
LLM_SITES = {
    "qkv": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
    "o": ("self_attn.o_proj",),
    "gateup": ("mlp.gate_proj", "mlp.up_proj"),
    "down": ("mlp.down_proj",),
}
#: The same for the Pi action expert (``G{i}_<site>``).
EXPERT_SITES = {
    "qkv": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
    "o": ("self_attn.o_proj",),
    "gu": ("mlp.gate_proj", "mlp.up_proj"),
    "dn": ("mlp.down_proj",),
}


def bits_of(ms: ModuleQuantState) -> int:
    """Weight width of a module's scheme: FoldQuant schemes by name, a ModelOpt tower what
    its config records."""
    if "bits" in ms.config:
        return int(ms.config["bits"])
    return schemes.bits_of(ms.scheme)


# Which projections make up each site


def _module_path(root: Any, target: Any) -> str:
    for name, m in root.named_modules():
        if m is target:
            return name
    raise ValueError(f"{type(target).__name__} is not a submodule of {type(root).__name__}")


def _llm_table(module: Any, ms: ModuleQuantState) -> Dict[str, Tuple[int, List[List[str]]]]:
    from .llm import resolve_qwen3_decoder

    decoder = resolve_qwen3_decoder(module)
    prefix = _module_path(module, decoder)
    prefix = prefix + "." if prefix else ""
    bits = bits_of(ms)
    site_bits = dict(ms.config.get("site_bits") or {})
    table: Dict[str, Tuple[int, List[List[str]]]] = {}
    for i in range(len(decoder.layers)):
        for site, names in LLM_SITES.items():
            table[f"L{i}_{site}"] = (int(site_bits.get(site, bits)), [[f"{prefix}layers.{i}.{n}.weight" for n in names]])
    return table


def _dit_table(module: Any, ms: ModuleQuantState) -> Dict[str, Tuple[int, List[List[str]]]]:
    bits = bits_of(ms)
    n = len(module.transformer_blocks)
    t: Dict[str, Tuple[int, List[List[str]]]] = {}
    cross_kv: List[List[str]] = []
    for i in range(n):
        p = f"transformer_blocks.{i}."
        if i % 2 == 1:  # self-attention block
            t[f"block{i}_qkv"] = (bits, [[f"{p}attn1.to_q.weight", f"{p}attn1.to_k.weight", f"{p}attn1.to_v.weight"]])
        else:  # cross-attention block: Q reads x, K/V read the encoder
            t[f"block{i}_q"] = (bits, [[f"{p}attn1.to_q.weight"]])
            t[f"block{i}_kv"] = (bits, [[f"{p}attn1.to_k.weight", f"{p}attn1.to_v.weight"]])
            cross_kv.append([f"{p}attn1.to_k.weight", f"{p}attn1.to_v.weight"])
        t[f"block{i}_o"] = (bits, [[f"{p}attn1.to_out.0.weight"]])
        t[f"block{i}_ffn0"] = (bits, [[f"{p}ff.net.0.proj.weight"]])
        t[f"block{i}_ffn2"] = (bits, [[f"{p}ff.net.2.weight"]])
        t[f"block{i}_adaln"] = (4, [[f"{p}norm1.linear.weight"]])
    # GPTQ packs every cross block's K/V under the shared encoder site, one entry per block.
    t["encoder"] = (bits, cross_kv)
    return t


def _expert_table(module: Any, ms: ModuleQuantState) -> Dict[str, Tuple[int, List[List[str]]]]:
    bits = bits_of(ms)
    layers = module.expert_model.model.layers
    t: Dict[str, Tuple[int, List[List[str]]]] = {}
    for i in range(len(layers)):
        for site, names in EXPERT_SITES.items():
            t[f"G{i}_{site}"] = (bits, [[f"expert_model.model.layers.{i}.{n}.weight" for n in names]])
    return t


_TABLES = {"llm": _llm_table, "dit": _dit_table, "expert": _expert_table}


def site_tables(modules: Mapping[str, Any], state: QuantState, checkpoint_root: Any) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """``{module: {recorded pack key: {"bits", "params": [[checkpoint keys]]}}}`` for *state*.

    *modules* are the live modules the state was recorded from (the family
    adapter's ``module_paths``), *checkpoint_root* the module whose
    ``named_parameters()`` names are the checkpoint's tensor keys. Every
    recorded pack must be claimed by a site, every site's entries must match the
    pack's, and every projection's rows must add up to the codes' rows.
    """
    names = {id(p): n for n, p in checkpoint_root.named_parameters()}
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for name, ms in state.modules.items():
        if name not in modules:
            raise KeyError(f"quant state covers {name!r}, which was not passed in")
        if not ms.gptq:  # a ModelOpt tower: no packed weights
            out[name] = {}
            continue
        if name not in _TABLES:
            raise NotImplementedError(f"no site table for the {name!r} module")
        module = modules[name]
        table = _TABLES[name](module, ms)
        claimed: Dict[str, Dict[str, Any]] = {}
        for site, (bits, groups) in table.items():
            for key in (f"{name}.{site}", f"{name}.rtn.{site}"):
                entries = ms.gptq.get(key)
                if entries is None:
                    continue
                if len(entries) != len(groups):
                    raise ValueError(f"{key}: {len(entries)} recorded pack(s) for {len(groups)} projection group(s)")
                params: List[List[str]] = []
                for (codes, _scale), rel_names in zip(entries, groups):
                    keys, rows = [], 0
                    for rel in rel_names:
                        p = module.get_parameter(rel)
                        if id(p) not in names:
                            raise ValueError(f"{rel} of the {name} module is not a parameter of the checkpoint root")
                        keys.append(names[id(p)])
                        rows += int(p.shape[0])
                    if rows != int(codes.shape[0]):
                        raise ValueError(f"{key}: projections have {rows} rows, the recorded codes {int(codes.shape[0])}")
                    if int(codes.abs().max()) > (7 if bits == 4 else 127):
                        raise ValueError(f"{key}: codes exceed {bits}-bit range")
                    params.append(keys)
                claimed[key] = {"bits": bits, "params": params}
        unclaimed = sorted(set(ms.gptq) - set(claimed))
        if unclaimed:
            raise ValueError(f"{name}: recorded packs with no projection: {unclaimed[:4]}")
        out[name] = claimed
    return out


# Writing


def _base_weight_files(base: Path) -> Tuple[List[Path], bool]:
    """The safetensors files holding the base weights, and whether they were sharded."""
    index = base / INDEX_NAME
    if index.is_file():
        files = sorted({base / f for f in json.loads(index.read_text())["weight_map"].values()})
        return files, True
    if (base / SINGLE_FILE).is_file():
        return [base / SINGLE_FILE], False
    files = sorted(p for p in base.glob("*.safetensors") if p.is_file())
    if not files:
        raise FileNotFoundError(f"{base} holds no safetensors weights")
    return files, len(files) > 1


class _ShardWriter:
    """Accumulates tensors and writes them as ``model-XXXXX-of-YYYYY.safetensors`` shards."""

    def __init__(self, directory: Path, single: bool) -> None:
        self.dir = directory
        self.single = single
        self.buf: Dict[str, Any] = {}
        self.buf_bytes = 0
        self.written: List[Path] = []
        self.weight_map: Dict[str, str] = {}
        self.total = 0

    def add(self, key: str, tensor: Any) -> None:
        nbytes = tensor.numel() * tensor.element_size()
        if self.buf and self.buf_bytes + nbytes > MAX_SHARD_BYTES:
            self._flush()
        self.buf[key] = tensor.contiguous()
        self.buf_bytes += nbytes
        self.total += nbytes

    def _flush(self) -> None:
        from safetensors.torch import save_file

        path = self.dir / f"model-{len(self.written) + 1:05d}.safetensors"
        save_file(self.buf, str(path), metadata={"format": "pt"})
        for key in self.buf:
            self.weight_map[key] = path.name
        self.written.append(path)
        self.buf, self.buf_bytes = {}, 0

    def finish(self) -> List[str]:
        if self.buf or not self.written:
            self._flush()
        if self.single and len(self.written) == 1:
            final = self.dir / SINGLE_FILE
            self.written[0].rename(final)
            return [final.name]
        n = len(self.written)
        names = []
        for i, path in enumerate(self.written, 1):
            final = self.dir / f"model-{i:05d}-of-{n:05d}.safetensors"
            path.rename(final)
            for key, f in self.weight_map.items():
                if f == path.name:
                    self.weight_map[key] = final.name
            names.append(final.name)
        (self.dir / INDEX_NAME).write_text(
            json.dumps({"metadata": {"total_size": self.total}, "weight_map": dict(sorted(self.weight_map.items()))}, indent=2)
        )
        return names


def _pack_codes_for(codes: Any, bits: int) -> Any:
    import torch

    from .rotation import pack_int4_nibbles

    if bits == 4:
        if codes.shape[1] % 2:
            raise ValueError("INT4 codes need an even number of columns to nibble-pack")
        return torch.from_numpy(pack_int4_nibbles(codes)).contiguous()
    return codes.to(torch.int8).contiguous()


def _tensor_key(module: str, key: str) -> str:
    return f"foldquant.{module}.{key.replace('/', '.')}"


def write_quantized_checkpoint(
    directory: Any,
    *,
    base_dir: Any,
    state: QuantState,
    sites: Mapping[str, Mapping[str, Mapping[str, Any]]],
    overrides: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Write *state* over the base checkpoint at *base_dir* as a quantized checkpoint in *directory*.

    The base's weight files are streamed tensor by tensor: every tensor a site
    claims is left out and replaced by the site's ``qweight`` / ``weight_scale``
    split by projection; a tensor in *overrides* (``{checkpoint key: tensor}``,
    the weights ModelOpt smoothed in place) is written from there instead of
    the base; every other tensor is copied. The base's other files
    (configs, tokenizer and processor files, assets) are copied from disk as
    they are; nothing is saved from the loaded policy, whose tokenizer the
    calibration has run. The only files written fresh are the weight files,
    ``foldquant_quant.json`` (``state.manifest`` plus the weight layout),
    ``hf_quant_config.json`` and the model card.
    """
    import shutil

    import torch
    from safetensors import safe_open

    out = Path(directory)
    base = Path(base_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    weight_files, sharded = _base_weight_files(base)

    quantized: Dict[str, Tuple[str, str, int]] = {}  # checkpoint key -> (module, pack key, entry index)
    for module, table in sites.items():
        for key, info in table.items():
            for e, group in enumerate(info["params"]):
                for ckpt_key in group:
                    if ckpt_key in quantized:
                        raise ValueError(f"{ckpt_key} is claimed by two sites")
                    quantized[ckpt_key] = (module, key, e)

    overrides = dict(overrides or {})
    if set(overrides) & set(quantized):
        raise ValueError(f"{sorted(set(overrides) & set(quantized))[:3]} both quantized and overridden")
    writer = _ShardWriter(out, single=not sharded)
    shapes: Dict[str, Tuple[int, ...]] = {}
    seen_overrides = set()
    for wf in weight_files:
        with safe_open(str(wf), framework="pt") as f:
            for key in f.keys():
                if key in quantized:
                    shapes[key] = tuple(f.get_slice(key).get_shape())
                    continue
                if key in overrides:
                    value = overrides[key].detach().to("cpu")
                    if tuple(value.shape) != tuple(f.get_slice(key).get_shape()):
                        raise ValueError(f"{key}: override shape {tuple(value.shape)} differs from the base")
                    writer.add(key, value)
                    seen_overrides.add(key)
                    continue
                writer.add(key, f.get_tensor(key))
    missing = sorted((set(quantized) | set(overrides)) - set(shapes) - seen_overrides)
    if missing:
        raise ValueError(f"the base checkpoint has no tensor {missing[:3]}; the quant state belongs to another model")

    n_quantized = 0
    for module, table in sites.items():
        ms = state.modules[module]
        for key, info in table.items():
            for (codes, scale), group in zip(ms.gptq[key], info["params"]):
                start = 0
                for ckpt_key in group:
                    rows = int(shapes[ckpt_key][0])
                    head = ckpt_key[: -len(".weight")]
                    writer.add(f"{head}.qweight", _pack_codes_for(codes[start : start + rows], int(info["bits"])))
                    writer.add(f"{head}.weight_scale", scale[start : start + rows].to(torch.float32))
                    start += rows
                    n_quantized += 1
        for tkey, value in ms.tensors.items():
            writer.add(_tensor_key(module, tkey), torch.as_tensor(value).detach().to("cpu"))
    files = writer.finish()

    # A ModelOpt tower's mode state (quantizer layout and config) is not JSON: it goes
    # beside the weights as its own file, named in the manifest.
    configs: Dict[str, Dict[str, Any]] = {}
    for name, ms in state.modules.items():
        cfg = dict(ms.config)
        if "modelopt_state" in cfg:
            fname = MODELOPT_STATE_FILE.format(module=name)
            torch.save(cfg.pop("modelopt_state"), out / fname)
            cfg["modelopt_state_file"] = fname
        configs[name] = cfg

    # The base's other files: configs, tokenizer and processor files, assets. Copied byte for
    # byte from the base directory on disk, never re-serialized from the loaded policy. The
    # calibration runs the tokenizer and processor, and a tokenizer saved from that state can
    # carry the calibration's padding / max_length settings, which would truncate at inference.
    skip = {p.name for p in weight_files} | {INDEX_NAME, "README.md", MANIFEST_NAME, HF_QUANT_CONFIG_NAME}
    skip |= {MODELOPT_STATE_FILE.format(module=name) for name in state.modules}
    for src in sorted(base.rglob("*")):
        rel = src.relative_to(base)
        if any(part.startswith(".") for part in rel.parts) or not src.is_file() or (len(rel.parts) == 1 and rel.name in skip):
            continue
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    manifest = dict(state.manifest)
    manifest.update(
        format=FORMAT,
        format_version=FORMAT_VERSION,
        weights={"files": files, "sharded": bool(sharded), "quantized_projections": n_quantized,
                 "overridden": sorted(seen_overrides)},
        modules={
            name: {
                "scheme": ms.scheme,
                "config": configs[name],
                "tensors": sorted(ms.tensors),
                "sites": {k: {"bits": v["bits"], "params": v["params"]} for k, v in sites.get(name, {}).items()},
            }
            for name, ms in state.modules.items()
        },
    )
    (out / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
    (out / HF_QUANT_CONFIG_NAME).write_text(json.dumps(_hf_quant_config(state, n_quantized), indent=2))
    logger.info(
        "wrote quantized checkpoint %s: %d quantized projections, %s", out, n_quantized,
        ", ".join(f"{k} {v.scheme}" for k, v in state.modules.items()),
    )
    return out


def _hf_quant_config(state: QuantState, n_quantized: int) -> Dict[str, Any]:
    """The summary NVIDIA ModelOpt's Hugging Face export writes, for readers that look for it."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        ver = version("foldquant")
    except PackageNotFoundError:
        ver = "unknown"
    arms = {(bits_of(ms), int(ms.config.get("act_bits") or bits_of(ms))) for ms in state.modules.values()}
    algo = "W{}A{}".format(*next(iter(arms))) if len(arms) == 1 else "MIXED"
    return {
        "producer": {"name": "foldquant", "version": ver},
        "quantization": {
            "quant_algo": algo,
            "modules": {
                name: {
                    "scheme": ms.scheme,
                    "weight_bits": bits_of(ms),
                    "site_bits": ms.config.get("site_bits") or {},
                    "act_bits": ms.config.get("act_bits") or bits_of(ms),
                    "params": ms.config.get("params") or {},
                }
                for name, ms in state.modules.items()
            },
            "quantized_projections": n_quantized,
            "weight_format": "qweight (int8 codes, or int4 codes nibble-packed as uint8) + weight_scale (fp32 per row)",
        },
    }


# Reading


def is_quantized_checkpoint(path: Any) -> bool:
    """True when *path* is a directory written by :func:`write_quantized_checkpoint`."""
    if path is None:
        return False
    manifest = Path(path) / MANIFEST_NAME
    if not manifest.is_file():
        return False
    try:
        return json.loads(manifest.read_text()).get("format") == FORMAT
    except (OSError, ValueError):
        return False


def load_quantized_checkpoint(directory: Any) -> QuantState:
    """Read the quant state back out of a quantized checkpoint: codes, scales and site tensors."""
    import torch
    from safetensors import safe_open

    src = Path(directory)
    manifest = json.loads((src / MANIFEST_NAME).read_text())
    if manifest.get("format") != FORMAT:
        raise ValueError(f"{src / MANIFEST_NAME} is not a FoldQuant quantized checkpoint (format {manifest.get('format')!r})")
    if int(manifest.get("format_version", 0)) > FORMAT_VERSION:
        raise ValueError(f"{src} was written by a newer FoldQuant (format {manifest['format_version']} > {FORMAT_VERSION})")
    files = [src / f for f in manifest["weights"]["files"]]
    handles = [safe_open(str(f), framework="pt") for f in files]
    where: Dict[str, Any] = {}
    for h in handles:
        for key in h.keys():
            where[key] = h

    def get(key: str) -> Any:
        if key not in where:
            raise KeyError(f"{src}: tensor {key!r} missing from the checkpoint")
        return where[key].get_tensor(key)

    modules_meta = manifest.pop("modules")
    state = QuantState(manifest=manifest)
    for name, meta in modules_meta.items():
        ms = ModuleQuantState(module=name, scheme=meta["scheme"], config=dict(meta["config"]))
        if ms.config.get("modelopt_state_file"):
            # our own torch.save (dtypes, calibration tensors), never a foreign pickle
            ms.config["modelopt_state"] = torch.load(src / ms.config["modelopt_state_file"], weights_only=False, map_location="cpu")
        for key in meta["tensors"]:
            ms.tensors[key] = get(_tensor_key(name, key))
        for pack_key, info in meta["sites"].items():
            bits = int(info["bits"])
            entries = []
            for group in info["params"]:
                codes_parts, scale_parts = [], []
                for ckpt_key in group:
                    head = ckpt_key[: -len(".weight")]
                    stored = get(f"{head}.qweight")
                    codes_parts.append(_unpack_codes(stored, "int4" if bits == 4 else "int8"))
                    scale_parts.append(get(f"{head}.weight_scale").to(torch.float32))
                entries.append((torch.cat(codes_parts, dim=0), torch.cat(scale_parts, dim=0)))
            ms.gptq[pack_key] = entries
        state.modules[name] = ms
    return state


def quantized_keys(directory: Any) -> List[str]:
    """The checkpoint keys of the replaced weights (each now a ``qweight`` / ``weight_scale`` pair)."""
    manifest = json.loads((Path(directory) / MANIFEST_NAME).read_text())
    return sorted(k for m in manifest["modules"].values() for s in m["sites"].values() for g in s["params"] for k in g)
