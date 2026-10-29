# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""The FoldQuant pipeline: quantized checkpoint -> real-quant ONNX -> TensorRT engines.

A quantized checkpoint (:mod:`foldquant.quantized_checkpoint`) is the base
checkpoint with every quantized projection's weight replaced by its integer
codes and per-row scale, plus the fold settings and activation transforms of
each site. Loading it means loading the policy from that directory as usual,
which leaves the quantized projections uninitialised, and calling
:func:`install_fake_quant` on its quantized modules, which replaces those
projections with the arithmetic the TensorRT engine runs (the same codes,
scales, rotations and per-token quantization). The family tools do that by
themselves when ``--model-path`` names a quantized checkpoint. The same
directory converts to the plugin ONNX graphs with
:func:`foldquant.export.export_llm` / ``export_dit`` / ``export_expert`` and
``state=``, without calibration data. A ModelOpt baseline tower has its
quantizers restored from the checkpoint and is traced from the example inputs
the checkpoint recorded.

Coverage: the LLM backbones (Qwen3 / Qwen3-VL, Gemma prefix) for the folded
schemes, the GR00T DiT for every DiT scheme, and the π₀.₅ action expert for
every expert scheme (:mod:`foldquant.expert_fake_quant`).

Command line, run inside a model family's environment (``models/<family>``, with
``foldquant`` and the family's ``foldquant_integration`` importable)::

    python -m foldquant.quantized info    --quantized-model D
    python -m foldquant.quantized push    --quantized-model D --repo-id org/name [--public] [--dry-run]
    python -m foldquant.quantized to-onnx --quantized-model D --output-dir ONNX
    python -m foldquant.quantized build   --onnx-dir ONNX --engine-dir ENGINES [family engine options]
    python -m foldquant.quantized convert --quantized-model D --output-dir OUT [...]

The families wrap ``to-onnx`` as ``foldquant_integration.export`` and ``build``
as ``foldquant_integration.build_engines``. ``to-onnx`` writes ``OUT/onnx`` (the
real-quant plugin graphs, byte-identical to the ones the quantization built),
``build`` compiles ``OUT/onnx`` into ``OUT/engines``, ``convert`` does both. The
family-specific parts (loading the policy, which modules are quantized, how
the engine directory is assembled) come from the family's adapter,
``foldquant_integration.quantized``, which defines ``FAMILY``,
``load_policy(model_path, embodiment_tag, device)``, ``module_paths(policy)``,
``checkpoint_root(policy)``, ``build_engines(onnx_dir, engine_dir, **options)``
and, for a family whose engine set also holds float components,
``complete_onnx(policy, onnx_dir, state)``.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

from . import schemes
from .quant_state import MANIFEST_NAME, ModuleQuantState, QuantState
from .quantized_checkpoint import (
    LLM_SITES,
    is_quantized_checkpoint,
    load_quantized_checkpoint,
    site_tables,
    write_quantized_checkpoint,
)

logger = logging.getLogger(__name__)

__all__ = [
    "EXPORT_MANIFEST_NAME",
    "FakeQuantHandle",
    "quantized_arm",
    "is_quantized_model",
    "resolve_model_path",
    "export_onnx",
    "install_fake_quant",
    "load_adapter",
    "load_quantized_model",
    "push_to_hub",
    "save_arm_state",
    "verify_base_checkpoint",
    "write_model_card",
]

def load_quantized_model(directory: Any) -> QuantState:
    """The quant state of a quantized checkpoint."""
    if not is_quantized_checkpoint(directory):
        raise ValueError(f"{directory} is not a FoldQuant quantized checkpoint (no {MANIFEST_NAME} in the packed format)")
    return load_quantized_checkpoint(directory)


class _ModelOptHandle:
    """A restored ModelOpt tower stays quantized: its weights are the smoothed ones the
    checkpoint holds, so there is no float arm of it to go back to."""

    def __init__(self, name: str) -> None:
        self.name = name

    def remove(self) -> None:
        logger.info("%s: ModelOpt quantizers stay in place (the checkpoint holds the smoothed weights)", self.name)


def _install_modelopt(name: str, module: Any, ms: ModuleQuantState) -> _ModelOptHandle:
    from .modelopt_int8 import restore_quantizers

    restore_quantizers(module, ms.config["modelopt_state"], ms.tensor_group("modelopt"))
    return _ModelOptHandle(name)

#: The export manifest every family writes beside its graphs (``build_engines`` and
#: the runtimes read the plugin libraries off it).
EXPORT_MANIFEST_NAME = "foldquant_export.json"


class FakeQuantHandle:
    """Undo token for :func:`install_fake_quant`."""

    def __init__(self, handles: List[Any]) -> None:
        self._handles = handles

    def remove(self) -> None:
        for h in reversed(self._handles):
            h.remove()
        self._handles = []


#: The row order each LLM site's weights are concatenated in by the emitter
#: (``llm._emit_layer``): the order its recorded codes are in.
_LLM_SITES = LLM_SITES


def install_llm_fake_quant(module: Any, ms: ModuleQuantState) -> Any:
    """Fake-quant a Qwen / Gemma LLM from its state, with the engine's codes, in fp32.

    Every projection becomes a :class:`foldquant.fake_quant_linear.FakeQuantLinear`
    over the site's weight codes and per-row scales: the recorded GPTQ codes for
    the INT4 sites, round-to-nearest of the folded weight (the emitter's rounding)
    for the INT8 ones. The SmoothQuant fold the kernel absorbs into the norm gain
    is applied as a division of the (unchanged) norm output, which is the same
    function; the down site's scale is already inside the up codes.
    """
    import torch

    from .fake_quant_linear import FakeQuantLinear, SwapHandle, swap_module
    from .llm import resolve_qwen3_decoder
    from .llm_rotation_sq import apply_rot_fold, apply_sq_fold
    from .rotation import hadamard_blocks
    from .weights import quant_weight_per_row

    if ms.scheme not in schemes.LLM_FOLDED_SCHEMES:
        raise NotImplementedError(
            f"LLM fake-quant covers the folded schemes {sorted(schemes.LLM_FOLDED_SCHEMES)}; {ms.scheme!r} is not one"
        )
    cfg = ms.config
    bits = int(cfg["bits"])
    rot_bs = int(cfg["rot_bs"])
    site_bits = dict(cfg.get("site_bits") or {})
    # The emitter's width rule (foldquant.llm, _width): a site kept at 8 bit is the W8A8
    # plugin pair, 8-bit activations whatever the arm's act_bits; a 4-bit site quantizes its
    # activations to act_bits (8 for a W4A8 fold, else 4).
    arm_act_bits = int(cfg.get("act_bits") or bits)
    act_clip = cfg.get("act_clip", 1.0)
    sq = ms.tensor_group("sq")
    decoder = resolve_qwen3_decoder(module)
    gemma = "Gemma" in type(decoder).__name__
    qmax = {4: 7.0, 8: 127.0}
    handle = SwapHandle()
    try:
        for i, layer in enumerate(decoder.layers):
            state = {k: v.detach() for k, v in layer.state_dict().items()}
            if gemma:
                for g in ("input_layernorm.weight", "post_attention_layernorm.weight"):
                    state[g] = state[g] + 1.0
            dev = state["input_layernorm.weight"].device
            s_qkv, s_gu, s_dn = (sq[f"L{i}_{k}"].to(dev) for k in ("qkv", "gateup", "down"))
            folded = apply_sq_fold(state, s_qkv=s_qkv, s_gu=s_gu, s_dn=s_dn)
            if rot_bs > 1:
                folded = apply_rot_fold(folded, rot_bs)
            s_pre = {"qkv": s_qkv, "gateup": s_gu}
            for site, names in _LLM_SITES.items():
                w_bits = int(site_bits.get(site, bits))
                a_bits = 8 if (w_bits == 8 or arm_act_bits == 8) else 4
                clip = float(act_clip.get(f"L{i}_{site}", 1.0)) if isinstance(act_clip, dict) else float(act_clip)
                clip = clip if a_bits == 4 else 1.0
                entries = ms.gptq.get(f"llm.L{i}_{site}") or ms.gptq.get(f"llm.rtn.L{i}_{site}")
                rows = [int(folded[n + ".weight"].shape[0]) for n in names]
                if entries:
                    codes, scale = entries[0]
                    if int(codes.shape[0]) != sum(rows):
                        raise ValueError(f"llm.L{i}_{site}: recorded codes have {codes.shape[0]} rows, the layer {sum(rows)}")
                else:
                    # A format-1 state: recompute the emitter's round-to-nearest.
                    if w_bits == 4:
                        raise ValueError(f"llm.L{i}_{site}: an INT4 site without recorded GPTQ codes")
                    merged = torch.cat([folded[n + ".weight"] for n in names], dim=0)
                    c8, sc8 = quant_weight_per_row(merged)
                    codes, scale = torch.from_numpy(c8), torch.from_numpy(sc8)
                k = int(folded[names[0] + ".weight"].shape[1])
                rot = {}
                if rot_bs > 1:
                    perm, R = hadamard_blocks(k, rot_bs)
                    rot = {"perm": perm, "rot": R, "rot_bs": rot_bs}
                start = 0
                for name, n in zip(names, rows):
                    parent_name, _, leaf = name.rpartition(".")
                    parent = layer.get_submodule(parent_name)
                    lin = getattr(parent, leaf)
                    if name == "mlp.up_proj" and lin.bias is not None:
                        raise NotImplementedError("an up_proj bias would need the down site's scale folded in")
                    fq = FakeQuantLinear(
                        codes[start : start + n], scale[start : start + n],
                        None if lin.bias is None else lin.bias.detach(),
                        a_qmax=qmax[a_bits], s_pre=s_pre.get(site), a_clip=clip, **rot,
                    )
                    swap_module(handle, parent, leaf, fq)
                    start += n
    except Exception:
        handle.remove()
        raise
    logger.info(
        "LLM fake-quant (%s) installed: %d layers, %d projections, %s", ms.scheme, len(decoder.layers), len(handle),
        "recorded GPTQ codes" if ms.gptq else "round-to-nearest",
    )
    return handle


def install_fake_quant(modules: Mapping[str, Any], state: QuantState) -> FakeQuantHandle:
    """Install the fake-quant of every module *state* quantized.

    Args:
        modules: ``{"llm": ..., "dit": ...}`` (or ``"expert"``), the live modules
            at the same paths the export read.
        state: a loaded quant state.

    Raises:
        KeyError: *state* quantized a module *modules* does not provide.
        NotImplementedError: a module kind or scheme without a PyTorch reader.
    """
    handles: List[Any] = []
    try:
        for name, ms in state.modules.items():
            if name not in modules:
                raise KeyError(f"quant state covers {name!r}, which was not passed in")
            if ms.scheme in schemes.MODELOPT_SCHEMES:
                handles.append(_install_modelopt(name, modules[name], ms))
            elif name == "llm":
                handles.append(install_llm_fake_quant(modules[name], ms))
            elif name == "dit":
                from .dit_fake_quant import install_dit_fake_quant

                handles.append(install_dit_fake_quant(modules[name], ms))
            elif name == "expert":
                from .expert_fake_quant import install_expert_fake_quant

                handles.append(install_expert_fake_quant(modules[name], ms))
            else:
                raise NotImplementedError(
                    f"no PyTorch fake-quant for the {name!r} module yet; its state converts to ONNX only"
                )
    except Exception:
        for h in reversed(handles):
            h.remove()
        raise
    return FakeQuantHandle(handles)


def verify_base_checkpoint(state: QuantState, checkpoint: Any) -> None:
    """Refuse a base checkpoint whose content is not the one *state* was calibrated on.

    Every file the state recorded (weights, index, configs) must be present with
    the same content; files it did not record (a README, a licence, Hub metadata)
    are ignored, so a download or a copy of the checkpoint passes.
    """
    from .eval_protocol import artifact_digest, file_hashes

    if is_quantized_checkpoint(checkpoint):
        return  # the codes are in the checkpoint itself; there is no separate base to match
    recorded = (state.manifest.get("base") or {}).get("file_hashes")
    if recorded:
        root = Path(checkpoint)
        if not root.is_dir():
            raise FileNotFoundError(f"base checkpoint {checkpoint} not found")
        have = file_hashes(root)
        missing = sorted(set(recorded) - set(have))
        changed = sorted(k for k in recorded if k in have and have[k] != recorded[k])
        if missing or changed:
            raise ValueError(
                f"{checkpoint} is not the checkpoint this quant state was calibrated on: "
                + (f"missing {missing[:3]} " if missing else "")
                + (f"different content in {changed[:3]}" if changed else "")
                + ". Its codes and scales would be applied to other weights."
            )
        return
    want = (state.manifest.get("base") or {}).get("digest")
    if not want:
        raise ValueError("the quant state records no base-checkpoint digest")
    got = artifact_digest(checkpoint)
    if got is None or got.get("missing"):
        raise FileNotFoundError(f"base checkpoint {checkpoint} not found")
    if got["digest"] != want:
        raise ValueError(
            f"{checkpoint} is not the checkpoint this quant state was calibrated on "
            f"(content digest {got['digest'][:12]}..., expected {want[:12]}...). "
            "Its codes and scales would be applied to other weights."
        )


def write_model_card(directory: Any, state: QuantState, *, repo_id: Optional[str] = None) -> Path:
    """A ``README.md`` for the Hub: what the checkpoint is, what it needs, how to use it."""
    m = state.manifest
    base = m.get("base") or {}
    mods = {k: v.scheme for k, v in state.modules.items()}
    family = m.get("family", "?")
    n_proj = (m.get("weights") or {}).get("quantized_projections")
    lines = [
        "---",
        "library_name: foldquant",
        "tags: [foldquant, quantization, vision-language-action]",
        *( [f"base_model: {base['model_id']}"] if base.get("model_id") and "/" in str(base.get("model_id")) else [] ),
        "---",
        "",
        f"# FoldQuant quantized checkpoint: {family} {' / '.join(f'{k} {v}' for k, v in mods.items())}",
        "",
        f"The base checkpoint with {n_proj or 'every'} quantized projection's weight replaced by its integer",
        "codes (`qweight`, INT8, or INT4 nibble-packed) and per-row scale (`weight_scale`): the bytes the",
        "TensorRT plugins carry. Everything not quantized (vision tower, encoders, norms, biases) is the",
        "base's. Each site's activation transform (SmoothQuant vector, rotation) is stored beside them",
        "as `foldquant.<module>.*` tensors; `foldquant_quant.json` and `hf_quant_config.json` describe it.",
        "A ModelOpt baseline tower keeps its smoothed weights, its quantizers' scales and the example",
        "inputs its graph is traced from.",
        "Load it like the base checkpoint to run the quantized policy in PyTorch",
        "(fake-quant layers over the engine's own codes, scales and activation transforms, computed in fp32),",
        "or export the FoldQuant plugin ONNX graphs from it, byte-identical to the ones the quantization built,",
        "and build TensorRT engines, with no calibration data. The bf16 weights of the quantized projections",
        "are not in this checkpoint; a bf16 baseline runs from the base checkpoint.",
        "",
        "| module | scheme | params |",
        "|---|---|---|",
        *[f"| `{k}` | `{v.scheme}` | `{json.dumps(v.config.get('params') or {})}` |" for k, v in state.modules.items()],
        "",
        f"Base checkpoint: `{base.get('model_id', '?')}` (content digest `{str(base.get('digest', '?'))[:16]}`).",
        f"Calibration: {m.get('calibration', {}).get('num_samples', '?')} observations, "
        f"seed {m.get('calibration', {}).get('seed', '?')}, from `{m.get('dataset_path', '?')}`.",
        "",
        "## Use",
        "",
        "With the FoldQuantVLA repository and the family's environment:",
        "",
        "```bash",
        f"huggingface-cli download {repo_id or '<this repo>'} --local-dir quantized",
        "# PyTorch fake-quant policy, e.g. LIBERO or a robot server",
        "python -m foldquant_integration.eval_libero --protocol p3 --model-path quantized --output <out>",
        "# real-quant plugin ONNX graphs, then TensorRT engines on the target device",
        "python -m foldquant_integration.export --model-path quantized --output-dir exports/<arm>/onnx",
        "python -m foldquant_integration.build_engines --onnx-dir exports/<arm>/onnx --engine-dir exports/<arm>/engines",
        "```",
        "",
        "The base checkpoint's license governs any use of these weights.",
        "",
    ]
    path = Path(directory) / "README.md"
    path.write_text("\n".join(lines))
    return path


def is_quantized_model(path: Any) -> bool:
    """True when *path* is a FoldQuant quantized checkpoint."""
    return is_quantized_checkpoint(path)


def quantized_arm(model_path: Any, quantized_model: Optional[str], *, no_fakequant: bool = False, other_arms: Any = ()) -> Optional[str]:
    """The fake-quant state a family tool should run: *quantized_model* if given, else *model_path*
    when it is a fake-quant model and no other arm (engines, a baseline pack) was asked for."""
    if no_fakequant and is_quantized_checkpoint(model_path):
        raise SystemExit(
            f"{model_path} is a quantized checkpoint: its quantized projections hold codes, not bf16 weights, "
            "so there is no bf16 arm in it. Pass the base checkpoint for the bf16 baseline."
        )
    if quantized_model or no_fakequant or any(other_arms):
        return quantized_model
    if is_quantized_model(model_path):
        logger.info("%s is a FoldQuant fake-quant model; running it fake-quantized (--no-fakequant for bf16)", model_path)
        return str(model_path)
    return None


def resolve_model_path(state: QuantState, quantized_model: Any, model_path: Optional[str]) -> str:
    """The base checkpoint to load: *model_path*, else the fake-quant model directory itself."""
    return str(model_path) if model_path else str(quantized_model)


def push_to_hub(
    directory: Any,
    repo_id: str,
    *,
    private: bool = True,
    dry_run: bool = False,
    token: Optional[str] = None,
    commit_message: Optional[str] = None,
) -> Dict[str, Any]:
    """Upload a quantized checkpoint to a Hugging Face model repo (private by default).

    Every file of the checkpoint goes (hidden files excluded). ``dry_run`` lists
    the files without contacting the Hub.
    """
    src = Path(directory)
    if not is_quantized_checkpoint(src):
        raise FileNotFoundError(f"{src} is not a FoldQuant quantized checkpoint")
    files = sorted(p for p in src.rglob("*") if p.is_file() and not any(x.startswith(".") for x in p.relative_to(src).parts))
    listing = [{"path": p.relative_to(src).as_posix(), "bytes": p.stat().st_size} for p in files]
    report: Dict[str, Any] = {"repo_id": repo_id, "private": private, "files": listing, "dry_run": dry_run,
                              "self_contained": True}
    if dry_run:
        return report
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    info = api.upload_folder(
        folder_path=str(src),
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=[f["path"] for f in listing],
        commit_message=commit_message or "Upload FoldQuant quantized checkpoint",
    )
    report["commit"] = getattr(info, "oid", None) or str(info)
    return report


# The pipeline: state -> ONNX -> engines


def save_arm_state(
    directory: Any,
    *,
    family: str,
    model_path: str,
    results: List[Any],
    modules: Mapping[str, Any],
    checkpoint_root: Any,
    export_manifest: Dict[str, Any],
    extra_files: Optional[Dict[str, Any]] = None,
    base_model_id: Optional[str] = None,
    embodiment_tag: Optional[str] = None,
    load_kwargs: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write the quantized checkpoint of an export that ran with ``record=True``.

    *modules* are the live modules the export quantized (the adapter's
    ``module_paths``) and *checkpoint_root* the module whose parameter names are
    the checkpoint's tensor keys (the adapter's ``checkpoint_root``); together
    they say which checkpoint tensors each recorded pack replaces. The base
    checkpoint at *model_path* is streamed into *directory* with those tensors
    swapped for codes and scales (:func:`foldquant.quantized_checkpoint.write_quantized_checkpoint`).

    ``base_model_id`` names the base checkpoint where others can get it (a Hub
    ``org/name``, optionally ``@revision``); it defaults to the path's basename.
    ``embodiment_tag`` / ``load_kwargs`` are what the family adapter's
    ``load_policy`` needs to rebuild the same policy (a data config, a train
    config name); both default to what the export manifest says. The digest of
    every graph the export built is recorded, and :func:`export_onnx` refuses to
    finish when its rebuild differs.

    *export_manifest* is the family's ``foldquant_export.json`` content;
    *extra_files* are other JSON files the family writes beside its graphs
    (``{"export_metadata.json": {...}}``), rewritten verbatim by :func:`export_onnx`.
    """
    import torch

    from .eval_protocol import file_hashes
    from .provenance import public_path

    if not Path(model_path).is_dir():
        raise FileNotFoundError(f"{model_path}: the base checkpoint must be a local directory")
    hashes = file_hashes(model_path)
    import hashlib

    base = {"digest": hashlib.sha256(json.dumps(sorted(hashes.items())).encode()).hexdigest(), "files": len(hashes)}
    state = QuantState(
        manifest={
            "family": family,
            "base": {
                "model_id": base_model_id or public_path(model_path),
                "digest": base["digest"],
                "files": base["files"],
                "file_hashes": hashes,
            },
            "graphs": {r.module: r.graph_digest for r in results if r.state is not None},
            "graph_digest": "inline",
            "embodiment_tag": embodiment_tag if embodiment_tag is not None else export_manifest.get("embodiment_tag"),
            "load_kwargs": dict(load_kwargs or {}),
            "dataset_path": export_manifest.get("dataset_path"),
            "calibration": {k: (export_manifest.get("calibration") or {}).get(k) for k in ("seed", "num_samples")},
            "cascade": export_manifest.get("cascade", False),
            "export_manifest": export_manifest,
            "extra_files": dict(extra_files or {}),
        },
        modules={r.module: r.state for r in results if r.state is not None},
    )
    sites = site_tables(modules, state, checkpoint_root)
    # A ModelOpt tower's calibration rescales the weights of the linears it quantizes
    # (SmoothQuant and AWQ both fold their scale into the weight): the checkpoint holds
    # those as they are now. Nothing else in the tower changes, and a tied weight (Gemma's
    # embeddings, stored under the lm_head name) must keep the base checkpoint's layout.
    names = {id(p): n for n, p in checkpoint_root.named_parameters()}
    overrides: Dict[str, Any] = {}
    for name, ms in state.modules.items():
        if ms.scheme in schemes.MODELOPT_SCHEMES:
            for rel, sub in modules[name].named_modules():
                p = getattr(sub, "weight", None)
                if not hasattr(sub, "weight_quantizer") or not isinstance(p, torch.nn.Parameter):
                    continue
                if id(p) not in names:
                    raise ValueError(f"{rel}.weight of the {name} module is not a parameter of the checkpoint root")
                overrides[names[id(p)]] = p
    write_quantized_checkpoint(directory, base_dir=model_path, state=state, sites=sites, overrides=overrides)
    write_model_card(directory, load_quantized_checkpoint(directory))
    return Path(directory)


class GraphMismatchError(RuntimeError):
    """A graph rebuilt from a quant state differs from the one its calibrating export wrote."""


def export_onnx(
    state: QuantState,
    modules: Mapping[str, Any],
    output_dir: Any,
    *,
    source: Any = None,
    adapter: Any = None,
    policy: Any = None,
) -> Path:
    """Write the real-quant plugin graphs of *state* to *output_dir*.

    Each module's graph is emitted from its state (no calibration data, no GPTQ
    solve), under the file name the calibrating export used, together with the
    export manifest and the family's other JSON files. A ModelOpt tower is
    traced by the family *adapter* (``export_modelopt_graph``) from the example
    inputs the checkpoint recorded, after its quantizers are restored on the
    live module.
    """
    from .eval_protocol import artifact_digest
    from .export import export_module
    from .provenance import public_path

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = dict(state.manifest.get("export_manifest") or {})
    files = manifest.get("files") or {}
    results = []
    for name, ms in state.modules.items():
        if name not in modules:
            raise KeyError(f"quant state covers {name!r}, which the family adapter does not provide")
        kwargs: Dict[str, Any] = {}
        if name == "llm" and ms.config.get("max_seq_len"):
            kwargs["max_seq_len"] = int(ms.config["max_seq_len"])
        path = out / files.get(name, f"{name}.onnx")
        if ms.scheme in schemes.MODELOPT_SCHEMES:
            if adapter is None:
                raise ValueError(f"{name} is a {ms.scheme} tower; exporting it needs the family adapter and policy")
            _install_modelopt(name, modules[name], ms)
            result, record = adapter.export_modelopt_graph(name, policy, modules[name], path, ms)
            manifest.setdefault("modelopt", {})[name] = {**(ms.config.get("modelopt") or {}), **record}
            results.append(result)
            continue
        result = export_module(name, modules[name], path, scheme=ms.scheme, state=ms, **kwargs)
        results.append(result)
        # The digest of the graph in memory (foldquant.onnx_io.graph_digest), as recorded.
        want = (state.manifest.get("graphs") or {}).get(name)
        got = result.graph_digest
        if want is not None and got != want:
            raise GraphMismatchError(
                f"{path.name} rebuilt from the quant state differs from the graph the calibrating export "
                "wrote. The weights, the FoldQuant version or the numerical libraries differ from the "
                "machine that recorded the state (a dense rotation's SVD, for one); this graph would not "
                "match the fake-quant model. Rebuild with the recording environment or recalibrate."
            )
    for fname, content in (state.manifest.get("extra_files") or {}).items():
        (out / fname).write_text(json.dumps(content, indent=2))
    libs: List[str] = list(manifest.get("plugin_libs") or [])
    for r in results:
        for lib in r.plugin_libs:
            if lib not in libs:
                libs.append(lib)
    manifest["plugin_libs"] = libs
    if state.manifest.get("family"):
        manifest["family"] = state.manifest["family"]
    if source is not None:
        manifest["from_quantized_model"] = {
            "path": public_path(str(source)),
            "digest": (artifact_digest(source) or {}).get("digest"),
        }
    (out / EXPORT_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
    logger.info("wrote %s from the quant state (%s)", out, ", ".join(r.onnx_path.name for r in results))
    return out


def load_adapter(family: Optional[str] = None) -> Any:
    """The active family's adapter, ``foldquant_integration.quantized``.

    Raises:
        ImportError: no family integration is importable (run from ``models/<family>``).
        ValueError: the importable integration is not *family*.
    """
    try:
        adapter = importlib.import_module("foldquant_integration.quantized")
    except ImportError as exc:
        raise ImportError(
            "no family adapter importable: run inside a model family's environment with "
            "models/<family> on PYTHONPATH (it provides foldquant_integration.quantized)"
        ) from exc
    have = getattr(adapter, "FAMILY", None)
    if family is not None and have != family:
        raise ValueError(f"the quant state is for {family!r}, but the importable integration is {have!r}")
    return adapter


def install_on_policy(policy: Any, quantized_model: Any, model_path: Optional[str] = None, *, check_checkpoint: bool = True) -> Any:
    """Load a state, check the checkpoint *policy* came from, install the fake-quant: ``(handle, state)``."""
    state = load_quantized_model(quantized_model)
    adapter = load_adapter(state.manifest.get("family"))
    if check_checkpoint:
        verify_base_checkpoint(state, resolve_model_path(state, quantized_model, model_path))
    handle = install_fake_quant(adapter.module_paths(policy), state)
    logger.info("fake-quant arm from %s: %s", quantized_model, ", ".join(f"{k} {v.scheme}" for k, v in state.modules.items()))
    return handle, state


# Command line


@dataclass
class Info:
    """Print what a quantized checkpoint holds."""

    quantized_model: str


@dataclass
class Push:
    """Upload a fake-quant model to a Hugging Face model repo."""

    quantized_model: str
    repo_id: str
    public: bool = False
    """Create the repo public; private by default."""
    dry_run: bool = False
    """List what would be uploaded, without contacting the Hub."""
    token: Optional[str] = None


@dataclass
class ToOnnx:
    """Fake-quant state -> real-quant plugin ONNX in ``output_dir`` (no dataset, no GPTQ)."""

    quantized_model: str
    output_dir: str
    """Where the graphs and manifests are written."""
    embodiment_tag: Optional[str] = None
    """Defaults to the tag the model was quantized with."""
    device: str = "cuda"


@dataclass
class Build:
    """ONNX graphs -> TensorRT engines, by the family's builder."""

    onnx_dir: str
    """The directory ``to-onnx`` / ``export`` wrote."""
    engine_dir: str
    """Destination engine directory."""
    max_batch: int = 1
    """Batch bound of every optimization profile; 1 for a single-robot deployment."""
    workspace_mb: int = 8192


@dataclass
class Convert(ToOnnx):
    """``to-onnx`` then ``build``: fake-quant state -> real-quant ONNX in ``output_dir`` -> TensorRT
    engines in ``engine_dir``."""

    engine_dir: str = ""
    max_batch: int = 1
    workspace_mb: int = 8192


def info(args: Info) -> Dict[str, Any]:
    state = load_quantized_model(args.quantized_model)
    summary = {
        "family": state.manifest.get("family"),
        "base": state.manifest.get("base"),
        "embodiment_tag": state.manifest.get("embodiment_tag"),
        "calibration": state.manifest.get("calibration"),
        "cascade": state.manifest.get("cascade"),
        "modules": {
            k: {"scheme": v.scheme, "params": v.config.get("params"), "gptq_sites": len(v.gptq), "tensors": len(v.tensors)}
            for k, v in state.modules.items()
        },
    }
    print(json.dumps(summary, indent=2))
    return summary


def push(args: Push) -> Dict[str, Any]:
    state = load_quantized_model(args.quantized_model)
    if not args.dry_run:
        write_model_card(args.quantized_model, state, repo_id=args.repo_id)  # names this repo in its commands
    report = push_to_hub(args.quantized_model, args.repo_id, private=not args.public, dry_run=args.dry_run, token=args.token)
    size = sum(f["bytes"] for f in report["files"])
    logger.info(
        "%s %d files (%.1f MB) to %s (%s)", "would upload" if args.dry_run else "uploaded", len(report["files"]),
        size / 1e6, args.repo_id, "public" if args.public else "private",
    )
    return report


def to_onnx(args: ToOnnx) -> Path:
    state = load_quantized_model(args.quantized_model)
    adapter = load_adapter(state.manifest.get("family"))
    # The quantized model is self-contained: the policy is loaded from it, never from the base.
    policy = adapter.load_policy(
        str(args.quantized_model),
        args.embodiment_tag or state.manifest.get("embodiment_tag"),
        args.device,
        **(state.manifest.get("load_kwargs") or {}),
    )
    out = export_onnx(state, adapter.module_paths(policy), args.output_dir, source=args.quantized_model,
                      adapter=adapter, policy=policy)
    # A family whose engine set also holds float components (GR00T N1.7's ViT, encoders and
    # decoder) exports them here too, so one directory carries everything build needs.
    complete = getattr(adapter, "complete_onnx", None)
    if complete is not None:
        complete(policy, out, state)
    return out


def build(args: Union[Build, Convert]) -> Path:
    onnx_dir = Path(args.onnx_dir if isinstance(args, Build) else args.output_dir)
    manifest = json.loads((onnx_dir / EXPORT_MANIFEST_NAME).read_text())
    adapter = load_adapter(manifest.get("family"))
    if not args.engine_dir:
        raise SystemExit("--engine-dir is required")
    engine_dir = Path(args.engine_dir)
    adapter.build_engines(onnx_dir, engine_dir, max_batch=args.max_batch, workspace_mb=args.workspace_mb)
    logger.info("engines in %s", engine_dir)
    return engine_dir


def convert(args: Convert) -> Path:
    to_onnx(args)
    return build(args)


def main(argv: Optional[List[str]] = None) -> Any:
    import tyro

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cmd = tyro.extras.subcommand_cli_from_dict(
        {"info": Info, "push": Push, "to-onnx": ToOnnx, "build": Build, "convert": Convert}, args=argv
    )
    handlers = {Info: info, Push: push, ToOnnx: to_onnx, Build: build, Convert: convert}
    return handlers[type(cmd)](cmd)


if __name__ == "__main__":
    main()
