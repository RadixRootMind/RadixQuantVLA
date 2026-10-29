# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""Capture module inputs during calibration replay.

Forward pre-hooks record calls made by ``forward_loop(module)`` for
SmoothQuant and GPTQ calibration. This module also detects LLM architectures
from their configs, extracts Qwen3-VL graph parameters from captured calls,
and loads learned SmoothQuant scales.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .dit_common import resolve_attend_n

#: The DiT inputs :func:`foldquant.dit_int4.compute_dit_sq_scales` unpacks, in its unpack order.
DIT_SQ_INPUT_NAMES: Tuple[str, ...] = (
    "hidden_states",
    "encoder_hidden_states",
    "timestep",
    "image_mask",
    "backbone_attention_mask",
)
#: The two of them a plain DiT's forward may not declare.
DIT_MASK_INPUT_NAMES: Tuple[str, ...] = ("image_mask", "backbone_attention_mask")


def capture_dit_inputs(
    module: nn.Module, forward_loop: Any, *, input_names: Tuple[str, ...] = DIT_SQ_INPUT_NAMES
) -> list:
    """Replay *forward_loop* and return the DiT's inputs as ``input_names``-ordered tuples.

    The hook takes kwargs and binds against the signature: GR00T's DiT is invoked
    entirely by keyword, so a positional-only capture would record an empty tuple
    per call. Binding normalizes either calling convention into one fixed order.
    """
    captured: list = []
    signature = inspect.signature(module.forward)
    attend_all = resolve_attend_n(module) is None

    def _hook(_m: nn.Module, args: tuple, kwargs: dict) -> None:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = {name: bound.arguments.get(name) for name in input_names}
        if attend_all:
            # A plain DiT attends the whole encoder sequence and its forward
            # carries no masks (or ignores them). The graph contract still
            # names both, so record what "no mask" means: attend everything.
            vl = values["encoder_hidden_states"]
            ones = torch.ones(vl.shape[:2], dtype=torch.bool, device=vl.device)
            for name in DIT_MASK_INPUT_NAMES:
                if values.get(name) is None:
                    values[name] = ones
        captured.append(tuple(values[name] for name in input_names))

    handle = module.register_forward_pre_hook(_hook, with_kwargs=True)
    try:
        forward_loop(module)
    finally:
        handle.remove()

    if not captured:
        raise RuntimeError(
            "calibration captured no DiT calls; the replay never reached the module, so "
            "SmoothQuant scales cannot be measured."
        )
    absent = [name for name, value in zip(input_names, captured[0]) if value is None]
    if absent:
        raise RuntimeError(
            f"calibration captured DiT calls missing {absent}; every one of {list(input_names)} "
            "must be present in the replayed call."
        )
    return captured


def dit_inputs_for(module: nn.Module, sample: Tuple[Any, ...]) -> Tuple[Any, ...]:
    """Move one captured DiT call onto *module*'s device, floating inputs in *module*'s dtype.

    The host may run its action head under ``torch.autocast`` (N1.5 does), so a
    captured ``encoder_hidden_states`` can be fp32 straight out of a LayerNorm while
    the DiT's weights are bf16. Autocast reconciled the two at every matmul and a
    bare replay cannot. The deployed graph's inputs are declared in the module's
    dtype, so the replay feeds exactly what the engine will see. Integer and bool
    inputs (timestep, masks) only change device.
    """
    param = next(module.parameters())
    return tuple(t.to(param.device, param.dtype) if torch.is_floating_point(t) else t.to(param.device) for t in sample)


def dit_accepts_masks(module: nn.Module) -> bool:
    """Whether the DiT's forward takes ``image_mask`` / ``backbone_attention_mask``.

    ``AlternateVLDiT`` does; a plain ``DiT`` may (and ignore them) or may not
    declare them at all. A replay of captured inputs passes the masks only when
    the signature has somewhere to put them.
    """
    params = inspect.signature(module.forward).parameters
    return all(name in params for name in DIT_MASK_INPUT_NAMES)


def capture_llm_snapshots(decoder: nn.Module, forward_loop: Any) -> list:
    """Replay *forward_loop* and return each raw ``(args, kwargs)`` call to *decoder*.

    Hook the DECODER (the module exposing ``.layers``), not a CausalLM-style
    wrapper around it: an export wrapper that calls ``language_model.model``
    directly never fires a hook on ``language_model``. Callers pass
    :func:`foldquant.llm.resolve_qwen3_decoder` output.
    """
    captured: list = []

    def _hook(_m: nn.Module, args: tuple, kwargs: dict) -> None:
        captured.append((args, dict(kwargs)))

    handle = decoder.register_forward_pre_hook(_hook, with_kwargs=True)
    try:
        forward_loop(decoder)
    finally:
        handle.remove()

    if not captured:
        raise RuntimeError(
            "calibration captured no LLM calls; the replay never reached the decoder, so "
            f"SmoothQuant scales cannot be measured. (Hook target: {type(decoder).__name__}.)"
        )
    return captured


def captured_prefix_len(snapshots: list) -> int:
    """Sequence length of the first captured call's hidden states ([B, S, K] → S)."""
    for args, kwargs in snapshots:
        for value in list(args) + list(kwargs.values()):
            if isinstance(value, torch.Tensor) and value.dim() == 3 and value.is_floating_point():
                return int(value.shape[1])
    raise ValueError("No captured decoder call carries a 3-D hidden-states tensor to pin the prefix length from.")


# LLM architecture, decided from the module's config


def _model_type(module: nn.Module) -> str:
    cfg = getattr(module, "config", None)
    return str(getattr(cfg, "model_type", "")).lower()


def rope_scaling(module: nn.Module) -> Dict[str, Any]:
    """The module's ``config.rope_scaling`` mapping, or an empty dict."""
    config: Any = getattr(module, "config", None)
    scaling: Any = getattr(config, "rope_scaling", None)
    return dict(scaling) if isinstance(scaling, dict) else {}


def is_gemma(module: nn.Module) -> bool:
    """Pi0 / Pi0.5's PaliGemma decoder reports ``model_type == "gemma"``."""
    return _model_type(module) == "gemma"


def is_qwen3_vl(module: nn.Module) -> bool:
    """GR00T N1.7's Qwen3-VL text tower, decided by ``rope_scaling.mrope_section``, not by name."""
    return bool(rope_scaling(module).get("mrope_section"))


def qwen3_vl_graph_params(module: nn.Module, snapshots: list) -> Optional[Dict[str, Any]]:
    """Emitter kwargs for a Qwen3-VL LLM, or ``None`` for a plain Qwen decoder.

    Qwen3 (N1.5 / N1.6) bakes a 1-D RoPE table; Qwen3-VL (N1.7) computes
    interleaved M-RoPE in-graph from ``position_ids`` and injects deepstack
    residuals after its first layers. Every value comes off the captured
    calibration call the fold itself uses, so graph and scales describe one pass.
    """
    scaling = rope_scaling(module)
    mrope_section = scaling.get("mrope_section")
    if not mrope_section:
        return None
    deepstack: list = []
    batch = 1
    for args, kwargs in snapshots:
        deepstack = list(kwargs.get("deepstack_visual_embeds") or []) or deepstack
        embeds = args[0] if args else kwargs.get("inputs_embeds")
        if embeds is not None and getattr(embeds, "ndim", 0) == 3:
            batch = int(embeds.shape[0])
    if not deepstack:
        raise ValueError(
            "This LLM declares rope_scaling.mrope_section (Qwen3-VL) but the calibration replay "
            "captured no 'deepstack_visual_embeds'; the plugin graph injects those residuals as "
            "explicit inputs and cannot be emitted without knowing how many there are."
        )
    return {
        "n1d7_mode": True,
        "mrope_section": list(mrope_section),
        "attention_scaling": float(scaling.get("attention_factor", 1.0)),
        "num_deepstack": len(deepstack),
        "num_vis_tokens": int(deepstack[0].shape[0]),
        "batch": batch,
    }


def load_learned_calib(path: Optional[str], *, num_layers: int, device: Any) -> Optional[Dict[str, Any]]:
    """Load learned SmoothQuant scales / clips and check they match the decoder.

    Returns ``{"sq_scales", "act_clip", "weight_clip"}`` with tensors on *device*,
    or ``None`` when *path* is empty. The layer count must match: a file learned
    on a different LLM would fold silently into the wrong sites.
    """
    if not path:
        return None
    data = torch.load(path, map_location="cpu")
    sq = {k: torch.as_tensor(v).float().to(device) for k, v in data["sq_scales"].items()}
    layers = {int(k.split("_")[0][1:]) for k in sq}
    if layers != set(range(num_layers)):
        raise ValueError(f"learned_calib {path}: scales cover layers {sorted(layers)}, decoder has {num_layers}")
    act_clip = {k: float(v) for k, v in (data.get("act_clip") or {}).items()}
    wc = data.get("weight_clip") or None
    weight_clip = {k: torch.as_tensor(v).float() for k, v in wc.items()} if wc else None
    return {"sq_scales": sq, "act_clip": act_clip, "weight_clip": weight_clip}


__all__: List[str] = [
    "DIT_SQ_INPUT_NAMES",
    "capture_dit_inputs",
    "dit_inputs_for",
    "capture_llm_snapshots",
    "captured_prefix_len",
    "is_gemma",
    "is_qwen3_vl",
    "load_learned_calib",
    "qwen3_vl_graph_params",
    "rope_scaling",
]
