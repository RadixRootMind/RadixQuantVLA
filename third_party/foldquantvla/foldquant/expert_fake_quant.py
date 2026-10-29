# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""PyTorch fake-quant of the Pi (Gemma-300M) action expert, read off its plugin graph.

As for the DiT (:mod:`.dit_fake_quant`), nothing is re-derived: the expert
emitter (:func:`foldquant.gemma_expert.build_gemma_expert_plugin_onnx`) runs in
memory with the state's weight packs replayed, and every
``PerRowInt{4,8}LinearResidual`` node is decoded back into the projections it
serves. Each layer has four such GEMMs: ``G{i}_qkv`` (q, k, v stacked by rows),
``G{i}_o``, ``G{i}_gu`` (gate, up) and ``G{i}_dn``. Their activation transform
is whatever the node declares:

* ``rot_block_size > 1``: the fixed Sylvester butterfly, with the SmoothQuant
  vector divided out before it (``act_scale_pre``) or after it (``act_scale_ch``);
* ``perm`` + ``rotation``: the learned dense block rotation with the scale folded
  in, which the kernel applies in bf16 through cuBLAS, so the matrix and the
  rotated activation are rounded to bf16 here too;
* neither: the plain per-row path (``w8a8``).

The norms (AdaRMS or vanilla), attention, the time MLP and the in/out
projections stay in the module's own precision, as they do in the engine.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import torch
import torch.nn as nn

from . import schemes
from .dit_fake_quant import _attrs, _codes, _f32
from .fake_quant_linear import FakeQuantLinear, SwapHandle, swap_module
from .quant_state import ModuleQuantState, pack_scope, replay_sites

logger = logging.getLogger(__name__)

__all__ = ["build_expert_graph", "install_expert_fake_quant"]

#: Projection names behind each per-layer GEMM, in the emitter's row order.
_EXPERT_SITES = {
    "qkv": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
    "o": ("self_attn.o_proj",),
    "gu": ("mlp.gate_proj", "mlp.up_proj"),
    "dn": ("mlp.down_proj",),
}

_EXPERT_SCHEMES = frozenset({schemes.W8A8, schemes.W8A8_SH, schemes.W4A4_SR, schemes.W4A4_SH, schemes.W4A4_SHG})


def build_expert_graph(module: nn.Module, state: ModuleQuantState) -> Any:
    """The expert plugin ``GraphProto`` for *state*, built in memory with its weight packs replayed.

    The keyword arguments mirror :func:`foldquant.export.export_expert` exactly;
    a knob taken differently here would decode a graph the engine is not built from.
    """
    from .export import _act_fold_knobs
    from .gemma_expert import build_gemma_expert_plugin_onnx

    scheme = state.scheme
    params = dict(state.config.get("params") or {})
    replay = replay_sites(state)
    kwargs: Dict[str, Any] = {}
    if scheme != schemes.W8A8:
        knobs = _act_fold_knobs(scheme, params)
        kwargs = {
            "int4": scheme in schemes.ACT_W4A4_SCHEMES,
            "sq_scales": state.tensor_group("sq"),
            "fold_order": knobs["fold_order"],
        }
        if scheme == schemes.W4A4_SHG:
            kwargs["gptq"] = replay
        if knobs["fwht"]:
            kwargs["fwht"] = True
    with pack_scope("expert", replay=replay):
        model = build_gemma_expert_plugin_onnx(module, None, **kwargs)
    replay.assert_consumed()
    return model.graph


def _transform(a: Dict[str, Any], k: int) -> Dict[str, Any]:
    """The activation transform one GEMM node declares, as :class:`FakeQuantLinear` kwargs."""
    from .rotation import hadamard_blocks

    rot_bs = int(a.get("rot_block_size", 0))
    has_dense = bool(a.get("perm")) or bool(a.get("rotation"))
    if rot_bs > 1:
        if has_dense:
            raise ValueError("a node carries both a butterfly block and a dense rotation")
        if k % rot_bs:
            raise ValueError(f"butterfly block {rot_bs} does not divide K={k}; the fake-quant does not pad")
        perm, rot = hadamard_blocks(k, rot_bs)
        t: Dict[str, Any] = {"perm": perm, "rot": rot, "rot_bs": rot_bs}
        if a.get("act_scale_pre") and a.get("act_scale_ch"):
            raise ValueError("a node carries both SmoothQuant orders; the kernel refuses that")
        if a.get("act_scale_pre"):
            t["s_pre"] = _f32(a["act_scale_pre"])
        if a.get("act_scale_ch"):
            t["s_post"] = _f32(a["act_scale_ch"])
        return t
    if has_dense:
        rot = _f32(a["rotation"])
        bs = rot.numel() // k  # (K/bs) blocks of bs x bs: K * bs values
        if bs < 1 or bs * k != rot.numel() or k % bs:
            raise ValueError(f"dense rotation of {rot.numel()} values does not tile K={k}")
        # The plugin uploads the matrix as bf16 and rotates the bf16 activation into bf16.
        rot = rot.to(torch.bfloat16).float()
        from .dit_fake_quant import _i32

        return {"perm": _i32(a["perm"]), "rot": rot.reshape(k // bs, bs, bs), "rot_bs": bs, "stage_bf16": True}
    if a.get("act_scale_pre") or a.get("act_scale_ch"):
        raise ValueError("a SmoothQuant vector without a rotation; the kernel has no such mode")
    return {}


def install_expert_fake_quant(module: nn.Module, state: ModuleQuantState) -> SwapHandle:
    """Replace the expert's quantized Linears with :class:`FakeQuantLinear` s; returns an undo handle.

    *module* is the expert as the export read it: an object exposing
    ``expert_model`` (the Gemma decoder) with the emitter's state-dict layout,
    such as the Pi0.5 integration's ``Pi05ExpertView``.
    """
    if state.scheme not in _EXPERT_SCHEMES:
        raise NotImplementedError(f"expert fake-quant has no reader for scheme {state.scheme!r}")
    int4 = state.scheme in schemes.ACT_W4A4_SCHEMES
    bits, qmax = (4, 7.0) if int4 else (8, 127.0)
    op = "PerRowInt4LinearResidual" if int4 else "PerRowInt8LinearResidual"
    tail = "_plr4" if int4 else "_plr"
    graph = build_expert_graph(module, state)
    nodes = {n.name: _attrs(n) for n in graph.node if n.op_type == op}
    layers = module.expert_model.model.layers
    if len(nodes) != 4 * len(layers):
        raise ValueError(f"expected {4 * len(layers)} {op} nodes for {len(layers)} layers, the graph has {len(nodes)}")
    swaps = SwapHandle()
    try:
        for i, layer in enumerate(layers):
            for site, names in _EXPERT_SITES.items():
                a = nodes[f"G{i}_{site}{tail}"]
                n_out, k = int(a["N"]), int(a["K"])
                lins = [layer.get_submodule(n) for n in names]
                rows = [int(lin.out_features) for lin in lins]
                if sum(rows) != n_out or any(int(lin.in_features) != k for lin in lins):
                    raise ValueError(f"G{i}_{site}: node is {n_out}x{k}, the layer's {names} are {rows}x{lins[0].in_features}")
                if any(lin.bias is not None for lin in lins):
                    raise NotImplementedError(f"G{i}_{site}: the residual plugin has no bias; a biased projection cannot be served")
                codes = _codes(a[f"weight_i{bits}"], n_out, k, bits)
                w_scale = _f32(a["weight_scale"])
                t = _transform(a, k)
                # The dense-rotation branch of the INT4 kernel quantizes unclipped.
                clip = 1.0 if t.get("stage_bf16") else float(a.get("act_clip_ratio", 1.0))
                start = 0
                for name, n in zip(names, rows):
                    parent_name, _, leaf = name.rpartition(".")
                    swap_module(swaps, layer.get_submodule(parent_name), leaf, FakeQuantLinear(
                        codes[start : start + n], w_scale[start : start + n], None, a_qmax=qmax, a_clip=clip, **t,
                    ))
                    start += n
    except Exception:
        swaps.remove()
        raise
    logger.info("expert fake-quant (%s) installed: %d layers, %d projections", state.scheme, len(layers), len(swaps))
    return swaps
