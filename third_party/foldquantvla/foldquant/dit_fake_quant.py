# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""PyTorch fake-quant of a GR00T DiT, read off the plugin graph its emitter builds.

The fake-quant does not re-derive any quantization number. It runs the DiT
emitter (:mod:`.dit_int4` / :mod:`.dit_int8`) in memory, with the recorded
GPTQ codes of a :class:`foldquant.quant_state.ModuleQuantState` replayed, and
reads every plugin node's attributes back: the packed weight codes, the
per-row weight scales, the channel permutation, the rotation exactly as
shipped (bf16 on the dense arm's per-site matrices, fp32 on the encoder's),
the SmoothQuant vector a butterfly site carries, and the biases. Each
quantized ``nn.Linear`` of the DiT is then replaced by a
:class:`FakeQuantLinear` that computes what the kernel computes::

    x -> (/ s_pre) -> permute + block-rotate -> per-token quantize -> integer GEMM -> * s_tok * s_w + b

so the PyTorch model and the TensorRT engine start from the same bytes.
Attention, the norms, the timestep encoder and the output head stay in the
module's own precision, as they do in the engine.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from . import schemes
from .quant_state import ModuleQuantState, replay_sites

logger = logging.getLogger(__name__)

__all__ = ["DitFakeQuantHandle", "FakeQuantLinear", "build_dit_graph", "install_dit_fake_quant"]

from .fake_quant_linear import FakeQuantLinear, SwapHandle, swap_module  # noqa: E402


# Reading plugin node attributes


def _attrs(node: Any) -> Dict[str, Any]:
    import onnx

    out: Dict[str, Any] = {}
    for a in node.attribute:
        if a.type == onnx.AttributeProto.STRING:
            out[a.name] = a.s
        elif a.type == onnx.AttributeProto.INT:
            out[a.name] = int(a.i)
        elif a.type == onnx.AttributeProto.FLOAT:
            out[a.name] = float(a.f)
    return out


def _f32(b: bytes) -> torch.Tensor:
    return torch.from_numpy(np.frombuffer(b, dtype=np.float32).copy())


def _i32(b: bytes) -> torch.Tensor:
    return torch.from_numpy(np.frombuffer(b, dtype=np.int32).copy())


def _bf16(b: bytes) -> torch.Tensor:
    return torch.from_numpy(np.frombuffer(b, dtype=np.int16).copy()).view(torch.bfloat16).float()


def _codes(b: bytes, rows: int, cols: int, bits: int) -> torch.Tensor:
    if bits == 8:
        return torch.from_numpy(np.frombuffer(b, dtype=np.int8).copy()).reshape(rows, cols)
    packed = torch.from_numpy(np.frombuffer(b, dtype=np.uint8).copy()).to(torch.int16)
    half = (cols + 1) // 2
    packed = packed.reshape(rows, half)
    lo, hi = packed & 0xF, (packed >> 4) & 0xF
    both = torch.stack([lo, hi], dim=-1).reshape(rows, 2 * half)[:, :cols]
    return torch.where(both >= 8, both - 16, both).to(torch.int8)


def _rotation(a: Dict[str, Any], perm_key: str, rot_key: str, k: int, fwht: bool, dense_dtype: str) -> Dict[str, Any]:
    """The activation transform a node declares for one input."""
    if fwht:
        from .rotation import hadamard_blocks

        bs = int(a["rot_block_size"])
        if k % bs:
            raise ValueError(f"butterfly block {bs} does not divide K={k}; the fake-quant does not pad")
        perm, rot = hadamard_blocks(k, bs)
        return {"perm": perm, "rot": rot, "rot_bs": bs, "stage_bf16": False}
    blob = a.get(rot_key, b"")
    if not blob:
        return {}
    rot = _bf16(blob) if dense_dtype == "bf16" else _f32(blob)
    bs = int(a["block_size"])
    return {
        "perm": _i32(a[perm_key]),
        "rot": rot.reshape(k // bs, bs, bs),
        "rot_bs": bs,
        "stage_bf16": dense_dtype == "bf16",
    }


# Building the graph the engine is built from


def build_dit_graph(module: nn.Module, state: ModuleQuantState) -> Any:
    """The DiT plugin ``GraphProto`` for *state*, built in memory with its GPTQ codes replayed."""
    from .dit_common import DiTWeights, resolve_attend_n
    from .export import _act_fold_knobs

    scheme = state.scheme
    params = dict(state.config.get("params") or {})
    from .quant_state import pack_scope

    w = DiTWeights(module, resolve_attend_n(module))
    replay = replay_sites(state)
    with pack_scope("dit", replay=replay):
        if scheme == schemes.W8A8:
            from .dit_int8 import _build_v2_dynamic_graph

            graph = _build_v2_dynamic_graph(w)
        else:
            knobs = _act_fold_knobs(scheme, params)
            sq = state.tensor_group("sq")
            if scheme == schemes.W8A8_SH:
                from .dit_int8 import _build_v2_dynamic_graph

                graph = _build_v2_dynamic_graph(w, sq_scales=sq, sq_fold_order=knobs["fold_order"], fwht=True)
            else:
                from .dit_int4 import _DEFAULT_BLOCK_SIZE, _build_w4a4_graph

                graph = _build_w4a4_graph(
                    w, sq, _DEFAULT_BLOCK_SIZE, 16, sq_fold_order=knobs["fold_order"], fwht=knobs["fwht"],
                    gptq=replay if scheme == schemes.W4A4_SHG else None,
                )
    replay.assert_consumed()
    return graph


# Installing


#: Kept under its first name for callers of the DiT reader.
DitFakeQuantHandle = SwapHandle


def _split_rows(codes: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor, sizes: List[int]) -> list:
    out, start = [], 0
    for n in sizes:
        out.append((codes[start : start + n], scale[start : start + n], bias[start : start + n]))
        start += n
    if start != codes.shape[0]:
        raise ValueError(f"node rows {codes.shape[0]} do not match the module's {sizes}")
    return out


def install_dit_fake_quant(module: nn.Module, state: ModuleQuantState) -> DitFakeQuantHandle:
    """Replace the DiT's quantized Linears with :class:`FakeQuantLinear` s; returns an undo handle."""
    scheme = state.scheme
    int4 = scheme in (schemes.W4A4_SHG, schemes.W4A4_SH, schemes.W4A4_SR)
    if not int4 and scheme not in (schemes.W8A8, schemes.W8A8_SH):
        raise NotImplementedError(f"DiT fake-quant has no reader for scheme {scheme!r}")
    bits = 4 if int4 else 8
    qmax = 7.0 if int4 else 127.0
    fwht = schemes.uses_fwht(scheme) if scheme != schemes.W8A8 else False
    suffix = {"self": "_selfattn_int4" if int4 else "_selfattn_full",
              "cross": "_crossattn_int4" if int4 else "_crossattn_full",
              "ffn": "_ffn_int4" if int4 else "_ffn"}
    graph = build_dit_graph(module, state)
    nodes = {n.name: _attrs(n) for n in graph.node if n.domain == "trt.plugins"}
    enc = nodes.get("encoder_prequant_int4" if int4 else "encoder_prequant")
    if enc is None:
        raise ValueError("the DiT graph has no encoder pre-quant node")
    blocks = module.transformer_blocks
    swaps = SwapHandle()

    def site(a: Dict[str, Any], wkey: str, skey: str, bkey: str, rows: int, cols: int) -> Tuple[Any, Any, Any]:
        return _codes(a[wkey], rows, cols, bits), _f32(a[skey]), _f32(a[bkey])

    def transform(a: Dict[str, Any], perm_key: str, rot_key: str, pre_key: str, k: int) -> Dict[str, Any]:
        if scheme == schemes.W8A8:
            return {}
        t = _rotation(a, perm_key, rot_key, k, fwht, "bf16")
        if fwht and pre_key in a:
            t["s_pre"] = _f32(a[pre_key])
        return t

    enc_k = int(enc["K_enc"])
    if scheme == schemes.W8A8:
        enc_t: Dict[str, Any] = {}
    elif fwht:
        enc_t = _rotation(enc, "perm_enc", "rotation_enc", enc_k, True, "fp32")
        if "act_scale_pre_enc" in enc:
            enc_t["s_pre"] = _f32(enc["act_scale_pre_enc"])
    else:
        enc_t = _rotation(enc, "perm_enc", "rotation_enc", enc_k, False, "fp32")
        enc_t["stage_bf16"] = False

    for idx, blk in enumerate(blocks):
        b = f"block{idx}"
        attn = blk.attn1
        if int4:
            ad = nodes[f"{b}_adaln_int4"]
            lin = blk.norm1.linear
            in_d, out_d = int(ad["in_dim"]), int(ad["out_dim"])
            codes = _codes(ad["weight_i4"], out_d, in_d + (in_d % 2), 4)[:, :in_d]
            swap_module(swaps, blk.norm1, "linear", FakeQuantLinear(
                codes, _bf16(ad["weight_scale"]), _bf16(ad["bias"]), a_qmax=None if int(ad.get("act_bits", 16)) >= 16 else 7.0,
            ))

        k = int(attn.to_q.weight.shape[1])
        if idx % 2 == 1:  # self-attention
            a = nodes[b + suffix["self"]]
            sizes = [attn.to_q.out_features, attn.to_k.out_features, attn.to_v.out_features]
            c, s, bias = site(a, f"weight_qkv_i{bits}", "weight_qkv_scale", "bias_qkv", sum(sizes), k)
            t = transform(a, "perm_qkv", "rotation_qkv", "act_scale_pre_in", k)
            for name, (ci, si, bi) in zip(("to_q", "to_k", "to_v"), _split_rows(c, s, bias, sizes)):
                swap_module(swaps, attn, name, FakeQuantLinear(ci, si, bi, a_qmax=qmax, **t))
        else:  # cross-attention: Q reads x, K/V read the encoder under the shared encoder transform
            a = nodes[b + suffix["cross"]]
            c, s, bias = site(a, f"weight_q_i{bits}", "weight_q_scale", "bias_q", attn.to_q.out_features, k)
            swap_module(swaps, attn, "to_q", FakeQuantLinear(c, s, bias, a_qmax=qmax, **transform(a, "perm_q", "rotation_q", "act_scale_pre_in", k)))
            sizes = [attn.to_k.out_features, attn.to_v.out_features]
            c, s, bias = site(a, f"weight_kv_i{bits}", "weight_kv_scale", "bias_kv", sum(sizes), enc_k)
            for name, (ci, si, bi) in zip(("to_k", "to_v"), _split_rows(c, s, bias, sizes)):
                swap_module(swaps, attn, name, FakeQuantLinear(ci, si, bi, a_qmax=qmax, **enc_t))
        o = attn.to_out[0]
        c, s, bias = site(a, f"weight_o_i{bits}", "weight_o_scale", "bias_o", o.out_features, o.in_features)
        swap_module(swaps, attn.to_out, "0", FakeQuantLinear(c, s, bias, a_qmax=qmax, **transform(a, "perm_o", "rotation_o", "act_scale_pre_o", o.in_features)))

        f = nodes[b + suffix["ffn"]]
        p0, p2 = blk.ff.net[0].proj, blk.ff.net[2]
        c, s, bias = site(f, f"weight_proj0_i{bits}", "weight_proj0_scale", "bias_proj0", p0.out_features, p0.in_features)
        swap_module(swaps, blk.ff.net[0], "proj", FakeQuantLinear(c, s, bias, a_qmax=qmax, **transform(f, "perm0", "rotation0", "act_scale_pre0", p0.in_features)))
        c, s, bias = site(f, f"weight_proj2_i{bits}", "weight_proj2_scale", "bias_proj2", p2.out_features, p2.in_features)
        swap_module(swaps, blk.ff.net, "2", FakeQuantLinear(c, s, bias, a_qmax=qmax, **transform(f, "perm2", "rotation2", "act_scale_pre2", p2.in_features)))

    logger.info("DiT fake-quant (%s) installed: %d projections replaced", scheme, len(swaps))
    return swaps
