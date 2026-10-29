# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""Dynamic per-row INT8/INT4 (width-selectable) plugin ONNX construction for the GR00T N1.6 Qwen3 LLM.

Builds an ONNX graph that replaces every Qwen3 Linear layer of the truncated
16-layer LLM backbone with two custom TensorRT IPluginV3 plugins (namespace
``gr00t::v1``), with INT8 weights baked into the graph as plugin attributes.

Per decoder layer (x16):

  1. ``FusedRmsNormLinearInt8`` (merged Q+K+V GEMM) - replaces
     ``input_layernorm`` + ``q_proj`` + ``k_proj`` + ``v_proj``.
  2. ONNX ``Split`` -> Q, K, V.
  3. ONNX ``Reshape`` + ``Transpose`` -> ``(B, H, S, D)`` / ``(B, H_kv, S, D)``.
  4. ONNX ``q_norm`` / ``k_norm`` (RMSNorm on ``head_dim``, BF16).
  5. ONNX RoPE (cos/sin baked as initializers, sliced to current S).
  6. ONNX ``repeat_kv`` (Unsqueeze + Tile + Reshape for GQA).
  7. ONNX SDPA (causal) - Myelin fuses to a single flash-attention kernel.
  8. ONNX ``Transpose`` + ``Reshape`` -> attention output.
  9. ``PerRowInt8LinearResidual`` (o_proj + residual = pre-attn x).
  10. ``FusedRmsNormLinearInt8`` (merged gate+up GEMM) - replaces
      ``post_attention_layernorm`` + ``gate_proj`` + ``up_proj``.
  11. ONNX ``Split`` -> gate, up.
  12. ONNX ``Sigmoid`` + ``Mul`` + ``Mul`` -> ``silu(gate) * up``.
  13. ``PerRowInt8LinearResidual`` (down_proj + residual = post-attn x).

Final RMSNorm (``qwen3.norm``) in BF16 over the hidden dim, emitted only when the
tower carries a real final norm. A tower whose norm is an identity bypass (N1.7,
whose action head consumes the pre-norm residual stream) ends one node earlier.

Activation amax is computed per row inside each plugin (no static activation
scale). The engine is dynamic-S: RoPE tables and the causal mask are baked at
``max_seq_len`` and sliced at runtime (see :func:`build_llm_plugin_onnx`).

Passing *sq_scales* (optionally with *rot_bs*) selects the
``w8a8_{s,sr}`` variants: each layer folds through
``llm_rotation_sq.apply_sq_fold``/``apply_rot_fold`` before
``quant_weight_per_row`` bakes it, and the plugin nodes carry the matching
``rot_block_size`` attribute.

Graph I/O names follow the upstream deployment exports so the engine is a
drop-in for the float LLM engine in each model's own TensorRT glue: plain
Qwen3 emits ``inputs_embeds``/``attention_mask`` -> ``hidden_states``; Qwen3-VL
(``n1d7_mode``) adds ``position_ids``/``visual_pos_masks``/``deepstack_i`` and
emits ``embeddings``; the Gemma prefix graph emits ``kv_stack``.

This module performs ONNX construction only. It does not import ``tensorrt``,
load any ``.so``, or run TensorRT.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional

import numpy as np
import onnx
import onnx.helper as oh

from foldquant.llm_rotation_sq import apply_rot_fold, apply_sq_fold
from foldquant.weights import (
    quant_weight_per_row,
    to_bytes_i8,
)

__all__ = ["build_llm_plugin_onnx", "resolve_qwen3_decoder"]


# Byte / initializer helpers


def _bytes_f32(arr: Any) -> bytes:
    """Flatten to contiguous FP32 bytes (PluginField ``kFLOAT32`` payload)."""
    data: bytes = np.asarray(arr).flatten().astype(np.float32).tobytes()
    return data


def _bytes_bf16(t: Any) -> bytes:
    """Flatten a torch tensor to raw BF16 bytes (PluginField payload)."""
    arr = t.to(_torch().bfloat16).contiguous().view(_torch().int16).cpu().numpy().astype(np.uint16)
    data: bytes = arr.tobytes()
    return data


def _bf16_initializer(name: str, t: Any) -> "onnx.TensorProto":
    """Build a BFLOAT16 ONNX initializer from a torch tensor."""
    torch = _torch()
    flat = t.to(torch.bfloat16).contiguous().view(torch.int16).cpu().numpy().astype(np.uint16)
    proto = onnx.TensorProto()
    proto.name = name
    proto.data_type = onnx.TensorProto.BFLOAT16
    for d in t.shape:
        proto.dims.append(int(d))
    proto.raw_data = flat.tobytes()
    return proto


def _i64_init(name: str, arr: Any) -> "onnx.TensorProto":
    """Build an INT64 ONNX initializer from an array-like."""
    a = np.asarray(arr, dtype=np.int64)
    return oh.make_tensor(name, onnx.TensorProto.INT64, list(a.shape), a.tobytes(), raw=True)


def _torch() -> Any:
    """Lazy torch import (kept out of module top-level)."""
    import torch

    return torch


# Quantization width (W8A8 vs W4A4)


# The graph topology is identical at both widths: ``bits`` selects only the
# plugin pair, the weight-attribute name, and the weight packer. INT4 weights
# are GPTQ-rounded (round-to-nearest measured chan_rel_err 0.4199 vs GPTQ's
# 0.1586 on the N1.6 LLM), which changes ONLY the rounding: GPTQ keeps the same
# per-output-row scale RTN would pick, so the packed layout and the s4 epilogue
# are untouched.

# Must stay in lockstep with rotation._QMAX_I4; both feed pack_int4_nibbles.
_QMAX_I4 = 7.0


class _WidthSpec(NamedTuple):
    rms_op: str  # fused RMSNorm+quant+GEMM plugin (qkv / gateup sites)
    res_op: str  # quant+GEMM+residual plugin (o / down sites)
    weight_key: str  # plugin weight attribute name
    pack: Any  # (weight, gptq_prep|None) -> (weight_bytes, scale_bytes)
    needs_gptq: bool
    act_bits: int  # activation width the plugin pair quantizes to


def _pack_int8(weight: Any, prep: "dict | None", row_clip: Any = None) -> "tuple[bytes, bytes]":
    """Per-output-row symmetric INT8 RTN: ``(N, K)`` int8 bytes + ``(N,)`` scales."""
    if prep is not None:
        raise ValueError("GPTQ prep supplied to the INT8 packer (width routing bug).")
    if row_clip is not None:
        raise ValueError("learned weight clips are an INT4 (GPTQ) weight knob; the INT8 packer takes none.")
    w_i8, scale = quant_weight_per_row(weight)
    return to_bytes_i8(w_i8), _bytes_f32(scale)


def _pack_int4(weight: Any, prep: "dict | None", row_clip: Any = None) -> "tuple[bytes, bytes]":
    """Per-output-row symmetric INT4 via GPTQ: ``(N, K/2)`` nibbles + ``(N,)`` scales.

    The nibble packing is :func:`rotation.pack_int4_nibbles`, the one
    place the byte order lives, shared with the DiT/expert packers, because a
    flipped order builds, loads and runs, and produces noise rather than an
    error.
    """
    from foldquant.llm_gptq import (
        MissingHessianError,
        gptq_quant_codes,
    )
    from foldquant.rotation import pack_int4_nibbles

    if prep is None:
        raise MissingHessianError(
            "INT4 LLM weights are GPTQ-rounded and need this site's calibration Hessian. "
            "Build through the per-row scheme adapter, which computes them."
        )
    codes, scale = gptq_quant_codes(weight, prep, qmax=_QMAX_I4, row_clip=row_clip)
    return pack_int4_nibbles(codes).tobytes(), _bytes_f32(scale.cpu().numpy())


# An unknown width raises KeyError in :func:`_emit_layer` rather than silently
# emitting an INT8 graph. INT4 has no static-scale counterpart: at 4 bits the
# scheme is dynamic-per-row only.
_LLM_WIDTH: "dict[int, _WidthSpec]" = {
    8: _WidthSpec("FusedRmsNormLinearInt8", "PerRowInt8LinearResidual", "weight_i8", _pack_int8, False, 8),
    4: _WidthSpec("FusedRmsNormLinearInt4", "PerRowInt4LinearResidual", "weight_i4", _pack_int4, True, 4),
}
# W4A8: the INT8 plugin pair fed nibble-packed INT4 weights (``weight_i4``);
# the plugin unpacks them to INT8 at load, so the GEMM is INT8xINT8 over the
# INT4 weight grid with the INT4 per-row scale, and the activation is the
# INT8 plugins' per-token dynamic quantizer. Same packer as W4A4 (GPTQ).
_LLM_W4A8 = _WidthSpec("FusedRmsNormLinearInt8", "PerRowInt8LinearResidual", "weight_i4", _pack_int4, True, 8)


# RoPE / causal mask precomputation


def _compute_qwen3_rope(max_s: int, head_dim: int, theta: float = 1000000.0) -> Any:
    """Bake Qwen3 RoPE (cos, sin) tensors of shape ``(1, 1, max_s, head_dim)`` BF16.

    The engine is dynamic-S; the baked tensors are at ``max_s`` and sliced down
    to the current seq_len at runtime by ``_emit_dynamic_slice_helpers``.
    """
    torch = _torch()
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    pos = torch.arange(max_s, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos().to(torch.bfloat16).view(1, 1, max_s, head_dim)
    sin = emb.sin().to(torch.bfloat16).view(1, 1, max_s, head_dim)
    return cos.contiguous(), sin.contiguous()


#: Name of the runtime-computed repeat_kv target shape ``(B, H, -1, D)``, emitted once
#: by :func:`_emit_kv_shape_dyn` and shared by every layer. Batch is read from
#: ``Shape(inputs_embeds)`` because a Reshape target may hold only one ``-1``, which
#: the seq dim needs so the engine stays dynamic-S.
KV_SHAPE_DYN = "_flat_kv_shape_dyn"


def _emit_kv_shape_dyn(nodes: list, inits: list, *, num_heads: int, head_dim: int, source: str) -> None:
    """Emit ``KV_SHAPE_DYN = Concat(Shape(source)[0], [H], [-1], [D])``."""
    nodes.append(oh.make_node("Shape", [source], ["_bh_ie_shape"]))
    inits.append(_i64_init("_bh_batch_idx", [0]))
    nodes.append(oh.make_node("Gather", ["_bh_ie_shape", "_bh_batch_idx"], ["_bh_batch_1d"], axis=0))
    inits.append(_i64_init("_bh_heads_1d", [num_heads]))
    inits.append(_i64_init("_bh_neg1_1d", [-1]))
    inits.append(_i64_init("_bh_head_dim_1d", [head_dim]))
    nodes.append(
        oh.make_node(
            "Concat",
            ["_bh_batch_1d", "_bh_heads_1d", "_bh_neg1_1d", "_bh_head_dim_1d"],
            [KV_SHAPE_DYN],
            axis=0,
        )
    )


def _causal_mask(max_s: int) -> Any:
    """Additive causal mask ``(1, 1, max_s, max_s)`` BF16 for the dynamic-S engine.

    Sliced to ``(1, 1, current_S, current_S)`` at runtime by
    ``_emit_dynamic_slice_helpers``.
    """
    torch = _torch()
    dtype = torch.bfloat16
    mask_val = torch.finfo(dtype).min * 0.5
    m = torch.triu(torch.full((max_s, max_s), mask_val, dtype=dtype), diagonal=1)
    return m.view(1, 1, max_s, max_s).contiguous()


def _emit_dynamic_slice_helpers(nodes: list, inits: list) -> None:
    """Slice baked rope/causal_mask down to current_S at runtime.

    Inputs (must exist as initializers before this is called):
      ``rope_cos_full`` BF16 ``(1, 1, max_s, D)``
      ``rope_sin_full`` BF16 ``(1, 1, max_s, D)``
      ``causal_mask_full`` BF16 ``(1, 1, max_s, max_s)``
    Outputs (named tensors usable by the layer body):
      ``rope_cos`` / ``rope_sin`` BF16 ``(1, 1, current_S, D)``
      ``causal_mask`` BF16 ``(1, 1, current_S, current_S)``

    ``current_S`` is derived from ``Shape(inputs_embeds)[1]``.
    """
    # current_S = Shape(inputs_embeds)[1] -> 1D int64 [1] tensor
    nodes.append(oh.make_node("Shape", ["inputs_embeds"], ["_ie_shape"]))
    inits.append(_i64_init("_slice_seq_idx", [1]))
    nodes.append(oh.make_node("Gather", ["_ie_shape", "_slice_seq_idx"], ["_current_S_scalar"], axis=0))
    # Gather with 1D idx [1] returns shape [1] - use _current_S_scalar directly as slice ends.

    # Slice rope cos/sin: axes=[2], starts=[0], ends=[current_S]
    inits.append(_i64_init("_slice_zero_1d", [0]))
    inits.append(_i64_init("_slice_axes_2", [2]))
    nodes.append(
        oh.make_node(
            "Slice",
            ["rope_cos_full", "_slice_zero_1d", "_current_S_scalar", "_slice_axes_2"],
            ["rope_cos"],
        )
    )
    nodes.append(
        oh.make_node(
            "Slice",
            ["rope_sin_full", "_slice_zero_1d", "_current_S_scalar", "_slice_axes_2"],
            ["rope_sin"],
        )
    )

    # Slice causal mask: axes=[2,3], starts=[0,0], ends=[current_S, current_S]
    inits.append(_i64_init("_slice_zero_2d", [0, 0]))
    inits.append(_i64_init("_slice_axes_2_3", [2, 3]))
    nodes.append(oh.make_node("Concat", ["_current_S_scalar", "_current_S_scalar"], ["_mask_ends"], axis=0))
    nodes.append(
        oh.make_node(
            "Slice",
            ["causal_mask_full", "_slice_zero_2d", "_mask_ends", "_slice_axes_2_3"],
            ["causal_mask"],
        )
    )


def resolve_qwen3_decoder(qwen3_model: Any) -> Any:
    """Unwrap the causal-LM wrapper (``Qwen3ForCausalLM``) down to the decoder exposing ``.layers``."""
    if not hasattr(qwen3_model, "layers") and hasattr(qwen3_model, "model"):
        qwen3_model = qwen3_model.model
    if not hasattr(qwen3_model, "layers"):
        raise AttributeError(
            f"resolve_qwen3_decoder: expected a Qwen3 decoder exposing '.layers', got "
            f"{type(qwen3_model).__name__} with no '.layers' and no '.model' to unwrap."
        )
    return qwen3_model


# Per-layer ONNX emitter


def _emit_gelu_tanh(nodes: list, inits: list, name: str, x: str, out: str) -> None:
    """``gelu_pytorch_tanh``: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))), opset-17-safe."""
    import torch as _t

    inits.append(_bf16_initializer(f"{name}_c0", _t.tensor(0.7978845608028654, dtype=_t.bfloat16)))  # sqrt(2/pi)
    inits.append(_bf16_initializer(f"{name}_c1", _t.tensor(0.044715, dtype=_t.bfloat16)))
    inits.append(_bf16_initializer(f"{name}_half", _t.tensor(0.5, dtype=_t.bfloat16)))
    inits.append(_bf16_initializer(f"{name}_one", _t.tensor(1.0, dtype=_t.bfloat16)))
    nodes.append(oh.make_node("Mul", [x, x], [f"{name}_x2"]))
    nodes.append(oh.make_node("Mul", [f"{name}_x2", x], [f"{name}_x3"]))
    nodes.append(oh.make_node("Mul", [f"{name}_x3", f"{name}_c1"], [f"{name}_x3c"]))
    nodes.append(oh.make_node("Add", [x, f"{name}_x3c"], [f"{name}_inner"]))
    nodes.append(oh.make_node("Mul", [f"{name}_inner", f"{name}_c0"], [f"{name}_scaled"]))
    nodes.append(oh.make_node("Tanh", [f"{name}_scaled"], [f"{name}_tanh"]))
    nodes.append(oh.make_node("Add", [f"{name}_tanh", f"{name}_one"], [f"{name}_t1"]))
    nodes.append(oh.make_node("Mul", [x, f"{name}_t1"], [f"{name}_xt"]))
    nodes.append(oh.make_node("Mul", [f"{name}_xt", f"{name}_half"], [out]))


def _emit_layer(
    layer_state: dict,
    prefix: str,
    cur_x_name: str,
    eps: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    hidden_size: int,
    intermediate_size: int,
    cos_init_name: str,
    sin_init_name: str,
    causal_mask_init_name: str,
    nodes: list,
    inits: list,
    rot_bs: int = 0,
    gemma_mode: bool = False,
    bits: int = 8,
    gptq: "Any | None" = None,
    act_clip_ratio: Any = 1.0,
    site_bits: "Dict[str, int] | None" = None,
    act_bits: "int | None" = None,
    weight_clip: "Dict[str, Any] | None" = None,
) -> str:
    """Emit ONNX nodes for one Qwen2/Qwen3 decoder layer. Returns the output tensor name.

    Qwen2 differences are read off ``layer_state`` structurally: q/k/v biases
    (added after the INT8 GEMM), and no per-head q/k RMSNorm.

    Dynamic-S body: no hardcoded seq_len in any constant. The caller passes
    sliced rope/causal_mask names (sized to current_S at runtime).
    """
    b = prefix
    # Width per SITE: ``site_bits`` (qkv / o / gateup / down) overrides the layer
    # width, so a mixed layer can keep its FWHT residual sites (o, down) at INT8
    # inside an INT4 stack. Both plugin pairs exist; only the op name, weight
    # attribute and packer differ per site (measured: o8+down8 recovers 47% of
    # the held-out action error of full W4A4 on N1.6).
    _site_bits = {k: int(v) for k, v in (site_bits or {}).items()}
    unknown = sorted(set(_site_bits) - {"qkv", "o", "gateup", "down"})
    if unknown:
        raise KeyError(f"layer {prefix}: unknown site(s) in site_bits: {unknown}")

    _act_bits = int(bits) if act_bits is None else int(act_bits)

    def _width(site: str) -> _WidthSpec:
        w = _site_bits.get(site, int(bits))
        return _LLM_W4A8 if (w == 4 and _act_bits == 8) else _LLM_WIDTH[w]

    def _clip_kw(site: str) -> Dict[str, Any]:
        # INT4-activation-only attribute: the INT8 plugins (INT8 and W4A8 modes)
        # do not declare it, and 1.0 is the plugins' default, so emit nothing in
        # either case so INT8 graphs and untuned INT4 graphs stay byte-identical.
        # A dict carries learned per-(layer, site) clips keyed ``L{i}_{site}``.
        ratio = (
            float(act_clip_ratio.get(f"{prefix}_{site}", 1.0))
            if isinstance(act_clip_ratio, dict)
            else float(act_clip_ratio)
        )
        return {"act_clip_ratio": ratio} if _width(site).act_bits == 4 and ratio != 1.0 else {}

    _SITE_WEIGHTS = {
        "qkv": ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"),
        "o": ("self_attn.o_proj.weight",),
        "gateup": ("mlp.gate_proj.weight", "mlp.up_proj.weight"),
        "down": ("mlp.down_proj.weight",),
    }

    def _row_clip(site: str) -> Any:
        """Learned per-row weight clips for *site*, concatenated in the merged-weight row order."""
        if not weight_clip:
            return None
        keys = [f"{prefix}_{w}" for w in _SITE_WEIGHTS[site]]
        if not all(k in weight_clip for k in keys):
            missing = [k for k in keys if k not in weight_clip]
            raise KeyError(f"layer {prefix}: learned weight clips missing {missing}")
        return _torch().cat([_torch().as_tensor(weight_clip[k]).float().reshape(-1) for k in keys])

    if any(_width(s).needs_gptq for s in ("qkv", "o", "gateup", "down")) and gptq is None:
        from foldquant.llm_gptq import MissingHessianError

        raise MissingHessianError(f"layer {prefix}: an INT4 site requires GPTQ site factors and none were supplied.")

    def _weight_kw(weight: Any, site: str) -> "dict[str, Any]":
        width = _width(site)
        prep = gptq.take(f"{prefix}_{site}") if (width.needs_gptq and gptq is not None) else None
        from foldquant.quant_state import site_scope

        with site_scope(f"{prefix}_{site}"):
            wb, sb = width.pack(weight, prep, _row_clip(site) if width.needs_gptq else None)
        return {width.weight_key: wb, "weight_scale": sb}

    k_dim = hidden_size
    ff = intermediate_size
    h = num_heads
    hkv = num_kv_heads
    d = head_dim
    kv_groups = h // hkv
    q_dim = h * d
    kv_dim = hkv * d
    qkv_out = q_dim + 2 * kv_dim

    # Plugin 1: input_layernorm + merged QKV
    torch = _torch()
    w_q = layer_state["self_attn.q_proj.weight"]
    w_k = layer_state["self_attn.k_proj.weight"]
    w_v = layer_state["self_attn.v_proj.weight"]
    w_qkv = torch.cat([w_q, w_k, w_v], dim=0)  # (q_dim + 2*kv_dim, K)
    gamma_pre = layer_state["input_layernorm.weight"]

    # Qwen2 carries q/k/v biases; Qwen3 does not. A bias is added
    # after the INT8 GEMM as a plain BF16 Add; it lives outside both the
    # SmoothQuant fold (which rescales input channels) and the rotation fold
    # (which transforms the input space), so neither touches it.
    has_qkv_bias = "self_attn.q_proj.bias" in layer_state

    nodes.append(
        oh.make_node(
            _width("qkv").rms_op,
            [cur_x_name],
            [f"{b}_qkv"],
            name=f"{b}_rmsnorm_qkv",
            domain="trt.plugins",
            plugin_namespace="gr00t::v1",
            plugin_version="1",
            gamma=_bytes_bf16(gamma_pre),
            N=int(qkv_out),
            K=int(k_dim),
            eps=float(eps),
            rot_block_size=int(rot_bs),
            **_clip_kw("qkv"),
            **_weight_kw(w_qkv, "qkv"),
        )
    )

    qkv_name = f"{b}_qkv"
    if has_qkv_bias:
        bias_qkv = torch.cat(
            [
                layer_state["self_attn.q_proj.bias"],
                layer_state["self_attn.k_proj.bias"],
                layer_state["self_attn.v_proj.bias"],
            ],
            dim=0,
        )
        inits.append(_bf16_initializer(f"{b}_qkv_bias", bias_qkv))
        nodes.append(oh.make_node("Add", [qkv_name, f"{b}_qkv_bias"], [f"{b}_qkv_biased"]))
        qkv_name = f"{b}_qkv_biased"

    # Split (q_dim, kv_dim, kv_dim) on last axis.
    inits.append(_i64_init(f"{b}_qkv_split", [q_dim, kv_dim, kv_dim]))
    nodes.append(
        oh.make_node(
            "Split",
            [qkv_name, f"{b}_qkv_split"],
            [f"{b}_q_flat", f"{b}_k_flat", f"{b}_v_flat"],
            axis=-1,
        )
    )

    # Reshape Q (B,S,q_dim) -> (B,S,H,D), transpose to (B,H,S,D).
    inits.append(_i64_init(f"{b}_q_shape4", [0, 0, h, d]))
    inits.append(_i64_init(f"{b}_k_shape4", [0, 0, hkv, d]))
    nodes.append(oh.make_node("Reshape", [f"{b}_q_flat", f"{b}_q_shape4"], [f"{b}_q_4"], allowzero=0))
    nodes.append(oh.make_node("Reshape", [f"{b}_k_flat", f"{b}_k_shape4"], [f"{b}_k_4"], allowzero=0))
    nodes.append(oh.make_node("Reshape", [f"{b}_v_flat", f"{b}_k_shape4"], [f"{b}_v_4"], allowzero=0))
    nodes.append(oh.make_node("Transpose", [f"{b}_q_4"], [f"{b}_q_t"], perm=[0, 2, 1, 3]))
    nodes.append(oh.make_node("Transpose", [f"{b}_k_4"], [f"{b}_k_t"], perm=[0, 2, 1, 3]))
    nodes.append(oh.make_node("Transpose", [f"{b}_v_4"], [f"{b}_v_t"], perm=[0, 2, 1, 3]))

    # q_norm / k_norm: head-dim RMSNorm in BF16 ONNX ops
    # Qwen3 only. Qwen2 has no per-head norms; RoPE reads the raw heads.
    has_qk_norm = "self_attn.q_norm.weight" in layer_state
    inits.append(_bf16_initializer(f"{b}_eps_bf16", torch.tensor(eps, dtype=torch.bfloat16)))
    if has_qk_norm:
        inits.append(_bf16_initializer(f"{b}_qnorm_gamma", layer_state["self_attn.q_norm.weight"]))
        inits.append(_bf16_initializer(f"{b}_knorm_gamma", layer_state["self_attn.k_norm.weight"]))

    def _emit_rmsnorm_headdim(in_name: str, gamma_name: str, out_name: str) -> None:
        inits.append(_bf16_initializer(f"{out_name}_two", torch.tensor(2.0, dtype=torch.bfloat16)))
        nodes.append(oh.make_node("Pow", [in_name, f"{out_name}_two"], [f"{out_name}_sq"]))
        nodes.append(oh.make_node("ReduceMean", [f"{out_name}_sq"], [f"{out_name}_mean"], axes=[-1], keepdims=1))
        nodes.append(oh.make_node("Add", [f"{out_name}_mean", f"{b}_eps_bf16"], [f"{out_name}_var"]))
        nodes.append(oh.make_node("Sqrt", [f"{out_name}_var"], [f"{out_name}_std"]))
        nodes.append(oh.make_node("Div", [in_name, f"{out_name}_std"], [f"{out_name}_normed"]))
        nodes.append(oh.make_node("Mul", [f"{out_name}_normed", gamma_name], [out_name]))

    if has_qk_norm:
        _emit_rmsnorm_headdim(f"{b}_q_t", f"{b}_qnorm_gamma", f"{b}_q_n")
        _emit_rmsnorm_headdim(f"{b}_k_t", f"{b}_knorm_gamma", f"{b}_k_n")
        q_for_rope, k_for_rope = f"{b}_q_n", f"{b}_k_n"
    else:
        q_for_rope, k_for_rope = f"{b}_q_t", f"{b}_k_t"

    # RoPE
    half = d // 2
    inits.append(_i64_init(f"{b}_split_half", [half, half]))

    def _emit_rope(qk_in: str, qk_out: str) -> None:
        nodes.append(oh.make_node("Split", [qk_in, f"{b}_split_half"], [f"{qk_in}_a", f"{qk_in}_b"], axis=-1))
        nodes.append(oh.make_node("Neg", [f"{qk_in}_b"], [f"{qk_in}_b_neg"]))
        nodes.append(oh.make_node("Concat", [f"{qk_in}_b_neg", f"{qk_in}_a"], [f"{qk_in}_rot"], axis=-1))
        nodes.append(oh.make_node("Mul", [qk_in, cos_init_name], [f"{qk_in}_cos"]))
        nodes.append(oh.make_node("Mul", [f"{qk_in}_rot", sin_init_name], [f"{qk_in}_sin"]))
        nodes.append(oh.make_node("Add", [f"{qk_in}_cos", f"{qk_in}_sin"], [qk_out]))

    _emit_rope(q_for_rope, f"{b}_q_r")
    _emit_rope(k_for_rope, f"{b}_k_r")

    # repeat_kv (GQA): (B, HKV, S, D) -> (B, H, S, D)
    # Target shape is a runtime tensor (KV_SHAPE_DYN), not a constant: a baked
    # [1, h, -1, d] would pin the graph to batch 1, with every later sample reading
    # sample 0's K/V and no shape error to signal it.
    inits.append(_i64_init(f"{b}_axes_2", [2]))

    def _emit_repeat_kv(in_name: str, out_name: str) -> None:
        nodes.append(oh.make_node("Unsqueeze", [in_name, f"{b}_axes_2"], [f"{out_name}_u"]))
        inits.append(_i64_init(f"{out_name}_reps", [1, 1, kv_groups, 1, 1]))
        nodes.append(oh.make_node("Tile", [f"{out_name}_u", f"{out_name}_reps"], [f"{out_name}_tiled"]))
        nodes.append(oh.make_node("Reshape", [f"{out_name}_tiled", KV_SHAPE_DYN], [out_name], allowzero=1))

    _emit_repeat_kv(f"{b}_k_r", f"{b}_k_full")
    _emit_repeat_kv(f"{b}_v_t", f"{b}_v_full")

    # SDPA (BF16 ONNX, Myelin fuses to a single flash-attention kernel)
    nodes.append(oh.make_node("Transpose", [f"{b}_k_full"], [f"{b}_k_T"], perm=[0, 1, 3, 2]))
    nodes.append(oh.make_node("MatMul", [f"{b}_q_r", f"{b}_k_T"], [f"{b}_qk"]))
    inits.append(_bf16_initializer(f"{b}_attn_scale", torch.tensor(1.0 / math.sqrt(d), dtype=torch.bfloat16)))
    nodes.append(oh.make_node("Mul", [f"{b}_qk", f"{b}_attn_scale"], [f"{b}_qk_s"]))
    nodes.append(oh.make_node("Add", [f"{b}_qk_s", causal_mask_init_name], [f"{b}_qk_m"]))
    nodes.append(oh.make_node("Softmax", [f"{b}_qk_m"], [f"{b}_attn_w"], axis=-1))
    nodes.append(oh.make_node("MatMul", [f"{b}_attn_w", f"{b}_v_full"], [f"{b}_attn_4"]))
    nodes.append(oh.make_node("Transpose", [f"{b}_attn_4"], [f"{b}_attn_perm"], perm=[0, 2, 1, 3]))
    inits.append(_i64_init(f"{b}_attn_flat_shape", [0, 0, q_dim]))
    nodes.append(oh.make_node("Reshape", [f"{b}_attn_perm", f"{b}_attn_flat_shape"], [f"{b}_attn_flat"], allowzero=0))
    attn_flat_name = f"{b}_attn_flat"

    # Plugin 2: o_proj + residual
    w_o = layer_state["self_attn.o_proj.weight"]
    nodes.append(
        oh.make_node(
            _width("o").res_op,
            [attn_flat_name, cur_x_name],
            [f"{b}_post_attn"],
            name=f"{b}_o_proj_res",
            domain="trt.plugins",
            plugin_namespace="gr00t::v1",
            plugin_version="1",
            N=int(k_dim),
            K=int(q_dim),
            rot_block_size=int(rot_bs),
            **_clip_kw("o"),
            **_weight_kw(w_o, "o"),
        )
    )

    # Plugin 3: post_attention_layernorm + merged gate+up GEMM
    w_g = layer_state["mlp.gate_proj.weight"]
    w_u = layer_state["mlp.up_proj.weight"]
    w_gu = torch.cat([w_g, w_u], dim=0)
    gamma_mlp = layer_state["post_attention_layernorm.weight"]
    nodes.append(
        oh.make_node(
            _width("gateup").rms_op,
            [f"{b}_post_attn"],
            [f"{b}_gateup"],
            name=f"{b}_rmsnorm_gateup",
            domain="trt.plugins",
            plugin_namespace="gr00t::v1",
            plugin_version="1",
            gamma=_bytes_bf16(gamma_mlp),
            N=int(2 * ff),
            K=int(k_dim),
            eps=float(eps),
            rot_block_size=int(rot_bs),
            **_clip_kw("gateup"),
            **_weight_kw(w_gu, "gateup"),
        )
    )
    inits.append(_i64_init(f"{b}_gu_split", [ff, ff]))
    nodes.append(oh.make_node("Split", [f"{b}_gateup", f"{b}_gu_split"], [f"{b}_gate", f"{b}_up"], axis=-1))
    if gemma_mode:
        # GemmaMLP: gelu_pytorch_tanh(gate) * up (not silu). Opset 17 has no
        # Gelu op, so it decomposes into basic ops.
        _emit_gelu_tanh(nodes, inits, f"{b}_gelu", f"{b}_gate", f"{b}_gact")
    else:
        nodes.append(oh.make_node("Sigmoid", [f"{b}_gate"], [f"{b}_gate_sig"]))
        nodes.append(oh.make_node("Mul", [f"{b}_gate", f"{b}_gate_sig"], [f"{b}_gact"]))
    nodes.append(oh.make_node("Mul", [f"{b}_gact", f"{b}_up"], [f"{b}_ff_mid"]))

    # Plugin 4: down_proj + residual
    w_d = layer_state["mlp.down_proj.weight"]
    out_name = f"{b}_layer_out"
    nodes.append(
        oh.make_node(
            _width("down").res_op,
            [f"{b}_ff_mid", f"{b}_post_attn"],
            [out_name],
            name=f"{b}_down_proj_res",
            domain="trt.plugins",
            plugin_namespace="gr00t::v1",
            plugin_version="1",
            N=int(k_dim),
            K=int(ff),
            rot_block_size=int(rot_bs),
            **_clip_kw("down"),
            **_weight_kw(w_d, "down"),
        )
    )
    return out_name


# Public entry point


# GR00T N1.7 (Qwen3-VL) extras: M-RoPE computed in-graph + deepstack residuals


def _emit_mrope_compute(
    nodes: list,
    inits: list,
    head_dim: int,
    theta: float,
    mrope_section: Any,
    attention_scaling: float = 1.0,
) -> None:
    """Emit ONNX nodes computing Qwen3-VL M-RoPE cos/sin from a ``position_ids`` input.

    Replaces the N1.6 baked ``rope_cos``/``rope_sin`` tensors. Mirrors
    ``Qwen3VLTextRotaryEmbedding.forward + apply_interleaved_mrope``:
    ``position_ids [3, B, S]`` → ``rope_cos``/``rope_sin`` ``[B, 1, S, head_dim]``
    BF16. Output tensor names match the N1.6 path so ``_emit_layer`` is unchanged.
    """
    assert head_dim % 2 == 0
    half = head_dim // 2
    h_len = int(mrope_section[1])
    w_len = int(mrope_section[2])
    # apply_interleaved_mrope overwrites the first 3*min(h,w) T-axis freqs with
    # stride-3 interleaved H/W contributions; the tail stays from T.
    interleaved_len = min(h_len, w_len) * 3
    tail_len = half - interleaved_len
    assert tail_len >= 0, f"head_dim/2={half} < interleaved={interleaved_len} (mrope_section={mrope_section})"
    groups = interleaved_len // 3

    # inv_freq as [3, 1, head_dim/2, 1] so MatMul with [3, B, 1, S] → [3, B, head_dim/2, S].
    inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    inv_freq_4d = np.broadcast_to(inv_freq[None, None, :, None], (3, 1, half, 1)).copy().astype(np.float32)
    inits.append(
        oh.make_tensor("mrope_inv_freq", onnx.TensorProto.FLOAT, [3, 1, half, 1], inv_freq_4d.flatten().tolist())
    )
    inits.append(oh.make_tensor("mrope_scale", onnx.TensorProto.FLOAT, [1], [float(attention_scaling)]))

    nodes.append(oh.make_node("Cast", ["position_ids"], ["mrope_pid_f32"], to=int(onnx.TensorProto.FLOAT)))
    inits.append(_i64_init("_mrope_axes_2", [2]))
    nodes.append(oh.make_node("Unsqueeze", ["mrope_pid_f32", "_mrope_axes_2"], ["mrope_pid_expanded"]))
    nodes.append(oh.make_node("MatMul", ["mrope_inv_freq", "mrope_pid_expanded"], ["mrope_freqs_raw"]))
    nodes.append(oh.make_node("Transpose", ["mrope_freqs_raw"], ["mrope_freqs"], perm=[0, 1, 3, 2]))

    # apply_interleaved_mrope via slice/reshape/concat (no ScatterND).
    inits.append(_i64_init("_mrope_axis0", [0]))
    for axis_name, idx in (("T", 0), ("H", 1), ("W", 2)):
        inits.append(_i64_init(f"_mrope_{axis_name}_start", [idx]))
        inits.append(_i64_init(f"_mrope_{axis_name}_end", [idx + 1]))
        nodes.append(
            oh.make_node(
                "Slice",
                ["mrope_freqs", f"_mrope_{axis_name}_start", f"_mrope_{axis_name}_end", "_mrope_axis0"],
                [f"mrope_axis_{axis_name}"],
            )
        )
        nodes.append(
            oh.make_node("Squeeze", [f"mrope_axis_{axis_name}", "_mrope_axis0"], [f"mrope_axis_{axis_name}_3d"])
        )

    inits.append(_i64_init("_mrope_lastdim", [-1]))
    inits.append(_i64_init("_mrope_lastdim_start_0", [0]))
    inits.append(_i64_init("_mrope_inter_end", [interleaved_len]))
    inits.append(_i64_init("_mrope_half_end", [half]))
    for axis_name in ("T", "H", "W"):
        nodes.append(
            oh.make_node(
                "Slice",
                [f"mrope_axis_{axis_name}_3d", "_mrope_lastdim_start_0", "_mrope_inter_end", "_mrope_lastdim"],
                [f"mrope_{axis_name}_inter"],
            )
        )
    nodes.append(
        oh.make_node(
            "Slice", ["mrope_axis_T_3d", "_mrope_inter_end", "_mrope_half_end", "_mrope_lastdim"], ["mrope_T_tail"]
        )
    )

    inits.append(oh.make_tensor("_mrope_reshape_groups", onnx.TensorProto.INT64, [4], [-1, 0, groups, 3]))
    for axis_name in ("T", "H", "W"):
        nodes.append(
            oh.make_node(
                "Reshape",
                [f"mrope_{axis_name}_inter", "_mrope_reshape_groups"],
                [f"mrope_{axis_name}_groups"],
                allowzero=0,
            )
        )
    for col, (nm, s, e) in enumerate((("T", 0, 1), ("H", 1, 2), ("W", 2, 3))):
        inits.append(_i64_init(f"_mrope_col{col}_start", [s]))
        inits.append(_i64_init(f"_mrope_col{col}_end", [e]))
        nodes.append(
            oh.make_node(
                "Slice",
                [f"mrope_{nm}_groups", f"_mrope_col{col}_start", f"_mrope_col{col}_end", "_mrope_lastdim"],
                [f"mrope_col_{nm}"],
            )
        )
    nodes.append(oh.make_node("Concat", ["mrope_col_T", "mrope_col_H", "mrope_col_W"], ["mrope_inter_mixed"], axis=-1))
    inits.append(oh.make_tensor("_mrope_reshape_back", onnx.TensorProto.INT64, [3], [-1, 0, interleaved_len]))
    nodes.append(
        oh.make_node("Reshape", ["mrope_inter_mixed", "_mrope_reshape_back"], ["mrope_inter_flat"], allowzero=0)
    )
    if tail_len > 0:
        nodes.append(oh.make_node("Concat", ["mrope_inter_flat", "mrope_T_tail"], ["mrope_freqs_t"], axis=-1))
    else:
        nodes.append(oh.make_node("Identity", ["mrope_inter_flat"], ["mrope_freqs_t"]))

    nodes.append(oh.make_node("Concat", ["mrope_freqs_t", "mrope_freqs_t"], ["mrope_emb"], axis=-1))
    nodes.append(oh.make_node("Cos", ["mrope_emb"], ["mrope_cos_raw"]))
    nodes.append(oh.make_node("Sin", ["mrope_emb"], ["mrope_sin_raw"]))
    nodes.append(oh.make_node("Mul", ["mrope_cos_raw", "mrope_scale"], ["mrope_cos_scaled"]))
    nodes.append(oh.make_node("Mul", ["mrope_sin_raw", "mrope_scale"], ["mrope_sin_scaled"]))
    nodes.append(oh.make_node("Cast", ["mrope_cos_scaled"], ["mrope_cos_bf16"], to=int(onnx.TensorProto.BFLOAT16)))
    nodes.append(oh.make_node("Cast", ["mrope_sin_scaled"], ["mrope_sin_bf16"], to=int(onnx.TensorProto.BFLOAT16)))
    inits.append(_i64_init("_mrope_axes_1", [1]))
    nodes.append(oh.make_node("Unsqueeze", ["mrope_cos_bf16", "_mrope_axes_1"], ["rope_cos"]))
    nodes.append(oh.make_node("Unsqueeze", ["mrope_sin_bf16", "_mrope_axes_1"], ["rope_sin"]))


def _emit_mrope_causal_mask(nodes: list, inits: list) -> None:
    """Slice the baked ``causal_mask_full`` to ``[.., current_S, current_S]`` (N1.7).

    N1.6 slices rope + mask via ``_emit_dynamic_slice_helpers``; N1.7 computes rope
    in-graph, so only the mask is sliced here.
    """
    nodes.append(oh.make_node("Shape", ["inputs_embeds"], ["_n1d7_ie_shape"]))
    inits.append(_i64_init("_n1d7_seq_idx", [1]))
    nodes.append(oh.make_node("Gather", ["_n1d7_ie_shape", "_n1d7_seq_idx"], ["_n1d7_s_1d"], axis=0))
    inits.append(_i64_init("_n1d7_zero_1d", [0]))
    inits.append(_i64_init("_n1d7_axes_2", [2]))
    inits.append(_i64_init("_n1d7_axes_3", [3]))
    nodes.append(
        oh.make_node("Slice", ["causal_mask_full", "_n1d7_zero_1d", "_n1d7_s_1d", "_n1d7_axes_2"], ["_n1d7_cm2"])
    )
    nodes.append(oh.make_node("Slice", ["_n1d7_cm2", "_n1d7_zero_1d", "_n1d7_s_1d", "_n1d7_axes_3"], ["causal_mask"]))


def _emit_deepstack_add(
    nodes: list, inits: list, cur_x_name: str, deepstack_name: str, out_name: str, suffix: str
) -> None:
    """One deepstack residual add: ``out = cur_x + ScatterND(0, NonZero(visual_pos_masks)ᵀ, deepstack)``.

    ``cur_x`` ``[B, S, hidden]`` BF16; ``deepstack`` ``[num_visual_tokens, hidden]``
    BF16; ``visual_pos_masks`` ``[B, S]`` BOOL (fixed graph-input name).
    """
    nodes.append(oh.make_node("NonZero", ["visual_pos_masks"], [f"_ds{suffix}_nz"]))
    nodes.append(oh.make_node("Transpose", [f"_ds{suffix}_nz"], [f"_ds{suffix}_idx"], perm=[1, 0]))
    nodes.append(oh.make_node("Shape", [cur_x_name], [f"_ds{suffix}_shape"]))
    nodes.append(
        oh.make_node(
            "ConstantOfShape",
            [f"_ds{suffix}_shape"],
            [f"_ds{suffix}_zeros"],
            value=oh.make_tensor("value", onnx.TensorProto.BFLOAT16, [1], [0]),
        )
    )
    nodes.append(
        oh.make_node("ScatterND", [f"_ds{suffix}_zeros", f"_ds{suffix}_idx", deepstack_name], [f"_ds{suffix}_delta"])
    )
    nodes.append(oh.make_node("Add", [cur_x_name, f"_ds{suffix}_delta"], [out_name]))


# Public entry point


def build_llm_plugin_onnx(
    qwen3_model: Any,
    output_path: "str | Path | None",
    *,
    max_seq_len: int,
    opset: int = 17,
    sq_scales: Optional[Dict[str, Any]] = None,
    rot_bs: int = 0,
    n1d7_mode: bool = False,
    gemma_mode: bool = False,
    mrope_section: Optional[list] = None,
    attention_scaling: float = 1.0,
    num_deepstack: int = 0,
    batch: int = 1,
    num_vis_tokens: Optional[int] = None,
    pin_seq_len: Optional[int] = None,
    bits: int = 8,
    gptq_hessians: Optional[Dict[str, Any]] = None,
    act_clip_ratio: Any = 1.0,
    site_bits: Optional[Dict[str, int]] = None,
    act_bits: Optional[int] = None,
    weight_clip: Optional[Dict[str, Any]] = None,
    final_norm: Optional[bool] = None,
) -> Path:
    """Build the dynamic per-row INT8/INT4 plugin ONNX for a supported LLM.

    ``bits`` selects the plugin pair (8: FusedRmsNormLinearInt8 +
    PerRowInt8LinearResidual; 4: FusedRmsNormLinearInt4 + the FWHT mode of
    PerRowInt4LinearResidual) and the weight packer. At 4 bits the weights are
    GPTQ-rounded, so ``gptq_hessians`` (from
    :func:`llm_gptq.compute_gptq_hessians_llm`, keyed ``f"L{i}_{site}"``) is
    required and its absence fails closed rather than falling back to RTN.

    One engine handles any seq_len in ``[1, max_seq_len]``. The causal mask is
    baked at ``max_seq_len`` then sliced to the current seq_len at runtime; the
    rest of the graph uses symbolic seq_len. Activation amax is computed at
    runtime inside each plugin (dynamic per-row), so no static activation-scale
    field is baked.

    Args:
        gemma_mode: emit the Pi0/Pi0.5 PaliGemma prefix-pass contract instead of
            GR00T's hidden-states one: inputs ``prefix_embs`` [B, S, hidden] /
            the additive ``attention_mask`` [B, 1, S, S] / ``position_ids``
            INT64 [B, S]; RoPE gathered from a baked table at theta=10000;
            output is the stacked post-RoPE KV cache ``kv_stack``
            [L, 2, B, H_kv, S, D] with no final norm (only the cache crosses the
            engine boundary).
        qwen3_model: live PyTorch LLM. Must expose ``.config`` and ``.layers`` and
            a ``state_dict()`` keyed ``layers.<i>.<...>`` plus ``norm.weight``.
        output_path: path to save the ``.onnx`` (external data written
            alongside when tensors exceed the size threshold); ``None`` returns
            the ``ModelProto`` instead of writing it.
        max_seq_len: upper bound on supported seq_len. Sets the baked
            rope/mask size; the engine optimization profile derives its max
            from the ONNX symbolic dim.
        opset: ONNX opset for the default domain. The plugin domain
            ``trt.plugins`` is always imported at version 1.
        sq_scales: per-layer SmoothQuant scales from
            ``llm_rotation_sq.compute_sq_scales_llm``, keyed ``f"L{i}_{qkv,gateup,down}"``.
            ``None`` (default) builds the plain ``w8a8`` graph.
        rot_bs: block-diagonal Hadamard rotation size (``llm_rotation_sq.apply_rot_fold``).
            0 disables rotation; requires *sq_scales*.
        final_norm: whether the graph ends with the tower's final RMSNorm.
            ``None`` (default) reads it off the module - a ``norm`` flagged
            ``is_identity_norm`` emits none. Pass ``False`` for a backbone that
            consumes the pre-norm residual stream while the module still carries
            a real ``norm``: GR00T N1.7 reads ``hidden_states[-1]``, which under
            its pinned transformers is the last decoder layer's output *before*
            ``norm``, so emitting the norm would feed the action head features
            the checkpoint was never trained on.

    Returns:
        The saved ONNX :class:`pathlib.Path`.
    """
    if rot_bs and sq_scales is None:
        raise ValueError("build_llm_plugin_onnx: rot_bs>0 requires sq_scales (rotation always follows an SQ fold).")

    torch = _torch()

    out_path = Path(output_path) if output_path is not None else None
    qwen3_model = resolve_qwen3_decoder(qwen3_model)

    cfg = qwen3_model.config
    k_dim = cfg.hidden_size
    ff = cfg.intermediate_size
    h = cfg.num_attention_heads
    hkv = cfg.num_key_value_heads
    d = getattr(cfg, "head_dim", k_dim // h)
    eps = cfg.rms_norm_eps
    theta = getattr(cfg, "rope_theta", 1000000.0)
    # Gemma genuinely uses rope_theta=10000; keep whatever the config gives
    # (10000) but pin it so a PaliGemma override cannot drift the emitter.
    if gemma_mode:
        theta = 10000.0
    num_layers = len(qwen3_model.layers)
    # gemma_mode emits the prefix-KV contract (prefix_embs in, additive block
    # mask, position_ids RoPE, kv_stack out); the Gemma specifics (RMSNorm +1,
    # gelu-tanh MLP) are applied inline below.
    kv_stack_mode = gemma_mode
    if kv_stack_mode and n1d7_mode:
        raise ValueError("gemma_mode is exclusive with n1d7_mode.")

    inits: list = []
    nodes: list = []

    # Dynamic batch + seq dims - names match adapter.py's ModuleExportSpec
    # dynamic_axes for the "llm" module ("batch"/"seq_len") so the graph reads
    # consistently with the rest of the export pipeline. RoPE cos/sin and
    # causal_mask are baked at (1, 1, max_seq_len, ...) - they broadcast over
    # batch naturally in the ONNX ops that consume them.
    # N1.6 opens the batch dim (symbolic "batch"); N1.7 pins it to the captured batch,
    # the deepstack ScatterND indexes a flat [B, S] mask, and a symbolic batch makes the
    # TensorRT optimization profile inconsistent (batch 1 vs seq max). This matches the
    # bf16 Qwen3-VL export, whose engine also builds with a fixed batch.
    batch_dim: Any = int(batch) if n1d7_mode else (1 if kv_stack_mode else "batch")
    # Gemma (Pi0/Pi0.5) pins the prefix length to the captured value: the Pi
    # processor pads text to a fixed max_length and the camera count is fixed,
    # so the runtime prefix is constant, and a symbolic seq_len here breaks
    # the TensorRT profile the same way a symbolic batch broke N1.7's (the
    # manifest's observed shapes describe the ORIGINAL float graph, whose
    # additive mask is 4-D, so derive_shapes maps the plugin graph's 3-D bool
    # mask axes wrong and myelin rejects the fused attention profile).
    if pin_seq_len is not None and not kv_stack_mode:
        raise ValueError("pin_seq_len is only meaningful for kv_stack_mode graphs.")
    if pin_seq_len is not None and int(pin_seq_len) > int(max_seq_len):
        raise ValueError(f"pin_seq_len={pin_seq_len} exceeds max_seq_len={max_seq_len}.")
    seq_dim: Any = int(pin_seq_len) if pin_seq_len is not None else "seq_len"
    seq_dim2: Any = int(pin_seq_len) if pin_seq_len is not None else "seq_len2"
    # Output name follows each family's ModuleExportSpec: N1.6's "llm" module declares
    # `hidden_states`, N1.7's declares `embeddings`. The drop-in engine module reads the
    # graph by name, so this is a contract, not a label.
    out_tensor = "embeddings" if n1d7_mode else ("kv_stack" if kv_stack_mode else "hidden_states")
    if kv_stack_mode:
        x_in = oh.make_tensor_value_info("prefix_embs", onnx.TensorProto.BFLOAT16, [batch_dim, seq_dim, k_dim])
        # The Pi prefix seam hands the engine the SAME 4-D ADDITIVE mask
        # the float export captured ([B, 1, S, S], HF `_prepare_4d_mask`
        # output); declare that contract instead of a bool mask so the
        # drop-in module needs no conversion and the graph rank matches
        # the manifest's observed input_features.
        am_in = oh.make_tensor_value_info(
            "attention_mask", onnx.TensorProto.BFLOAT16, [batch_dim, 1, seq_dim, seq_dim2]
        )
        # Gemma's stack_kv_cache keeps HF-native [B, H_kv, S, D] per layer.
        y_out = oh.make_tensor_value_info(
            out_tensor, onnx.TensorProto.BFLOAT16, [num_layers, 2, batch_dim, hkv, seq_dim, d]
        )
    else:
        x_in = oh.make_tensor_value_info("inputs_embeds", onnx.TensorProto.BFLOAT16, [batch_dim, "seq_len", k_dim])
        # ``attention_mask`` is declared for IO parity and to size the dynamic seq_len, but
        # attention is causal-only (no key-padding bias) - correct for GR00T's single
        # un-padded VL sequence per request.
        am_in = oh.make_tensor_value_info("attention_mask", onnx.TensorProto.INT64, [batch_dim, "seq_len"])
        y_out = oh.make_tensor_value_info(out_tensor, onnx.TensorProto.BFLOAT16, [batch_dim, "seq_len", k_dim])

    # N1.7 (Qwen3-VL): extra graph inputs for M-RoPE + deepstack injection. The deepstack
    # token count is fixed (as in the bf16 export) so the engine profile is self-consistent;
    # it varies per checkpoint, so the engine is rebuilt per checkpoint.
    extra_inputs: list = []
    if kv_stack_mode:
        extra_inputs.append(oh.make_tensor_value_info("position_ids", onnx.TensorProto.INT64, [batch_dim, seq_dim]))
    if n1d7_mode:
        if mrope_section is None:
            raise ValueError("n1d7_mode requires mrope_section (Qwen3-VL rope_scaling.mrope_section).")
        vis_dim: Any = int(num_vis_tokens) if num_vis_tokens else "num_vis_tokens"
        extra_inputs.append(
            oh.make_tensor_value_info("position_ids", onnx.TensorProto.INT64, [3, batch_dim, "seq_len"])
        )
        extra_inputs.append(
            oh.make_tensor_value_info("visual_pos_masks", onnx.TensorProto.BOOL, [batch_dim, "seq_len"])
        )
        for dsi in range(num_deepstack):
            extra_inputs.append(
                oh.make_tensor_value_info(f"deepstack_{dsi}", onnx.TensorProto.BFLOAT16, [vis_dim, k_dim])
            )

    if kv_stack_mode:
        # RoPE gathered by position_ids from tables baked at max_seq_len.
        cos, sin = _compute_qwen3_rope(max_seq_len, d, theta)
        inits.append(_bf16_initializer("rope_cos_table", cos.view(max_seq_len, d)))
        inits.append(_bf16_initializer("rope_sin_table", sin.view(max_seq_len, d)))
        nodes.append(oh.make_node("Gather", ["rope_cos_table", "position_ids"], ["rope_cos_g"], axis=0))
        nodes.append(oh.make_node("Gather", ["rope_sin_table", "position_ids"], ["rope_sin_g"], axis=0))
        inits.append(_i64_init("_rope_axes_1", [1]))
        nodes.append(oh.make_node("Unsqueeze", ["rope_cos_g", "_rope_axes_1"], ["rope_cos"]))
        nodes.append(oh.make_node("Unsqueeze", ["rope_sin_g", "_rope_axes_1"], ["rope_sin"]))
        # Input is already the additive [B, 1, S, S] bias; pass through.
        nodes.append(oh.make_node("Identity", ["attention_mask"], ["block_mask"]))
    else:
        # Bake the causal mask at max_seq_len; slice down at runtime (both families).
        inits.append(_bf16_initializer("causal_mask_full", _causal_mask(max_seq_len)))
    if n1d7_mode:
        # M-RoPE is computed in-graph from position_ids; only the mask is sliced.
        _emit_mrope_compute(
            nodes, inits, head_dim=d, theta=theta, mrope_section=mrope_section, attention_scaling=attention_scaling
        )
        _emit_mrope_causal_mask(nodes, inits)
    elif not kv_stack_mode:
        # N1.5/N1.6: bake 1D RoPE tables at max_seq_len + rope/mask slice helpers.
        cos, sin = _compute_qwen3_rope(max_seq_len, d, theta)
        inits.append(_bf16_initializer("rope_cos_full", cos))
        inits.append(_bf16_initializer("rope_sin_full", sin))
        _emit_dynamic_slice_helpers(nodes, inits)

    # Key-padding bias, folded into the causal mask: a batch of parallel environments
    # is padded to the longest prompt, and the causal mask alone says nothing about
    # pad positions. The kv-stack prefix graph takes its own block mask, so this
    # covers the plain causal families.
    attn_mask_name = "causal_mask"
    if not kv_stack_mode:
        _t = _torch()
        key_pad_val = float(_t.finfo(_t.bfloat16).min * 0.5)
        inits.append(_bf16_initializer("_kp_one", _t.tensor(1.0, dtype=_t.bfloat16)))
        inits.append(_bf16_initializer("_kp_val", _t.tensor(key_pad_val, dtype=_t.bfloat16)))
        inits.append(_i64_init("_kp_axes", [1, 2]))
        nodes.append(oh.make_node("Cast", ["attention_mask"], ["_kp_am"], to=onnx.TensorProto.BFLOAT16))
        nodes.append(oh.make_node("Sub", ["_kp_one", "_kp_am"], ["_kp_pad"]))
        nodes.append(oh.make_node("Mul", ["_kp_pad", "_kp_val"], ["_kp_bias"]))
        nodes.append(oh.make_node("Unsqueeze", ["_kp_bias", "_kp_axes"], ["_kp_bias_4d"]))
        nodes.append(oh.make_node("Add", ["causal_mask", "_kp_bias_4d"], ["attn_mask"]))
        attn_mask_name = "attn_mask"

    cur_x = "prefix_embs" if kv_stack_mode else "inputs_embeds"
    # Batch-aware repeat_kv target, shared by every layer (see KV_SHAPE_DYN).
    _emit_kv_shape_dyn(nodes, inits, num_heads=h, head_dim=d, source=cur_x)
    kv_layer_names: list = []
    sd_full = qwen3_model.state_dict()
    _sites = ("qkv", "o", "gateup", "down")
    _sb = {k: int(v) for k, v in (site_bits or {}).items()}
    if set(_sb) - set(_sites) or any(v not in (4, 8) for v in _sb.values()):
        raise ValueError(f"site_bits must map a subset of {_sites} to 4 or 8, got {site_bits!r}")
    if act_bits is not None and int(act_bits) not in (4, 8):
        raise ValueError(f"act_bits must be 4 or 8, got {act_bits!r}")
    if act_bits is not None and int(act_bits) == 4 and int(bits) == 8:
        raise ValueError("act_bits=4 with INT8 weights (W8A4) has no plugin pair.")
    any_int4 = int(bits) == 4 or any(_sb.get(s, int(bits)) == 4 for s in _sites)
    if not any_int4 and gptq_hessians is not None:
        raise ValueError(
            f"gptq_hessians supplied for a width-{bits} build. GPTQ serves the INT4 weight "
            "axis only; an INT8 build receiving Hessians means the scheme routing is wrong."
        )
    gptq = None
    if any_int4:
        from foldquant.llm_gptq import GPTQSiteFactors

        gptq = GPTQSiteFactors(gptq_hessians or {})

    for i in range(num_layers):
        prefix = f"L{i}"
        layer_state = {
            key.replace(f"layers.{i}.", ""): v for key, v in sd_full.items() if key.startswith(f"layers.{i}.")
        }
        if gemma_mode:
            # GemmaRMSNorm applies its weight as (1 + w); the plugin computes
            # normed*gamma, so materialize gamma = w + 1 here, BEFORE any
            # SmoothQuant fold, which rescales gamma and must see (1+w).
            layer_state = dict(layer_state)
            layer_state["input_layernorm.weight"] = layer_state["input_layernorm.weight"] + 1.0
            layer_state["post_attention_layernorm.weight"] = layer_state["post_attention_layernorm.weight"] + 1.0
        if sq_scales is not None:
            # compute_sq_scales_llm returns CPU tensors; move to the weights' device.
            dev = layer_state["input_layernorm.weight"].device
            layer_state = apply_sq_fold(
                layer_state,
                s_qkv=sq_scales[f"L{i}_qkv"].to(dev),
                s_gu=sq_scales[f"L{i}_gateup"].to(dev),
                s_dn=sq_scales[f"L{i}_down"].to(dev),
            )
            if rot_bs > 1:
                layer_state = apply_rot_fold(layer_state, rot_bs)
        cur_x = _emit_layer(
            layer_state=layer_state,
            prefix=prefix,
            cur_x_name=cur_x,
            eps=eps,
            num_heads=h,
            num_kv_heads=hkv,
            head_dim=d,
            hidden_size=k_dim,
            intermediate_size=ff,
            cos_init_name="rope_cos",
            sin_init_name="rope_sin",
            causal_mask_init_name="block_mask" if kv_stack_mode else attn_mask_name,
            nodes=nodes,
            inits=inits,
            rot_bs=rot_bs,
            gemma_mode=gemma_mode,
            bits=bits,
            gptq=gptq,
            act_clip_ratio=act_clip_ratio,
            site_bits=_sb or None,
            act_bits=act_bits,
            weight_clip=weight_clip,
        )
        if kv_stack_mode:
            # Collect post-RoPE K and raw V (both pre-repeat_kv) into the
            # runtime KV-stack layout. Gemma keeps HF-native [B, HKV, S, D]
            # (k_r and v_t are already that).
            k_export, v_export = f"{prefix}_k_r", f"{prefix}_v_t"
            nodes.append(oh.make_node("Unsqueeze", [k_export, "_kv_axes_0"], [f"{prefix}_k_u"]))
            nodes.append(oh.make_node("Unsqueeze", [v_export, "_kv_axes_0"], [f"{prefix}_v_u"]))
            nodes.append(oh.make_node("Concat", [f"{prefix}_k_u", f"{prefix}_v_u"], [f"{prefix}_kv"], axis=0))
            nodes.append(oh.make_node("Unsqueeze", [f"{prefix}_kv", "_kv_axes_0"], [f"{prefix}_kv_u"]))
            kv_layer_names.append(f"{prefix}_kv_u")
        # N1.7: deepstack residual after each of the first ``num_deepstack`` layers.
        if n1d7_mode and i < num_deepstack:
            ds_out = f"{prefix}_plus_deepstack"
            _emit_deepstack_add(
                nodes, inits, cur_x_name=cur_x, deepstack_name=f"deepstack_{i}", out_name=ds_out, suffix=str(i)
            )
            cur_x = ds_out

    if kv_stack_mode:
        # Stack the per-layer KV; no final norm: the prefix hidden states are
        # consumed nowhere (only the cache crosses the engine boundary).
        inits.append(_i64_init("_kv_axes_0", [0]))
        nodes.append(oh.make_node("Concat", kv_layer_names, [out_tensor], axis=0))
        graph = oh.make_graph(
            nodes, "llm_gemma_prefix_plugin", [x_in, am_in] + extra_inputs, [y_out], initializer=inits
        )
        model = oh.make_model(
            graph,
            opset_imports=[oh.make_opsetid("", opset), oh.make_opsetid("trt.plugins", 1)],
        )
        model.ir_version = 9
        if output_path is None:
            return model
        from .onnx_io import save_plugin_onnx

        save_plugin_onnx(model, out_path, size_threshold=1024)
        return out_path

    # Whether the graph ends with a final norm is read off the tower unless the
    # caller states it: a ``norm`` flagged ``is_identity_norm`` ends the tower at
    # the residual stream. The explicit flag exists for backbones that read the
    # pre-norm stream off a module that still carries a real norm (see the Args).
    if final_norm is None:
        _final_norm = getattr(qwen3_model, "norm", None)
        final_norm = not bool(getattr(_final_norm, "is_identity_norm", False))
    if not final_norm:
        nodes.append(oh.make_node("Identity", [cur_x], [out_tensor]))
    else:
        # Final RMSNorm (qwen3.norm), for towers whose head consumes post-norm.
        inits.append(_bf16_initializer("final_eps", torch.tensor(eps, dtype=torch.bfloat16)))
        inits.append(_bf16_initializer("final_two", torch.tensor(2.0, dtype=torch.bfloat16)))
        inits.append(_bf16_initializer("final_gamma", sd_full["norm.weight"]))
        nodes.append(oh.make_node("Pow", [cur_x, "final_two"], ["final_sq"]))
        nodes.append(oh.make_node("ReduceMean", ["final_sq"], ["final_mean"], axes=[-1], keepdims=1))
        nodes.append(oh.make_node("Add", ["final_mean", "final_eps"], ["final_var"]))
        nodes.append(oh.make_node("Sqrt", ["final_var"], ["final_std"]))
        nodes.append(oh.make_node("Div", [cur_x, "final_std"], ["final_normed"]))
        nodes.append(oh.make_node("Mul", ["final_normed", "final_gamma"], [out_tensor]))

    graph = oh.make_graph(nodes, "llm_qwen3_plugin", [x_in, am_in] + extra_inputs, [y_out], initializer=inits)
    model = oh.make_model(
        graph,
        opset_imports=[oh.make_opsetid("", opset), oh.make_opsetid("trt.plugins", 1)],
    )
    model.ir_version = 9
    if output_path is None:
        return model
    from .onnx_io import save_plugin_onnx

    save_plugin_onnx(model, out_path, size_threshold=1024)
    return out_path
