# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""INT8 per-row plugin graph for the Pi0.5 / Pi0 Gemma-300M denoise step.

Replaces the exported ``expert.onnx`` (one flow-matching denoise step) with a
graph whose transformer GEMMs run through the ``gr00t::v1`` INT8 per-row
plugins, with the same I/O contract as the graph it replaces (``x_t [1,H,32]``,
``timestep [1]``, ``prefix_pad_masks [1,S]`` BOOL,
``kv_stack [18,2,1,1,S,256]`` BF16, Pi0 additionally ``state [1,32]`` ->
``velocity [1,H,32]`` FP32).

No fused macro plugin fits this block: the norm is ``GemmaAdaRMSNorm``,
RMSNorm-based (no mean subtraction, so the DiT AdaLN plugin's LayerNorm
prologue does NOT match) modulated by *runtime* ``(1+scale)*normed + shift``
tensors from ``dense(adarms_cond)``, with an AdaLN-Zero ``x + y*gate``
residual. So the graph decomposes the block: the AdaRMS
modulation, RoPE, GQA repeat, SDPA and gated residuals stay BF16 ONNX ops,
and every projection GEMM runs through ``PerRowInt8LinearResidual`` (with a
static zeros residual, since the suffix length is fixed per build, and the gated
add applied *outside* the plugin, since the gate multiply must sit between
GEMM output and residual add).

There is **no layer alternation**: all 18 layers run full
attention over ``[prefix_S + suffix]`` keys, consuming the cached prefix K/V
raw (``kv_stack[i]``, HF-native ``[B, H_kv, S, D]``, K cached post-RoPE) and
RoPE-ing the fresh suffix K at ABSOLUTE positions (prefix_len + 0..L-1).

Variant differences (read off ``use_adarms``):
- Pi0.5: suffix = action tokens only (L = H); AdaRMS with
  ``cond = silu(time_mlp_out(silu(time_mlp_in(sin_emb))))``; suffix attends
  bidirectionally (att cumsum is constant).
- Pi0: suffix = ``[state_token, action_tokens]`` (L = H+1); vanilla
  ``(1+w)`` RMSNorm, un-gated residuals; the state token attends only itself
  (+ prefix), the action tokens attend everything.

What stays BF16/FP32 ONNX besides the glue: the suffix embedding (the
period-parametrised sinusoidal time embedding with constants baked from exact
float64 tables, the time/state/action MLPs), the per-norm ``dense`` modulation
matmuls, and ``action_out_proj`` (bf16 matmul, fp32 output cast, matching the
runtime). Constructs ONNX only: no ``tensorrt`` import, no ``.so`` load.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np
import onnx
import onnx.helper as oh

from .llm import (
    _bf16_initializer,
    _emit_gelu_tanh,
    _i64_init,
    _torch,
)

logger = logging.getLogger(__name__)

__all__ = ["build_gemma_expert_plugin_onnx"]

# The plugin contract lives in dit_common ("must match the compiled plugin .so.
# Do not change"), imported rather than redeclared so the four emitters cannot drift.
# PLUGIN_VERSION is a STRING: TensorRT reads plugin_version as a string
# attribute, and an INT attr fails creator lookup ("Plugin not found") at parse.
from . import foldq  # noqa: E402  (contract imports kept next to their use)
from . import rotation as rotations  # noqa: E402
from .dit_common import PLUGIN_DOMAIN as _PLUGIN_DOMAIN  # noqa: E402
from .dit_common import PLUGIN_NAMESPACE as _PLUGIN_NAMESPACE  # noqa: E402
from .dit_common import PLUGIN_VERSION as _PLUGIN_VERSION  # noqa: E402


def _f32_initializer(name: str, tensor: Any) -> onnx.TensorProto:
    arr = np.ascontiguousarray(tensor.detach().float().cpu().numpy(), dtype=np.float32)
    return oh.make_tensor(name, onnx.TensorProto.FLOAT, list(arr.shape), arr.tobytes(), raw=True)


def build_gemma_expert_plugin_onnx(
    action_expert: Any,
    out_path: "str | Path | None",
    *,
    rope_max_seq_len: int = 4096,
    int4: bool = False,
    sq_scales: "dict | None" = None,
    block_size: int = 64,
    fwht: bool = False,
    fold_order: str = "after",
    gptq: "dict | None" = None,
) -> Any:
    """Write the INT8 per-row denoise-step plugin graph for *action_expert*.

    Args:
        action_expert: Live ``Pi05FlowMatchingExpert`` (read-only; weights are
            snapshotted into initializers). ``config.use_adarms`` selects the
            Pi0.5 (AdaRMS) vs Pi0 (vanilla RMS + state token) graph.
        out_path: Destination ``.onnx``; ``None`` returns the ``ModelProto`` instead
            of writing it (the fake-quant reads its nodes in memory).
        rope_max_seq_len: Size of the baked absolute-position RoPE table.
        int4: Emit the FoldQuant W4A4 variant (``w4a4_sr``): every
            projection GEMM through ``PerRowInt4LinearResidual`` (rotated
            per-row INT4). Requires *sq_scales*.
        sq_scales: Per-site rotated-activation amax from
            :func:`compute_gemma_expert_sq_scales`.
        block_size: FoldQuant rotation block width (int4 only).
    """
    if int4 and sq_scales is None:
        raise ValueError("int4=True requires sq_scales (the W4A4 scheme is SmoothQuant-folded).")
    torch = _torch()
    sd = {k: v.detach() for k, v in action_expert.state_dict().items()}
    cfg = action_expert.config
    use_adarms = bool(cfg.use_adarms)
    variant = action_expert._variant
    hidden = int(variant.width)  # 1024
    num_layers = int(variant.depth)  # 18
    h = int(variant.num_heads)  # 8
    hkv = int(variant.num_kv_heads)  # 1
    d = int(variant.head_dim)  # 256
    q_dim = h * d  # 2048
    kv_dim = hkv * d  # 256
    kv_groups = h // hkv
    ff = int(variant.mlp_dim)  # 4096
    horizon = int(cfg.action_horizon)
    action_dim = int(cfg.action_dim)
    suffix_len = horizon if use_adarms else horizon + 1
    eps = 1e-6  # GemmaDecoderLayer default; GemmaAdaRMSNorm uses the same
    theta = 10000.0

    ep = "expert_model.model."  # decoder prefix inside the expert state_dict

    inits: list = []
    nodes: list = []

    x_in = oh.make_tensor_value_info("x_t", onnx.TensorProto.FLOAT, [1, horizon, action_dim])
    t_in = oh.make_tensor_value_info("timestep", onnx.TensorProto.FLOAT, [1])
    p_in = oh.make_tensor_value_info("prefix_pad_masks", onnx.TensorProto.BOOL, [1, "prefix_len"])
    kv_in = oh.make_tensor_value_info("kv_stack", onnx.TensorProto.BFLOAT16, [num_layers, 2, 1, hkv, "prefix_len", d])
    graph_inputs = [x_in, t_in, p_in, kv_in]
    if not use_adarms:
        graph_inputs.append(oh.make_tensor_value_info("state", onnx.TensorProto.FLOAT, [1, action_dim]))
    y_out = oh.make_tensor_value_info("velocity", onnx.TensorProto.FLOAT, [1, horizon, action_dim])

    def _linear(name: str, x: str, w_key: str, out: str, b_key: "str | None" = None) -> None:
        inits.append(_bf16_initializer(f"{name}_w", sd[w_key].t().contiguous()))
        nodes.append(oh.make_node("MatMul", [x, f"{name}_w"], [f"{name}_mm"]))
        bias_key = b_key if b_key is not None else w_key.replace(".weight", ".bias")
        if bias_key in sd:
            inits.append(_bf16_initializer(f"{name}_b", sd[bias_key]))
            nodes.append(oh.make_node("Add", [f"{name}_mm", f"{name}_b"], [out]))
        else:
            nodes.append(oh.make_node("Identity", [f"{name}_mm"], [out]))

    def _silu(name: str, x: str, out: str) -> None:
        nodes.append(oh.make_node("Sigmoid", [x], [f"{name}_sig"]))
        nodes.append(oh.make_node("Mul", [x, f"{name}_sig"], [out]))

    # time embedding (constants baked from exact float64)
    half = hidden // 2
    fraction = torch.linspace(0.0, 1.0, half, dtype=torch.float64)
    period = 4e-3 * (4.0 / 4e-3) ** fraction
    scaling = (1.0 / period * 2 * math.pi).to(torch.float64)
    inits.append(_f32_initializer("time_scaling", scaling.reshape(1, half)))
    inits.append(_i64_init("_axes_1", [1]))
    nodes.append(oh.make_node("Unsqueeze", ["timestep", "_axes_1"], ["t_2d"]))  # [1,1]
    nodes.append(oh.make_node("Mul", ["t_2d", "time_scaling"], ["sin_input"]))
    nodes.append(oh.make_node("Sin", ["sin_input"], ["t_sin"]))
    nodes.append(oh.make_node("Cos", ["sin_input"], ["t_cos"]))
    nodes.append(oh.make_node("Concat", ["t_sin", "t_cos"], ["time_emb_f32"], axis=1))  # [1,hidden]
    nodes.append(oh.make_node("Cast", ["time_emb_f32"], ["time_emb"], to=onnx.TensorProto.BFLOAT16))

    # suffix embedding
    nodes.append(oh.make_node("Cast", ["x_t"], ["x_t_bf16"], to=onnx.TensorProto.BFLOAT16))
    _linear("act_in", "x_t_bf16", "action_in_proj.weight", "action_emb")  # [1,H,hidden]

    if use_adarms:
        # Pi0.5: cond = silu(time_mlp_out(silu(time_mlp_in(time_emb)))).
        _linear("tmlp_in", "time_emb", "time_mlp_in.weight", "tmlp_h")
        _silu("tmlp_s1", "tmlp_h", "tmlp_hs")
        _linear("tmlp_out", "tmlp_hs", "time_mlp_out.weight", "tmlp_o")
        _silu("cond_s", "tmlp_o", "adarms_cond")  # [1,hidden]
        suffix_name = "action_emb"
    else:
        # Pi0: fuse time into action tokens; prepend the state token.
        inits.append(
            oh.make_tensor(
                "time_expand_shape",
                onnx.TensorProto.INT64,
                [3],
                np.array([1, horizon, hidden], dtype=np.int64).tobytes(),
                raw=True,
            )
        )
        nodes.append(oh.make_node("Unsqueeze", ["time_emb", "_axes_1"], ["time_emb_3d"]))
        nodes.append(oh.make_node("Expand", ["time_emb_3d", "time_expand_shape"], ["time_emb_x"]))
        nodes.append(oh.make_node("Concat", ["action_emb", "time_emb_x"], ["action_time_cat"], axis=2))
        _linear("atm_in", "action_time_cat", "action_time_mlp_in.weight", "atm_h")
        _silu("atm_s", "atm_h", "atm_hs")
        _linear("atm_out", "atm_hs", "action_time_mlp_out.weight", "atm_o")
        nodes.append(oh.make_node("Cast", ["state"], ["state_bf16"], to=onnx.TensorProto.BFLOAT16))
        _linear("state_p", "state_bf16", "state_proj.weight", "state_emb_2d")
        nodes.append(oh.make_node("Unsqueeze", ["state_emb_2d", "_axes_1"], ["state_emb"]))  # [1,1,hidden]
        nodes.append(oh.make_node("Concat", ["state_emb", "atm_o"], ["suffix_embs"], axis=1))  # [1,H+1,hidden]
        suffix_name = "suffix_embs"

    # positions and masks (shared by all layers)
    nodes.append(oh.make_node("Shape", ["prefix_pad_masks"], ["_pp_shape"]))
    inits.append(_i64_init("_idx_1", [1]))
    nodes.append(oh.make_node("Gather", ["_pp_shape", "_idx_1"], ["_S"], axis=0))
    nodes.append(oh.make_node("Cast", ["prefix_pad_masks"], ["_pp_i64"], to=onnx.TensorProto.INT64))
    inits.append(_i64_init("_sum_axes_1", [1]))
    nodes.append(oh.make_node("ReduceSum", ["_pp_i64", "_sum_axes_1"], ["_offset"], keepdims=0))  # [1]
    inits.append(
        oh.make_tensor(
            "suffix_arange",
            onnx.TensorProto.INT64,
            [1, suffix_len],
            np.arange(suffix_len, dtype=np.int64).tobytes(),
            raw=True,
        )
    )
    nodes.append(oh.make_node("Unsqueeze", ["_offset", "_axes_1"], ["_offset_2d"]))
    nodes.append(oh.make_node("Add", ["_offset_2d", "suffix_arange"], ["abs_positions"]))  # [1,L]

    inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
    pos = torch.arange(rope_max_seq_len, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    inits.append(_bf16_initializer("rope_cos_table", emb.cos().to(torch.bfloat16)))
    inits.append(_bf16_initializer("rope_sin_table", emb.sin().to(torch.bfloat16)))
    nodes.append(oh.make_node("Gather", ["rope_cos_table", "abs_positions"], ["cos_g"], axis=0))
    nodes.append(oh.make_node("Gather", ["rope_sin_table", "abs_positions"], ["sin_g"], axis=0))
    nodes.append(oh.make_node("Unsqueeze", ["cos_g", "_axes_1"], ["cos_abs"]))  # [1,1,L,d]
    nodes.append(oh.make_node("Unsqueeze", ["sin_g", "_axes_1"], ["sin_abs"]))

    # Additive masks. Prefix part from the runtime pad mask; suffix part static.
    mask_val = float(torch.finfo(torch.bfloat16).min * 0.5)
    inits.append(_bf16_initializer("m_one", torch.tensor(1.0, dtype=torch.bfloat16)))
    inits.append(_bf16_initializer("m_scale", torch.tensor(mask_val, dtype=torch.bfloat16)))
    nodes.append(oh.make_node("Cast", ["prefix_pad_masks"], ["_pp_bf16"], to=onnx.TensorProto.BFLOAT16))
    nodes.append(oh.make_node("Sub", ["m_one", "_pp_bf16"], ["_pp_inv"]))
    nodes.append(oh.make_node("Mul", ["_pp_inv", "m_scale"], ["_pp_bias_2d"]))  # [1,S]
    inits.append(_i64_init("_bias_axes", [1, 2]))
    nodes.append(oh.make_node("Unsqueeze", ["_pp_bias_2d", "_bias_axes"], ["prefix_bias"]))  # [1,1,1,S]
    inits.append(_i64_init("_exp_const_11L", [1, 1, suffix_len]))
    nodes.append(oh.make_node("Concat", ["_exp_const_11L", "_S"], ["_exp_shape"], axis=0))
    nodes.append(oh.make_node("Expand", ["prefix_bias", "_exp_shape"], ["prefix_bias_q"]))  # [1,1,L,S]
    # Suffix block, from make_att_2d_masks(ones, att_masks):
    #   Pi0.5: att = [1, 0, 0, ...] -> cumsum constant -> fully bidirectional (zeros).
    #   Pi0:   att = [1, 1, 0, ...] -> the state token (row 0) attends only itself.
    suffix_mask = torch.zeros(suffix_len, suffix_len)
    if not use_adarms:
        suffix_mask[0, 1:] = mask_val
    inits.append(_bf16_initializer("suffix_mask", suffix_mask.to(torch.bfloat16).view(1, 1, suffix_len, suffix_len)))
    nodes.append(oh.make_node("Concat", ["prefix_bias_q", "suffix_mask"], ["full_mask"], axis=3))

    inits.append(_bf16_initializer("attn_scale", torch.tensor(d**-0.5, dtype=torch.bfloat16)))
    inits.append(_bf16_initializer("one_bf16", torch.tensor(1.0, dtype=torch.bfloat16)))
    inits.append(_bf16_initializer("eps_bf16", torch.tensor(eps, dtype=torch.bfloat16)))
    inits.append(_bf16_initializer("two_bf16", torch.tensor(2.0, dtype=torch.bfloat16)))
    inits.append(_i64_init("_axes_2", [2]))
    inits.append(_i64_init("split_half_d", [d // 2, d // 2]))
    inits.append(
        oh.make_tensor("_scalar_0", onnx.TensorProto.INT64, [], np.array(0, dtype=np.int64).tobytes(), raw=True)
    )
    inits.append(
        oh.make_tensor("_scalar_1", onnx.TensorProto.INT64, [], np.array(1, dtype=np.int64).tobytes(), raw=True)
    )

    def _rms_normed(name: str, x: str, out: str) -> None:
        """Bare RMS normalization (no weight): x * rsqrt(mean(x^2) + eps)."""
        nodes.append(oh.make_node("Pow", [x, "two_bf16"], [f"{name}_sq"]))
        nodes.append(oh.make_node("ReduceMean", [f"{name}_sq"], [f"{name}_mean"], axes=[-1], keepdims=1))
        nodes.append(oh.make_node("Add", [f"{name}_mean", "eps_bf16"], [f"{name}_var"]))
        nodes.append(oh.make_node("Sqrt", [f"{name}_var"], [f"{name}_std"]))
        nodes.append(oh.make_node("Div", [x, f"{name}_std"], [out]))

    def _adarms(name: str, x: str, norm_key: str, out: str) -> "str | None":
        """GemmaAdaRMSNorm: returns the gate tensor name (None for vanilla)."""
        _rms_normed(name, x, f"{name}_normed")
        if use_adarms:
            _linear(f"{name}_dense", "adarms_cond", norm_key + ".dense.weight", f"{name}_mod")  # [1,3*hidden]
            inits.append(_i64_init(f"{name}_mod_split", [hidden, hidden, hidden]))
            nodes.append(
                oh.make_node(
                    "Split",
                    [f"{name}_mod", f"{name}_mod_split"],
                    [f"{name}_scale", f"{name}_shift", f"{name}_gate_2d"],
                    axis=-1,
                )
            )
            nodes.append(oh.make_node("Unsqueeze", [f"{name}_scale", "_axes_1"], [f"{name}_scale_3d"]))
            nodes.append(oh.make_node("Unsqueeze", [f"{name}_shift", "_axes_1"], [f"{name}_shift_3d"]))
            nodes.append(oh.make_node("Unsqueeze", [f"{name}_gate_2d", "_axes_1"], [f"{name}_gate"]))
            nodes.append(oh.make_node("Add", [f"{name}_scale_3d", "one_bf16"], [f"{name}_scale1"]))
            nodes.append(oh.make_node("Mul", [f"{name}_normed", f"{name}_scale1"], [f"{name}_scaled"]))
            nodes.append(oh.make_node("Add", [f"{name}_scaled", f"{name}_shift_3d"], [out]))
            return f"{name}_gate"
        # Vanilla Gemma RMSNorm: normed * (1 + weight).
        w = sd[norm_key + ".weight"]
        inits.append(_bf16_initializer(f"{name}_g1", w + 1.0))
        nodes.append(oh.make_node("Mul", [f"{name}_normed", f"{name}_g1"], [out]))
        return None

    def _int8_linear(name: str, x: str, w: Any, n_out: int, k_in: int, zeros_name: str, out: str) -> None:
        """Bare INT GEMM: PerRowInt{8,4}LinearResidual with a static zeros residual.
        *name* is the site's key in the quant state."""
        from .quant_state import site_scope

        with site_scope(name):
            _int8_linear_site(name, x, w, n_out, k_in, zeros_name, out)

    def _int8_linear_site(name: str, x: str, w: Any, n_out: int, k_in: int, zeros_name: str, out: str) -> None:
        if int4:
            s_ch = sq_scales[name] if sq_scales is not None else None
            if fwht:
                # The fold itself lives in foldq: it picks the rotation, folds the
                # SmoothQuant scale onto the axis fold_order names, packs for the
                # bit width, and returns the attributes that describe all of it.
                packed_b, scale_b, fold_attrs = foldq.fold_site(
                    w,
                    bits=4,
                    block_size=block_size,
                    s_ch=s_ch,
                    fold_order=fold_order,
                    fwht=True,
                    gptq=(gptq or {}).get(name),
                )
                nodes.append(
                    oh.make_node(
                        "PerRowInt4LinearResidual",
                        [x, zeros_name],
                        [out],
                        name=f"{name}_plr4",
                        domain=_PLUGIN_DOMAIN,
                        plugin_namespace=_PLUGIN_NAMESPACE,
                        plugin_version=_PLUGIN_VERSION,
                        N=int(n_out),
                        K=int(k_in),
                        block_size=int(block_size),
                        weight_i4=packed_b,
                        weight_scale=scale_b,
                        **fold_attrs,
                    )
                )
                return
            # Dense (legacy) rotation, same single source: foldq returns the perm
            # and the SmoothQuant-folded matrix as node attributes.
            packed_b, scale_b, fold_attrs = foldq.fold_site(
                w, bits=4, block_size=block_size, s_ch=s_ch, fold_order=fold_order, fwht=False
            )
            nodes.append(
                oh.make_node(
                    "PerRowInt4LinearResidual",
                    [x, zeros_name],
                    [out],
                    name=f"{name}_plr4",
                    domain=_PLUGIN_DOMAIN,
                    plugin_namespace=_PLUGIN_NAMESPACE,
                    plugin_version=_PLUGIN_VERSION,
                    N=int(n_out),
                    K=int(k_in),
                    block_size=int(block_size),
                    weight_i4=packed_b,
                    weight_scale=scale_b,
                    **fold_attrs,
                )
            )
            return
        # INT8 with the same fold as the INT4 twin when calibration is available:
        # rotate the weight, absorb SmoothQuant on the matching axis, and hand the
        # kernel the vector a fixed butterfly cannot absorb. Without scales this
        # stays the naive per-row path the unfolded arms use.
        w_i8_b, s_b, extra8 = foldq.fold_site(
            w,
            bits=8,
            block_size=block_size,
            s_ch=sq_scales[name] if sq_scales is not None else None,
            fold_order=fold_order,
            fwht=True,
        )
        nodes.append(
            oh.make_node(
                "PerRowInt8LinearResidual",
                [x, zeros_name],
                [out],
                name=f"{name}_plr",
                domain=_PLUGIN_DOMAIN,
                plugin_namespace=_PLUGIN_NAMESPACE,
                plugin_version=_PLUGIN_VERSION,
                N=int(n_out),
                K=int(k_in),
                weight_i8=w_i8_b,
                weight_scale=s_b,
                **extra8,
            )
        )

    def _gated_add(name: str, x: str, y: str, gate: "str | None", out: str) -> None:
        if gate is None:
            nodes.append(oh.make_node("Add", [x, y], [out]))
        else:
            nodes.append(oh.make_node("Mul", [y, gate], [f"{name}_gy"]))
            nodes.append(oh.make_node("Add", [x, f"{name}_gy"], [out]))

    # Static zeros residuals for the bare GEMMs (suffix length is fixed).
    for zname, n_out in (("z_qkv", q_dim + 2 * kv_dim), ("z_o", hidden), ("z_gu", 2 * ff), ("z_dn", hidden)):
        inits.append(_bf16_initializer(zname, torch.zeros(1, suffix_len, n_out)))

    cur = suffix_name
    for i in range(num_layers):
        b = f"G{i}"
        p = f"{ep}layers.{i}."

        n1_out = f"{b}_n1"
        gate1 = _adarms(f"{b}_ln1", cur, p + "input_layernorm", n1_out)

        w_q = sd[p + "self_attn.q_proj.weight"]
        w_k = sd[p + "self_attn.k_proj.weight"]
        w_v = sd[p + "self_attn.v_proj.weight"]
        w_qkv = torch.cat([w_q, w_k, w_v], dim=0)
        _int8_linear(f"{b}_qkv", n1_out, w_qkv, q_dim + 2 * kv_dim, hidden, "z_qkv", f"{b}_qkv_o")
        inits.append(_i64_init(f"{b}_qkv_split", [q_dim, kv_dim, kv_dim]))
        nodes.append(
            oh.make_node("Split", [f"{b}_qkv_o", f"{b}_qkv_split"], [f"{b}_qf", f"{b}_kf", f"{b}_vf"], axis=-1)
        )
        inits.append(_i64_init(f"{b}_q_shape", [1, suffix_len, h, d]))
        inits.append(_i64_init(f"{b}_kv_shape", [1, suffix_len, hkv, d]))
        nodes.append(oh.make_node("Reshape", [f"{b}_qf", f"{b}_q_shape"], [f"{b}_q4"], allowzero=0))
        nodes.append(oh.make_node("Reshape", [f"{b}_kf", f"{b}_kv_shape"], [f"{b}_k4"], allowzero=0))
        nodes.append(oh.make_node("Reshape", [f"{b}_vf", f"{b}_kv_shape"], [f"{b}_v4"], allowzero=0))
        nodes.append(oh.make_node("Transpose", [f"{b}_q4"], [f"{b}_qt"], perm=[0, 2, 1, 3]))
        nodes.append(oh.make_node("Transpose", [f"{b}_k4"], [f"{b}_kt"], perm=[0, 2, 1, 3]))
        nodes.append(oh.make_node("Transpose", [f"{b}_v4"], [f"{b}_vt"], perm=[0, 2, 1, 3]))

        def _rope(qk_in: str, qk_out: str) -> None:
            nodes.append(oh.make_node("Split", [qk_in, "split_half_d"], [f"{qk_in}_a", f"{qk_in}_b"], axis=-1))
            nodes.append(oh.make_node("Neg", [f"{qk_in}_b"], [f"{qk_in}_bn"]))
            nodes.append(oh.make_node("Concat", [f"{qk_in}_bn", f"{qk_in}_a"], [f"{qk_in}_rot"], axis=-1))
            nodes.append(oh.make_node("Mul", [qk_in, "cos_abs"], [f"{qk_in}_c"]))
            nodes.append(oh.make_node("Mul", [f"{qk_in}_rot", "sin_abs"], [f"{qk_in}_s"]))
            nodes.append(oh.make_node("Add", [f"{qk_in}_c", f"{qk_in}_s"], [qk_out]))

        _rope(f"{b}_qt", f"{b}_qr")
        _rope(f"{b}_kt", f"{b}_kr")

        # Cached prefix K/V (raw, [1, HKV, S, D]) ++ fresh suffix K/V along S.
        inits.append(
            oh.make_tensor(f"{b}_gidx", onnx.TensorProto.INT64, [], np.array(i, dtype=np.int64).tobytes(), raw=True)
        )
        nodes.append(oh.make_node("Gather", ["kv_stack", f"{b}_gidx"], [f"{b}_kv_i"], axis=0))  # [2,1,HKV,S,D]
        nodes.append(oh.make_node("Gather", [f"{b}_kv_i", "_scalar_0"], [f"{b}_kc"], axis=0))  # [1,HKV,S,D]
        nodes.append(oh.make_node("Gather", [f"{b}_kv_i", "_scalar_1"], [f"{b}_vc"], axis=0))
        nodes.append(oh.make_node("Concat", [f"{b}_kc", f"{b}_kr"], [f"{b}_k_all"], axis=2))
        nodes.append(oh.make_node("Concat", [f"{b}_vc", f"{b}_vt"], [f"{b}_v_all"], axis=2))

        # repeat_kv 1 -> 8 (MQA).
        def _repeat_kv(in_name: str, out_name: str) -> None:
            nodes.append(oh.make_node("Unsqueeze", [in_name, "_axes_2"], [f"{out_name}_u"]))
            inits.append(_i64_init(f"{out_name}_reps", [1, 1, kv_groups, 1, 1]))
            nodes.append(oh.make_node("Tile", [f"{out_name}_u", f"{out_name}_reps"], [f"{out_name}_t"]))
            inits.append(_i64_init(f"{out_name}_shape", [1, h, -1, d]))
            nodes.append(oh.make_node("Reshape", [f"{out_name}_t", f"{out_name}_shape"], [out_name], allowzero=1))

        _repeat_kv(f"{b}_k_all", f"{b}_k_full")
        _repeat_kv(f"{b}_v_all", f"{b}_v_full")

        # SDPA
        nodes.append(oh.make_node("Transpose", [f"{b}_k_full"], [f"{b}_kT"], perm=[0, 1, 3, 2]))
        nodes.append(oh.make_node("MatMul", [f"{b}_qr", f"{b}_kT"], [f"{b}_qk"]))
        nodes.append(oh.make_node("Mul", [f"{b}_qk", "attn_scale"], [f"{b}_qks"]))
        nodes.append(oh.make_node("Add", [f"{b}_qks", "full_mask"], [f"{b}_qkm"]))
        nodes.append(oh.make_node("Softmax", [f"{b}_qkm"], [f"{b}_aw"], axis=-1))
        nodes.append(oh.make_node("MatMul", [f"{b}_aw", f"{b}_v_full"], [f"{b}_a4"]))
        nodes.append(oh.make_node("Transpose", [f"{b}_a4"], [f"{b}_ap"], perm=[0, 2, 1, 3]))
        inits.append(_i64_init(f"{b}_a_flat_shape", [1, suffix_len, q_dim]))
        nodes.append(oh.make_node("Reshape", [f"{b}_ap", f"{b}_a_flat_shape"], [f"{b}_a_flat"], allowzero=0))

        # o_proj (bare INT8) then the gated residual OUTSIDE the plugin.
        _int8_linear(f"{b}_o", f"{b}_a_flat", sd[p + "self_attn.o_proj.weight"], hidden, q_dim, "z_o", f"{b}_o_out")
        _gated_add(f"{b}_res1", cur, f"{b}_o_out", gate1, f"{b}_post_attn")

        n2_out = f"{b}_n2"
        gate2 = _adarms(f"{b}_ln2", f"{b}_post_attn", p + "post_attention_layernorm", n2_out)
        w_gu = torch.cat([sd[p + "mlp.gate_proj.weight"], sd[p + "mlp.up_proj.weight"]], dim=0)
        _int8_linear(f"{b}_gu", n2_out, w_gu, 2 * ff, hidden, "z_gu", f"{b}_gu_o")
        inits.append(_i64_init(f"{b}_gu_split", [ff, ff]))
        nodes.append(oh.make_node("Split", [f"{b}_gu_o", f"{b}_gu_split"], [f"{b}_gate_p", f"{b}_up"], axis=-1))
        _emit_gelu_tanh(nodes, inits, f"{b}_gelu", f"{b}_gate_p", f"{b}_gact")
        nodes.append(oh.make_node("Mul", [f"{b}_gact", f"{b}_up"], [f"{b}_ff_mid"]))
        _int8_linear(f"{b}_dn", f"{b}_ff_mid", sd[p + "mlp.down_proj.weight"], hidden, ff, "z_dn", f"{b}_dn_out")
        _gated_add(f"{b}_res2", f"{b}_post_attn", f"{b}_dn_out", gate2, f"{b}_out")
        cur = f"{b}_out"

    # final norm (AdaRMS with cond / vanilla) + action_out_proj
    _adarms("final_norm", cur, ep + "norm", "suffix_normed")
    if not use_adarms:
        # Pi0: drop the state token; velocity reads the last H tokens.
        inits.append(_i64_init("_slice_start", [1]))
        inits.append(_i64_init("_slice_end", [suffix_len]))
        inits.append(_i64_init("_slice_axes", [1]))
        nodes.append(
            oh.make_node("Slice", ["suffix_normed", "_slice_start", "_slice_end", "_slice_axes"], ["suffix_actions"])
        )
        suffix_out = "suffix_actions"
    else:
        suffix_out = "suffix_normed"
    # Runtime: action_out_proj runs at the model precision (bf16), then the
    # velocity is cast to fp32 for the Euler step.
    _linear("act_out", suffix_out, "action_out_proj.weight", "velocity_bf16")
    nodes.append(oh.make_node("Cast", ["velocity_bf16"], ["velocity"], to=onnx.TensorProto.FLOAT))

    graph = oh.make_graph(nodes, "gemma_expert_int8_per_row", graph_inputs, [y_out], initializer=inits)
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 17), oh.make_opsetid(_PLUGIN_DOMAIN, 1)])
    model.ir_version = 9
    if out_path is None:
        return model
    from .onnx_io import save_plugin_onnx

    path = Path(out_path)
    save_plugin_onnx(model, path)
    c = Counter(n.op_type for n in nodes)
    logger.info(
        "Built Gemma expert %s per-row plugin graph (%s): %d layers, %d nodes, ops=%s -> %s",
        "INT4" if int4 else "INT8",
        "pi05/adarms" if use_adarms else "pi0/vanilla",
        num_layers,
        len(nodes),
        dict(sorted(c.items(), key=lambda x: -x[1])),
        path,
    )
    return path


def compute_gemma_expert_sq_scales(
    action_expert: Any,
    forward_loop: Callable[[Any], None],
    *,
    block_size: int = 64,
    fwht: bool = False,
    fold_order: str = "after",
    alpha: float = 1.0,
    gptq_scales: "dict | None" = None,
) -> Dict[str, Any]:
    """Rotated-activation amax for every Gemma-expert INT4 GEMM site.

    The Gemma norms return ``(normed, gate)`` tuples; the hook reads element 0.

    Args:
        fwht: Calibrate against the fixed Hadamard butterfly instead of the
            learned dense rotation. The scale is the amax of the *rotated*
            activation, so it is only valid for the rotation it was measured
            under. Mixing the two silently mis-scales every site.
        fold_order: ``"after"`` measures the amax in the ROTATED frame (the fold
            divides the rotation's output axis); ``"before"`` measures it in the
            RAW frame (SmoothRot order, dividing the input axis). The two are not
            interchangeable: a scale measured in the wrong frame mis-scales
            every channel with no error.
            gptq_scales: when given, this call switches from measuring SCALES to
            measuring each site's GPTQ Hessian, through the SAME taps. Pass the
            scales this capture returned on its first pass: the Hessian must be
            built on the activation the STORED weight multiplies (rotated, and
            divided by that vector), so it needs a second replay after the scales
            exist. Returns ``{site: (K, K) float64}`` in that mode.
    """
    if fold_order not in ("before", "after"):
        raise ValueError(f"fold_order must be 'before' or 'after', got {fold_order!r}")
    torch = _torch()
    expert = action_expert.eval()
    sd = {k: v.detach() for k, v in expert.state_dict().items()}
    layers = expert.expert_model.model.layers

    # The weight that reads each group's input channel, for the SmoothQuant
    # term. Padded exactly like the rotation: the scale is per PADDED channel.
    group_w: Dict[str, Any] = {}
    rot: Dict[str, tuple] = {}
    for i, _ in enumerate(layers):
        p = f"expert_model.model.layers.{i}."
        b = f"G{i}"
        w_qkv = torch.cat(
            [sd[p + "self_attn.q_proj.weight"], sd[p + "self_attn.k_proj.weight"], sd[p + "self_attn.v_proj.weight"]],
            dim=0,
        )
        w_gu = torch.cat([sd[p + "mlp.gate_proj.weight"], sd[p + "mlp.up_proj.weight"]], dim=0)
        w_o = sd[p + "self_attn.o_proj.weight"]
        w_dn = sd[p + "mlp.down_proj.weight"]
        for key, wt in ((f"{b}_qkv", w_qkv), (f"{b}_o", w_o), (f"{b}_gu", w_gu), (f"{b}_dn", w_dn)):
            group_w[key] = rotations.pad_in_dim(wt, block_size)
            rot[key] = foldq.site_rotation(group_w[key], block_size, fwht)

    # One set of taps, two accumulators: amax on the first pass, the GPTQ
    # Hessian on the second. Duplicating the hook wiring is how the two passes
    # would drift on which tensor feeds which site.
    if gptq_scales is not None:
        amax, accum = foldq.hessian_accumulator(rot, gptq_scales, fold_order)
    else:
        amax, accum = foldq.scale_accumulator(rot, block_size, fold_order)

    def _norm_out(out: Any) -> Any:
        return out[0] if isinstance(out, tuple) else out

    handles = []
    for i, layer in enumerate(layers):
        b = f"G{i}"
        handles.append(
            layer.input_layernorm.register_forward_hook(lambda _m, _i, out, k=f"{b}_qkv": accum(k, _norm_out(out)))
        )
        handles.append(
            layer.post_attention_layernorm.register_forward_hook(
                lambda _m, _i, out, k=f"{b}_gu": accum(k, _norm_out(out))
            )
        )
        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(lambda _m, args, k=f"{b}_o": accum(k, args[0])))
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(lambda _m, args, k=f"{b}_dn": accum(k, args[0])))
    try:
        with torch.inference_mode():
            forward_loop(expert)
    finally:
        for h in handles:
            h.remove()
    logger.info("    Computed Gemma expert SmoothQuant scales for %d sites.", len(amax))
    if gptq_scales is not None:
        return dict(amax)  # Hessians, not scales
    return foldq.finalize_scales(amax, weights=group_w, alpha=alpha)
