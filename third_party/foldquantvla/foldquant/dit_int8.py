# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""Build the GR00T N1.6 DiT INT8 dynamic per-row (v2) custom-plugin ONNX graph.

This constructs an ONNX graph whose transformer blocks are replaced by the
``gr00t::v1`` custom TensorRT plugin nodes (``EncoderPreQuant``,
``FusedSelfAttnFull``, ``FusedCrossAttnFull``, ``FusedFfnBlock``), with INT8
per-row weights baked into each node's ``PluginField`` attributes. Activation
amax is computed at runtime inside each plugin (dynamic mode), so no static
calibration scales are emitted.

Graph IO names (``sa_embs``/``vl_embs``/``timestep``/``image_mask``/
``backbone_attention_mask`` -> ``output``) are the upstream GR00T deployment
export's (``dit_common.DIT_INPUT_NAMES``), so the engine is a drop-in for
``dit_bf16.engine`` in the upstream TensorRT glue.

Weights are read directly from the live DiT submodules (``to_q``/``to_k``/
``to_v``/``to_out``, ``ff.net[0].proj``/``ff.net[2]``, ``norm1.linear``,
``timestep_encoder``, ``proj_out_1``/``proj_out_2``). The output head
(``norm_out`` + ``proj_out_1`` + ``proj_out_2``) stays BF16.

The precision-agnostic scaffolding (graph IO, timestep encoder, mask routing,
output head, live-weight accessors) lives in :mod:`dit_common`. This module
constructs ONNX only: it does not import ``tensorrt``, load any ``.so``, or
run TensorRT, and it does not import from ``foldquant.runtime``.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnx.helper as oh

from . import foldq
from .dit_common import (
    ATTEND_ALL_MASK,
    emit_attend_all_mask,
    resolve_attend_n,
)
from .dit_common import (
    DIT_INPUT_NAMES as _DIT_INPUT_NAMES,
)
from .dit_common import (
    DIT_OUTPUT_NAME as _DIT_OUTPUT_NAME,
)
from .dit_common import (
    DIT_BATCH_DIM as _DIT_BATCH_DIM,
    DIT_SA_SEQ_DIM as _DIT_SA_SEQ_DIM,
)
from .dit_common import (
    DIT_VL_SEQ_DIM as _DIT_VL_SEQ_DIM,
)
from .dit_common import (
    EPS as _EPS,
)
from .dit_common import (
    PLUGIN_DOMAIN as _PLUGIN_DOMAIN,
)
from .dit_common import (
    PLUGIN_NAMESPACE as _PLUGIN_NAMESPACE,
)
from .dit_common import (
    PLUGIN_VERSION as _PLUGIN_VERSION,
)
from .dit_common import (
    DiTWeights as _DiTWeights,
)
from .dit_common import (
    bf16_initializer as _bf16_initializer,
)
from .dit_common import (
    bias_f32 as _bias_f32,
)
from .dit_common import (
    emit_mask_routing as _emit_mask_routing,
)
from .dit_common import (
    emit_output_head as _emit_output_head,
)
from .dit_common import (
    emit_timestep_encoding as _emit_timestep_encoding,
)
from .dit_common import (
    to_bytes_f32 as _to_bytes_f32,
)
from .onnx_io import save_plugin_onnx
from .quant_state import site_scope

logger = logging.getLogger(__name__)

__all__ = ["build_dit_plugin_onnx"]


# Per-block AdaLN scale/shift emitter


def _emit_adaln(w: _DiTWeights, idx: int, nodes: list, inits: list) -> None:
    """SiLU(temb) -> AdaLN.linear -> split into per-block (scale, shift) BF16."""
    wL, bL = w.adaln(idx)
    b = f"block{idx}"
    inits.append(_bf16_initializer(f"{b}_adaln_w", wL.t().contiguous()))
    inits.append(_bf16_initializer(f"{b}_adaln_b", bL))
    nodes.append(oh.make_node("Sigmoid", ["temb"], [f"{b}_sig_temb"]))
    nodes.append(oh.make_node("Mul", ["temb", f"{b}_sig_temb"], [f"{b}_silu_temb"]))
    nodes.append(oh.make_node("MatMul", [f"{b}_silu_temb", f"{b}_adaln_w"], [f"{b}_lin_pre"]))
    nodes.append(oh.make_node("Add", [f"{b}_lin_pre", f"{b}_adaln_b"], [f"{b}_lin"]))
    inits.append(
        oh.make_tensor(
            f"{b}_split", onnx.TensorProto.INT64, [2], np.array([w.dim, w.dim], dtype=np.int64).tobytes(), raw=True
        )
    )
    nodes.append(oh.make_node("Split", [f"{b}_lin", f"{b}_split"], [f"{b}_scale", f"{b}_shift"], axis=-1))


# Graph builder


def _build_v2_dynamic_graph(
    w: _DiTWeights,
    *,
    sq_scales: "dict | None" = None,
    block_size: int = 64,
    sq_fold_order: str = "after",
    fwht: bool = False,
) -> onnx.GraphProto:
    """Assemble the full v2 dynamic-per-row plugin graph from the live DiT.

    With ``sq_scales=None`` this is the unfolded graph this module always built,
    node for node. With scales it folds through the same :mod:`foldq` calls the
    INT4 emitter uses. Bit width is the only difference between the two.
    """
    dim = w.dim
    num_heads = w.num_heads
    head_dim = w.head_dim
    ff_inner = w.ff_inner
    kv_dim = w.kv_dim
    output_dim = w.output_dim
    attend_n = w.attend_text_every_n_blocks

    inits: list = []
    nodes: list = []

    import torch  # concatenating weight rows before the fold, as the INT4 emitter does

    def _scale(key: str) -> Any:
        return sq_scales[key] if sq_scales is not None else None

    def _macro(site: str, weight: Any, s_ch: Any, rotation: Any = None) -> tuple:
        """One site of a fused macro node -> ``(i8_bytes, scale_bytes, perm, R_use)``.

        The same shared fold the INT4 emitter uses; bit width is the only
        difference, which is why it is a parameter and not a second code path.
        *site* names the pack in the quant state.
        """
        with site_scope(site):
            return _macro_site(weight, s_ch, rotation)

    def _macro_site(weight: Any, s_ch: Any, rotation: Any = None) -> tuple:
        if sq_scales is None:
            # Unfolded W8A8: these nodes carry no rotation slot, so the weight has to
            # be packed in its own frame. Packing it under a dense rotation (what
            # fold_macro_site does for any supplied rotation) builds an engine that
            # runs and returns unrelated actions (action cosine ~0.27 on N1.7).
            i8, sc, _ = foldq.fold_site(weight, bits=8, block_size=block_size, s_ch=None, fwht=False)
            return i8, sc, None, None
        perm, R = rotation if rotation is not None else foldq.site_rotation(weight, block_size, fwht)
        i8, sc, r_use = foldq.fold_macro_site(weight, perm, R, block_size, s_ch, fold_order=sq_fold_order, bits=8)
        return i8, sc, perm, r_use

    def _pre_vec(field: str, s_ch: Any) -> dict:
        """The raw-frame vector a butterfly site ships instead of a folded matrix."""
        if not fwht or s_ch is None:
            return {}
        return {field: _to_bytes_f32(s_ch.detach().float().cpu().numpy())}

    def _rot_attr() -> dict:
        return {"rot_block_size": int(block_size)} if (fwht and sq_scales is not None) else {}

    # Graph inputs / output
    #
    # Batch is pinned to 1 - the fused attention plugins' cuBLAS strided-batched
    # addressing is only correct at B == 1.
    x_name, e_name, ts_name, im_name, bam_name = _DIT_INPUT_NAMES
    x_t = oh.make_tensor_value_info(x_name, onnx.TensorProto.BFLOAT16, [_DIT_BATCH_DIM, _DIT_SA_SEQ_DIM, dim])
    e_t = oh.make_tensor_value_info(e_name, onnx.TensorProto.BFLOAT16, [_DIT_BATCH_DIM, _DIT_VL_SEQ_DIM, kv_dim])
    ts_t = oh.make_tensor_value_info(ts_name, onnx.TensorProto.INT64, [_DIT_BATCH_DIM])
    # The two boolean masks are the upstream export's; the DiT's own
    # ``attention_mask``/``encoder_attention_mask`` kwargs are not graph inputs -
    # see dit_common.emit_mask_routing()'s docstring for why that is correct here.
    im_t = oh.make_tensor_value_info(im_name, onnx.TensorProto.BOOL, [_DIT_BATCH_DIM, _DIT_VL_SEQ_DIM])
    bam_t = oh.make_tensor_value_info(bam_name, onnx.TensorProto.BOOL, [_DIT_BATCH_DIM, _DIT_VL_SEQ_DIM])
    y_t = oh.make_tensor_value_info(_DIT_OUTPUT_NAME, onnx.TensorProto.BFLOAT16, [_DIT_BATCH_DIM, _DIT_SA_SEQ_DIM, output_dim])

    _emit_timestep_encoding(w, nodes, inits)
    _emit_mask_routing(nodes, inits)
    # A plain DiT routes neither half; it needs the attend-everything mask instead.
    if attend_n is None:
        emit_attend_all_mask(nodes, inits)

    # EncoderPreQuant once -> (encoder_i8 INT32-packed, encoder_scale FP32);
    # all cross-attn blocks share one INT8 quant of the encoder. Dynamic mode =>
    # no `static_act_scale_enc` field (plugin computes encoder amax at runtime).
    nodes.append(
        oh.make_node(
            "EncoderPreQuant",
            [e_name],
            ["encoder_i8", "encoder_scale"],
            name="encoder_prequant",
            domain=_PLUGIN_DOMAIN,
            plugin_namespace=_PLUGIN_NAMESPACE,
            plugin_version=_PLUGIN_VERSION,
            **_rot_attr(),
            **_pre_vec("act_scale_pre_enc", _scale("encoder")),
            K_enc=int(kv_dim),
        )
    )

    cur_x = x_name
    for idx in range(w.num_blocks):
        b = f"block{idx}"
        attn_kind = "self" if (idx % 2 == 1) else "cross"

        _emit_adaln(w, idx, nodes, inits)

        # QKV weights + bias (per-row INT8 weight quant; activation amax dynamic).
        (wQ, bQ), (wK, bK), (wV, bV) = w.qkv(idx)
        bQf, bKf, bVf = _bias_f32(bQ), _bias_f32(bK), _bias_f32(bV)
        wO, bO = w.out_proj(idx)
        bOf = _bias_f32(bO)

        # The scale keys are the DiT capture's, shared with the INT4 emitter:
        # a self block merges Q/K/V into one group, a cross block keeps Q alone
        # (its K/V read the encoder, under the shared encoder rotation).
        s_in = _scale(f"{b}_qkv") if attn_kind == "self" else _scale(f"{b}_q")
        s_o = _scale(f"{b}_o")

        # Cross-attention mask. An alternating DiT switches text/image on its own
        # schedule; a plain DiT has no split, so every cross block attends the whole
        # encoder sequence (ATTEND_ALL_MASK) exactly as its unmasked forward does.
        if attn_kind != "cross":
            attn_mask_name = None
        elif attend_n is None:
            attn_mask_name = ATTEND_ALL_MASK
        else:
            attn_mask_name = "non_img_mask_add" if (idx % (2 * attend_n) == 0) else "img_mask_add"

        if attn_kind == "self":
            qkv_i8, sQKV_b, _, _ = _macro(f"{b}_qkv", torch.cat([wQ, wK, wV], dim=0), s_in)
            o_i8, sO_b, _, _ = _macro(f"{b}_o", wO, s_o)
            bQKV = np.concatenate([bQf, bKf, bVf])
            nodes.append(
                oh.make_node(
                    "FusedSelfAttnFull",
                    [cur_x, f"{b}_scale", f"{b}_shift"],
                    [f"{b}_post_attn"],
                    name=f"{b}_selfattn_full",
                    domain=_PLUGIN_DOMAIN,
                    plugin_namespace=_PLUGIN_NAMESPACE,
                    plugin_version=_PLUGIN_VERSION,
                    inner_dim=int(dim),
                    K=int(dim),
                    num_heads=int(num_heads),
                    head_dim=int(head_dim),
                    eps=_EPS,
                    **_rot_attr(),
                    **_pre_vec("act_scale_pre_in", s_in),
                    **_pre_vec("act_scale_pre_o", s_o),
                    weight_qkv_i8=qkv_i8,
                    weight_qkv_scale=sQKV_b,
                    bias_qkv=_to_bytes_f32(bQKV),
                    weight_o_i8=o_i8,
                    weight_o_scale=sO_b,
                    bias_o=_to_bytes_f32(bOf),
                )
            )
        else:
            assert attn_mask_name is not None  # cross-attn always routes a mask
            # Q reads the block's own X; K/V read the encoder, so they fold under
            # the SHARED encoder rotation, the same pairing the INT4 emitter uses.
            q_i8, sQ_b, _, _ = _macro(f"{b}_q", wQ, s_in)
            o_i8, sO_b, _, _ = _macro(f"{b}_o", wO, s_o)
            kv_i8, sKV_b, _, _ = _macro(f"{b}_kv", torch.cat([wK, wV], dim=0), _scale("encoder"))
            bKV = np.concatenate([bKf, bVf])
            nodes.append(
                oh.make_node(
                    "FusedCrossAttnFull",
                    [cur_x, f"{b}_scale", f"{b}_shift", "encoder_i8", "encoder_scale", attn_mask_name],
                    [f"{b}_post_attn"],
                    name=f"{b}_crossattn_full",
                    domain=_PLUGIN_DOMAIN,
                    plugin_namespace=_PLUGIN_NAMESPACE,
                    plugin_version=_PLUGIN_VERSION,
                    inner_dim=int(dim),
                    K=int(dim),
                    K_enc=int(wK.shape[1]),
                    num_heads=int(num_heads),
                    head_dim=int(head_dim),
                    eps=_EPS,
                    **_rot_attr(),
                    **_pre_vec("act_scale_pre_in", s_in),
                    **_pre_vec("act_scale_pre_o", s_o),
                    weight_q_i8=q_i8,
                    weight_q_scale=sQ_b,
                    bias_q=_to_bytes_f32(bQf),
                    weight_kv_i8=kv_i8,
                    weight_kv_scale=sKV_b,
                    bias_kv=_to_bytes_f32(bKV),
                    weight_o_i8=o_i8,
                    weight_o_scale=sO_b,
                    bias_o=_to_bytes_f32(bOf),
                )
            )

        # FFN plugin.
        (wP0, bP0), (wP2, bP2) = w.ffn(idx)
        s0 = _scale(f"{b}_ffn0")
        s2 = _scale(f"{b}_ffn2")
        p0_i8, sP0_b, _, _ = _macro(f"{b}_ffn0", wP0, s0)
        p2_i8, sP2_b, _, _ = _macro(f"{b}_ffn2", wP2, s2)
        nodes.append(
            oh.make_node(
                "FusedFfnBlock",
                [f"{b}_post_attn"],
                [f"{b}_out"],
                name=f"{b}_ffn",
                domain=_PLUGIN_DOMAIN,
                plugin_namespace=_PLUGIN_NAMESPACE,
                plugin_version=_PLUGIN_VERSION,
                inner_dim=int(ff_inner),
                K=int(dim),
                eps=_EPS,
                **_rot_attr(),
                **_pre_vec("act_scale_pre0", s0),
                **_pre_vec("act_scale_pre2", s2),
                weight_proj0_i8=p0_i8,
                weight_proj0_scale=sP0_b,
                bias_proj0=_to_bytes_f32(_bias_f32(bP0)),
                weight_proj2_i8=p2_i8,
                weight_proj2_scale=sP2_b,
                bias_proj2=_to_bytes_f32(_bias_f32(bP2)),
            )
        )
        cur_x = f"{b}_out"

    _emit_output_head(w, cur_x, nodes, inits)

    graph = oh.make_graph(
        nodes,
        "groot_dit_int8_per_row_v2_dynamic",
        [x_t, e_t, ts_t, im_t, bam_t],
        [y_t],
        initializer=inits,
    )
    c = Counter(n.op_type for n in nodes)
    logger.info(
        "    Built v2 dynamic plugin graph: nodes=%d ops=%s", len(nodes), dict(sorted(c.items(), key=lambda x: -x[1]))
    )
    return graph


# Public entry point


def build_dit_plugin_onnx(
    dit_module: Any,
    output_path: Any,
    *,
    opset: int = 17,
    sq_scales: "dict | None" = None,
    block_size: int = 64,
    sq_fold_order: str = "after",
    fwht: bool = False,
) -> Path:
    """Construct the v2 dynamic-per-row INT8 plugin ONNX from a live DiT module.

    Args:
        dit_module: The live GR00T N1.6 DiT module
            (``action_expert.model``, an ``AlternateVLDiT`` instance). Read-only;
            weights are extracted, not mutated.
        output_path: Destination ``.onnx`` path. External data is written
            alongside as ``<name>.data``.
        opset: Default ONNX opset for the empty domain; the ``trt.plugins``
            domain is always opset 1.

    Returns:
        The saved ONNX ``Path``.
    """
    attend_n = resolve_attend_n(dit_module)
    w = _DiTWeights(dit_module, attend_n)
    graph = _build_v2_dynamic_graph(
        w, sq_scales=sq_scales, block_size=block_size, sq_fold_order=sq_fold_order, fwht=fwht
    )

    model = oh.make_model(
        graph,
        opset_imports=[oh.make_opsetid("", opset), oh.make_opsetid("trt.plugins", 1)],
    )

    if output_path is None:
        return model
    out_path = Path(output_path)
    save_plugin_onnx(model, out_path)
    logger.info("    Exported v2 dynamic plugin ONNX (5-input/1-output): %s", out_path)
    return out_path
