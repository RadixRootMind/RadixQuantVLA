# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""Build the GR00T DiT FoldQuant W4A4 (v4) custom-plugin ONNX graph.

INT4 analogue of :mod:`dit_int8`. Each transformer block collapses to
the ``gr00t::v1`` W4A4 macro plugins (``EncoderPreQuantInt4``,
``FusedSelfAttnFullInt4``, ``FusedCrossAttnFullInt4``, ``FusedFfnBlockInt4``,
``AdaLNModInt4``): a rotated-INT4 GEMM + BF16-folded rotation + cuBLAS SDPA +
INT4 weight-only AdaLN. The BF16 scaffolding (graph IO, timestep encoder, mask
routing, output head, live-weight accessors) is shared verbatim with the INT8
builder via :mod:`dit_common`; only the quantized macro nodes differ.

The shipped scheme is **SmoothQuant-folded** W4A4 (``w4a4_sr``): per-row
INT4 weights + composite SVD·Hadamard rotation, with a static per-channel activation
scale absorbed into each rotation + weight (:func:`rotation.fold_rotation_sq` /
``pack_int4_colmajor_sq``). The scales come from :func:`compute_dit_sq_scales`, which the
GR00T N1.6 exporter runs in-process off the build's calibration replay; the builder
only bakes what it is handed.

Passing ``sq_scales=None`` still builds the uncalibrated **dynamic** variant
(``w4a4``, activation amax computed per row at runtime). No build path
does (it measurably loses accuracy, see the kernel notes in ``kernels/tensorrt/int4_per_row``), and it is kept
only so the rotation/packing math can be exercised without a calibration set.

AdaLN modulation is INT4 weight-only (``AdaLNModInt4``); with ``adaln_act_bits``
defaulting to 16 the activation stays BF16 (W4A16 AdaLN), matching
``w4a4_sr``.

Constructs ONNX only: no ``tensorrt`` import, no ``.so`` load, no
``foldquant.runtime`` import.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import onnx
import onnx.helper as oh

from . import foldq
from . import rotation as rotations
from .calibrate import dit_accepts_masks, dit_inputs_for
from .dit_common import (
    ATTEND_ALL_MASK,
    DIT_INPUT_NAMES,
    DIT_OUTPUT_NAME,
    DIT_BATCH_DIM,
    DIT_SA_SEQ_DIM,
    DIT_VL_SEQ_DIM,
    EPS,
    PLUGIN_DOMAIN,
    PLUGIN_NAMESPACE,
    PLUGIN_VERSION,
    DiTWeights,
    bias_f32,
    emit_attend_all_mask,
    emit_mask_routing,
    emit_output_head,
    emit_timestep_encoding,
    resolve_attend_n,
    to_bytes_f32,
)
from .onnx_io import save_plugin_onnx
from .quant_state import site_scope

logger = logging.getLogger(__name__)

_DEFAULT_BLOCK_SIZE = 64

# Floor on a SmoothQuant per-channel scale, relative to the largest channel in the
# same group. Bounds how far the fold can amplify a channel that calibration saw as


# Per-weight FoldQuant rotation + INT4 pack


def _cross_kv_weights(w: DiTWeights) -> Any:
    """Stack every cross-attn block's ``[to_k; to_v]`` weight → ``(sum_out, kv_dim)``.

    All cross-attn blocks consume the same VL encoder, so one shared rotation is
    built from their stacked KV weights and reused by ``EncoderPreQuantInt4`` and
    every cross block's KV pack (preserving the once-per-forward encoder quant).
    """
    import torch

    kv_stack = []
    for idx in range(w.num_blocks):
        if idx % 2 == 0:  # cross-attn block
            (_, _), (wK, _), (wV, _) = w.qkv(idx)
            kv_stack.append(wK)
            kv_stack.append(wV)
    return torch.cat(kv_stack, dim=0)


# AdaLN (INT4 weight-only) emitter


def _emit_adaln_int4(w: DiTWeights, idx: int, nodes: list, adaln_act_bits: int) -> None:
    """SiLU(temb) → ``AdaLNModInt4`` (INT4 weight-only GEMV) → split (scale, shift)."""
    wL, bL = w.adaln(idx)
    b = f"block{idx}"
    w_bytes, sc_bytes, in_d, out_d = rotations.adaln_pack_int4(wL)
    nodes.append(oh.make_node("Sigmoid", ["temb"], [f"{b}_sig_temb"]))
    nodes.append(oh.make_node("Mul", ["temb", f"{b}_sig_temb"], [f"{b}_silu_temb"]))
    nodes.append(
        oh.make_node(
            "AdaLNModInt4",
            [f"{b}_silu_temb"],
            [f"{b}_lin"],
            name=f"{b}_adaln_int4",
            domain=PLUGIN_DOMAIN,
            plugin_namespace=PLUGIN_NAMESPACE,
            plugin_version=PLUGIN_VERSION,
            weight_i4=w_bytes,
            weight_scale=sc_bytes,
            bias=rotations.to_bytes_bf16(bL),
            in_dim=int(in_d),
            out_dim=int(out_d),
            act_bits=int(adaln_act_bits),
        )
    )


# Graph builder


def _build_w4a4_graph(
    w: DiTWeights,
    sq_scales: Optional[Dict[str, Any]],
    block_size: int,
    adaln_act_bits: int,
    sq_fold_order: str = "after",
    fwht: bool = False,
    gptq: "dict | None" = None,
) -> onnx.GraphProto:
    """Assemble the full FoldQuant W4A4 (v4) plugin graph from the live DiT."""
    import torch

    dim = w.dim
    num_heads = w.num_heads
    head_dim = w.head_dim
    ff_inner = w.ff_inner
    kv_dim = w.kv_dim
    output_dim = w.output_dim
    attend_n = w.attend_text_every_n_blocks
    is_sq = sq_scales is not None

    def _g(key: str) -> Any:
        """This site's GPTQ factors, or None for round-to-nearest."""
        return (gptq or {}).get(key)

    def _rot_attr() -> Dict[str, Any]:
        """The block width a butterfly node rotates in; absent on the dense arm."""
        return {"rot_block_size": int(block_size)} if fwht else {}

    def _pre_vec(field: str, s_ch: Any) -> Dict[str, Any]:
        """The raw-frame vector a butterfly site ships instead of a folded matrix."""
        if not fwht or s_ch is None:
            return {}
        return {field: to_bytes_f32(s_ch.detach().float().cpu().numpy())}

    inits: list = []
    nodes: list = []

    # Graph inputs / output  (identical IO contract to the INT8 v2 graph)
    #
    # Batch is pinned to 1, not symbolic. The fused attention plugins address Q/K/V
    # with a single cuBLAS strided-batched call whose per-batch stride is only
    # correct at B == 1 (the required offset b*S*3*inner + h*D is not affine in
    # b*H + h, so one strided call cannot express it). A static dim makes the
    # profile min=opt=max=1 and lets TensorRT reject B > 1 up front; the plugins
    # carry a matching runtime guard in enqueue().
    x_name, e_name, ts_name, im_name, bam_name = DIT_INPUT_NAMES
    x_t = oh.make_tensor_value_info(x_name, onnx.TensorProto.BFLOAT16, [DIT_BATCH_DIM, DIT_SA_SEQ_DIM, dim])
    e_t = oh.make_tensor_value_info(e_name, onnx.TensorProto.BFLOAT16, [DIT_BATCH_DIM, DIT_VL_SEQ_DIM, kv_dim])
    ts_t = oh.make_tensor_value_info(ts_name, onnx.TensorProto.INT64, [DIT_BATCH_DIM])
    # The two boolean masks are the upstream export's; the DiT's own
    # ``attention_mask``/``encoder_attention_mask`` kwargs are not graph inputs -
    # see dit_common.emit_mask_routing()'s docstring for why that is correct here.
    im_t = oh.make_tensor_value_info(im_name, onnx.TensorProto.BOOL, [DIT_BATCH_DIM, DIT_VL_SEQ_DIM])
    bam_t = oh.make_tensor_value_info(bam_name, onnx.TensorProto.BOOL, [DIT_BATCH_DIM, DIT_VL_SEQ_DIM])
    y_t = oh.make_tensor_value_info(DIT_OUTPUT_NAME, onnx.TensorProto.BFLOAT16, [DIT_BATCH_DIM, DIT_SA_SEQ_DIM, output_dim])

    emit_timestep_encoding(w, nodes, inits)
    emit_mask_routing(nodes, inits)
    # A plain DiT routes neither half; it needs the attend-everything mask instead.
    if attend_n is None:
        emit_attend_all_mask(nodes, inits)

    # Shared encoder rotation from the stacked cross-attn KV weights, emitted once.
    # EncoderPreQuantInt4 outputs the INT4-rotated encoder shared by all cross
    # blocks. Under SmoothQuant the shared per-channel encoder scale is folded into
    # the encoder rotation (rotation_enc is FP32; the encoder kernel reads FP32).
    kv_stacked = _cross_kv_weights(w)
    assert kv_stacked.shape[1] % block_size == 0, (
        f"kv_dim {kv_stacked.shape[1]} not divisible by block_size {block_size}"
    )
    with site_scope("encoder"):
        enc_perm, enc_R = foldq.site_rotation(kv_stacked, block_size, fwht)
    # The encoder rotation is the one bake site NOT produced by fold_macro_site
    # (EncoderPreQuantInt4 shares it across all cross blocks), so it must fold
    # by the SAME order the KV weights pack with. A mismatch here breaks every
    # cross-attention KV product (measured: dit cosine 0.9996 -> 0.9923).
    enc_R_use = foldq.fold_rotation(
        enc_R, enc_perm, sq_scales["encoder"] if sq_scales is not None else None, sq_fold_order
    )
    nodes.append(
        oh.make_node(
            "EncoderPreQuantInt4",
            [e_name],
            ["encoder_i4", "encoder_scale"],
            name="encoder_prequant_int4",
            domain=PLUGIN_DOMAIN,
            plugin_namespace=PLUGIN_NAMESPACE,
            plugin_version=PLUGIN_VERSION,
            **_rot_attr(),
            **_pre_vec("act_scale_pre_enc", (sq_scales or {}).get("encoder")),
            perm_enc=rotations.to_bytes_i32(enc_perm.detach().cpu().numpy()),
            rotation_enc=(b"" if fwht else to_bytes_f32(enc_R_use.detach().float().cpu().numpy())),
            K_enc=int(kv_dim),
            block_size=int(block_size),
        )
    )

    cur_x = x_name
    for idx in range(w.num_blocks):
        b = f"block{idx}"
        attn_kind = "self" if (idx % 2 == 1) else "cross"

        # AdaLN modulation (INT4 weight-only) → {b}_lin, split into (scale, shift).
        with site_scope(f"{b}_adaln"):
            _emit_adaln_int4(w, idx, nodes, adaln_act_bits)
        inits.append(
            oh.make_tensor(
                f"{b}_split", onnx.TensorProto.INT64, [2], np.array([dim, dim], dtype=np.int64).tobytes(), raw=True
            )
        )
        nodes.append(oh.make_node("Split", [f"{b}_lin", f"{b}_split"], [f"{b}_scale", f"{b}_shift"], axis=-1))

        (wQ, bQ), (wK, bK), (wV, bV) = w.qkv(idx)
        bQf, bKf, bVf = bias_f32(bQ), bias_f32(bK), bias_f32(bV)
        wO, bO = w.out_proj(idx)
        bOf = bias_f32(bO)

        # attn_O rotation (input = inner_dim), shared by self and cross blocks.
        s_o = sq_scales[f"block{idx}_o"] if sq_scales is not None else None
        with site_scope(f"{b}_o"):
            permO, RO = foldq.site_rotation(wO, block_size, fwht)
            o_i4, o_sc, RO_use = foldq.fold_macro_site(
                wO, permO, RO, block_size, s_o, fold_order=sq_fold_order, gptq=_g(f"block{idx}_o")
            )
        perm_o_b = rotations.to_bytes_i32(permO.detach().cpu().numpy())
        rot_o_b = b"" if fwht else rotations.to_bytes_bf16(RO_use)
        # Butterfly sites carry rot_block_size plus the SmoothQuant vector
        # instead of a baked matrix; the scale rides on the raw channel.
        bf_o = _pre_vec("act_scale_pre_o", s_o)

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
            wQKV = torch.cat([wQ, wK, wV], dim=0)  # (3*inner, dim), input = x
            s_qkv = sq_scales[f"block{idx}_qkv"] if sq_scales is not None else None
            with site_scope(f"{b}_qkv"):
                permQKV, RQKV = foldq.site_rotation(wQKV, block_size, fwht)
                qkv_i4, qkv_sc, RQKV_use = foldq.fold_macro_site(
                    wQKV,
                    permQKV,
                    RQKV,
                    block_size,
                    s_qkv,
                    fold_order=sq_fold_order,
                    gptq=_g(f"block{idx}_qkv"),
                )
            bf_qkv = _pre_vec("act_scale_pre_in", s_qkv)
            bQKV = np.concatenate([bQf, bKf, bVf])
            nodes.append(
                oh.make_node(
                    "FusedSelfAttnFullInt4",
                    [cur_x, f"{b}_scale", f"{b}_shift"],
                    [f"{b}_post_attn"],
                    name=f"{b}_selfattn_int4",
                    domain=PLUGIN_DOMAIN,
                    plugin_namespace=PLUGIN_NAMESPACE,
                    plugin_version=PLUGIN_VERSION,
                    **_rot_attr(),
                    inner_dim=int(dim),
                    K=int(dim),
                    num_heads=int(num_heads),
                    head_dim=int(head_dim),
                    block_size=int(block_size),
                    eps=EPS,
                    weight_qkv_i4=qkv_i4,
                    weight_qkv_scale=qkv_sc,
                    bias_qkv=to_bytes_f32(bQKV),
                    weight_o_i4=o_i4,
                    weight_o_scale=o_sc,
                    bias_o=to_bytes_f32(bOf),
                    perm_qkv=rotations.to_bytes_i32(permQKV.detach().cpu().numpy()),
                    rotation_qkv=(b"" if fwht else rotations.to_bytes_bf16(RQKV_use)),
                    **bf_qkv,
                    perm_o=perm_o_b,
                    rotation_o=rot_o_b,
                    **bf_o,
                )
            )
        else:
            assert attn_mask_name is not None  # cross-attn always routes a mask
            s_q = sq_scales[f"block{idx}_q"] if sq_scales is not None else None
            with site_scope(f"{b}_q"):
                permQ, RQ = foldq.site_rotation(wQ, block_size, fwht)  # cross-attn Q input = x
                q_i4, q_sc, RQ_use = foldq.fold_macro_site(
                    wQ, permQ, RQ, block_size, s_q, fold_order=sq_fold_order, gptq=_g(f"block{idx}_q")
                )
            bf_q = _pre_vec("act_scale_pre_in", s_q)
            # KV weight rotated by the SHARED encoder rotation (enc_perm/enc_R); under
            # SQ the same shared encoder per-channel scale is folded into the KV weight.
            wKV = torch.cat([wK, wV], dim=0)  # (2*inner, kv_dim)
            s_enc = sq_scales["encoder"] if sq_scales is not None else None
            with site_scope(f"{b}_kv"):
                kv_i4, kv_sc, _ = foldq.fold_macro_site(
                    wKV, enc_perm, enc_R, block_size, s_enc, fold_order=sq_fold_order, gptq=_g("encoder")
                )
            bKV = np.concatenate([bKf, bVf])
            nodes.append(
                oh.make_node(
                    "FusedCrossAttnFullInt4",
                    [cur_x, f"{b}_scale", f"{b}_shift", "encoder_i4", "encoder_scale", attn_mask_name],
                    [f"{b}_post_attn"],
                    name=f"{b}_crossattn_int4",
                    domain=PLUGIN_DOMAIN,
                    plugin_namespace=PLUGIN_NAMESPACE,
                    plugin_version=PLUGIN_VERSION,
                    **_rot_attr(),
                    inner_dim=int(dim),
                    K=int(dim),
                    K_enc=int(wK.shape[1]),
                    num_heads=int(num_heads),
                    head_dim=int(head_dim),
                    block_size=int(block_size),
                    eps=EPS,
                    weight_q_i4=q_i4,
                    weight_q_scale=q_sc,
                    bias_q=to_bytes_f32(bQf),
                    weight_kv_i4=kv_i4,
                    weight_kv_scale=kv_sc,
                    bias_kv=to_bytes_f32(bKV),
                    weight_o_i4=o_i4,
                    weight_o_scale=o_sc,
                    bias_o=to_bytes_f32(bOf),
                    perm_q=rotations.to_bytes_i32(permQ.detach().cpu().numpy()),
                    rotation_q=(b"" if fwht else rotations.to_bytes_bf16(RQ_use)),
                    **bf_q,
                    perm_o=perm_o_b,
                    rotation_o=rot_o_b,
                    **bf_o,
                    emit_kv=0,
                )
            )

        # FFN plugin: proj0 input = dim (post-LN), proj2 input = ff_inner (post-GELU).
        (wP0, bP0), (wP2, bP2) = w.ffn(idx)
        s0 = sq_scales[f"block{idx}_ffn0"] if sq_scales is not None else None
        s2 = sq_scales[f"block{idx}_ffn2"] if sq_scales is not None else None
        with site_scope(f"{b}_ffn0"):
            perm0, R0 = foldq.site_rotation(wP0, block_size, fwht)
            p0_i4, p0_sc, R0_use = foldq.fold_macro_site(
                wP0, perm0, R0, block_size, s0, fold_order=sq_fold_order, gptq=_g(f"block{idx}_ffn0")
            )
        with site_scope(f"{b}_ffn2"):
            perm2, R2 = foldq.site_rotation(wP2, block_size, fwht)
            p2_i4, p2_sc, R2_use = foldq.fold_macro_site(
                wP2, perm2, R2, block_size, s2, fold_order=sq_fold_order, gptq=_g(f"block{idx}_ffn2")
            )
        bf_2 = _pre_vec("act_scale_pre2", s2)
        bf_0 = _pre_vec("act_scale_pre0", s0)
        nodes.append(
            oh.make_node(
                "FusedFfnBlockInt4",
                [f"{b}_post_attn"],
                [f"{b}_out"],
                name=f"{b}_ffn_int4",
                domain=PLUGIN_DOMAIN,
                plugin_namespace=PLUGIN_NAMESPACE,
                plugin_version=PLUGIN_VERSION,
                **_rot_attr(),
                inner_dim=int(ff_inner),
                K=int(dim),
                block_size=int(block_size),
                eps=EPS,
                weight_proj0_i4=p0_i4,
                weight_proj0_scale=p0_sc,
                bias_proj0=to_bytes_f32(bias_f32(bP0)),
                weight_proj2_i4=p2_i4,
                weight_proj2_scale=p2_sc,
                bias_proj2=to_bytes_f32(bias_f32(bP2)),
                perm0=rotations.to_bytes_i32(perm0.detach().cpu().numpy()),
                rotation0=(b"" if fwht else rotations.to_bytes_bf16(R0_use)),
                **bf_0,
                perm2=rotations.to_bytes_i32(perm2.detach().cpu().numpy()),
                rotation2=(b"" if fwht else rotations.to_bytes_bf16(R2_use)),
                **bf_2,
            )
        )
        cur_x = f"{b}_out"

    emit_output_head(w, cur_x, nodes, inits)

    graph = oh.make_graph(
        nodes,
        "groot_dit_foldquant_w4a4",
        [x_t, e_t, ts_t, im_t, bam_t],
        [y_t],
        initializer=inits,
    )
    c = Counter(n.op_type for n in nodes)
    logger.info(
        "    Built W4A4 (%s) plugin graph: nodes=%d ops=%s",
        "sq" if is_sq else "dynamic",
        len(nodes),
        dict(sorted(c.items(), key=lambda x: -x[1])),
    )
    return graph


# SmoothQuant calibration (offline; produces the sq_scales dict)


def compute_dit_sq_scales(
    dit_module: Any,
    calib_inputs: Any,
    *,
    block_size: int = _DEFAULT_BLOCK_SIZE,
    max_samples: Optional[int] = None,
    alpha: float = 1.0,
    fwht: bool = False,
    fold_order: str = "after",
    gptq_scales: "dict | None" = None,
) -> Dict[str, Any]:
    """Per-channel amax of the ROTATED activation for each W4A4 rotation group.

    Runs the live DiT forward over calibration inputs with hooks that capture each
    group's raw activation, rotates it by the SAME rotation
    :func:`build_dit_plugin_onnx_int4` bakes (deterministic ``build_rotation`` →
    bit-identical downstream), and accumulates the per-channel amax over all
    tokens and samples. The returned dict feeds ``build_dit_plugin_onnx_int4(...,
    sq_scales=...)`` for the deployable ``w4a4_sr`` scheme.

    Group keys (matching the builder's ``sq_scales`` indexing):
      ``encoder`` (shared cross-attn KV), ``block{idx}_qkv`` (self-attn merged
      QKV), ``block{idx}_q`` (cross-attn Q), ``block{idx}_o`` (attn_O),
      ``block{idx}_ffn0`` (FFN proj0 input), ``block{idx}_ffn2`` (FFN proj2 input).

    Args:
        dit_module: The live GR00T action-head DiT (read-only).
        calib_inputs: Iterable of
            ``(sa_embs, vl_embs, timestep, image_mask, backbone_attention_mask)``
            tensors (the DiT plugin-graph inputs).
        block_size: FoldQuant rotation block width (must match the build call).
        max_samples: Optional cap on the number of calibration samples consumed.

    Returns:
        ``{group_key: 1-D per-channel amax tensor (CPU, clamped ≥ 1e-8)}``.
    """
    import torch

    if fold_order not in ("after", "before"):
        raise ValueError(f"fold_order must be 'before' or 'after', got {fold_order!r}")
    if fold_order == "after" and alpha != 1.0:
        # The deployed post-rotation fold is a pure rotated-activation amax; a
        # SmoothQuant alpha only has meaning in the raw frame, before rotation.
        raise ValueError("alpha is a fold_order='before' knob; the 'after' fold is amax-only (alpha=1.0)")

    dit = dit_module.eval()
    w = DiTWeights(dit, resolve_attend_n(dit_module))
    # A plain DiT's forward may declare no mask parameters at all (N1.5).
    pass_masks = dit_accepts_masks(dit)

    # Rotations identical to the builder's: same constructor, same fwht flag. A
    # capture that builds a dense rotation while the builder bakes a butterfly
    # measures the scale in a frame the engine never enters.
    # The shared encoder site goes through the same accumulator as every block
    # site. A hand-rolled amax here kept working for the scale pass but silently
    # handed the GPTQ pass an amax VECTOR where it expected a Hessian, so the
    # cross-attention KV weights were never GPTQ-rounded.
    rot: Dict[str, tuple] = {"encoder": foldq.site_rotation(_cross_kv_weights(w), block_size, fwht)}
    for idx in range(w.num_blocks):
        (wQ, _), (wK, _), (wV, _) = w.qkv(idx)
        wO, _ = w.out_proj(idx)
        (wP0, _), (wP2, _) = w.ffn(idx)
        if idx % 2 == 1:
            rot[f"block{idx}_qkv"] = foldq.site_rotation(torch.cat([wQ, wK, wV], dim=0), block_size, fwht)
        else:
            rot[f"block{idx}_q"] = foldq.site_rotation(wQ, block_size, fwht)
        rot[f"block{idx}_o"] = foldq.site_rotation(wO, block_size, fwht)
        rot[f"block{idx}_ffn0"] = foldq.site_rotation(wP0, block_size, fwht)
        rot[f"block{idx}_ffn2"] = foldq.site_rotation(wP2, block_size, fwht)

    # One set of taps, two accumulators: amax on the first pass, the GPTQ
    # Hessian on the second. Duplicating the hook wiring is how the two passes
    # would drift on which tensor feeds which site.
    if gptq_scales is not None:
        amax, accum = foldq.hessian_accumulator(rot, gptq_scales, fold_order)
    else:
        amax, accum = foldq.scale_accumulator(rot, block_size, fold_order)

    handles = []
    for b_idx, block in enumerate(dit.transformer_blocks):
        qkv_key = f"block{b_idx}_qkv" if b_idx % 2 == 1 else f"block{b_idx}_q"
        # norm1 output = the QKV/Q input; norm3 output = the FFN proj0 input.
        handles.append(block.norm1.register_forward_hook(lambda _m, _i, out, k=qkv_key: accum(k, out)))
        handles.append(block.norm3.register_forward_hook(lambda _m, _i, out, k=f"block{b_idx}_ffn0": accum(k, out)))
        # Pre-hooks capture the Linear input: proj2 (post-GELU) and attn_O (post-SDPA).
        handles.append(
            block.ff.net[2].register_forward_pre_hook(lambda _m, args, k=f"block{b_idx}_ffn2": accum(k, args[0]))
        )
        handles.append(
            block.attn1.to_out[0].register_forward_pre_hook(lambda _m, args, k=f"block{b_idx}_o": accum(k, args[0]))
        )

    try:
        with torch.inference_mode():
            for i, sample in enumerate(calib_inputs):
                if max_samples is not None and i >= max_samples:
                    break
                sa_embs, vl_embs, timestep, image_mask, bam = dit_inputs_for(dit, sample)
                # Shared encoder site: vl_embs is the EncoderPreQuantInt4 input.
                accum("encoder", vl_embs)
                masks = {"image_mask": image_mask, "backbone_attention_mask": bam} if pass_masks else {}
                dit(hidden_states=sa_embs, encoder_hidden_states=vl_embs, timestep=timestep, **masks)
    finally:
        for h in handles:
            h.remove()

    if gptq_scales is not None:
        return dict(amax)  # Hessians, not scales
    if fold_order == "before":
        # SmoothRot scale in the raw frame: s = a^alpha / w^(1-alpha), per raw
        # input channel, with the group's weight amax taken over every weight
        # that shares the group's input (Q+K+V merged; every cross-attn KV for
        # the shared encoder group).
        group_weights: Dict[str, Any] = {"encoder": _cross_kv_weights(w)}
        for idx in range(w.num_blocks):
            (wQ, _), (wK, _), (wV, _) = w.qkv(idx)
            wO, _ = w.out_proj(idx)
            (wP0, _), (wP2, _) = w.ffn(idx)
            if idx % 2 == 1:
                group_weights[f"block{idx}_qkv"] = torch.cat([wQ, wK, wV], dim=0)
            else:
                group_weights[f"block{idx}_q"] = wQ
            group_weights[f"block{idx}_o"] = wO
            group_weights[f"block{idx}_ffn0"] = wP0
            group_weights[f"block{idx}_ffn2"] = wP2
        scales = foldq.finalize_scales(amax, weights=group_weights, alpha=alpha)
        logger.info("    Computed W4A4 scales (fold-before, alpha=%.2f) for %d groups.", alpha, len(scales))
        return scales

    logger.info("    Computed SmoothQuant scales for %d rotation groups.", len(amax))
    # Relative floor, not absolute. The fold divides R by s_ch, so a channel that
    # happened to be near-zero over the calibration set would be amplified without
    # bound; at inference the activation quantizer picks a per-token amax over all
    # channels, so one off-distribution value in such a channel would set the row
    # scale and drive every other channel to q=0. Bounding the spread costs nothing
    # on channels that carry signal.
    return foldq.finalize_scales(amax)


# Public entry point


def build_dit_plugin_onnx_int4(
    dit_module: Any,
    output_path: Any,
    *,
    sq_scales: Optional[Dict[str, Any]] = None,
    sq_fold_order: str = "after",
    fwht: bool = False,
    block_size: int = _DEFAULT_BLOCK_SIZE,
    adaln_act_bits: int = 16,
    opset: int = 17,
    gptq: "dict | None" = None,
) -> Path:
    """Construct the FoldQuant W4A4 (v4) plugin ONNX from a live DiT module.

    Args:
        dit_module: The live GR00T action-head DiT module (the reference's
            ``policy.model.action_head.model``). Read-only; weights are
            extracted, not mutated.
        output_path: Destination ``.onnx`` path. External data is written
            alongside as ``<name>.data``.
        sq_scales: Per-group SmoothQuant per-channel activation scales (see
            :func:`compute_dit_sq_scales`) → the calibrated ``w4a4_sr``
            graph, which is what every build path produces. ``None`` builds the
            uncalibrated ``w4a4`` graph and is test-only.
        block_size: FoldQuant rotation block width (input dims must divide it).
        adaln_act_bits: ``AdaLNModInt4`` activation bits (16 = BF16 / W4A16
            AdaLN, the default; 4 = INT4 activation W4A4 AdaLN).
        opset: Default ONNX opset for the empty domain (``trt.plugins`` is
            always opset 1).

    Returns:
        The saved ONNX ``Path``.
    """
    if sq_fold_order not in ("after", "before"):
        raise ValueError(f"sq_fold_order must be 'before' or 'after', got {sq_fold_order!r}")
    if sq_fold_order == "before" and sq_scales is None:
        # The uncalibrated dynamic graph has no scale to fold; silently
        # accepting the knob would build an arm that ignores it.
        raise ValueError("sq_fold_order='before' requires sq_scales (the raw-frame SmoothRot scales)")
    attend_n = resolve_attend_n(dit_module)
    w = DiTWeights(dit_module, attend_n)
    graph = _build_w4a4_graph(
        w, sq_scales, int(block_size), int(adaln_act_bits), sq_fold_order=sq_fold_order, fwht=fwht, gptq=gptq
    )

    model = oh.make_model(
        graph,
        opset_imports=[oh.make_opsetid("", opset), oh.make_opsetid("trt.plugins", 1)],
    )

    if output_path is None:
        return model
    out_path = Path(output_path)
    save_plugin_onnx(model, out_path)
    logger.info(
        "    Exported W4A4 (%s) plugin ONNX (5-input/1-output): %s",
        "sq" if sq_scales is not None else "dynamic",
        out_path,
    )
    return out_path
