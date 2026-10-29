# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""Shared GR00T DiT plugin-graph scaffolding (precision-agnostic).

Graph IO, live-weight accessors, and the BF16 timestep encoder / mask routing /
output head are identical across every quantized DiT plugin scheme that reads
from live weights; those shared pieces live here so :mod:`.dit_int8`
emits them consistently.

Constructs ONNX only: no ``tensorrt`` import, no ``.so`` load, no
``foldquant.runtime`` import.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
import onnx
import onnx.helper as oh

# Custom plugin contract - must match the compiled plugin ``.so``. Do not change.
PLUGIN_DOMAIN = "trt.plugins"
PLUGIN_NAMESPACE = "gr00t::v1"
PLUGIN_VERSION = "1"
EPS = 1e-5

#: Graph I/O contract every DiT plugin scheme emits. The names, order and dtypes
#: are those of the upstream GR00T deployment export
#: (``scripts/deployment/export_onnx_n1d7.py::export_dit_to_onnx``), so a
#: FoldQuant DiT engine is a drop-in for ``dit_bf16.engine`` in the upstream
#: ``trt_model_forward.py`` glue: ``sa_embs`` (state + action tokens),
#: ``vl_embs`` (backbone features), ``timestep``, and the two boolean
#: ``[B, vl_seq_len]`` masks the backbone emits -> ``output``. The DiT's own
#: ``attention_mask`` / ``encoder_attention_mask`` kwargs are not graph inputs:
#: ``AlternateVLDiT`` ignores the first and only reaches the second when the two
#: masks below are absent, which the GR00T backbone never leaves them.
DIT_INPUT_NAMES = (
    "sa_embs",
    "vl_embs",
    "timestep",
    "image_mask",
    "backbone_attention_mask",
)
DIT_OUTPUT_NAME = "output"
#: Symbolic dim names of the two dynamic axes, spelled the way the upstream
#: ``export_onnx_n1d7.py`` graph spells them so the upstream engine builder's
#: ``export_metadata.json`` shape hints (``sa_seq_len`` / ``vl_seq_len``) apply
#: to a FoldQuant DiT graph unchanged.
DIT_SA_SEQ_DIM = "sa_seq_len"
DIT_VL_SEQ_DIM = "vl_seq_len"
#: Batch is symbolic too, so one graph serves both a robot client (batch 1) and
#: upstream's vectorised simulation clients, which send one observation per
#: environment. The plugins already derive their sample count from the runtime
#: tensor shape rather than assuming 1; only the graph's declared I/O pinned it.
#: ``build_engines --max-batch`` decides the profile, and defaults to 1.
DIT_BATCH_DIM = "batch"


# Byte / initializer helpers


def to_bytes_f32(arr_or_tensor: Any) -> bytes:
    """Flatten to contiguous FP32 bytes (PluginField ``kFLOAT32`` payload)."""
    import torch

    if isinstance(arr_or_tensor, torch.Tensor):
        arr_or_tensor = arr_or_tensor.float().cpu().numpy()
    data: bytes = arr_or_tensor.flatten().astype(np.float32).tobytes()
    return data


def bias_f32(bias: Any) -> np.ndarray:
    """Return a Linear bias as a contiguous FP32 numpy vector."""
    out: np.ndarray = bias.float().cpu().numpy().astype(np.float32)
    return out


def bf16_initializer(name: str, t: Any) -> onnx.TensorProto:
    """Build a BFLOAT16 ONNX initializer from a torch tensor (raw uint16).

    Casts to bf16 first: a no-op for a bf16 module, but a module that reaches its
    emitter in fp32 would otherwise be viewed as int16 at double the element count,
    and TensorRT rejects the initializer with a size mismatch.
    """
    import torch

    flat = t.detach().to(torch.bfloat16).contiguous().view(torch.int16).cpu().numpy().astype(np.uint16)
    proto = onnx.TensorProto()
    proto.name = name
    proto.data_type = onnx.TensorProto.BFLOAT16
    for d in t.shape:
        proto.dims.append(int(d))
    proto.raw_data = flat.tobytes()
    return proto


def resolve_attend_n(dit_module: Any) -> Optional[int]:
    """The DiT's ``attend_text_every_n_blocks``, or ``None`` for a plain DiT.

    ``AlternateVLDiT`` alternates its cross-attention between the text-only and
    the image tokens on a schedule this value sets, and the plugin graph bakes
    that routing in (``idx % (2 * attend_n) == 0`` picks the text mask). Reading
    the schedule off the module is the only safe source: a wrong value produces a
    graph that builds and runs at full speed while attending to the wrong tokens.

    A plain non-alternating ``DiT`` (GR00T N1.5) has no schedule because it has
    no split: every cross block attends the whole encoder sequence, and its
    forward is called with no mask at all. ``None`` says exactly that, and the
    emitters give those blocks :func:`emit_attend_all_mask`'s all-zero additive
    mask rather than either half of a split that does not exist.
    """
    cfg = getattr(dit_module, "config", None)
    if cfg is not None and hasattr(cfg, "attend_text_every_n_blocks"):
        return int(cfg.attend_text_every_n_blocks)
    if hasattr(dit_module, "attend_text_every_n_blocks"):
        return int(dit_module.attend_text_every_n_blocks)
    return None


#: Name of the additive cross-attention mask a non-alternating DiT uses: zeros
#: everywhere, i.e. attend every encoder position. Matches the PyTorch reference,
#: which calls a plain ``DiT`` with ``mask=None``.
ATTEND_ALL_MASK = "all_mask_add"


def emit_attend_all_mask(nodes: list, inits: list) -> None:
    """Emit :data:`ATTEND_ALL_MASK`, a broadcastable all-zero additive mask.

    Shaped ``(1, 1, 1, 1)`` so it broadcasts over any (B, H, S_q, S_kv) score
    block. Zeros, not ``backbone_attention_mask``: the plain-DiT reference
    attends padded encoder positions too, and masking them here would make the
    engine disagree with the model it is exported from.
    """
    import torch

    inits.append(bf16_initializer(ATTEND_ALL_MASK, torch.zeros(1, 1, 1, 1, dtype=torch.bfloat16)))


# Live-module weight access: reads the real DiT submodule names directly.


class DiTWeights:
    """Thin view over the live GR00T DiT module exposing the tensors the plugin
    graph bakes, plus the shape/head metadata it needs."""

    def __init__(self, dit_module: Any, attend_text_every_n_blocks: Optional[int]) -> None:
        self.dit = dit_module
        blocks = dit_module.transformer_blocks
        self.blocks = blocks
        self.num_blocks = len(blocks)

        self.dim = int(dit_module.inner_dim)
        self.num_heads = int(dit_module.config.num_attention_heads)
        self.head_dim = int(dit_module.config.attention_head_dim)
        self.inner_dim = self.num_heads * self.head_dim
        self.ff_inner = int(blocks[0].ff.net[0].proj.weight.shape[0])
        self.kv_dim = int(blocks[0].attn1.to_k.weight.shape[1])
        self.output_dim = int(dit_module.proj_out_2.out_features)
        #: ``None`` on a plain DiT: no text/image alternation to schedule.
        self.attend_text_every_n_blocks = (
            None if attend_text_every_n_blocks is None else int(attend_text_every_n_blocks)
        )

        # Output head + timestep encoder (BF16, reused verbatim).
        self.timestep_encoder = dit_module.timestep_encoder
        self.proj_out_1 = dit_module.proj_out_1
        self.proj_out_2 = dit_module.proj_out_2

    # -- per-block accessors -------------------------------------------------
    def adaln(self, idx: int) -> Any:
        lin = self.blocks[idx].norm1.linear
        return lin.weight.data, lin.bias.data

    def qkv(self, idx: int) -> Any:
        a = self.blocks[idx].attn1
        return (
            (a.to_q.weight.data, a.to_q.bias.data),
            (a.to_k.weight.data, a.to_k.bias.data),
            (a.to_v.weight.data, a.to_v.bias.data),
        )

    def out_proj(self, idx: int) -> Any:
        a = self.blocks[idx].attn1
        return a.to_out[0].weight.data, a.to_out[0].bias.data

    def ffn(self, idx: int) -> Any:
        ff = self.blocks[idx].ff
        return (
            (ff.net[0].proj.weight.data, ff.net[0].proj.bias.data),
            (ff.net[2].weight.data, ff.net[2].bias.data),
        )


# Timestep encoder + mask routing emitters


def emit_timestep_encoding(w: DiTWeights, nodes: list, inits: list) -> None:
    """Sinusoidal timestep embedding + 2-layer MLP -> ``temb`` (BF16).

    Replicates diffusers ``Timesteps(num_channels=256, flip_sin_to_cos=True,
    downscale_freq_shift=1)`` followed by ``TimestepEmbedding`` (Linear-SiLU-
    Linear), reading the live ``timestep_encoder`` weights.
    """
    te = w.timestep_encoder
    te_lin1 = te.timestep_embedder.linear_1
    te_lin2 = te.timestep_embedder.linear_2
    te_in_ch = int(te_lin1.in_features)

    half = te_in_ch // 2
    freq = np.exp(-math.log(10000.0) * np.arange(half, dtype=np.float32) / max(half - 1, 1)).astype(np.float32)
    inits.append(oh.make_tensor("freq_const", onnx.TensorProto.FLOAT, [half], freq.tobytes(), raw=True))
    inits.append(
        oh.make_tensor("_ax_0", onnx.TensorProto.INT64, [1], np.array([0], dtype=np.int64).tobytes(), raw=True)
    )
    inits.append(
        oh.make_tensor("_ax_1", onnx.TensorProto.INT64, [1], np.array([1], dtype=np.int64).tobytes(), raw=True)
    )
    nodes.append(oh.make_node("Cast", ["timestep"], ["ts_f"], to=onnx.TensorProto.FLOAT))
    nodes.append(oh.make_node("Unsqueeze", ["ts_f", "_ax_1"], ["ts_f_u"]))
    nodes.append(oh.make_node("Unsqueeze", ["freq_const", "_ax_0"], ["freq_u"]))
    nodes.append(oh.make_node("Mul", ["ts_f_u", "freq_u"], ["te_arg"]))
    nodes.append(oh.make_node("Sin", ["te_arg"], ["te_sin"]))
    nodes.append(oh.make_node("Cos", ["te_arg"], ["te_cos"]))
    # flip_sin_to_cos=True -> concat order is (cos, sin), NOT (sin, cos).
    nodes.append(oh.make_node("Concat", ["te_cos", "te_sin"], ["te_concat_f"], axis=-1))
    nodes.append(oh.make_node("Cast", ["te_concat_f"], ["te_concat"], to=onnx.TensorProto.BFLOAT16))
    inits.append(bf16_initializer("te_lin1_w", te_lin1.weight.data.t().contiguous()))
    inits.append(bf16_initializer("te_lin1_b", te_lin1.bias.data))
    nodes.append(oh.make_node("MatMul", ["te_concat", "te_lin1_w"], ["te_l1_pre"]))
    nodes.append(oh.make_node("Add", ["te_l1_pre", "te_lin1_b"], ["te_l1"]))
    # SiLU
    nodes.append(oh.make_node("Sigmoid", ["te_l1"], ["te_l1_sig"]))
    nodes.append(oh.make_node("Mul", ["te_l1", "te_l1_sig"], ["te_silu"]))
    inits.append(bf16_initializer("te_lin2_w", te_lin2.weight.data.t().contiguous()))
    inits.append(bf16_initializer("te_lin2_b", te_lin2.bias.data))
    nodes.append(oh.make_node("MatMul", ["te_silu", "te_lin2_w"], ["te_l2_pre"]))
    nodes.append(oh.make_node("Add", ["te_l2_pre", "te_lin2_b"], ["temb"]))


def emit_mask_routing(nodes: list, inits: list) -> None:
    """Build the BF16 additive image/text attention masks (B, 1, 1, Senc).

    ``image_attn_mask  = image_mask & backbone_mask``
    ``non_image_attn   = (~image_mask) & backbone_mask``
    Each becomes an additive mask (0 = attend, -1e4 = mask-out).

    The DiT's ``attention_mask``/``encoder_attention_mask`` kwargs (self-attn
    mask and the cross-attn fallback mask on ``AlternateVLDiT.forward()``) are
    not routed here: ``attention_mask`` is unused on that DiT variant regardless
    of value, and ``encoder_attention_mask`` is only a fallback the DiT reaches
    when ``image_mask``/``backbone_attention_mask`` are absent - the GR00T
    backbone always supplies both, so that fallback never fires. The masks
    arrive as BOOL (the upstream export's dtype); the Casts below are no-ops
    that also accept an integer mask.
    """
    nodes.append(oh.make_node("Cast", ["image_mask"], ["im_bool"], to=onnx.TensorProto.BOOL))
    nodes.append(oh.make_node("Cast", ["backbone_attention_mask"], ["bm_bool"], to=onnx.TensorProto.BOOL))
    nodes.append(oh.make_node("And", ["im_bool", "bm_bool"], ["img_attn_bool"]))
    nodes.append(oh.make_node("Not", ["im_bool"], ["non_im_bool"]))
    nodes.append(oh.make_node("And", ["non_im_bool", "bm_bool"], ["non_img_attn_bool"]))

    import torch

    inits.append(bf16_initializer("mask_zero", torch.tensor(0.0, dtype=torch.bfloat16)))
    inits.append(bf16_initializer("mask_neg", torch.tensor(-1e4, dtype=torch.bfloat16)))
    nodes.append(oh.make_node("Where", ["img_attn_bool", "mask_zero", "mask_neg"], ["img_mask_add_bs"]))
    nodes.append(oh.make_node("Where", ["non_img_attn_bool", "mask_zero", "mask_neg"], ["non_img_mask_add_bs"]))
    inits.append(
        oh.make_tensor(
            "_attn_mask_shape", onnx.TensorProto.INT64, [4], np.array([0, 1, 1, -1], dtype=np.int64).tobytes(), raw=True
        )
    )
    nodes.append(oh.make_node("Reshape", ["img_mask_add_bs", "_attn_mask_shape"], ["img_mask_add"], allowzero=0))
    nodes.append(
        oh.make_node("Reshape", ["non_img_mask_add_bs", "_attn_mask_shape"], ["non_img_mask_add"], allowzero=0)
    )


# Output head emitter


def emit_output_head(w: DiTWeights, cur_x: str, nodes: list, inits: list) -> None:
    """norm_out + proj_out_1 (AdaLN-style modulate) + proj_out_2 -> the output tensor."""
    import torch

    nodes.append(oh.make_node("Sigmoid", ["temb"], ["po_sig"]))
    nodes.append(oh.make_node("Mul", ["temb", "po_sig"], ["po_silu"]))
    inits.append(bf16_initializer("po1_w", w.proj_out_1.weight.data.t().contiguous()))
    inits.append(bf16_initializer("po1_b", w.proj_out_1.bias.data))
    nodes.append(oh.make_node("MatMul", ["po_silu", "po1_w"], ["po1_pre"]))
    nodes.append(oh.make_node("Add", ["po1_pre", "po1_b"], ["po1"]))
    inits.append(
        oh.make_tensor(
            "_po_split", onnx.TensorProto.INT64, [2], np.array([w.dim, w.dim], dtype=np.int64).tobytes(), raw=True
        )
    )
    nodes.append(oh.make_node("Split", ["po1", "_po_split"], ["po_shift", "po_scale"], axis=-1))
    # norm_out (elementwise_affine=False, eps=1e-6) via LayerNormalization.
    inits.append(bf16_initializer("norm_out_w_dummy", torch.ones(w.dim, dtype=torch.bfloat16)))
    inits.append(bf16_initializer("norm_out_b_dummy", torch.zeros(w.dim, dtype=torch.bfloat16)))
    nodes.append(
        oh.make_node(
            "LayerNormalization", [cur_x, "norm_out_w_dummy", "norm_out_b_dummy"], ["norm_out"], axis=-1, epsilon=1e-6
        )
    )
    inits.append(
        oh.make_tensor("_ax_1_b", onnx.TensorProto.INT64, [1], np.array([1], dtype=np.int64).tobytes(), raw=True)
    )
    nodes.append(oh.make_node("Unsqueeze", ["po_scale", "_ax_1_b"], ["po_scale_u"]))
    nodes.append(oh.make_node("Unsqueeze", ["po_shift", "_ax_1_b"], ["po_shift_u"]))
    inits.append(bf16_initializer("one_bf16", torch.tensor(1.0, dtype=torch.bfloat16)))
    nodes.append(oh.make_node("Add", ["po_scale_u", "one_bf16"], ["po_scale_p1"]))
    nodes.append(oh.make_node("Mul", ["norm_out", "po_scale_p1"], ["po_mul"]))
    nodes.append(oh.make_node("Add", ["po_mul", "po_shift_u"], ["po_modulated"]))
    inits.append(bf16_initializer("po2_w", w.proj_out_2.weight.data.t().contiguous()))
    inits.append(bf16_initializer("po2_b", w.proj_out_2.bias.data))
    nodes.append(oh.make_node("MatMul", ["po_modulated", "po2_w"], ["po2_pre"]))
    nodes.append(oh.make_node("Add", ["po2_pre", "po2_b"], [DIT_OUTPUT_NAME]))
