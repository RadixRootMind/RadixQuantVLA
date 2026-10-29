# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""Quantize and pack weights in the byte layouts required by TensorRT plugins."""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np


def quant_weight_per_row(weight: "Any") -> Tuple[np.ndarray, np.ndarray]:
    """Symmetric per-output-row INT8 weight quant for a ``(N, K)`` tensor.

    ``amax`` over the input dim gives one scale per output row; this is the
    ``weight_i8`` / ``weight_scale`` pair baked into the INT8 per-row plugins.

    Returns ``(int8 [N, K], float32 scale [N])``.
    """
    import torch

    from .quant_state import rtn_pack

    def compute():
        wf = weight.float()
        amax = wf.abs().amax(dim=1)
        scale = (amax / 127.0).clamp(min=1e-12)
        return torch.round(wf / scale.unsqueeze(1)).clamp(-127, 127).to(torch.int8), scale

    codes, scale = rtn_pack(weight, compute)
    return codes.cpu().numpy().astype(np.int8), scale.cpu().numpy().astype(np.float32)


def pack_intweights(unpacked_qweight: np.ndarray) -> np.ndarray:
    """Pack unsigned 4-bit AWQ weights ``(N, K)`` -> ``(N/4, K)`` int16.

    AWQ interleaved layout expected by the INT4 groupwise GEMM kernel. Requires
    ``N % 4 == 0`` and ``K % 64 == 0``. Ported verbatim from the reference
    ``int4_gemm_plugin.pack_intweights``.
    """
    interleave = 4
    kstride = 64
    n, k = unpacked_qweight.shape
    assert n % interleave == 0, f"N={n} not divisible by 4"
    assert k % kstride == 0, f"K={k} not divisible by 64"

    pk: Any = unpacked_qweight.reshape(n, k // 32, 32)
    pk = pk.reshape(n, k // 32, 4, 4, 2).transpose(0, 1, 3, 2, 4)
    pk = pk.reshape(n, k // 32, 32)

    pk = pk.reshape(n, k // 32, 4, 8)
    pk = pk.reshape(n, k // 32, 4, 4, 2).transpose(0, 1, 2, 4, 3)
    pk = pk.reshape(n, k)

    pk = pk.reshape(n // interleave, interleave, k // kstride, kstride)
    pk = pk.transpose(0, 2, 1, 3)
    pk = pk.reshape(n // interleave, k // kstride, kstride, interleave)
    pk = (
        pk[..., 0].astype(np.int32)
        | (pk[..., 1].astype(np.int32) << 4)
        | (pk[..., 2].astype(np.int32) << 8)
        | (pk[..., 3].astype(np.int32) << 12)
    )
    packed: np.ndarray = pk.reshape(n // interleave, k).astype(np.int16)
    return packed


def to_bytes_i8(arr: np.ndarray) -> bytes:
    """Flatten to contiguous INT8 bytes (PluginField ``kINT8`` payload)."""
    # bytes(ndarray) goes through the builtin buffer-protocol constructor rather than
    # .tobytes(), whose return type mypy sees as Any/bytes depending on numpy-stub version.
    return bytes(arr.flatten().astype(np.int8))
