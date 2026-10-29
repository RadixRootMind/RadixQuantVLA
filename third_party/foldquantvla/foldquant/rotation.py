# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""Rotation, SmoothQuant folding, and INT4 packing for FoldQuant exports.

Provides Hadamard and SVD-based block rotations, channel padding, scale
folding, and weight serialization. Shared by calibration, action-module
folding, and the LLM INT4 packer.

Packed weights use the column-major layout expected by the TensorRT plugins.
Byte helpers serialize their INT32 and BF16 fields. Torch is imported inside
the functions that need it; this module does not load TensorRT plugins.
"""

from __future__ import annotations

import logging
from typing import Any, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_QMAX_I4 = 7  # symmetric INT4: values in [-7, 7]


# PluginField byte helpers (exact layouts the plugins read)


def to_bytes_i32(arr: Any) -> bytes:
    """Flatten to contiguous INT32 bytes (PluginField ``kINT32`` payload)."""
    # Keep the return type explicit for NumPy stubs that type tobytes() as Any.
    return bytes(np.ascontiguousarray(arr).flatten().astype(np.int32).tobytes())


def to_bytes_bf16(arr_or_tensor: Any) -> bytes:
    """Flatten to raw BF16 bytes with round-to-nearest-even.

    BIT-IDENTICAL to the plugin's ``f32_to_bf16`` (``int4_host_util.h``):
    ``rounded = (x + 0x7FFF + ((x >> 16) & 1)) >> 16`` wrapping at uint32. The
    plugins up-convert the rotation ``R`` to BF16 at runtime, so storing it BF16
    here changes nothing numerically while halving the FP32 byte footprint.
    """
    import torch

    if isinstance(arr_or_tensor, torch.Tensor):
        arr_or_tensor = arr_or_tensor.float().cpu().numpy()
    u = np.ascontiguousarray(arr_or_tensor).flatten().astype(np.float32).view(np.uint32)
    inc = (np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))).astype(np.uint64)
    rounded = (((u.astype(np.uint64) + inc) & np.uint64(0xFFFFFFFF)) >> np.uint64(16)).astype(np.uint16)
    return bytes(rounded.tobytes())


# Rotation builders


def normalized_hadamard(n: int) -> Any:
    """``n×n`` normalized Sylvester Hadamard (``n`` must be a power of 2)."""
    import torch

    assert n & (n - 1) == 0, f"Hadamard size {n} must be a power of 2"
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / (n**0.5)


def pad_in_dim(t: Any, block_size: int) -> Any:
    """Zero-pad a tensor's last axis up to a multiple of ``block_size``.

    Some experts are not a multiple of the rotation block: a 480- or 720-wide
    projection (480 = 2^5*3*5, 720 = 2^4*3^2*5), where 720 additionally fails the
    INT4 GEMM's 128-bit alignment (K % 32 != 0), so a smaller block would not
    rescue it. Padding is exact for a linear: with the weight's extra columns zero,
    W_pad·x_pad = W·x, and an orthonormal rotation stays orthonormal on the
    padded space.

    The calibration capture and the emitter must pad IDENTICALLY: the scale is
    measured in the rotated frame, so a rotation built on a different width
    mis-scales every channel with no error anywhere.
    """
    k = int(t.shape[-1])
    k_pad = ((k + block_size - 1) // block_size) * block_size
    if k_pad == k:
        return t
    import torch

    out = torch.zeros(*t.shape[:-1], k_pad, dtype=t.dtype, device=t.device)
    out[..., :k] = t
    return out


def hadamard_blocks(k_in: int, block_size: int) -> Tuple[Any, Any]:
    """Identity permutation plus a per-block Sylvester Hadamard.

    Shaped exactly like :func:`build_rotation`'s return value, so every consumer
    of ``(perm, R)`` (the weight fold, the SmoothQuant capture, the packers)
    works unchanged for a butterfly rotation. The runtime never receives this
    matrix; the kernel recomputes it as a butterfly. It exists so the offline
    side can fold ``W·Hᵀ`` with the same code path the dense rotation uses.
    """
    import torch

    perm = torch.arange(k_in, dtype=torch.int32)
    h = normalized_hadamard(block_size)
    rot = h.unsqueeze(0).expand(k_in // block_size, block_size, block_size).contiguous()
    return perm, rot


def build_rotation(weight: Any, block_size: int) -> Tuple[Any, Any]:
    """Build ``(perm, R)`` for the composite SVD·Hadamard input rotation.

    Args:
        weight: ``(out, in)`` float tensor (on any device; SVD runs on CPU).
        block_size: input-channel block width (``in`` must be divisible by it).

    Returns:
        ``perm``: ``(in,)`` long. Zigzag channel permutation by descending
        column norm (round-robin into blocks → each block mixes high/low energy
        channels, preventing per-block-scale domination).
        ``R``: ``(num_blocks, block_size, block_size)`` float. Per-block
        orthogonal rotation ``R_b = V·H`` (``V`` = right singular vectors of the
        block's columns; ``H`` = normalized Hadamard). Falls back to pure ``H``
        on the (rare) blocks where SVD fails to converge (still orthogonal).
    """
    import torch

    out_f, in_f = weight.shape
    assert in_f % block_size == 0, f"in dim {in_f} not divisible by block_size {block_size}"
    num_blocks = in_f // block_size

    # Normalize to FP32/CPU up front so the whole function is a deterministic
    # function of the weight values alone. The column norms below feed an argsort
    # whose ties break differently in BF16-on-CUDA than in FP32-on-CPU, which would
    # make the permutation depend on where the caller happened to hold the weights.
    # (This also hoists the per-block `.float().cpu()` the SVD needed anyway, so it
    # is one host transfer instead of num_blocks of them.)
    orig_device = weight.device
    weight = weight.detach().float().cpu()

    col_norm = weight.norm(dim=0)  # (in,)
    order = torch.argsort(col_norm, descending=True, stable=True)
    perm = torch.empty(in_f, dtype=torch.long)
    for i in range(in_f):  # round-robin into blocks
        perm[(i % num_blocks) * block_size + (i // num_blocks)] = order[i]

    # SVD on CPU: a one-time offline cost that avoids the broken CUDA cuSOLVER
    # path on some Jetson torch builds (libtorch_cuda_linalg undefined-symbol).
    # Blocks are tiny (out × block_size), so CPU is cheap.
    H = normalized_hadamard(block_size).to(torch.float32)
    Wp = weight.index_select(1, perm)  # (out, in) permuted cols
    R = torch.empty(num_blocks, block_size, block_size, dtype=torch.float32)
    svd_failed: list[int] = []
    for b in range(num_blocks):
        blk = Wp[:, b * block_size : (b + 1) * block_size]  # (out, bs), out >= bs
        try:
            _, _, Vh = torch.linalg.svd(blk, full_matrices=False)  # Vh (bs, bs)
            R[b] = Vh.transpose(0, 1) @ H  # V·H, orthogonal
        except torch.linalg.LinAlgError:
            # Narrow: only non-convergence falls back. A broader catch would report
            # an OOM or a dtype bug as "still orthogonal/lossless", which is untrue.
            R[b] = H  # pure Hadamard (still orthogonal / lossless)
            svd_failed.append(b)
    if svd_failed:
        logger.warning(
            "  FoldQuant rotation: SVD did not converge on %d/%d blocks %s; used pure-Hadamard rotation "
            "there (still orthogonal/lossless).",
            len(svd_failed),
            num_blocks,
            svd_failed,
        )
    return perm.to(orig_device), R.to(orig_device)


def apply_rotation(x: Any, perm: Any, R: Any, block_size: int) -> Any:
    """``(x permuted by perm) block-rotated by R``, over the last axis.

    One implementation for both sides of the rotation contract: the weight side
    (``(out, in)``) that gets baked, and the activation side (any leading dims) that
    the plugin prologue reproduces at runtime. They must agree exactly (the
    SmoothQuant scales are measured through this on activations and folded into
    weights rotated through the same code), so they share it rather than mirroring
    each other.
    """
    import torch

    x = x.index_select(-1, perm.to(x.device))
    lead = x.shape[:-1]
    in_f = x.shape[-1]
    num_blocks = in_f // block_size
    xb = x.reshape(*lead, num_blocks, block_size)
    xr = torch.einsum("...nb,nbc->...nc", xb, R.to(x.dtype).to(x.device))
    return xr.reshape(*lead, in_f)


# Weights are the rank-2 case of the same operation.
apply_weight_rotation = apply_rotation


def fold_rotation_sq(R: Any, s_ch: Any) -> Any:
    """SmoothQuant fold of a per-channel activation scale into the rotation.

    ``R'[b,i,c] = R[b,i,c] / s_ch[b*bs + c]`` so that at runtime
    ``rotate(x, R') = rotate(x, R) / s_ch``: the per-channel activation scale is
    absorbed into the (baked) rotation, leaving the prologue + INT4 GEMM kernels
    unchanged. Paired with :func:`pack_int4_colmajor_sq` on the weight side.
    """
    nbk, _, bs = R.shape
    return R / s_ch.to(R.device).float().reshape(nbk, 1, bs)


# INT4 weight packing


def pack_int4_nibbles(codes: Any) -> np.ndarray:
    """Nibble-pack INT4 codes ``(out, in)`` → uint8 ``(out, in/2)``.

    For output channel ``n``, byte ``b`` holds ``nibble(2b)`` = low,
    ``nibble(2b+1)`` = high of ``codes[n, :]``, matching CUTLASS
    ``ColumnMajor int4b_t`` (column ``n`` = ``in/2`` contiguous bytes; see
    ``dit_int4_rowwise.h``).

    The nibble order lives here and nowhere else. Every INT4 producer goes
    through it (the DiT/expert RTN packers below and the LLM's GPTQ path)
    because a flipped order builds, loads and runs, and produces noise rather
    than an error.
    """
    import torch

    qi = codes.to(torch.int32) & 0xF
    packed = (qi[:, 0::2] | (qi[:, 1::2] << 4)).to(torch.uint8)  # (out, in/2)
    out: np.ndarray = packed.cpu().numpy()
    return out


def pack_int4_colmajor(weight_rot: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Per-output-channel symmetric RTN INT4 of a ``(out, in)`` weight.

    Returns:
        ``packed``: uint8 ``(out, in/2)`` via :func:`pack_int4_nibbles`.
        ``scale``: float32 ``(out,)`` = ``amax_row / 7``.
    """
    import torch

    from .quant_state import rtn_pack

    def compute():
        W = weight_rot.detach().float()
        scale = (W.abs().amax(dim=1) / _QMAX_I4).clamp(min=1e-8)  # (out,)
        return torch.clamp(torch.round(W / scale.unsqueeze(1)), -_QMAX_I4, _QMAX_I4).to(torch.int32), scale

    q, scale = rtn_pack(weight_rot, compute)
    return pack_int4_nibbles(q), scale.cpu().numpy().astype(np.float32)


def fold_rotation_sq_before(R: Any, s_raw: Any, perm: Any) -> Any:
    """Pre-rotation SmoothQuant fold (SmoothRot order): scale RAW channels, then rotate.

    ``rotate(x / s_raw) == apply_rotation(x, perm, R')`` with
    ``R'[b, i, c] = R[b, i, c] / s_perm[b*bs + i]``: the division lands on the
    rotation's INPUT axis (the permuted raw channel), where :func:`fold_rotation_sq`
    divides the OUTPUT axis (the rotated channel). Measured on the N1.6 DiT
    (Zen weekly 17-22/08): folding before the rotation at alpha=0.5 cuts output
    error 36-40% at unchanged size/latency. Caveat carried from the same report:
    the folded rotation is baked in BF16, and fold-before at alpha >= 0.6 pushes
    ``R/s`` past an 8-bit mantissa on the engine (simulation does not model it);
    hold alpha at the engine-verified 0.5.
    """
    nbk, bs, _ = R.shape
    s_perm = s_raw.to(R.device).float()[perm.to(s_raw.device)]
    return R / s_perm.reshape(nbk, bs, 1)


def fold_weight_sq(weight: Any, perm: Any, R: Any, s_ch: Any, block_size: int, fold_order: str) -> Any:
    """The FoldQuant-folded FLOAT weight, before any packing.

    ``"after"``  -> ``rotate(W) · diag(s_ch)``   pairs with fold_rotation_sq
    ``"before"`` -> ``rotate(W · diag(s_raw))``  pairs with fold_rotation_sq_before

    Split out of the INT4 packers so the INT8 path folds identically: the fold
    is the algorithm, the packer is only the bit width. Keeping two copies is how
    the INT8 action arms ended up folding nothing while their scheme name still
    said ``_sr``.
    """
    if fold_order not in ("before", "after"):
        raise ValueError(f"fold_order must be 'before' or 'after', got {fold_order!r}")
    wf = weight.float()
    if fold_order == "before":
        return apply_weight_rotation(wf * s_ch.to(wf.device).float().reshape(1, -1), perm, R, block_size)
    return apply_weight_rotation(wf, perm, R, block_size) * s_ch.to(wf.device).float().reshape(1, -1)


def pack_int4_colmajor_sq_before(
    weight: Any, perm: Any, R: Any, s_raw: Any, block_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Weight-side pair of :func:`fold_rotation_sq_before`.

    ``W' = rotate(W · diag(s_raw))``: the scale multiplies the RAW input columns
    before the permutation and rotation, so ``W'·rot(x/s) = (W·s)·(x/s) = W·x``
    exactly (baking-only, like the post-rotation fold).
    """
    Ws = weight.float() * s_raw.to(weight.device).float().reshape(1, -1)
    Wr = apply_weight_rotation(Ws, perm, R, block_size)
    return pack_int4_colmajor(Wr)


def pack_int4_colmajor_sq(weight: Any, perm: Any, R: Any, s_ch: Any, block_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """SmoothQuant weight-side pack: rotate ``W`` by the ORIGINAL ``R`` (not the
    folded ``R'``), multiply by ``s_ch`` per input channel, then RTN INT4.

    The activation-side fold (:func:`fold_rotation_sq`) divides by ``s_ch`` and
    the weight-side pack multiplies by ``s_ch``, so the product ``W·x`` is
    unchanged; this is baking-only. Must be called with the same ``perm``/``R``
    :func:`build_rotation` produced for this weight.
    """
    Wr = apply_weight_rotation(weight.float(), perm, R, block_size)
    Wr_s = Wr * s_ch.to(Wr.device).float().reshape(1, -1)
    return pack_int4_colmajor(Wr_s)


def adaln_pack_int4(wL: Any) -> Tuple[bytes, bytes, int, int]:
    """Pack a ``norm1.linear`` weight (PyTorch ``[out, in]``) for ``AdaLNModInt4``.

    Per-output-channel symmetric INT4 (``qmax=7``), packed 2 nibbles/byte along
    ``in`` (``i`` even = low, ``i`` odd = high; ``in`` padded to even). Matches
    the GEMV kernel ``adaln_gemv_int4_cuda.cu``
    (``out[j] = scale[j] · Σ x[i] · int4(w[j,i])``).

    Returns ``(weight_bytes, scale_bf16_bytes, in_dim, out_dim)``.

    Pads ``in`` to even and delegates the quantization + nibble packing to
    :func:`pack_int4_colmajor`: the nibble order has to stay in lockstep with the
    CUDA kernels, so it is defined in exactly one place.
    """
    import torch

    w = wL.detach().float().cpu()  # [out, in]
    out_dim, in_dim = w.shape
    if in_dim % 2:  # the packer consumes channel pairs
        w = torch.cat([w, torch.zeros(out_dim, 1)], dim=1)
    packed, scale = pack_int4_colmajor(w)  # [out, (in+1)//2], [out]
    return packed.tobytes(), to_bytes_bf16(scale), int(in_dim), int(out_dim)
