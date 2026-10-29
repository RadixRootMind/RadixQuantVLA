# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""Shared rotation, scale folding, and weight packing for action modules.

INT4 and INT8 emitters use the same folding contract:

    weight side (offline)   W' = rotate(W)·diag(s)      fold_order="after"
                            W' = rotate(W·diag(s))      fold_order="before"
    runtime                 quantize(rotate(x)/s)  resp. quantize(rotate(x/s))

Dense rotations absorb the scale into their coefficients. Fixed Hadamard
butterflies pass a separate scale vector to the kernel, which applies it on
the side selected by ``fold_order``.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from . import rotation as rotations
from .dit_common import to_bytes_f32
from .weights import quant_weight_per_row

__all__ = [
    "site_rotation",
    "rotation_block_for",
    "fold_site",
    "fold_rotation",
    "scale_accumulator",
    "finalize_scales",
    "SQ_SCALE_REL_FLOOR",
    "FoldSpec",
]

SQ_SCALE_REL_FLOOR = 1e-3

#: Node attribute carrying the SmoothQuant vector, per fold order.
_SCALE_FIELD = {"before": "act_scale_pre", "after": "act_scale_ch"}


def rotation_block_for(k_in: int, block_size: int) -> int:
    """Largest power-of-two rotation block ``<= block_size`` that divides ``k_in``.

    Sites whose input width is not a multiple of the nominal block still get a
    rotation, just a narrower one: a 480-wide projection drops from 64 to 32.
    Callers that can pad the input axis (the INT4 path) should do that instead
    and keep the full width; callers that cannot (a fused-norm plugin
    normalises over K internally and cannot take a padded activation) use this.
    """
    bs = int(block_size)
    while bs > 1 and k_in % bs != 0:
        bs //= 2
    return bs


def site_rotation(weight: Any, block_size: int, fwht: bool) -> Tuple[Any, Any]:
    """The ``(perm, R)`` pair for one site: fixed butterfly, or learned dense."""
    if fwht:
        return rotations.hadamard_blocks(int(weight.shape[1]), block_size)
    from .quant_state import dense_rotation

    import torch

    perm, R = dense_rotation(lambda: rotations.build_rotation(weight, block_size))
    return perm.to(device=weight.device, dtype=torch.long), R.to(device=weight.device, dtype=torch.float32)


def fold_macro_site(
    weight: Any,
    perm: Any,
    R: Any,
    block_size: int,
    s_ch: Any | None,
    fold_order: str = "after",
    bits: int = 4,
    gptq: Any | None = None,
) -> Tuple[bytes, bytes, Any]:
    """Fold and pack one site for a fused attention or FFN plugin.

    Returns ``(packed_bytes, weight_scale_bytes, R_use)`` with INT4 nibbles or
    INT8 bytes according to ``bits``. Dense callers bake the folded ``R_use``
    matrix. Butterfly callers ignore it and supply an empty rotation blob,
    ``rot_block_size``, and the raw scale vector instead.
    """
    packed_b, scale_b, attrs = fold_site(
        weight,
        bits=bits,
        block_size=block_size,
        s_ch=s_ch,
        fold_order=fold_order,
        fwht=False,
        rotation=(perm, R),
        gptq=gptq,
    )
    return packed_b, scale_b, (attrs.rotation_tensor if attrs.rotation_tensor is not None else R)


def fold_rotation(R: Any, perm: Any, s_ch: Any | None, fold_order: str) -> Any:
    """Fold SmoothQuant into a rotation matrix on the weight's matching axis.

    Used by sites such as DiT encoder pre-quantization, which shares one
    rotation across the cross-attention KV weights.
    """
    if s_ch is None:
        return R
    if fold_order == "before":
        return rotations.fold_rotation_sq_before(R, s_ch, perm)
    if fold_order == "after":
        return rotations.fold_rotation_sq(R, s_ch)
    raise ValueError(f"fold_order must be 'before' or 'after', got {fold_order!r}")


class FoldSpec(dict):
    """Node attributes describing a folded site.

    Compatible with ``make_node(**spec)``. ``rotation_tensor`` is a Python
    attribute so it is available to emitters without becoming a node field.
    """

    rotation_tensor: Any | None = None


def fold_site(
    weight: Any,
    *,
    bits: int,
    block_size: int,
    s_ch: Any | None,
    fold_order: str = "after",
    fwht: bool = True,
    rotation: Tuple[Any, Any] | None = None,
    gptq: Any | None = None,
) -> Tuple[bytes, bytes, FoldSpec]:
    """Fold, rotate and pack one site.

    Args:
        weight: ``(N, K)`` float weight, already padded if the caller pads.
        bits: 4 or 8, the only thing that differs between the two paths.
        block_size: nominal rotation block; narrowed to fit ``K`` when needed.
        s_ch: SmoothQuant vector measured in the frame ``fold_order`` names, or
            None for an unfolded site.
        fold_order: ``"before"`` (raw frame) or ``"after"`` (rotated frame).
        fwht: butterfly (no matrix shipped) rather than the learned dense rotation.
        gptq: this site's factorized Hessian (:func:`llm_gptq.gptq_prepare`), to
            round the folded weight with GPTQ instead of round-to-nearest. The
            Hessian must be built on the SAME rotated, scaled activation the
            kernel sees, or the error it compensates is not the error that occurs.
        rotation: an already-built ``(perm, R)`` to fold with, for sites whose
            rotation is derived from a different tensor than the one being packed.
            The DiT's encoder pre-quant shares one rotation across every cross
            block's KV pack, so it cannot be re-derived from each weight.

    Returns:
        ``(weight_bytes, weight_scale_bytes, attrs)`` where *attrs* carries
        ``rot_block_size`` plus the scale vector, ready to splat into the node.
    """
    if bits not in (4, 8):
        raise ValueError(f"bits must be 4 or 8, got {bits}")
    if fold_order not in ("before", "after"):
        raise ValueError(f"fold_order must be 'before' or 'after', got {fold_order!r}")

    k_in = int(weight.shape[1])
    bs = rotation_block_for(k_in, block_size)
    if s_ch is None or bs <= 1:
        # Unfolded, but a caller-supplied rotation still has to be applied: the
        # weights are packed rotated whether or not a scale rides along.
        if rotation is not None:
            perm, rot = rotation
            return (
                *_pack(rotations.apply_weight_rotation(weight.float(), perm, rot, bs), bits),
                FoldSpec(rot_block_size=0 if not fwht else int(bs)),
            )
        return (*_pack(weight, bits, gptq), FoldSpec(rot_block_size=0))

    perm, rot = rotation if rotation is not None else site_rotation(weight, bs, fwht)
    folded = rotations.fold_weight_sq(weight, perm, rot, s_ch, bs, fold_order)
    attrs: Dict[str, Any] = {
        "rot_block_size": int(bs),
        _SCALE_FIELD[fold_order]: to_bytes_f32(s_ch.detach().float().cpu().numpy()),
    }
    if not fwht:
        # The dense rotation is shipped and absorbs the scale itself, so the
        # vector would double-count it.
        attrs.pop(_SCALE_FIELD[fold_order])
        attrs["rot_block_size"] = 0
        attrs["perm"] = rotations.to_bytes_i32(perm.detach().cpu().numpy())
        r_use = (
            rotations.fold_rotation_sq_before(rot, s_ch, perm)
            if fold_order == "before"
            else rotations.fold_rotation_sq(rot, s_ch)
        )
        attrs["rotation"] = to_bytes_f32(r_use.detach().float().cpu().numpy())
    spec = FoldSpec(**attrs)
    if not fwht:
        # Emitters that bake the matrix themselves need the tensor, not the bytes.
        spec.rotation_tensor = r_use
    return (*_pack(folded, bits, gptq), spec)


#: Symmetric code limits. The only numbers that differ between the two widths.
_QMAX = {4: 7.0, 8: 127.0}


def _pack(weight: Any, bits: int, gptq: Any | None = None) -> Tuple[bytes, bytes]:
    """Bit-width-specific packing, the only step that is not shared.

    ``gptq`` is this site's factorized Hessian from
    :func:`llm_gptq.gptq_prepare`. With it the weight is GPTQ-rounded instead of
    round-to-nearest: same fold, same rotation, same per-output-row scale, only
    the rounding changes, so nothing downstream (kernel, node attributes, byte
    order) is affected.

    Why it is worth a branch here rather than in each emitter: at 4 bits the
    error is grid-limited, not outlier-limited. Measured on a flow-matching action head,
    W4A4/W8A8 error came out at 18.3x against an ideal 127/7 = 18.14x, so the
    fold already extracts everything the grid allows. GPTQ adds no grid; it
    spends the same grid better by propagating each column's rounding error into
    the columns not yet quantized. Measured gain on those sites: 19% median,
    against 9% for grouping the weight scale.
    """
    if gptq is not None:
        from .llm_gptq import gptq_quant_codes

        codes, scale = gptq_quant_codes(weight, gptq, qmax=_QMAX[bits])
        sb = _to_np_f32(scale).tobytes()
        if bits == 4:
            return rotations.pack_int4_nibbles(codes).tobytes(), sb
        return _to_np_i8(codes).tobytes(), sb
    if bits == 4:
        packed, scale = rotations.pack_int4_colmajor(weight)
        return packed.tobytes(), scale.astype(np.float32).tobytes()
    w_i8, scale = quant_weight_per_row(weight)
    return w_i8.astype(np.int8).tobytes(), scale.astype(np.float32).tobytes()


def _to_np_f32(t: Any) -> Any:
    return (t.detach().float().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)).astype(np.float32)


def _to_np_i8(t: Any) -> Any:
    return (t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)).astype(np.int8)


def _torch() -> Any:
    import torch

    return torch


def finalize_scales(
    amax: Dict[str, Any], *, weights: Dict[str, Any] | None = None, alpha: float = 1.0
) -> Dict[str, Any]:
    """The per-channel scale each site ships, from its captured activation amax.

    ``alpha`` is SmoothQuant's migration strength, and it only has meaning in the
    RAW frame, that is, under ``fold_order="before"``. There the scale divides the
    activation and multiplies the weight, so how much of the outlier burden moves
    across is a choice::

        s = a^alpha / w^(1 - alpha)

    ``alpha=1.0`` (the default, and the only meaningful value for the
    post-rotation fold) is pure activation amax: the weights absorb everything,
    so a quiet channel's weight column grows without limit and, with per-ROW
    weight scales, costs the rest of the row its resolution. ``alpha=0.5``
    splits the burden, which is what SmoothQuant does.

    Args:
        amax: per-key captured activation amax (the accumulator's output).
        weights: per-key weight sharing that key's input channel, needed when
            ``alpha != 1.0``. For merged groups pass the concatenation (Q+K+V
            together, every cross-attention KV for a shared encoder group). The
            scale is per INPUT channel, so every weight reading that input must
            be in the amax.
        alpha: migration strength in ``[0, 1]``.
    """
    if not amax:
        raise RuntimeError(
            "W4A4 calibration replay produced zero forward passes; the SmoothQuant fold "
            "cannot be computed. Check the calibration manifest/capture."
        )
    if alpha != 1.0 and weights is None:
        raise ValueError(
            f"alpha={alpha} needs the group weights: s = a^alpha / w^(1-alpha) has a weight term. "
            "Pass weights=, or leave alpha=1.0 for the pure activation-amax fold."
        )
    out: Dict[str, Any] = {}
    for key, act in amax.items():
        if alpha == 1.0:
            s = act
        elif weights is None:  # unreachable: guarded above, but mypy cannot see it
            raise AssertionError("alpha != 1.0 requires weights")
        else:
            w_amax = weights[key].float().abs().amax(dim=0).clamp(min=1e-8).to(act.device)
            s = (act.clamp(min=1e-8).pow(alpha) / w_amax.pow(1.0 - alpha)).clamp(min=1e-5, max=1e5)
        # Relative floor, not absolute. The fold divides by s, so a channel that
        # happened to be near-zero over the calibration set would be amplified
        # without bound; at inference the activation quantizer takes a per-token
        # amax over all channels, so one off-distribution value there would set
        # the row scale and drive every other channel to q=0.
        out[key] = s.clamp(min=float(s.max()) * SQ_SCALE_REL_FLOOR).cpu()
    return out


def hessian_accumulator(rot: Dict[str, tuple], scales: Dict[str, Any], fold_order: str = "before") -> tuple:
    """Hooks that accumulate each site's GPTQ Hessian, in the frame the kernel sees.

    A second calibration pass, after :func:`scale_accumulator` has produced the
    scales: GPTQ compensates the rounding error of the weight the engine actually
    stores, so its Hessian has to be built on the activation that weight actually
    multiplies, rotated, and divided by the SmoothQuant vector. A Hessian taken
    on the raw activation compensates an error that never occurs.

    Returns ``(hessians, accum)``; feed each site's entry to
    :func:`llm_gptq.gptq_prepare`, then hand the result to :func:`fold_site` as
    ``gptq=``.
    """
    torch = _torch()
    hess: Dict[str, Any] = {}
    # The Hessians live on the host in float64 (129 DiT sites would not fit
    # next to the model on a 16 GB card), so every call ships a K x K matrix
    # across PCIe. Staging it as fp32 in pinned memory and adding in place is
    # 3x faster than ``.double().cpu()`` plus an out-of-place add; the per-call
    # GEMM is fp32 either way, so the accumulated value is bit-identical.
    staging: Dict[int, Any] = {}

    def _to_host(h32: Any) -> Any:
        if h32.device.type == "cpu":
            return h32
        buf = staging.get(int(h32.shape[-1]))
        if buf is None:
            try:
                buf = torch.empty(h32.shape, dtype=torch.float32, pin_memory=True)
            except RuntimeError:  # no pinned allocator (CPU-only torch)
                return h32.cpu()
            staging[int(h32.shape[-1])] = buf
        buf.copy_(h32)
        return buf

    def accum(key: str, x: Any) -> None:
        perm, rmat = rot[key]
        width, bs = int(perm.numel()), int(rmat.shape[-1])
        xf = x.detach().float()
        k_in = int(xf.shape[-1])
        if k_in < width:
            xf = torch.nn.functional.pad(xf, (0, width - k_in))
        elif k_in > width:
            raise ValueError(f"{key}: activation is {k_in} wide but its rotation covers {width}.")
        xf = xf.reshape(-1, xf.shape[-1])
        s = scales.get(key)
        if s is not None:
            s = s.to(xf.device)
            xr = (
                rotations.apply_rotation(xf / s, perm, rmat, bs)
                if fold_order == "before"
                else (rotations.apply_rotation(xf, perm, rmat, bs) / s)
            )
        else:
            xr = rotations.apply_rotation(xf, perm, rmat, bs)
        h32 = xr.T @ xr
        if key not in hess:
            hess[key] = torch.zeros(h32.shape, dtype=torch.float64)
        hess[key].add_(_to_host(h32))

    return hess, accum


def scale_accumulator(rot: Dict[str, tuple], block_size: int = 0, fold_order: str = "after") -> tuple:
    """Hooks that accumulate each site's activation amax, in the frame it is folded in.

    The width and the block come from the SITE'S OWN ROTATION, not from a
    caller-supplied ``block_size``: ``perm`` is exactly as wide as the weight the
    emitter folds, and ``R.shape[-1]`` is exactly the block it rotates in.

    That matters because the two bit widths handle an awkward width differently.
    A 480-wide projection is zero-PADDED to 512 on the INT4 path, while
    the INT8 path NARROWS the block to 32 and stays at 480: a fused norm+GEMM
    plugin has nowhere to put a Pad node. A capture that assumed one policy
    produced a 512-long scale for a 480-wide site and the build died on a shape
    mismatch. Deriving both from the rotation makes the capture agree with
    whatever the emitter actually built, with no second flag to keep in sync.

    ``block_size`` is accepted and ignored; callers still pass it positionally.
    """
    torch = _torch()
    amax: Dict[str, Any] = {}

    def accum(key: str, x: Any) -> None:
        perm, rmat = rot[key]
        width = int(perm.numel())
        bs = int(rmat.shape[-1])
        xf = x.detach().float()
        k_in = int(xf.shape[-1])
        if k_in < width:
            # The rotation was built on a padded weight; the activation has to
            # reach the same width or the frames do not line up.
            xf = torch.nn.functional.pad(xf, (0, width - k_in))
        elif k_in > width:
            raise ValueError(
                f"{key}: activation is {k_in} wide but its rotation covers {width}. "
                "The capture and the emitter disagree on the site width."
            )
        if fold_order == "before":
            # SmoothRot order: the scale lands on the raw channel, so measure there.
            v = xf.abs().reshape(-1, xf.shape[-1]).amax(dim=0)
        else:
            xr = rotations.apply_rotation(xf, perm, rmat, bs)
            v = xr.abs().reshape(-1, xr.shape[-1]).amax(dim=0)
        amax[key] = v if key not in amax else torch.maximum(amax[key], v)

    return amax, accum
