# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""The fake-quant projection shared by every module kind, and the swap bookkeeping.

A FoldQuant projection runs, in the plugin kernels::

    x -> (/ s) -> permute + block-rotate -> per-token quantize -> integer GEMM -> * s_tok * s_w + b

:class:`FakeQuantLinear` computes exactly that in fp32 from the integer weight
codes and per-row scales the engine is built from, so the PyTorch module and
the engine start from the same numbers and differ only by fp32 against INT32
accumulation. The LLM (:mod:`foldquant.quantized`) and the GR00T DiT
(:mod:`foldquant.dit_fake_quant`) replace their quantized ``nn.Linear`` s with it.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

__all__ = ["FakeQuantLinear", "SwapHandle", "swap_module"]


class FakeQuantLinear(nn.Module):
    """A quantized projection: activation transform, per-token quantization, integer GEMM.

    Args:
        codes: ``(N, K)`` integer weight codes (int8 storage).
        w_scale: ``(N,)`` per-output-row weight scale.
        bias: ``(N,)`` bias or ``None``.
        a_qmax: activation code limit (7 / 127), or ``None`` for a weight-only site
            whose activation stays in floating point (the INT4 AdaLN GEMV).
        perm, rot, rot_bs: ``x[..., perm]`` then a block rotation by ``rot``
            (``(K/bs, bs, bs)``) over blocks of ``rot_bs``; ``rot=None`` for no rotation.
        s_pre: raw-frame SmoothQuant vector divided out before the rotation
            (butterfly sites), or ``None`` when it is folded into ``rot``.
        s_post: rotated-frame SmoothQuant vector divided out after the rotation
            (a butterfly site folded "after", the kernel's ``act_scale_ch``).
        a_clip: activation clip ratio; the per-token scale is ``a_clip * amax / a_qmax``
            and codes clamp (the INT4 LLM sites' ``act_clip_ratio``).
        stage_bf16: round the rotated activation to bf16 before quantizing, as the
            dense arm's cuBLAS prologue does.
    """

    def __init__(
        self,
        codes: torch.Tensor,
        w_scale: torch.Tensor,
        bias: Optional[torch.Tensor],
        *,
        a_qmax: Optional[float],
        perm: Optional[torch.Tensor] = None,
        rot: Optional[torch.Tensor] = None,
        rot_bs: int = 0,
        s_pre: Optional[torch.Tensor] = None,
        s_post: Optional[torch.Tensor] = None,
        stage_bf16: bool = False,
        a_clip: float = 1.0,
    ) -> None:
        super().__init__()
        self.out_features, self.in_features = int(codes.shape[0]), int(codes.shape[1])
        self.register_buffer("codes", codes.to(torch.int8).contiguous())
        self.register_buffer("w_scale", w_scale.float().contiguous())
        self.register_buffer("bias", None if bias is None else bias.float().contiguous())
        self.register_buffer("perm", None if perm is None else perm.long().contiguous())
        self.register_buffer("rot", None if rot is None else rot.float().contiguous())
        self.register_buffer("s_pre", None if s_pre is None else s_pre.float().contiguous())
        self.register_buffer("s_post", None if s_post is None else s_post.float().contiguous())
        # The dtype the replaced Linear computed in. Upstream code reads
        # ``proj.weight.dtype`` to pick a cast (openpi's Gemma); ``weight`` answers
        # that and nothing else: an empty tensor, since this module's weight is
        # integer codes in the rotated frame, not a dense matrix.
        self.register_buffer("_dtype_probe", torch.empty(0), persistent=False)
        self.a_qmax = a_qmax
        self.rot_bs = int(rot_bs)
        self.stage_bf16 = bool(stage_bf16)
        self.a_clip = float(a_clip)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """The activation the kernel quantizes, in fp32."""
        from .rotation import apply_rotation

        xf = x.float()
        if self.s_pre is not None:
            xf = xf / self.s_pre
        if self.rot is not None:
            perm = self.perm if self.perm is not None else torch.arange(xf.shape[-1], device=xf.device)
            xf = apply_rotation(xf, perm, self.rot, self.rot_bs)
            if self.stage_bf16:
                xf = xf.to(torch.bfloat16).float()
        if self.s_post is not None:
            xf = xf / self.s_post
        return xf

    @property
    def weight(self) -> torch.Tensor:
        """An empty tensor in the replaced Linear's dtype, for callers that read ``.weight.dtype``."""
        return self._dtype_probe

    def set_io_dtype(self, dtype: torch.dtype) -> "FakeQuantLinear":
        self._dtype_probe = torch.empty(0, dtype=dtype, device=self.codes.device)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xt = self.transform(x)
        w = self.codes.float()
        if self.a_qmax is None:
            y = (xt @ w.t()) * self.w_scale
        else:
            amax = xt.abs().amax(dim=-1, keepdim=True)
            s_tok = (self.a_clip * amax / self.a_qmax).clamp(min=1e-8)
            q = torch.round(xt / s_tok).clamp(-self.a_qmax, self.a_qmax)
            y = (q @ w.t()) * s_tok * self.w_scale
        if self.bias is not None:
            y = y + self.bias
        return y.to(x.dtype)

    def extra_repr(self) -> str:
        kind = "weight-only" if self.a_qmax is None else f"a_qmax={self.a_qmax:g}" + (f", clip={self.a_clip:g}" if self.a_clip != 1.0 else "")
        rot = "none" if self.rot is None else f"block {self.rot_bs}"
        return f"in={self.in_features}, out={self.out_features}, {kind}, rotation={rot}"



class SwapHandle:
    """Undo token: puts the original modules back."""

    def __init__(self, swaps: Optional[List[Tuple[nn.Module, str, nn.Module]]] = None) -> None:
        self._swaps = list(swaps or [])

    def add(self, parent: nn.Module, name: str, original: nn.Module) -> None:
        self._swaps.append((parent, name, original))

    def remove(self) -> None:
        for parent, name, original in reversed(self._swaps):
            setattr(parent, name, original)
        self._swaps = []

    def __len__(self) -> int:
        return len(self._swaps)


def swap_module(handle: SwapHandle, parent: nn.Module, name: str, new: nn.Module) -> None:
    """``parent.<name> = new`` on the original's device, recorded on *handle*."""
    original = getattr(parent, name)
    ref = next(original.parameters())
    handle.add(parent, name, original)
    new = new.to(ref.device)
    if isinstance(new, FakeQuantLinear):
        new.set_io_dtype(ref.dtype)
    setattr(parent, name, new)
