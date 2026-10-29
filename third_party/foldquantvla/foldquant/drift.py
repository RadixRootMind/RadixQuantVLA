# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""Locate the channel with the largest action error.

Callers specify the flattened action layout: GR00T concatenates named channels
(channel-major, ``i // steps``), while pi flattens a ``(steps, width)`` chunk
(step-major, ``i % width``).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

__all__ = ["worst_channel"]


def worst_channel(
    delta: Any,
    *,
    order: str,
    width: Optional[int] = None,
    labels: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Where the largest ``|delta|`` sits, as ``{max_abs, channel, label, step}``.

    Args:
        delta: 1-D engine-minus-reference over one flattened action chunk.
        order: ``"channel_major"`` or ``"step_major"``; see the module
            docstring; the two index oppositely.
        width: channels per step (step-major), or steps per channel
            (channel-major). Inferred from *labels* when omitted.
        labels: channel names in concatenation order, when the family has them.

    Returns ``label: None`` for a family whose channels are unnamed.
    """
    import torch

    flat = torch.as_tensor(delta).flatten().abs()
    if flat.numel() == 0:
        raise ValueError("worst_channel: empty delta")
    index = int(torch.argmax(flat))
    total = int(flat.numel())

    if order == "channel_major":
        per = int(width) if width else (total // len(labels) if labels else 0)
        if per <= 0:
            raise ValueError("channel_major needs width (steps per channel) or labels")
        channel, step = divmod(index, per)
    elif order == "step_major":
        if not width:
            raise ValueError("step_major needs width (channels per step)")
        step, channel = divmod(index, int(width))
    else:
        raise ValueError(f"order must be 'channel_major' or 'step_major', got {order!r}")

    label = None
    if labels is not None and 0 <= channel < len(labels):
        label = str(labels[channel])
    return {
        "max_abs": float(flat[index]),
        "channel": int(channel),
        "label": label,
        "step": int(step),
    }
