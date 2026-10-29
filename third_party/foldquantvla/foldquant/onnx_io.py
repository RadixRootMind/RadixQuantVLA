# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""Save ONNX graphs without appending to stale external weights.

Plugin exports can overwrite a float graph at the same path. Because
``onnx.save_model`` appends to an existing sidecar, remove it before saving;
otherwise unused float weights inflate the file and can exceed parser limits.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["save_plugin_onnx"]


def save_plugin_onnx(model: Any, output_path: Path | str, **save_kwargs: Any) -> Path:
    """Save *model* to *output_path* with a freshly written external-data sidecar.

    Args:
        model: The ``onnx.ModelProto`` to write.
        output_path: Destination ``.onnx`` path. Any sidecar left by a previous
            write of this path is removed first.
        **save_kwargs: Extra ``onnx.save_model`` options (e.g. ``size_threshold``,
            ``convert_attribute``). The external-data options are fixed.

    Returns:
        The saved ONNX ``Path``.
    """
    import onnx

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sidecar = out_path.with_name(out_path.name + ".data")
    # Truncate rather than append: see the module docstring.
    sidecar.unlink(missing_ok=True)

    onnx.save_model(
        model,
        str(out_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=sidecar.name,
        **save_kwargs,
    )
    return out_path


def graph_digest(model: Any) -> str:
    """SHA-256 over a ``ModelProto``'s content, computed in memory before it is written.

    Every node, initializer, input, output and value_info is hashed in its
    deterministic serialization, plus the sorted opset imports: what the graph
    computes and the bytes of every weight it carries, but not the file header
    (``ir_version``, producer fields) an ``onnx`` release stamps. A quantization
    records this for each graph it built; an export of the same quant state
    rebuilds the graph and must reproduce it. Element by element, so a graph of
    several gigabytes is never serialized as one buffer.
    """
    import hashlib

    h = hashlib.sha256()
    g = model.graph
    h.update(g.name.encode() + b"\0")
    for field in ("node", "initializer", "input", "output", "value_info"):
        h.update(field.encode() + b"\0")
        for item in getattr(g, field):
            h.update(item.SerializeToString(deterministic=True))
            h.update(b"\0")
    for op in sorted((o.domain, o.version) for o in model.opset_import):
        h.update(repr(op).encode())
    return h.hexdigest()
