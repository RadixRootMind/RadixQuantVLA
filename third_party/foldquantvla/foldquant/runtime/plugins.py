# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""Load TensorRT plugins before graph parsing or engine deserialization.

``prepare_plugins`` resolves or builds libraries, then loads them for an
engine build. ``load_plugins`` only resolves and loads existing binaries;
missing libraries raise ``MissingPluginError`` at runtime.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List

from ..kernels import locator

logger = logging.getLogger(__name__)

__all__ = ["load_plugins", "prepare_plugins", "warn_tensor_core_fit"]


def warn_tensor_core_fit(libnames: Iterable[str]) -> None:
    """Log a warning for every lib whose tensor-core path this GPU lacks.

    A W4A4 engine on a device without s4 tensor cores (Hopper) is correct but
    ~15x slower, and the mismatch is otherwise silent. Best effort: a probe
    failure logs at debug and never blocks a build.
    """
    try:
        import torch

        from ..kernels.registry import tensor_core_support_warning

        cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else None
        for lib in libnames:
            msg = tensor_core_support_warning(lib, cap)
            if msg:
                logger.warning("%s", msg)
    except Exception:
        logger.debug("tensor-core support probe failed; continuing.", exc_info=True)


def prepare_plugins(libnames: Iterable[str]) -> List[Path]:
    """Resolve (rebuilding if needed) and load *libnames*; returns the loaded paths."""
    names = list(dict.fromkeys(libnames))
    if not names:
        return []
    from ..kernels.build import ensure_plugins_for_current_device

    warn_tensor_core_fit(names)
    paths = ensure_plugins_for_current_device(names)
    locator.load_plugin_libs(paths)
    return paths


def load_plugins(libnames: Iterable[str]) -> List[Path]:
    """Resolve and load *libnames* without rebuilding; raises ``MissingPluginError`` on a miss."""
    names = list(dict.fromkeys(libnames))
    if not names:
        return []
    return locator.load_required_plugins(names)
