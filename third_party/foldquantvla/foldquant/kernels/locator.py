# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""Resolve TensorRT plugin libraries for the current device.

Binaries are selected by CUDA SM, machine ABI, and TensorRT version.
Packaged libraries use ``<libname>.<target_slug>.so``; cached builds use
``<cache>/<slug>/<libname>.so``. Heavy dependencies are imported lazily,
and resolution never triggers a build.
"""

from __future__ import annotations

import ctypes
import logging
import os
import re
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

from foldquant.kernels.registry import TENSORRT_KERNELS

logger = logging.getLogger(__name__)

# Standardized logical library names. Committed: ``<libname>.<slug>.so``;
# rebuilt cache: ``<cache>/<slug>/<libname>.so``.
INT8_PER_ROW_LIB = "foldquant_int8_per_row"
INT4_GROUPWISE_LIB = "foldquant_int4_groupwise"
INT4_PER_ROW_LIB = "foldquant_int4_per_row"

# Derived from the registry rather than restated: adding a kernel there makes it
# resolvable and rebuildable here with no second edit.
KNOWN_PLUGIN_LIBS: Tuple[str, ...] = tuple(sorted(TENSORRT_KERNELS))

# Best-effort C entry points for static-init registration that may not fire under
# ``ctypes.CDLL``. Probed (``hasattr``) after dlopen; absent symbols are skipped.
# Re-calling an init after the load-time ``__attribute__((constructor))`` already
# registered the creator is a no-op (duplicate ``registerCreator`` returns false).
# The INT4 weight-only groupwise plugin registers via ``REGISTER_TENSORRT_PLUGIN``
# only and is covered by the trailing ``init_libnvinfer_plugins`` refresh.
KNOWN_INIT_SYMBOLS: Tuple[str, ...] = tuple(
    dict.fromkeys(sym for spec in TENSORRT_KERNELS.values() for sym in spec.init_symbols)
)

# Process-wide record of already-dlopened ``.so`` paths (registration is global).
_LOADED: set[str] = set()


class MissingPluginError(RuntimeError):
    """No usable plugin ``.so`` exists for the running device tuple."""


def current_target_tuple() -> Tuple[str, str, str]:
    """Return ``(sm, machine, trt)`` identifying a compatible plugin binary.

    e.g. ``("sm86", "x86_64", "trt10.15")``. Requires CUDA, torch, tensorrt.
    """
    import platform

    import tensorrt as trt
    import torch

    major, minor = torch.cuda.get_device_capability()
    sm = f"sm{major}{minor}"
    machine = platform.machine()
    ver = str(trt.__version__).split(".")
    return sm, machine, f"trt{ver[0]}.{ver[1]}"


def target_slug() -> str:
    """``sm<CC>-<machine>-trt<MAJ>.<MIN>`` for the running device."""
    return "-".join(current_target_tuple())


def sm_arch_number() -> str:
    """CUDA arch number for ``-DCMAKE_CUDA_ARCHITECTURES`` (e.g. ``"86"``)."""
    return current_target_tuple()[0][2:]


def committed_plugin_dir() -> Path:
    """Filesystem dir of the checked-in, target-named ``.so`` (package-data).

    Reached by path navigation from this package (``foldquant.kernels``)
    only, never from the export code, which keeps the build->runtime
    boundary intact.
    """
    from importlib.resources import files

    return Path(str(files("foldquant.kernels").joinpath("prebuilt")))


def plugin_cache_dir() -> Path:
    """Out-of-tree cache for device-mismatch rebuilds: ``<root>/plugins/<slug>``.

    Mirrors (without importing) the cache-root precedence used elsewhere:
    ``FOLDQUANT_CACHE_DIR`` -> ``$XDG_CACHE_HOME/foldquant`` -> ``~/.cache/foldquant``.
    Never the committed dir, so a rebuild is never accidentally committed.
    """
    env = os.environ.get("FOLDQUANT_CACHE_DIR")
    if env:
        root = Path(env).expanduser()
    else:
        xdg = os.environ.get("XDG_CACHE_HOME")
        root = (Path(xdg).expanduser() / "foldquant") if xdg else (Path.home() / ".cache" / "foldquant")
    return root / "plugins" / target_slug()


def resolve_plugin_so(libname: str) -> Optional[Path]:
    """Path to a usable ``.so`` for *libname* on this device, or ``None``.

    Resolution order: exact committed name -> version-tolerant committed match ->
    a prior rebuild in the cache. Version tolerance: a binary built for the same
    SM + machine + TensorRT **major** loads on a host with an equal-or-higher TRT
    **minor** (the IPluginV3 ABI is stable within a major); a minor mismatch is
    loaded with a warning. SM and machine must always match exactly (SASS / ELF).
    """
    sm, machine, trt = current_target_tuple()
    committed_dir = committed_plugin_dir()

    exact = committed_dir / f"{libname}.{sm}-{machine}-{trt}.so"
    if exact.is_file():
        return exact

    # Version-tolerant: highest committed minor <= host minor, same SM/machine/major.
    host_major, host_minor = (int(x) for x in trt[3:].split("."))
    pattern = re.compile(rf"^{re.escape(libname)}\.{sm}-{machine}-trt{host_major}\.(\d+)\.so$")
    best: Optional[Path] = None
    best_minor = -1
    for cand in committed_dir.glob(f"{libname}.{sm}-{machine}-trt{host_major}.*.so"):
        match = pattern.match(cand.name)
        if match is None:
            continue
        minor = int(match.group(1))
        if minor <= host_minor and minor > best_minor:
            best, best_minor = cand, minor
    if best is not None:
        if best_minor != host_minor:
            logger.warning(
                "Loading TRT plugin %s built for TRT %d.%d on host TRT %d.%d; forward-compatible "
                "within major %d, but verify if you hit plugin/ABI errors.",
                libname,
                host_major,
                best_minor,
                host_major,
                host_minor,
                host_major,
            )
        return best

    cached = plugin_cache_dir() / f"{libname}.so"
    if cached.is_file():
        return cached
    return None


def declared_plugin_libs(extra: Any) -> List[str]:
    """The plugin libraries a module's ``quantization.extra`` declares, as a list.

    ``plugin_lib`` is a single name for a homogeneous graph and a list for a
    mixed-width one (an INT4 LLM whose ``site_bits`` keep some sites at INT8
    carries nodes from both per-row libraries). Order is preserved; missing
    or empty declarations yield ``[]``.
    """
    declared = (extra or {}).get("plugin_lib")
    if not declared:
        return []
    return [declared] if isinstance(declared, str) else list(declared)


def is_plugin_loaded(path: Path) -> bool:
    """True once *path* has been ``dlopen``-ed into this process by :func:`load_plugin_libs`."""
    return str(path) in _LOADED


def load_plugin_libs(paths: Sequence[Path]) -> None:
    """``ctypes.CDLL`` each not-yet-loaded ``.so``, probe init symbols, refresh registry."""
    new = [p for p in paths if str(p) not in _LOADED]
    if not new:
        return
    import tensorrt as trt

    for path in new:
        handle = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        logger.info("Loaded TRT plugin library: %s", path)
        for sym in KNOWN_INIT_SYMBOLS:
            if hasattr(handle, sym):
                getattr(handle, sym).restype = ctypes.c_int
                rc = getattr(handle, sym)()
                logger.debug("  %s() -> %s", sym, rc)
        _LOADED.add(str(path))
    trt.init_libnvinfer_plugins(trt.Logger(trt.Logger.WARNING), "")


def load_required_plugins(libnames: Iterable[str] = KNOWN_PLUGIN_LIBS) -> List[Path]:
    """Resolve + load every requested lib; raise ``MissingPluginError`` on any miss.

    Used at runtime before engine deserialization: a missing device-matched
    binary is a deployment error, not something to silently rebuild mid-inference.
    """
    names = list(libnames)
    paths: List[Path] = []
    missing: List[str] = []
    for name in names:
        resolved = resolve_plugin_so(name)
        (paths.append(resolved) if resolved is not None else missing.append(name))
    if missing:
        raise MissingPluginError(
            f"No TensorRT plugin {missing} for target {target_slug()!r}. Searched committed dir "
            f"{committed_plugin_dir()} and cache {plugin_cache_dir()}. This host matches no committed "
            f"binary. Run `python -m foldquant.kernels build` (rebuilds from "
            f"foldquant/kernels into the cache) or commit a matching prebuilt .so."
        )
    load_plugin_libs(paths)
    return paths
