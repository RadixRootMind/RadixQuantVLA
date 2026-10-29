# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""Compile a Pi0.5 engine directory from FoldQuant graphs.

The graphs are compiled with :func:`foldquant.runtime.builder.build_engine`
(STRONGLY_TYPED; the plugin nodes declare their own dtypes), after the plugin
libraries the export manifest names are loaded (built for this device when no
cached binary matches). Both graphs are static in practice: the LLM prefix
graph was emitted with its sequence length pinned, and the expert graph's
one symbolic axis (``prefix_len``, the KV-stack length it cross-attends) is
profiled at exactly the captured prefix: upstream pads the prompt to a fixed
token count, so no other length ever reaches it.

A module the export left float (``--expert-scheme none`` / ``--llm-scheme
none``) stays in PyTorch.

Example::

    python -m foldquant_integration.build_engines \\
        --onnx-dir exports/pi05_w4a4/onnx --engine-dir exports/pi05_w4a4/engines
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import shutil
import time
from typing import Any

from foldquant.provenance import public_path
from foldquant.runtime.builder import build_engine
from foldquant.runtime.builder import profiles_from_onnx
from foldquant.runtime.plugins import prepare_plugins
import tyro

from ._upstream import COMPONENTS
from ._upstream import ENGINES_RECORD_NAME
from ._upstream import EXPORT_METADATA_NAME
from ._upstream import MANIFEST_NAME

logger = logging.getLogger("foldquant.pi05.build")


@dataclass
class BuildConfig:
    onnx_dir: str
    """The FoldQuant export's ``onnx/`` directory (holds foldquant_export.json and export_metadata.json)."""

    engine_dir: str
    """Destination engine directory."""

    workspace_mb: int = 8192


def load_manifest(onnx_dir: Path) -> dict[str, Any]:
    return json.loads((onnx_dir / MANIFEST_NAME).read_text())


def dim_ranges(metadata: dict[str, Any]) -> dict[str, Any]:
    """``{dim_name: int}`` for the symbolic dimensions the graphs may carry, all pinned to the capture."""
    prefix_len = int(metadata["prefix_len"])
    return {
        "batch": 1,
        "batch_size": 1,
        "prefix_len": prefix_len,
        "seq_len": prefix_len,
        "seq_len2": prefix_len,
    }


def build(args: BuildConfig) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    onnx_dir = Path(args.onnx_dir)
    engine_dir = Path(args.engine_dir)
    engine_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = onnx_dir / EXPORT_METADATA_NAME
    if not metadata_path.is_file():
        raise SystemExit(f"captured shapes not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    ranges = dim_ranges(metadata)
    logger.info("dimension ranges: %s", ranges)

    manifest = load_manifest(onnx_dir)
    plugin_libs: list[str] = list(manifest["plugin_libs"])
    if plugin_libs:
        prepare_plugins(plugin_libs)

    record: dict[str, Any] = {
        "onnx_dir": public_path(str(onnx_dir)),
        "metadata": metadata,
        "dim_ranges": dict(ranges),
        "plugin_libs": plugin_libs,
        "schemes": dict(manifest["schemes"]),
        "components": {},
    }
    for name, _onnx_name, engine_name in COMPONENTS:
        if name not in manifest["files"]:
            logger.info("%s: not exported, stays in PyTorch", name)
            continue
        src = onnx_dir / manifest["files"][name]
        if not src.is_file():
            raise FileNotFoundError(src)
        profiles = profiles_from_onnx(src, ranges)
        # A graph of the all-float baseline carries no plugin nodes; still built STRONGLY_TYPED,
        # so it honours its own bf16 dtypes and differs from the quantized arms only in the
        # precision of the projections.
        is_quantized = name in manifest.get("schemes", {})
        t0 = time.time()
        logger.info("%s: building %s from %s", name, engine_name, src.name)
        build_engine(
            src,
            engine_dir / engine_name,
            profiles=profiles,
            plugin_libs=plugin_libs if is_quantized else (),
            strongly_typed=True,
            int8=is_quantized,
            workspace_mb=args.workspace_mb,
        )
        record["components"][name] = {
            "onnx": str(src),
            "engine": engine_name,
            "source": "foldquant",
            "profiles": {k: {"min": v.min, "opt": v.opt, "max": v.max} for k, v in profiles.items()},
            "seconds": round(time.time() - t0, 1),
        }
        logger.info("%s: built in %.0fs", name, time.time() - t0)
    if not record["components"]:
        raise SystemExit("nothing to build")

    (engine_dir / ENGINES_RECORD_NAME).write_text(json.dumps(record, indent=2))
    shutil.copy2(onnx_dir / MANIFEST_NAME, engine_dir / MANIFEST_NAME)
    logger.info("engine directory complete: %s", engine_dir)
    return engine_dir


if __name__ == "__main__":
    build(tyro.cli(BuildConfig))
