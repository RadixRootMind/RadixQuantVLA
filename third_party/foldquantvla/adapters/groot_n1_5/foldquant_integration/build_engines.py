# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""Compile an N1.5 engine directory from FoldQuant graphs.

The graphs are compiled with :func:`foldquant.runtime.builder.build_engine`
(STRONGLY_TYPED; the plugin nodes declare their own dtypes), after the plugin
libraries the export manifest names are loaded (built for this device when no
cached binary matches). Shape profiles come from the ONNX value infos plus the
captured shapes the export recorded in ``export_metadata.json``: the state +
future-vision + action stream is static, the vision-language stream and the
LLM sequence get a range around the captured length because the task text
differs per task.

Only FoldQuant graphs are built here. Upstream's ``deployment_scripts/export_onnx.py``
writes a three-input fp16 DiT (``sa_embs``, ``vl_embs``, ``timesteps_tensor``)
and an fp16 LLM that are served by its own ``trt_model_forward.py``; those
contracts differ from FoldQuant's (see ``runtime.py``) and are not loaded by
:func:`foldquant_integration.runtime.install_engines`. A module the export
left float (``--dit-scheme none`` / ``--llm-scheme none``) stays in PyTorch.

Example::

    python -m foldquant_integration.build_engines \\
        --onnx-dir exports/n15_w4a4/onnx --engine-dir exports/n15_w4a4/engines
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tyro
from foldquant.provenance import public_path
from foldquant.runtime.builder import build_engine, profiles_from_onnx
from foldquant.runtime.plugins import prepare_plugins

from ._upstream import COMPONENTS, ENGINES_RECORD_NAME, EXPORT_METADATA_NAME, MANIFEST_NAME

logger = logging.getLogger("foldquant.groot_n1_5.build")

#: RoPE table length the FoldQuant LLM graph bakes (``foldquant.export.LLM_BAKE_MAX_SEQ_LEN``).
LLM_BAKED_MAX_SEQ_LEN = 4096


@dataclass
class BuildConfig:
    onnx_dir: str
    """The FoldQuant export's ``onnx/`` directory (holds foldquant_export.json and export_metadata.json)."""

    engine_dir: str
    """Destination engine directory."""

    llm_max_seq_len: int | None = None
    """Upper bound of the LLM sequence profile (default: ``max(2 * captured, captured + 64)``, capped at 4096)."""

    vl_max_seq_len: int | None = None
    """Upper bound of the DiT's vision-language sequence profile (same default rule)."""

    workspace_mb: int = 8192


def _default_max(opt: int) -> int:
    return max(2 * opt, opt + 64)


def load_manifest(onnx_dir: Path) -> dict[str, Any]:
    return json.loads((onnx_dir / MANIFEST_NAME).read_text())


def dim_ranges(metadata: dict[str, Any], args: BuildConfig) -> dict[str, Any]:
    """``{dim_name: int | (min, opt, max)}`` for every symbolic dimension the two graphs carry."""
    llm_opt = int(metadata["llm_seq_len"])
    vl_opt = int(metadata["vl_seq_len"])
    llm_max = args.llm_max_seq_len or min(_default_max(llm_opt), LLM_BAKED_MAX_SEQ_LEN)
    vl_max = args.vl_max_seq_len or _default_max(vl_opt)
    if not llm_opt <= llm_max <= LLM_BAKED_MAX_SEQ_LEN:
        raise ValueError(
            f"--llm-max-seq-len {llm_max} must be >= the captured {llm_opt} and <= {LLM_BAKED_MAX_SEQ_LEN}"
        )
    if vl_max < vl_opt:
        raise ValueError(f"--vl-max-seq-len {vl_max} is below the captured {vl_opt}")
    return {
        "batch": 1,
        "batch_size": 1,
        "seq_len": (1, llm_opt, llm_max),
        "vl_seq_len": (1, vl_opt, vl_max),
        "sa_seq_len": int(metadata["sa_seq_len"]),
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
    ranges = dim_ranges(metadata, args)
    logger.info("dimension ranges: %s", ranges)

    manifest = load_manifest(onnx_dir)
    plugin_libs: list[str] = list(manifest["plugin_libs"])
    if plugin_libs:
        prepare_plugins(plugin_libs)

    record: dict[str, Any] = {
        "onnx_dir": public_path(str(onnx_dir)),
        "metadata": metadata,
        "dim_ranges": {k: v for k, v in ranges.items()},
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
