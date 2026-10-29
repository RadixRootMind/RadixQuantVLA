# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""Locate the upstream Isaac GR00T tree this integration sits in."""

from __future__ import annotations

import os
import sys
from pathlib import Path

#: RadixQuantVLA's existing GR00T N1.5 tree. Keep the model implementation in
#: one place and layer FoldQuant's runtime adapter over it.
_BUNDLED_ROOT = Path(__file__).resolve().parents[5]
UPSTREAM_ROOT = Path(
    os.environ.get("FOLDQUANT_GROOT_ROOT")
    or os.environ.get("QUANTVLA_ROOT")
    or os.environ.get("AWESOME_QVLA_ROOT")
    or _BUNDLED_ROOT
).expanduser().resolve()
#: Upstream's fp16 / FP8 TensorRT path (plain scripts, not a package). Its
#: DiT graph takes three inputs (``sa_embs``, ``vl_embs``, ``timesteps_tensor``)
#: and its LLM graph is fp16, so neither is interchangeable with a FoldQuant
#: engine; the directory is referenced for documentation only.
DEPLOYMENT_DIR = UPSTREAM_ROOT / "deployment_scripts"

#: The data config upstream's LIBERO fine-tunes were trained and served with
#: (``examples/Libero/README.md``). N1.5 keeps the modality config and the
#: transforms in code, not in the checkpoint, so every tool takes
#: ``--data-config`` and defaults to this one.
LIBERO_DATA_CONFIG = "examples.Libero.custom_data_config:LiberoDataConfig"

#: The two modules FoldQuant replaces, the graph each is exported to and the
#: engine it is served from.
COMPONENTS = (
    ("llm", "llm_bf16.onnx", "llm_bf16.engine"),
    ("dit", "dit_bf16.onnx", "dit_bf16.engine"),
)

#: Names of the FoldQuant export manifest, the shape-hint file the export
#: writes beside it, and the record :mod:`.build_engines` leaves in the engine
#: directory.
MANIFEST_NAME = "foldquant_export.json"
EXPORT_METADATA_NAME = "export_metadata.json"
ENGINES_RECORD_NAME = "foldquant_engines.json"


def ensure_upstream_on_path() -> None:
    """Make ``examples.Libero.*`` importable (idempotent).

    Upstream addresses its LIBERO data config and evaluation helpers as
    ``examples.Libero...``, a package that exists only relative to the
    repository root, which upstream's own scripts assume is the working
    directory.
    """
    root = str(UPSTREAM_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _seed_libero_config(benchmark_root: Path) -> None:
    """Point LIBERO at this checkout (:func:`foldquant.libero_config.use_checkout`).

    ``libero.libero`` reads its task files and initial states from
    ``$LIBERO_CONFIG_PATH/config.yaml`` (default ``~/.libero``), which another
    project on the machine may have pointed at its own checkout; and with no
    config at all it asks on stdin, which a script cannot answer. The config
    used here names this checkout and lives in the FoldQuant cache.
    """
    from foldquant.libero_config import use_checkout

    use_checkout(benchmark_root)


#: Where a LIBERO checkout may be named, for the families whose release pins
#: none. The N1.5 release ships ``examples/Libero`` (the client loop) but not
#: the benchmark itself, so the checkout is the operator's to supply.
LIBERO_DIR_ENV = "FOLDQUANT_LIBERO_DIR"


def ensure_libero_on_path() -> None:
    """Make a LIBERO checkout importable (idempotent).

    Unlike the N1.6 / N1.7 releases this one pins no LIBERO submodule, so there
    is nothing in-tree to point at and the location cannot be hard-coded: it
    differs per machine and naming one would put somebody's filesystem in a
    public repository. An installed ``libero`` satisfies this outright;
    otherwise ``FOLDQUANT_LIBERO_DIR`` names a checkout, which is put on
    ``sys.path`` (LIBERO's ``libero/`` carries no ``__init__.py``, so an
    editable install of it leaves nothing importable; the directory itself has
    to be on the path).
    """
    import importlib.util

    spec = importlib.util.find_spec("libero")
    if spec is not None:
        for loc in spec.submodule_search_locations or ():
            if (Path(loc) / "libero").is_dir():
                _seed_libero_config(Path(loc) / "libero")
                break
        return
    named = os.environ.get(LIBERO_DIR_ENV)
    if not named:
        raise RuntimeError(
            "LIBERO is not importable and this release pins no copy of it. Point "
            f"{LIBERO_DIR_ENV} at a LIBERO checkout, e.g.\n"
            f"    {LIBERO_DIR_ENV}=/path/to/LIBERO python -m foldquant_integration.eval_libero ...\n"
            "and install the simulator stack (see this integration's README)."
        )
    root = Path(named).expanduser().resolve()
    if not (root / "libero").is_dir():
        raise RuntimeError(f"{LIBERO_DIR_ENV}={root} does not look like a LIBERO checkout (no libero/ inside)")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    _seed_libero_config(root / "libero" / "libero")
