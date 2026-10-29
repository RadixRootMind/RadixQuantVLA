# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""Point LIBERO at the checkout a family pins, not at whatever ``~/.libero`` names.

``libero.libero`` reads its paths (task BDDL files, initial states, assets)
from ``$LIBERO_CONFIG_PATH/config.yaml``, default ``~/.libero/config.yaml``,
when it is first imported. That file is per user, not per checkout: on a
machine where another project has set it, a FoldQuant rollout imports the
pinned LIBERO code but reads another checkout's tasks and initial states.
And with no file at all, LIBERO asks on stdin, which a script cannot answer.

:func:`use_checkout` gives each checkout a config of its own, under the
FoldQuant cache (``$FOLDQUANT_CACHE_DIR``, default ``~/.cache/foldquant``),
and sets ``LIBERO_CONFIG_PATH`` to it for this process and every child it
starts (the π₀.₅ LIBERO client). A ``LIBERO_CONFIG_PATH`` the user set is
honoured. Torch-free.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Any, Dict

__all__ = ["CONFIG_ENV", "config_for", "use_checkout"]

CONFIG_ENV = "LIBERO_CONFIG_PATH"
#: Set by :func:`use_checkout` beside ``LIBERO_CONFIG_PATH``, so a later call
#: can tell a path it chose from one the user set.
_OWNED_ENV = "FOLDQUANT_LIBERO_CONFIG_OWNED"


def config_for(benchmark_root: Any) -> Dict[str, str]:
    """LIBERO's default paths for the checkout whose ``libero/libero`` is *benchmark_root*
    (what upstream's ``setup_libero.sh`` writes)."""
    root = str(Path(benchmark_root).resolve())
    return {
        "benchmark_root": root,
        "bddl_files": os.path.join(root, "./bddl_files"),
        "init_states": os.path.join(root, "./init_files"),
        "datasets": os.path.join(root, "../datasets"),
        "assets": os.path.join(root, "./assets"),
    }


def _cache_root() -> Path:
    return Path(os.environ.get("FOLDQUANT_CACHE_DIR", Path.home() / ".cache" / "foldquant"))


def _write(config_file: Path, config: Dict[str, str]) -> None:
    text = "".join(f"{k}: {v}\n" for k, v in config.items())
    if config_file.is_file() and config_file.read_text() == text:
        return
    config_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = config_file.with_name(f".{config_file.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    tmp.replace(config_file)


def use_checkout(benchmark_root: Any) -> Path:
    """Make LIBERO read its paths from the checkout at *benchmark_root*; returns the config directory.

    Call it before anything imports ``libero.libero``. With a user-set
    ``LIBERO_CONFIG_PATH``, a missing config there is seeded for this checkout
    and an existing one is left as it is. Otherwise the config lives under the
    FoldQuant cache, keyed by the checkout, and is rewritten to match it.

    Raises:
        RuntimeError: ``libero.libero`` was already imported, reading another checkout.
    """
    config = config_for(benchmark_root)
    user_dir = os.environ.get(CONFIG_ENV)
    if user_dir and os.environ.get(_OWNED_ENV) != user_dir:
        config_dir = Path(user_dir).expanduser()
        if not (config_dir / "config.yaml").is_file():
            _write(config_dir / "config.yaml", config)
    else:
        key = hashlib.sha256(config["benchmark_root"].encode()).hexdigest()[:12]
        config_dir = _cache_root() / "libero" / key
        _write(config_dir / "config.yaml", config)
        os.environ[CONFIG_ENV] = str(config_dir)
        os.environ[_OWNED_ENV] = str(config_dir)
    loaded = sys.modules.get("libero.libero")
    if loaded is not None:
        have = loaded.get_libero_path("benchmark_root")
        if Path(have).resolve() != Path(config["benchmark_root"]):
            raise RuntimeError(
                f"libero.libero was imported before its config was set and reads {have}, "
                f"not the pinned checkout {config['benchmark_root']}"
            )
    return config_dir
