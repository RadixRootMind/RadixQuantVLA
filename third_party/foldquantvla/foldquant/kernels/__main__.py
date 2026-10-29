# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""``python -m foldquant.kernels build [--force] [LIB ...]``: build the TensorRT plugins for this device.

Binaries land in the out-of-tree cache (``FOLDQUANT_CACHE_DIR`` ->
``$XDG_CACHE_HOME/foldquant`` -> ``~/.cache/foldquant``) under a
``<sm>-<machine>-<trt>`` slug, where the locator finds them at build and run
time. ``python -m foldquant.kernels status`` prints what resolves for this
device without building anything.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional

from . import locator


def _cmd_status(_args: argparse.Namespace) -> int:
    print(f"target: {locator.target_slug()}")
    print(f"committed dir: {locator.committed_plugin_dir()}")
    print(f"cache dir: {locator.plugin_cache_dir()}")
    rc = 0
    for lib in locator.KNOWN_PLUGIN_LIBS:
        resolved = locator.resolve_plugin_so(lib)
        shown = resolved if resolved is not None else "(unresolved; run: python -m foldquant.kernels build)"
        print(f"  {lib:28s} {shown}")
        rc |= int(resolved is None)
    return rc


def _cmd_build(args: argparse.Namespace) -> int:
    from .build import ensure_plugins_for_current_device, rebuild_plugins

    libs: List[str] = list(args.libs) or list(locator.KNOWN_PLUGIN_LIBS)
    unknown = sorted(set(libs) - set(locator.KNOWN_PLUGIN_LIBS))
    if unknown:
        print(f"unknown plugin lib(s) {unknown}; known: {list(locator.KNOWN_PLUGIN_LIBS)}", file=sys.stderr)
        return 2
    if args.force:
        rebuild_plugins(locator.plugin_cache_dir(), libs)
    paths = ensure_plugins_for_current_device(libs)
    for lib, path in zip(libs, paths):
        print(f"  {lib:28s} {path}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m foldquant.kernels", description=__doc__.split("\n\n")[0])
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="build (or refresh) the plugin libraries for this device")
    b.add_argument("libs", nargs="*", help="library names (default: all)")
    b.add_argument("--force", action="store_true", help="rebuild even if a matching binary resolves")
    b.set_defaults(func=_cmd_build)
    s = sub.add_parser("status", help="show which binaries resolve for this device")
    s.set_defaults(func=_cmd_status)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
