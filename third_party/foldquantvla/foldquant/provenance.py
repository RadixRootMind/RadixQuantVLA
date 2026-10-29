# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""Normalize paths for committed export and verification records.

Hugging Face cache paths become ``ORG/NAME``. Other absolute paths retain only
their basename; hub IDs and repository-relative paths remain unchanged.
"""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath
from typing import Any

__all__ = ["public_path"]

#: ``models--ORG--NAME`` / ``datasets--ORG--NAME``, the Hugging Face cache layout.
_HF_CACHE = re.compile(r"^(?:models|datasets)--(.+?)--(.+)$")

#: This checkout, used to keep paths inside it in their repo-relative form.
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _repo_relative(text: str) -> "str | None":
    """*text* as a path relative to this checkout, or ``None`` if it is outside it."""
    try:
        resolved = Path(text).expanduser().resolve()
    except (OSError, RuntimeError):  # unresolvable path: treat as outside
        return None
    try:
        return resolved.relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        return None


def public_path(value: Any) -> Any:
    """A committable form of *value*.

    Non-strings and empty strings pass through, so a caller can hand this a
    field that may be ``None`` without checking first.

    >>> public_path("/home/someone/weights/GR00T-N1.7-LIBERO")
    'GR00T-N1.7-LIBERO'
    >>> public_path("/home/someone/.cache/huggingface/hub/models--nvidia--GR00T-N1.6-3B/snapshots/3ddc6c9c")
    'nvidia/GR00T-N1.6-3B'
    >>> public_path("lerobot/pi05_base")
    'lerobot/pi05_base'
    >>> public_path("exports/w4a4/onnx")
    'exports/w4a4/onnx'
    """
    if not isinstance(value, str) or not value:
        return value

    text = value.strip()
    absolute = text.startswith("/") or text.startswith("~")
    if not absolute:
        # A hub id ("org/name") or a repo-relative path: portable already.
        return value

    # A path inside this checkout keeps its whole repo-relative form: it names
    # the arm ("models/pi05/exports/w4a4/engines"), which the basename alone
    # would throw away; every engine directory is called "engines".
    inside = _repo_relative(text)
    if inside is not None:
        return inside

    parts = PurePosixPath(text).parts
    for part in reversed(parts):
        found = _HF_CACHE.match(part)
        if found:
            return f"{found.group(1)}/{found.group(2)}"

    name = PurePosixPath(text).name
    return name or value
