# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""The quantization state of a FoldQuant arm: calibrate once, replay anywhere.

A FoldQuant export is a deterministic function of the checkpoint weights and a
small calibration result: the SmoothQuant scales of every site, the fold
knobs, the graph-shape values the emitters read off a captured forward call,
and, for GPTQ schemes, the integer codes GPTQ chose. Everything else (the
rotations, the round-to-nearest sites, the byte layout) is recomputed from the
weights by the same code. This module holds that result:

* :class:`ModuleQuantState` is one module's share (``llm`` / ``dit`` /
  ``expert``); :func:`foldquant.export.export_llm` and friends fill it while
  they calibrate (``record=True``) and consume it instead of calibrating
  (``state=...``).
* Every weight pack is recorded, so a replay never recomputes a code: GPTQ
  codes at :func:`foldquant.llm_gptq.gptq_quant_codes` (a prep tagged with its
  site key is recorded there, and a :class:`ReplaySite` in its place hands the
  codes back), and round-to-nearest codes at :func:`rtn_pack`, which the two
  RTN packers (``weights.quant_weight_per_row``, ``rotation.pack_int4_colmajor``)
  go through, keyed ``<module>.rtn`` (``<module>.rtn.<site>`` inside a
  :func:`site_scope`) in call order. RTN codes of a folded weight depend on the
  device the fold's matmuls ran on; replayed, a graph is the same on any
  device. The emitters themselves do not change, so a replayed export writes
  the same bytes as the export that recorded it.
* :class:`QuantState` / :class:`ModuleQuantState` hold the result; the
  quantized checkpoint (:mod:`foldquant.quantized_checkpoint`) is how it is
  written to disk and pushed to the Hub.

Torch and safetensors are imported lazily.
"""

from __future__ import annotations

import contextlib
import contextvars
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

__all__ = [
    "MANIFEST_NAME",
    "ModuleQuantState",
    "QuantState",
    "ReplayError",
    "ReplaySite",
    "record_gptq",
    "replay_sites",
]

MANIFEST_NAME = "foldquant_quant.json"


class ReplayError(RuntimeError):
    """A replayed export asked for GPTQ codes the state does not hold, or left some unused."""


# GPTQ record / replay


_RECORDER: contextvars.ContextVar[Optional[Dict[str, list]]] = contextvars.ContextVar("foldquant_gptq_recorder", default=None)
_MODULE: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("foldquant_pack_module", default=None)
_SITE: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("foldquant_pack_site", default=None)
_REPLAY: contextvars.ContextVar[Optional["ReplaySession"]] = contextvars.ContextVar("foldquant_pack_replay", default=None)
_ROT_RECORDER: contextvars.ContextVar[Optional[Dict[str, Tuple[Any, Any]]]] = contextvars.ContextVar(
    "foldquant_rotation_recorder", default=None
)


@contextlib.contextmanager
def pack_scope(module: str, replay: Optional["ReplaySession"] = None) -> Iterator[None]:
    """Name the module whose weights are being packed, and optionally replay its recorded packs."""
    t_mod = _MODULE.set(module)
    t_rep = _REPLAY.set(replay)
    try:
        yield
    finally:
        _REPLAY.reset(t_rep)
        _MODULE.reset(t_mod)


@contextlib.contextmanager
def site_scope(site: str) -> Iterator[None]:
    """Key the round-to-nearest packs made inside by *site* (the LLM's ``L{i}_{site}``)."""
    token = _SITE.set(site)
    try:
        yield
    finally:
        _SITE.reset(token)


def rtn_pack(weight: Any, compute: Any) -> Tuple[Any, Any]:
    """``compute()`` -> ``(codes, scale)``, recorded or replayed inside a :func:`pack_scope`.

    Outside a pack scope (tests, ad-hoc packing) it simply computes.
    """
    module = _MODULE.get()
    if module is None:
        return compute()
    site = _SITE.get()
    local = "rtn" + (f".{site}" if site else "")
    replay = _REPLAY.get()
    if replay is not None and replay.has_rtn:
        entry = replay.get(local)
        if entry is None and site and "rtn" in replay:
            entry = replay["rtn"]  # a state recorded before the packs were named: call order
        if entry is None:
            raise ReplayError(
                f"{module}: no recorded round-to-nearest pack {local!r}; the state was recorded for another "
                "scheme, or the emitter changed"
            )
        return entry.take(weight)
    codes, scale = compute()
    record(f"{module}.{local}", codes, scale)
    return codes, scale


def dense_rotation(compute: Any) -> Tuple[Any, Any]:
    """``compute()`` -> ``(perm, R)`` of a learned dense rotation, recorded or replayed inside a
    :func:`pack_scope` + :func:`site_scope` like a weight pack.

    A dense rotation is an SVD of the site's weight. A quantized checkpoint holds the
    packed codes and no bf16 weight, so the rotation is recorded with the codes and
    handed back on replay rather than recomputed.
    """
    module, site = _MODULE.get(), _SITE.get()
    if module is None or site is None:
        return compute()
    replay = _REPLAY.get()
    if replay is not None and site in replay.rotations:
        return replay.rotations[site]
    perm, R = compute()
    sink = _ROT_RECORDER.get()
    if sink is not None:
        import torch

        sink[f"{module}.rot.{site}"] = (
            perm.detach().to("cpu", torch.int32).contiguous(),
            R.detach().to("cpu", torch.float32).contiguous(),
        )
    return perm, R


@contextlib.contextmanager
def record_rotations() -> Iterator[Dict[str, Tuple[Any, Any]]]:
    """Collect every dense rotation :func:`dense_rotation` computes while active, keyed
    ``<module>.rot.<site>``."""
    sink: Dict[str, Tuple[Any, Any]] = {}
    token = _ROT_RECORDER.set(sink)
    try:
        yield sink
    finally:
        _ROT_RECORDER.reset(token)


@contextlib.contextmanager
def record_gptq() -> Iterator[Dict[str, list]]:
    """Collect every ``(codes, scale)`` a site-tagged GPTQ prep produces while active.

    Yields ``{site: [(codes int8, scale fp32), ...]}`` in call order. Codes are
    stored as int8 (``|code| <= 127``), a quarter of the int32 the rounding
    returns, so recording a 1.5 B-weight arm costs about 1.5 GB of host memory.
    """
    sink: Dict[str, list] = {}
    token = _RECORDER.set(sink)
    try:
        yield sink
    finally:
        _RECORDER.reset(token)


def record(site: str, codes: Any, scale: Any) -> None:
    """Called by :func:`gptq_quant_codes` for a site-tagged prep; a no-op unless recording."""
    import torch

    sink = _RECORDER.get()
    if sink is None:
        return
    sink.setdefault(site, []).append(
        (codes.detach().to("cpu", torch.int8).contiguous(), scale.detach().to("cpu", torch.float32).contiguous())
    )


class ReplaySite:
    """Stands in for one site's GPTQ factors and returns the recorded codes instead.

    :func:`foldquant.llm_gptq.gptq_prepare` passes it through untouched and
    :func:`foldquant.llm_gptq.gptq_quant_codes` calls :meth:`take`, so every
    emitter that hands a site's prep to the rounding gets its recorded codes
    back in the order it asked for them the first time.
    """

    def __init__(self, site: str, entries: List[Tuple[Any, Any]]) -> None:
        self.site = site
        self._entries = list(entries)
        self._cursor = 0

    def take(self, weight: Any) -> Tuple[Any, Any]:
        """The next recorded ``(codes int32, scale fp32)`` for *weight* on this site."""
        import torch

        if self._cursor >= len(self._entries):
            raise ReplayError(
                f"GPTQ site {self.site!r}: call {self._cursor + 1} asks for codes the state holds only "
                f"{len(self._entries)} of. The emitter changed since the state was recorded, or the "
                "state belongs to another scheme."
            )
        codes, scale = self._entries[self._cursor]
        want = tuple(weight.shape)
        if tuple(codes.shape) != want:
            raise ReplayError(
                f"GPTQ site {self.site!r}, call {self._cursor + 1}: recorded codes are {tuple(codes.shape)}, "
                f"the weight is {want}. The checkpoint is not the one the state was calibrated on."
            )
        self._cursor += 1
        device = weight.device
        return codes.to(device, torch.int32), scale.to(device, torch.float32)

    @property
    def consumed(self) -> bool:
        return self._cursor == len(self._entries)

    def remaining(self) -> int:
        return len(self._entries) - self._cursor


class ReplaySession(dict):
    """``{site_key: ReplaySite}`` for one module, with a check that every entry was used.

    ``has_rtn`` is False for a state written before round-to-nearest packs were
    recorded (format 1); those packs are then recomputed.
    """

    #: ``{site: (perm, R)}`` of the recorded dense rotations, for :func:`dense_rotation`.
    rotations: Dict[str, Tuple[Any, Any]]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.rotations = {}

    @property
    def has_rtn(self) -> bool:
        return any(k == "rtn" or k.startswith("rtn.") for k in self)

    def assert_consumed(self) -> None:
        left = {k: s.remaining() for k, s in self.items() if not s.consumed}
        if left:
            raise ReplayError(
                f"the replayed export left recorded GPTQ codes unused at {len(left)} site(s) "
                f"({dict(list(left.items())[:4])}...). The emitter took a different path than when the "
                "state was recorded; the graph would not match the fake-quant model."
            )


def replay_sites(module_state: "ModuleQuantState") -> ReplaySession:
    """Replay factors for *module_state*'s GPTQ sites, keyed the way its emitter asks for them."""
    session = ReplaySession()
    prefix = f"{module_state.module}."
    for site, entries in module_state.gptq.items():
        key = site[len(prefix):] if site.startswith(prefix) else site
        session[key] = ReplaySite(site, entries)
    import torch

    rots, perms = module_state.tensor_group("rot"), module_state.tensor_group("perm")
    for site, R in rots.items():
        if site not in perms:
            raise ReplayError(f"{module_state.module}: rotation {site!r} is recorded without its permutation")
        session.rotations[site] = (perms[site].to(torch.int32), R.to(torch.float32))
    return session


# State containers


@dataclass
class ModuleQuantState:
    """One module's calibration result.

    ``config`` is JSON: scheme, params, resolved fold knobs, graph-shape values.
    ``tensors`` holds named float tensors (SmoothQuant scales, learned clips).
    ``gptq`` holds the recorded ``(codes, scale)`` list per site key, in call order.
    """

    module: str
    scheme: str
    config: Dict[str, Any] = field(default_factory=dict)
    tensors: Dict[str, Any] = field(default_factory=dict)
    gptq: Dict[str, List[Tuple[Any, Any]]] = field(default_factory=dict)

    def tensor_group(self, prefix: str) -> Dict[str, Any]:
        """``{key: tensor}`` for every tensor named ``<prefix>/<key>``."""
        head = prefix + "/"
        return {k[len(head):]: v for k, v in self.tensors.items() if k.startswith(head)}

    def put_group(self, prefix: str, values: Optional[Dict[str, Any]]) -> None:
        for k, v in (values or {}).items():
            self.tensors[f"{prefix}/{k}"] = v


@dataclass
class QuantState:
    """Every quantized module of one arm, plus what identifies the arm."""

    manifest: Dict[str, Any] = field(default_factory=dict)
    modules: Dict[str, ModuleQuantState] = field(default_factory=dict)


# Code packing


def _unpack_codes(stored: Any, kind: str) -> Any:
    import torch

    if kind == "int8":
        return stored.to(torch.int8)
    packed = stored.to(torch.int32)
    lo = packed & 0xF
    hi = (packed >> 4) & 0xF
    both = torch.stack([lo, hi], dim=-1).reshape(stored.shape[0], stored.shape[1] * 2)
    return torch.where(both >= 8, both - 16, both).to(torch.int8)
