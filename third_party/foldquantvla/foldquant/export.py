# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""Export live PyTorch modules as FoldQuant plugin-node ONNX graphs.

Use ``export_llm``, ``export_dit``, or ``export_expert`` directly, or dispatch
through ``export_module``. Folded schemes require ``forward_loop(module)``
to replay calibration observations through the host model.

Calibration measures SmoothQuant scales first, then GPTQ Hessians in a second
replay using the transformed activations. ``foldq.fold_site`` folds and packs
the weights before the emitter constructs the graph.

``record=True`` also returns the calibration result as a
:class:`foldquant.quant_state.ModuleQuantState` on ``ExportResult.state``;
passing that state back as ``state=`` emits the same graph with no
``forward_loop``, no calibration data and no GPTQ solve.

Action-module params are ``sq_alpha`` and ``sq_fold_order``. LLM params are
validated by ``llm_rotation_sq.resolve_llm_plugin_params``.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

import torch.nn as nn

from . import calibrate, schemes
from .quant_state import ModuleQuantState, pack_scope, record_gptq, record_rotations, replay_sites
from .llm_gptq import gptq_prepare, tag_site

logger = logging.getLogger(__name__)

#: Upper bound the LLM plugin graph bakes its RoPE / causal-mask tables at. The
#: engine handles any seq_len in ``[1, bound]`` by runtime slicing. Deliberately
#: a fixed constant rather than ``config.max_position_embeddings`` (32768 for
#: Qwen3): baking at that size allocates a multi-gigabyte mask initializer for a
#: bound a VLA prefix never approaches.
LLM_BAKE_MAX_SEQ_LEN = 4096

ForwardLoop = Callable[[nn.Module], None]


@dataclass
class ExportResult:
    """What an export produced and what running it needs."""

    module: str
    scheme: str
    #: Where the graph was written; ``None`` when it was only built in memory (quantizing).
    onnx_path: Optional[Path]
    #: Plugin libraries the engine build and the runtime must load.
    plugin_libs: List[str]
    #: Folded-LLM only: everything :func:`install_llm_emulation` needs to
    #: replay the deployed LLM numerics in PyTorch (cascade calibration of the
    #: modules downstream of the LLM). ``None`` otherwise.
    emulation: Optional[Dict[str, Any]] = field(default=None, repr=False)
    #: The calibration result, when the export ran with ``record=True`` or replayed one.
    state: Optional[ModuleQuantState] = field(default=None, repr=False)
    #: :func:`foldquant.onnx_io.graph_digest` of the graph, when it was recorded or replayed.
    graph_digest: Optional[str] = field(default=None, repr=False)


def _finish(model: Any, onnx_path: Optional[Path], want_digest: bool) -> Optional[str]:
    """Digest the in-memory graph (recording or replaying a quant state), then write it if
    *onnx_path* names a file. The digest comes before the write: saving with external data
    moves the tensors out of the proto."""
    from .onnx_io import graph_digest, save_plugin_onnx

    digest = graph_digest(model) if want_digest else None
    if onnx_path is not None:
        save_plugin_onnx(model, onnx_path, size_threshold=1024)
    return digest


def _check_state(state: ModuleQuantState, module: str, scheme: str, params: Mapping[str, Any]) -> Dict[str, Any]:
    """The params to export with: the state's, which a caller may repeat but not contradict."""
    if state.module != module:
        raise ValueError(f"quant state is for {state.module!r}, not {module!r}")
    if state.scheme != scheme:
        raise ValueError(f"quant state was calibrated for {state.scheme!r}; exporting {scheme!r} would not match it")
    recorded = dict(state.config.get("params") or {})
    if params and dict(params) != recorded:
        raise ValueError(f"{module} params {dict(params)} differ from the quant state's {recorded}")
    return recorded


@contextlib.contextmanager
def _packing(module: str, record: bool, ms: Optional[ModuleQuantState], replay: Any = None) -> Any:
    """Run an emitter so every weight pack it makes is recorded into *ms* (``record``) or
    replayed from *replay*; the recorded dict holds GPTQ and round-to-nearest packs alike."""
    with pack_scope(module, replay=replay):
        if not record:
            yield
            return
        with record_gptq() as sink, record_rotations() as rots:
            yield
        assert ms is not None
        ms.gptq.update(sink)
        head = f"{module}.rot."
        ms.put_group("perm", {k[len(head):]: perm for k, (perm, _R) in rots.items()})
        ms.put_group("rot", {k[len(head):]: R for k, (_perm, R) in rots.items()})


def _log_emitted(kind: str, scheme: str, onnx_path: Optional[Path], record: bool, detail: str = "") -> None:
    """While quantizing, the graph is built in memory only to pack and record the weight codes;
    say so rather than naming a file."""
    if record:
        logger.info("Quantized %s %s%s: weight codes recorded", kind, scheme, detail)
    else:
        logger.info("Emitted %s %s graph%s at %s", scheme, kind, detail, onnx_path)


def _prepare_gptq(hessians: Dict[str, Any], module: str) -> Dict[str, Any]:
    """Factorize each site's Hessian once, for :func:`foldquant.foldq.fold_site`.

    Each prep is tagged ``<module>.<site>`` so a recording export captures its codes.
    """
    return {k: tag_site(gptq_prepare(h), f"{module}.{k}") for k, h in hessians.items()}


def _require_loop(scheme: str, forward_loop: Optional[ForwardLoop]) -> ForwardLoop:
    if forward_loop is None:
        raise ValueError(
            f"{scheme!r} needs calibration data: its SmoothQuant scales are the per-channel amax of "
            "the (rotated) activation measured from real inputs, and GPTQ schemes additionally "
            "measure per-site Hessians there. Pass forward_loop."
        )
    return forward_loop


def _act_fold_knobs(scheme: str, params: Mapping[str, Any]) -> Dict[str, Any]:
    """``{"fwht", "fold_order", "alpha"?}`` for an action-module fold.

    Butterfly schemes default to scaling before rotation, allowing the divide
    to fuse into the kernel's load loop. Scaling after rotation requires a
    second shared-memory pass and remains available for ablation.

    ``alpha`` applies only before rotation: 1.0 moves activation outliers into
    the weights, while 0.5 balances activation and weight ranges.
    """
    unknown = sorted(set(params) - _ACT_FOLD_PARAMS)
    if unknown:
        raise ValueError(
            f"{scheme}: unknown action-module fold param(s) {unknown}; "
            f"the action-module folds read only {sorted(_ACT_FOLD_PARAMS)}."
        )
    fwht = schemes.uses_fwht(scheme)
    fold_order = str(params.get("sq_fold_order", "before" if fwht else "after"))
    if fold_order not in ("before", "after"):
        raise ValueError(f"sq_fold_order must be 'before' or 'after', got {fold_order!r}")
    knobs: Dict[str, Any] = {"fwht": fwht, "fold_order": fold_order}
    if fold_order == "before":
        knobs["alpha"] = float(params.get("sq_alpha", 0.5))
    elif "sq_alpha" in params:
        # The "after" fold is amax-only; dropping the value here would build an
        # alpha=1.0 graph while the manifest records the requested alpha.
        raise ValueError(
            f"{scheme}: sq_alpha applies to sq_fold_order='before' only; the 'after' fold "
            "(the default for the dense-rotation _r scheme) is amax-only. Pass "
            "sq_fold_order='before' with it, or drop sq_alpha."
        )
    return knobs


#: Supported action-module fold parameters.
_ACT_FOLD_PARAMS = frozenset({"sq_alpha", "sq_fold_order"})


def _refuse_params(module: str, scheme: str, params: Mapping[str, Any]) -> None:
    """A scheme that folds nothing has no knob to apply; refuse params instead of ignoring them."""
    if params:
        raise ValueError(
            f"{module} {scheme!r} folds nothing, so fold params {sorted(params)} would be ignored "
            "(and still recorded in the export manifest). Drop them or pick a folded scheme."
        )


# LLM


def export_llm(
    module: nn.Module,
    onnx_path: Path,
    *,
    scheme: str = schemes.W8A8,
    forward_loop: Optional[ForwardLoop] = None,
    params: Optional[Mapping[str, Any]] = None,
    max_seq_len: int = LLM_BAKE_MAX_SEQ_LEN,
    final_norm: Optional[bool] = None,
    record: bool = False,
    state: Optional[ModuleQuantState] = None,
) -> ExportResult:
    """Emit the LLM decoder as a plugin-node graph.

    The architecture is read off ``module.config``: Qwen2 / Qwen3 / Qwen3-VL
    (GR00T) get the dynamic-S decoder graph; Gemma (``gemma_mode``) gets the
    KV-stack prefix graph the expert cross-attends. A Qwen3-VL LLM needs a ``forward_loop`` even for ``w8a8``:
    its deepstack count and visual-token width come from a captured call.
    ``final_norm`` states whether the graph ends with the tower's final RMSNorm
    (``None`` reads it off the module; see :func:`foldquant.llm.build_llm_plugin_onnx`).

    ``record`` returns the calibration result on ``ExportResult.state``;
    ``state`` replays one instead of calibrating (no ``forward_loop`` needed).
    """
    from .llm import build_llm_plugin_onnx, resolve_qwen3_decoder
    from .llm_rotation_sq import compute_sq_scales_llm, resolve_llm_plugin_params

    schemes.validate("llm", scheme)
    params = dict(params or {})
    if state is not None:
        params = _check_state(state, "llm", scheme, params)
    onnx_path = Path(onnx_path) if onnx_path is not None else None
    folded = scheme in schemes.LLM_FOLDED_SCHEMES
    if not folded:
        _refuse_params("llm", scheme, params)
    decoder = resolve_qwen3_decoder(module)
    prefix_graph = calibrate.is_gemma(module)
    ms = ModuleQuantState("llm", scheme, config={"params": params, "max_seq_len": int(max_seq_len)}) if record else None

    mode_kwargs: Dict[str, Any] = {}
    snapshots: list = []
    loop: Optional[ForwardLoop] = None
    if state is not None:
        mode_kwargs = dict(state.config["mode_kwargs"])
    else:
        loop = _require_loop(scheme, forward_loop) if folded else forward_loop
        if prefix_graph:
            mode_kwargs = {"gemma_mode": True}
            # The Pi processor pads text to a fixed length and the camera count is
            # fixed, so the runtime prefix is constant; pin it from a captured
            # call. Hook layer 0's q_proj LEAF: the Pi prefix replay drives the
            # projection submodules directly and never runs the decoder's forward.
            if loop is None:
                raise ValueError(
                    "The Gemma plugin graph pins its prefix length from a captured forward call; "
                    "pass forward_loop even for the dynamic per-row scheme."
                )
            q0 = decoder.layers[0].self_attn.q_proj
            mode_kwargs["pin_seq_len"] = calibrate.captured_prefix_len(calibrate.capture_llm_snapshots(q0, loop))
        else:
            mode_kwargs["final_norm"] = final_norm
            if folded or calibrate.is_qwen3_vl(module):
                if loop is None:
                    raise ValueError(
                        "This LLM is Qwen3-VL (rope_scaling.mrope_section present). Its plugin graph injects "
                        "M-RoPE position_ids and deepstack residuals as explicit inputs whose count and width "
                        "are only knowable from a real forward call. Pass forward_loop even for w8a8."
                    )
                snapshots = calibrate.capture_llm_snapshots(decoder, loop)
                mode_kwargs.update(calibrate.qwen3_vl_graph_params(module, snapshots) or {})
    if ms is not None:
        ms.config["mode_kwargs"] = mode_kwargs

    if not folded:
        replay = replay_sites(state) if state is not None else None
        with _packing("llm", record, ms, replay):
            model = build_llm_plugin_onnx(module, None, max_seq_len=max_seq_len, **mode_kwargs)
        if replay is not None:
            replay.assert_consumed()
        digest = _finish(model, onnx_path, record or state is not None)
        _log_emitted("LLM", scheme, onnx_path, record)
        return ExportResult("llm", scheme, onnx_path, schemes.plugin_libs(scheme, params=params), state=ms or state, graph_digest=digest)

    bits = schemes.bits_of(scheme)
    fold = resolve_llm_plugin_params(scheme, bits=bits, overrides=params or None)
    replay = None
    if state is not None:
        sq_scales = state.tensor_group("sq")
        weight_clip = state.tensor_group("weight_clip") or None
        act_clip: Any = state.config["act_clip"]
        gptq_hessians: Any = None
        replay = replay_sites(state)
        if bits == 4:
            gptq_hessians = replay
    else:
        if prefix_graph:
            # The prefix replay drives the layer submodules directly, so the per-site
            # leaf hooks fire under it; there is no decoder-level snapshot to replay.
            # One snapshot therefore stands for the WHOLE forward loop: every
            # calibration observation runs inside it, which is what the "over 1
            # replay snapshot(s)" log lines below count.
            replay_snaps: list = [None]
            assert loop is not None

            def replay_fn(_snap: Any) -> None:
                loop(module)
        else:
            replay_snaps = snapshots

            def replay_fn(snap: Any) -> None:
                decoder(*snap[0], **snap[1])

        learned = calibrate.load_learned_calib(
            fold.get("learned_calib"), num_layers=len(decoder.layers), device=next(decoder.parameters()).device
        )
        if learned is not None:
            sq_scales = learned["sq_scales"]
        else:
            sq_scales = compute_sq_scales_llm(decoder, replay_snaps, alpha=fold["sq_alpha"], forward_fn=replay_fn)
        gptq_hessians = None
        if bits == 4:
            from .llm_gptq import compute_gptq_hessians_llm

            gptq_hessians = compute_gptq_hessians_llm(
                decoder, replay_snaps, sq_scales=sq_scales, rot_bs=int(fold["rot_block_size"]), forward_fn=replay_fn
            )
        act_clip = learned["act_clip"] if learned is not None else float(fold.get("act_clip_ratio", 1.0))
        weight_clip = learned["weight_clip"] if learned is not None else None
    fold_kwargs: Dict[str, Any] = {
        "sq_scales": sq_scales,
        "rot_bs": int(fold["rot_block_size"]),
        "bits": bits,
        "gptq_hessians": gptq_hessians,
        "act_clip_ratio": act_clip,
        "site_bits": fold.get("site_bits"),
        "act_bits": fold.get("act_bits"),
        "weight_clip": weight_clip,
    }
    if ms is not None:
        ms.config.update(
            act_clip=act_clip, rot_bs=int(fold["rot_block_size"]), bits=bits,
            site_bits=fold.get("site_bits"), act_bits=fold.get("act_bits"),
        )
        ms.put_group("sq", sq_scales)
        ms.put_group("weight_clip", weight_clip)
    with _packing("llm", record, ms, replay):
        model = build_llm_plugin_onnx(module, None, max_seq_len=max_seq_len, **fold_kwargs, **mode_kwargs)
    if replay is not None:
        replay.assert_consumed()
    digest = _finish(model, onnx_path, record or state is not None)
    family = "gemma" if prefix_graph else "qwen"
    _log_emitted("LLM", scheme, onnx_path, record, f" ({family}, {bits}-bit)" + (" from quant state" if state else ""))
    # A replayed export has no Hessians; cascade calibration needs a calibrating export.
    emulation = {"family": family, **fold_kwargs} if state is None else None
    return ExportResult(
        "llm",
        scheme,
        onnx_path,
        schemes.plugin_libs(scheme, params=params),
        emulation=emulation,
        state=ms or state,
        graph_digest=digest,
    )


def install_llm_emulation(module: nn.Module, result: ExportResult) -> Any:
    """Install a kernel-matched fake-quant of an exported LLM, for cascade calibration.

    Modules downstream of the LLM (the DiT / expert) should be calibrated on the
    activations the *quantized* LLM produces, not the float one. Returns an
    ``LlmEmulationHandle``; call ``.remove()`` after the downstream capture.

    Raises:
        ValueError: *result* carries no emulation data (not a folded LLM scheme).
        NotImplementedError: a decoder family the Qwen / Gemma emulation does
            not cover. A wrong emulation is worse than none.
    """
    data = result.emulation
    if data is None:
        raise ValueError(f"{result.scheme!r} on {result.module!r} has no LLM emulation to install.")
    if data["family"] not in ("qwen", "gemma"):
        raise NotImplementedError(
            f"cascade calibration: the PyTorch emulation covers Qwen3/Qwen3-VL and Gemma; "
            f"{data['family']!r} is not validated there."
        )
    from .llm_fake_quant import install_llm_per_row_emulation

    return install_llm_per_row_emulation(
        module,
        sq_scales=data["sq_scales"],
        gptq_hessians=data["gptq_hessians"],
        rot_bs=data["rot_bs"],
        bits=data["bits"],
        act_clip_ratio=data.get("act_clip_ratio", 1.0),
        site_bits=data.get("site_bits"),
        act_bits=data.get("act_bits"),
        weight_clip=data.get("weight_clip"),
    )


# DiT


def export_dit(
    module: nn.Module,
    onnx_path: Path,
    *,
    scheme: str = schemes.W8A8,
    forward_loop: Optional[ForwardLoop] = None,
    params: Optional[Mapping[str, Any]] = None,
    record: bool = False,
    state: Optional[ModuleQuantState] = None,
) -> ExportResult:
    """Emit the GR00T action-head DiT as a plugin-node graph.

    ``record`` / ``state`` as for :func:`export_llm`.
    """
    schemes.validate("dit", scheme)
    params = dict(params or {})
    if state is not None:
        params = _check_state(state, "dit", scheme, params)
    onnx_path = Path(onnx_path) if onnx_path is not None else None
    libs = schemes.plugin_libs(scheme, params=params)
    ms = ModuleQuantState("dit", scheme, config={"params": params}) if record else None

    replay = replay_sites(state) if state is not None else None
    if scheme == schemes.W8A8:
        from .dit_int8 import build_dit_plugin_onnx

        _refuse_params("dit", scheme, params)
        with _packing("dit", record, ms, replay):
            model = build_dit_plugin_onnx(module, None)
        if replay is not None:
            replay.assert_consumed()
        digest = _finish(model, onnx_path, record or state is not None)
        _log_emitted("DiT", scheme, onnx_path, record)
        return ExportResult("dit", scheme, onnx_path, libs, state=ms or state, graph_digest=digest)

    from .dit_int4 import compute_dit_sq_scales

    knobs = _act_fold_knobs(scheme, params)
    if knobs["fwht"] and knobs["fold_order"] != "before":
        raise ValueError(
            f"{scheme} on the DiT carries only the pre-rotation scale (act_scale_pre_*); "
            "sq_fold_order='after' would need a post-rotation vector the macro plugins do not declare."
        )
    alpha = knobs.get("alpha", 1.0)
    if state is not None:
        sq = state.tensor_group("sq")
    else:
        loop = _require_loop(scheme, forward_loop)
        # The scales are measured in-process from the same replay, so the fold
        # matches the weights (and rotation) this very graph bakes.
        calib_inputs = calibrate.capture_dit_inputs(module, loop)
        sq = compute_dit_sq_scales(module, calib_inputs, alpha=alpha, fwht=knobs["fwht"], fold_order=knobs["fold_order"])
    if ms is not None:
        ms.put_group("sq", sq)

    if scheme == schemes.W8A8_SH:
        from .dit_int8 import build_dit_plugin_onnx

        with _packing("dit", record, ms, replay):
            model = build_dit_plugin_onnx(module, None, sq_scales=sq, sq_fold_order=knobs["fold_order"], fwht=True)
        if replay is not None:
            replay.assert_consumed()
        digest = _finish(model, onnx_path, record or state is not None)
        _log_emitted("DiT", scheme, onnx_path, record)
        return ExportResult("dit", scheme, onnx_path, libs, state=ms or state, graph_digest=digest)

    from .dit_int4 import build_dit_plugin_onnx_int4

    gptq: Any = None
    if scheme == schemes.W4A4_SHG:
        if state is not None:
            gptq = replay
        else:
            # Second replay, after the scales exist: Hessians of rot(x / s).
            hess = compute_dit_sq_scales(
                module, calib_inputs, alpha=alpha, fwht=knobs["fwht"], fold_order=knobs["fold_order"], gptq_scales=sq
            )
            gptq = _prepare_gptq(hess, "dit")
    with _packing("dit", record, ms, replay):
        model = build_dit_plugin_onnx_int4(
            module, None, sq_scales=sq, sq_fold_order=knobs["fold_order"], fwht=knobs["fwht"], gptq=gptq
        )
    if replay is not None:
        replay.assert_consumed()
    digest = _finish(model, onnx_path, record or state is not None)
    _log_emitted("DiT", scheme, onnx_path, record)
    return ExportResult("dit", scheme, onnx_path, libs, state=ms or state, graph_digest=digest)


# expert


def export_expert(
    module: nn.Module,
    onnx_path: Path,
    *,
    scheme: str = schemes.W8A8,
    forward_loop: Optional[ForwardLoop] = None,
    params: Optional[Mapping[str, Any]] = None,
    record: bool = False,
    state: Optional[ModuleQuantState] = None,
) -> ExportResult:
    """Emit the Pi (Gemma-300M) action expert.

    The scale capture and the emitter take the SAME fold kwargs, from one
    mapping. A knob dropped on one side leaves the capture measuring in one
    frame while the emitter folds in another, with no error anywhere.
    ``record`` / ``state`` as for :func:`export_llm`.
    """
    schemes.validate("expert", scheme)
    params = dict(params or {})
    if state is not None:
        params = _check_state(state, "expert", scheme, params)
    onnx_path = Path(onnx_path) if onnx_path is not None else None
    if not hasattr(module, "expert_model"):
        raise TypeError(
            f"export_expert: expected a Pi action expert exposing 'expert_model', got {type(module).__name__}."
        )
    from .gemma_expert import build_gemma_expert_plugin_onnx as build
    from .gemma_expert import compute_gemma_expert_sq_scales as compute

    ms = ModuleQuantState("expert", scheme, config={"params": params}) if record else None
    kwargs: Dict[str, Any] = {}
    replay = replay_sites(state) if state is not None else None
    if scheme == schemes.W8A8:
        _refuse_params("expert", scheme, params)
    else:
        knobs = _act_fold_knobs(scheme, params)
        scale_kwargs: Dict[str, Any] = {"fold_order": knobs["fold_order"]}
        if knobs["fwht"]:
            scale_kwargs["fwht"] = True
        if "alpha" in knobs:
            scale_kwargs["alpha"] = knobs["alpha"]
        if state is not None:
            sq = state.tensor_group("sq")
        else:
            loop = _require_loop(scheme, forward_loop)
            sq = compute(module, loop, **scale_kwargs)
        if ms is not None:
            ms.put_group("sq", sq)
        kwargs = {"int4": scheme in schemes.ACT_W4A4_SCHEMES, "sq_scales": sq, "fold_order": knobs["fold_order"]}
        if scheme == schemes.W4A4_SHG:
            if state is not None:
                kwargs["gptq"] = replay
            else:
                kwargs["gptq"] = _prepare_gptq(compute(module, loop, **scale_kwargs, gptq_scales=sq), "expert")
        if knobs["fwht"]:
            kwargs["fwht"] = True
    with _packing("expert", record, ms, replay):
        model = build(module, None, **kwargs)
    if replay is not None:
        replay.assert_consumed()
    digest = _finish(model, onnx_path, record or state is not None)
    _log_emitted("expert", scheme, onnx_path, record)
    return ExportResult("expert", scheme, onnx_path, schemes.plugin_libs(scheme, params=params), state=ms or state, graph_digest=digest)


# dispatch

_EXPORTERS: Dict[str, Callable[..., ExportResult]] = {
    "llm": export_llm,
    "dit": export_dit,
    "expert": export_expert,
}


def export_module(
    name: str,
    module: nn.Module,
    onnx_path: Path,
    *,
    scheme: str,
    forward_loop: Optional[ForwardLoop] = None,
    params: Optional[Mapping[str, Any]] = None,
    record: bool = False,
    state: Optional[ModuleQuantState] = None,
    **kwargs: Any,
) -> ExportResult:
    """Export *module* under *scheme*, dispatching on the module kind *name*.

    Extra keyword arguments go to that exporter (``max_seq_len`` / ``final_norm`` for the LLM).
    """
    schemes.validate(name, scheme)
    return _EXPORTERS[name](
        module, onnx_path, scheme=scheme, forward_loop=forward_loop, params=params, record=record, state=state, **kwargs
    )


__all__ = [
    "LLM_BAKE_MAX_SEQ_LEN",
    "ExportResult",
    "export_dit",
    "export_expert",
    "export_llm",
    "export_module",
    "install_llm_emulation",
]
