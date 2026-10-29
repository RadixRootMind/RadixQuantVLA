# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""The ModelOpt INT8 SmoothQuant baseline arm for Pi0.5 (see :mod:`foldquant.modelopt_int8`).

Reproduces the framework preset ``pi05/tensorrt/modelopt_w8a8_smoothquant``: the
PaliGemma language model and the action expert are quantized **in place on the
live policy** by ``quantize`` (the checkpoint keeps the smoothed weights, the
quantizers' scales and one captured call), then, in ``export``, with the
quantizers restored, traced by the KV-stack contract wrappers
(:func:`.quantize.trace_llm_pi05` /
:func:`.quantize.trace_expert_pi05`), so ``llm_bf16.onnx`` and
``expert_bf16.onnx`` keep the FoldQuant file names, bindings and dtypes, and
``build_engines``, ``runtime.install_engines``, ``verify`` and ``serve`` load
them unchanged.

Where the quantizers live and what calibrates them follows the preset's
export plan, in which the capture seam differs from the quantized module:

* ``llm``: quantizers in ``paligemma.language_model`` (the framework
  ``backbone.model.model.language_model``, the same HF ``GemmaModel``), calibrated
  by replaying every captured prefix pass (``prefix_embs``, the 4-D additive
  mask, ``position_ids``) through ``paligemma_with_expert.forward``, as the framework
  replays its ``prefix_core`` captures through its prefix wrapper.
* ``expert``: quantizers in :class:`.runtime.Pi05ExpertView`, which exposes the
  live expert under the framework's ``action_expert`` names (``expert_model.model.layers.*``,
  ``action_in_proj``, ``action_out_proj``, ``time_mlp_in``, ``time_mlp_out``), so
  the preset's exclusion globs select the same leaves: the adaRMS ``dense``
  modulation layers (``*norm*``) stay float, the action and time projections are
  quantized (none of them contains ``action_proj``). Calibrated by replaying every
  captured denoise step (``x_t``, ``timestep``, ``prefix_pad_masks``, KV stack),
  the ``denoise_core`` captures the framework replays.

Both seams are captured in one float replay before anything is quantized, so the
expert calibrates on float-LLM KV caches, as in the preset (no cascade).
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import logging
from pathlib import Path
from typing import Any

from foldquant import modelopt_int8
from foldquant.int4_groupwise import apply_int4_surgery_in_place
import torch
from torch import nn

from .runtime import cache_from_stack
from .runtime import expert_view
from .runtime import llm_module
from .runtime import model_of
from .runtime import require_eager
from .runtime import stack_cache

logger = logging.getLogger("foldquant.pi05.modelopt")


@dataclass
class Captures:
    """Every prefix pass and every denoise step of one calibration replay."""

    llm: list[dict[str, torch.Tensor]] = field(default_factory=list)
    """Prefix passes: ``prefix_embs``, ``attention_mask`` (4-D additive), ``position_ids``."""

    expert: list[dict[str, torch.Tensor]] = field(default_factory=list)
    """Denoise steps: ``state``, ``prefix_pad_masks``, ``kv_stack``, ``x_t``, ``timestep``.
    The Euler steps of one observation share one ``kv_stack`` tensor (stored once)."""


def _clone(t: Any) -> Any:
    return t.detach().clone() if isinstance(t, torch.Tensor) else t


def capture(policy, forward_loop, towers) -> Captures:
    """One replay of *forward_loop*, recording the calls into each tower in *towers*.

    Both seams are method calls on the live model (``paligemma_with_expert.forward``
    and ``denoise_step``), which a module hook never sees; they are wrapped on the
    instance for the replay, call straight through, and are restored afterwards.
    """
    model = model_of(policy)
    require_eager(model)
    pwe = model.paligemma_with_expert
    if "forward" in pwe.__dict__ or "denoise_step" in model.__dict__:
        raise RuntimeError("a seam is already rebound on the policy (engines installed?); capture needs PyTorch")
    store = Captures()
    prefix_forward = pwe.forward
    denoise_step = model.denoise_step
    shared: dict[str, Any] = {}

    def prefix_spy(*args: Any, **kwargs: Any) -> Any:
        embeds = kwargs.get("inputs_embeds")
        if embeds is not None and embeds[1] is None:
            if args or kwargs.get("attention_mask") is None or kwargs.get("position_ids") is None:
                raise RuntimeError("prefix pass called without keyword attention_mask / position_ids")
            store.llm.append(
                {
                    "prefix_embs": _clone(embeds[0]),
                    "attention_mask": _clone(kwargs["attention_mask"]),
                    "position_ids": _clone(kwargs["position_ids"]),
                }
            )
        return prefix_forward(*args, **kwargs)

    def denoise_spy(state, prefix_pad_masks, past_key_values, x_t, timestep):
        # The cache object is the same for every Euler step of an observation:
        # stack (and copy) it once, keep the object so the identity test stays valid.
        if shared.get("cache") is not past_key_values:
            kv = past_key_values if torch.is_tensor(past_key_values) else stack_cache(past_key_values)
            shared.update(cache=past_key_values, stack=kv.detach().clone())
        store.expert.append(
            {
                "state": _clone(state),
                "prefix_pad_masks": _clone(prefix_pad_masks),
                "kv_stack": shared["stack"],
                "x_t": _clone(x_t),
                "timestep": _clone(timestep),
            }
        )
        return denoise_step(state, prefix_pad_masks, past_key_values, x_t, timestep)

    if "llm" in towers:
        pwe.forward = prefix_spy
    if "expert" in towers:
        model.denoise_step = denoise_spy
    try:
        forward_loop(None)
    finally:
        pwe.__dict__.pop("forward", None)
        model.__dict__.pop("denoise_step", None)
    for tower in towers:
        calls = getattr(store, tower)
        if not calls:
            raise RuntimeError(f"calibration replay never reached the {tower} seam; nothing to calibrate on")
        logger.info("captured %d %s calls", len(calls), tower)
    return store


def _llm_replay(policy, calls: list[dict[str, torch.Tensor]]):
    pwe = model_of(policy).paligemma_with_expert

    def prefix(prefix_embs, attention_mask, position_ids):
        return pwe.forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

    return modelopt_int8.replay_loop([((), c) for c in calls], target=prefix)


def _expert_replay(policy, calls: list[dict[str, torch.Tensor]]):
    model = model_of(policy)
    memo: dict[str, Any] = {}

    def step(state, prefix_pad_masks, kv_stack, x_t, timestep):
        # The PyTorch attention concatenates the cache without writing to it, so one
        # rebuilt cache serves every step that shares a stack.
        if memo.get("stack") is not kv_stack:
            memo.update(stack=kv_stack, cache=cache_from_stack(kv_stack))
        return model.denoise_step(state, prefix_pad_masks, memo["cache"], x_t, timestep)

    return modelopt_int8.replay_loop([((), c) for c in calls], target=step)


def _quantizable_leaves(module: nn.Module) -> list[str]:
    return sorted(n for n, m in module.named_modules() if n and isinstance(m, modelopt_int8.QUANTIZABLE_LEAF_TYPES))


def _require_live(policy, module: nn.Module, tower: str) -> None:
    """Every enabled quantizer of *module* must sit in the live policy, not in a detached copy."""
    live = {id(m) for m in model_of(policy).modules()}
    from modelopt.torch.quantization.nn import TensorQuantizer

    detached = [n for n, m in module.named_modules() if isinstance(m, TensorQuantizer) and id(m) not in live]
    if detached:
        raise RuntimeError(
            f"{tower}: {len(detached)} quantizers are not reachable from the live policy: {detached[:5]}"
        )


def _referenced_files(onnx_path: Path) -> set[str]:
    import onnx
    from onnx.external_data_helper import _get_all_tensors

    model = onnx.load(str(onnx_path), load_external_data=False)
    return {
        e.value
        for t in _get_all_tensors(model)
        if t.data_location == onnx.TensorProto.EXTERNAL
        for e in t.external_data
        if e.key == "location"
    }


def _finish_graph(onnx_path: Path, name: str, output_dtype: torch.dtype, before: set[str], *, algorithm: str) -> dict[str, Any]:
    """Post-process a traced graph; *before* lists the directory's files ahead of the trace."""
    import onnx

    to_onnx = {
        torch.bfloat16: onnx.TensorProto.BFLOAT16,
        torch.float32: onnx.TensorProto.FLOAT,
        torch.float16: onnx.TensorProto.FLOAT16,
    }
    # Order matters: the cast and repair passes save structure only, the
    # consolidation then rewrites every tensor into one sidecar.
    repairs = modelopt_int8.repair_onnx_dtypes(onnx_path, name)
    output_casts = modelopt_int8.cast_graph_outputs(onnx_path, name, to_onnx[output_dtype])
    merged = modelopt_int8.consolidate_external_data(onnx_path, name)
    # The exporter also writes files for constants it later folds away; nothing references them.
    keep = before | {onnx_path.name} | _referenced_files(onnx_path)
    orphans = sorted(f.name for f in onnx_path.parent.iterdir() if f.is_file() and f.name not in keep)
    for orphan in orphans:
        (onnx_path.parent / orphan).unlink()
    if orphans:
        logger.info("%s: removed %d unreferenced files the exporter left", name, len(orphans))
    stripped = modelopt_int8.strip_default_scatternd_reduction(onnx_path, name)
    # Counted BEFORE the INT4 surgery, which consumes the weight DQ nodes it rewrites.
    qdq = modelopt_int8.require_qdq(onnx_path, name)
    record = {
        "dtype_repairs": repairs,
        "output_casts": {c: str(output_dtype).removeprefix("torch.") for c in output_casts},
        "external_data_files_merged": merged,
        "unreferenced_files_removed": len(orphans),
        "scatternd_reduction_stripped": stripped,
        "qdq_nodes": qdq,
    }
    if modelopt_int8.is_weight_only(algorithm):
        record["int4_groupwise_surgery"] = apply_int4_surgery_in_place(onnx_path, name)
    return record


def quantize_llm(policy, captures: Captures, *, algorithm: str) -> dict[str, Any]:
    """Quantize the live language model in place on its captured prefix passes."""
    module = llm_module(policy)
    leaves = _quantizable_leaves(module)
    record = modelopt_int8.quantize_module(module, _llm_replay(policy, captures.llm), algorithm=algorithm)
    _require_live(policy, module, "llm")
    record.update(quantizable_leaves=len(leaves), calibration_calls=len(captures.llm))
    return record


def quantize_expert(policy, captures: Captures, *, algorithm: str) -> dict[str, Any]:
    """Quantize the live action expert in place on its captured denoise steps."""
    view = expert_view(policy)
    leaves = _quantizable_leaves(view)
    record = modelopt_int8.quantize_module(view, _expert_replay(policy, captures.expert), algorithm=algorithm)
    # The view's children are the model's own modules; a quantizer ModelOpt placed on a
    # replacement bound to the view alone would leave the served projection float.
    _require_live(policy, view, "expert")
    record.update(quantizable_leaves=len(leaves), calibration_calls=len(captures.expert))
    return record


def export_llm(policy, trace: dict[str, torch.Tensor], onnx_path: Path, *, algorithm: str, opset: int) -> dict[str, Any]:
    """Trace ``llm_bf16.onnx`` from the quantized live language model on the recorded prefix pass."""
    from .quantize import trace_llm_pi05

    seen = {
        "inputs_embeds": [trace["prefix_embs"], None],
        "attention_mask": trace["attention_mask"],
        "position_ids": trace["position_ids"],
    }
    logger.info("exporting ModelOpt LLM -> %s (opset %d)", onnx_path, opset)
    before = {f.name for f in Path(onnx_path).parent.iterdir()}
    trace_llm_pi05(policy, onnx_path, seen=seen, opset=opset)
    # The engine hands its KV stack to the expert, which binds it as bf16.
    record = _finish_graph(Path(onnx_path), "llm", torch.bfloat16, before, algorithm=algorithm)
    record["opset"] = opset
    return record


def export_expert(policy, trace: dict[str, torch.Tensor], onnx_path: Path, *, algorithm: str, opset: int) -> dict[str, Any]:
    """Trace ``expert_bf16.onnx`` from the quantized live action expert on the recorded denoise step."""
    from .quantize import trace_expert_pi05

    model = model_of(policy)
    with torch.inference_mode():
        out = model.denoise_step(
            trace["state"],
            trace["prefix_pad_masks"],
            cache_from_stack(trace["kv_stack"]),
            trace["x_t"],
            trace["timestep"],
        )
    logger.info("exporting ModelOpt expert -> %s (opset %d)", onnx_path, opset)
    before = {f.name for f in Path(onnx_path).parent.iterdir()}
    trace_expert_pi05(policy, onnx_path, seen=trace, opset=opset)
    # upstream's denoise_step returns this dtype; the runtime casts it to x_t's.
    record = _finish_graph(Path(onnx_path), "expert", out.dtype, before, algorithm=algorithm)
    record["opset"] = opset
    return record
