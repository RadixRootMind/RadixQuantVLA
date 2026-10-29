# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""π₀.₅ adapter for the FoldQuant fake-quant pipeline (:mod:`foldquant.quantized`).

The state converts to the real-quant ONNX graphs and engines like any family.
The PyTorch fake-quant covers both quantized modules: the Gemma prefix LLM and
the action expert (:mod:`foldquant.expert_fake_quant`, reached through
:class:`.runtime.Pi05ExpertView`, whose submodules are the policy's own).
``serve``, ``eval_libero`` and ``verify`` run a fake-quant model given as
``--checkpoint-dir``, or a state given with ``--quantized-model``.
Run from ``models/pi05``::

    python -m foldquant_integration.quantize ... --output-dir exports/<arm>/quantized
    python -m foldquant_integration.export --checkpoint-dir <dir> --output-dir exports/<arm>/onnx
    python -m foldquant_integration.build_engines --onnx-dir exports/<arm>/onnx --engine-dir exports/<arm>/engines
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from foldquant import quantized as _pipeline

from ._upstream import EXPORT_METADATA_NAME

FAMILY = "pi05"


def load_policy(model_path: str, embodiment_tag: str | None, device: Any = "cuda", **load_kwargs: Any) -> Any:
    """``embodiment_tag`` does not apply; ``config_name`` comes from the state."""
    from . import calibration

    return calibration.load_policy(model_path, device=device, compile=False, **load_kwargs)


def module_paths(policy: Any) -> dict[str, Any]:
    from .runtime import expert_view, llm_module

    return {"llm": llm_module(policy), "expert": expert_view(policy)}


def build_engines(
    onnx_dir: Path,
    engine_dir: Path,
    *,
    max_batch: int = 1,
    workspace_mb: int = 8192,
) -> Any:
    from .build_engines import BuildConfig, build

    if max_batch != 1:
        raise SystemExit("pi05 engines pin batch 1")
    return build(BuildConfig(onnx_dir=str(onnx_dir), engine_dir=str(engine_dir), workspace_mb=workspace_mb))

def checkpoint_root(policy: Any) -> Any:
    """The module whose parameter names are the checkpoint's tensor keys (``PI0Pytorch``)."""
    from .runtime import model_of

    return model_of(policy)


def save_arm_state(
    directory: Path,
    *,
    policy: Any,
    model_path: str,
    results: list,
    export_metadata: dict,
    export_manifest: dict,
    base_model_id: str | None = None,
    load_kwargs: dict | None = None,
) -> Path:
    """Called by ``quantize``."""
    return _pipeline.save_arm_state(
        directory,
        family=FAMILY,
        model_path=model_path,
        results=results,
        export_manifest=export_manifest,
        extra_files={EXPORT_METADATA_NAME: export_metadata},
        base_model_id=base_model_id,
        load_kwargs=load_kwargs,
        modules=module_paths(policy),
        checkpoint_root=checkpoint_root(policy),
    )


# A ModelOpt tower's graph: traced by the wrappers in :mod:`.quantize` from the call ``quantize``
# recorded in the quantized checkpoint.


def export_modelopt_graph(tower: str, policy: Any, module: Any, onnx_path: Path, ms: Any) -> Any:
    """The ModelOpt Q/DQ graph of *tower*, quantizers restored on the live module, traced from the
    recorded call. Returns ``(ExportResult, export record)``."""
    from foldquant import modelopt_int8
    from foldquant.export import ExportResult
    from foldquant.trace_export import trace_kwargs

    from . import modelopt_export

    t = trace_kwargs(ms, next(module.parameters()).device)
    opset = int(ms.config.get("opset", modelopt_int8.DEFAULT_OPSET))
    record = getattr(modelopt_export, f"export_{tower}")(policy, t, Path(onnx_path), algorithm=ms.scheme, opset=opset)
    libs = []
    if modelopt_int8.is_weight_only(ms.scheme):
        from foldquant.kernels.locator import INT4_GROUPWISE_LIB

        libs = [INT4_GROUPWISE_LIB]
    return ExportResult(tower, ms.scheme, Path(onnx_path), libs), record


if __name__ == "__main__":
    _pipeline.main()
