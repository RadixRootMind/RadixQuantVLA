# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""Export a Pi0 / Pi0.5 PyTorch checkpoint to the ONNX graphs the engine builder compiles.

Reads ``--checkpoint-dir`` and writes, into ``--output-dir``::

    llm_bf16.onnx / expert_bf16.onnx   the prefix pass and the denoise step
    export_metadata.json               shape hints for the engine builder
    foldquant_export.json              what was exported, needing which plugins

A **quantized model** (what ``quantize`` wrote) needs no dataset: the recorded
weight codes are replayed into the FoldQuant plugin graphs, byte-identical to
the ones the quantization built. An **unquantized checkpoint** is the all-float
baseline: the prefix pass and the denoise step are traced under the KV-stack
contract on one observation from ``--dataset-path``, with no plugin nodes.
``build_engines --onnx-dir <output-dir>`` then builds the engines either way.

Example::

    python -m foldquant_integration.export --checkpoint-dir exports/w4a4/quantized --output-dir exports/w4a4/onnx
    python -m foldquant_integration.export --checkpoint-dir <checkpoint> --dataset-path <dataset> \\
        --config pi05_libero --output-dir exports/float/onnx
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import time
from typing import Any

from foldquant.provenance import public_path
from foldquant.quantized import ToOnnx, to_onnx
from foldquant.quantized_checkpoint import is_quantized_checkpoint
import tyro

from ._upstream import EXPORT_METADATA_NAME, LIBERO_TRAIN_CONFIG, MANIFEST_NAME
from .quantized import FAMILY

logger = logging.getLogger("foldquant.pi05.export")


@dataclass
class ExportConfig:
    checkpoint_dir: str
    """A quantized model (``quantize``'s ``--output-dir``) or an unquantized PyTorch checkpoint (the all-float baseline)."""

    output_dir: str
    """Destination directory of the graphs and manifests."""

    dataset_path: str | None = None
    """LeRobot dataset; one observation traces the graphs of an unquantized checkpoint (unused for a quantized model)."""

    config: str = LIBERO_TRAIN_CONFIG
    """Upstream training config of an unquantized checkpoint (a quantized model records its own)."""

    seed: int = 0
    """Seed of the traced observation of an unquantized checkpoint."""

    device: str = "cuda"


def export_float(args: ExportConfig) -> Path:
    """The all-float baseline: prefix pass and denoise step traced under the KV-stack contract."""
    from . import calibration
    from .quantize import capture_denoise, capture_prefix, capture_shape_metadata, export_metadata, trace_expert_pi05, trace_llm_pi05
    from .runtime import model_of

    if not args.dataset_path:
        raise SystemExit(f"{args.checkpoint_dir} is not a quantized model; tracing its graphs needs --dataset-path")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    policy = calibration.load_policy(args.checkpoint_dir, config_name=args.config, device=args.device, compile=False)
    dataset = calibration.load_dataset(args.dataset_path)
    samples, observations = calibration.sample_observations(
        dataset, 1, seed=args.seed, keys=calibration.resolve_keys(dataset, args.config)
    )
    loop = calibration.make_forward_loop(policy, observations, seed=args.seed)
    logger.info("policy + dataset in %.0fs; tracing on %s", time.time() - t0, samples[0])
    shapes = capture_shape_metadata(policy, observations[0], seed=args.seed)
    model = model_of(policy)

    t1 = time.time()
    trace_llm_pi05(policy, out / "llm_bf16.onnx", seen=capture_prefix(model, loop))
    trace_expert_pi05(policy, out / "expert_bf16.onnx", seen=capture_denoise(model, loop))
    logger.info("two float graphs traced in %.0fs", time.time() - t1)

    (out / EXPORT_METADATA_NAME).write_text(json.dumps(export_metadata(shapes, args.config), indent=2))
    manifest: dict[str, Any] = {
        "family": FAMILY,
        "checkpoint_dir": public_path(args.checkpoint_dir),
        "config": args.config,
        "dataset_path": public_path(args.dataset_path),
        "schemes": {},
        "params": {},
        "cascade": False,
        "plugin_libs": [],
        "files": {"llm": "llm_bf16.onnx", "expert": "expert_bf16.onnx"},
        "float": True,
        "calibration": {"seed": args.seed, "num_samples": 1, "samples": [asdict(s) for s in samples]},
    }
    (out / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
    logger.info("wrote %s (all-float baseline, total %.0fs)", out, time.time() - t0)
    return out


def main(args: ExportConfig) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    if is_quantized_checkpoint(args.checkpoint_dir):
        return to_onnx(ToOnnx(quantized_model=args.checkpoint_dir, output_dir=args.output_dir, device=args.device))
    return export_float(args)


if __name__ == "__main__":
    main(tyro.cli(ExportConfig))
