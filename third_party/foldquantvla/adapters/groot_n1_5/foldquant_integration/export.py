# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""Export a GR00T N1.5 checkpoint to the ONNX graphs the engine builder compiles.

Reads ``--model-path`` and writes, into ``--output-dir``::

    llm_bf16.onnx / dit_bf16.onnx   the two towers
    export_metadata.json            shape hints for the engine builder
    foldquant_export.json           what was exported, needing which plugins

A **quantized model** (what ``quantize`` wrote) needs no dataset: the recorded
weight codes are replayed into the FoldQuant plugin graphs, byte-identical to
the ones the quantization built. An **unquantized checkpoint** is the all-float
baseline: each tower is traced under the same engine bindings on one
observation from ``--dataset-path``, with no plugin nodes.
``build_engines --onnx-dir <output-dir>`` then builds the engines either way.

Example::

    python -m foldquant_integration.export --model-path exports/w4a4/quantized --output-dir exports/w4a4/onnx
    python -m foldquant_integration.export --model-path <checkpoint> --dataset-path <dataset> \\
        --embodiment-tag <tag> --output-dir exports/float/onnx
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, Optional

from foldquant.provenance import public_path
from foldquant.quantized import ToOnnx, to_onnx
from foldquant.quantized_checkpoint import is_quantized_checkpoint
from foldquant.trace_export import Binding, causal_additive_mask_4d, trace_module
import torch
import tyro

from ._upstream import EXPORT_METADATA_NAME, LIBERO_DATA_CONFIG, MANIFEST_NAME
from .quantized import FAMILY, module_paths

logger = logging.getLogger("foldquant.groot_n1_5.export")


@dataclass
class ExportConfig:
    model_path: str
    """A quantized model (``quantize``'s ``--output-dir``) or an unquantized checkpoint (the all-float baseline)."""

    output_dir: str
    """Destination directory of the graphs and manifests."""

    dataset_path: Optional[str] = None
    """LeRobot dataset; one observation traces the graphs of an unquantized checkpoint (unused for a quantized model)."""

    embodiment_tag: Optional[str] = None
    """Defaults to the tag the model was quantized with, or the checkpoint's single embodiment."""

    data_config: str = LIBERO_DATA_CONFIG
    """Upstream data config of an unquantized checkpoint (a quantized model records its own)."""

    denoising_steps: Optional[int] = None

    seed: int = 0
    """Seed of the traced observation of an unquantized checkpoint."""

    video_backend: str = "torchcodec"

    device: str = "cuda"


def _ones_mask(kw: Dict[str, Any]) -> torch.Tensor:
    enc = kw["encoder_hidden_states"]
    return torch.ones(enc.shape[:2], dtype=torch.bool, device=enc.device)


def export_float(args: ExportConfig) -> Path:
    """The all-float baseline: both towers traced under the engine bindings, no plugin nodes."""
    from . import calibration
    from .quantize import capture_shape_metadata, export_metadata

    if not args.dataset_path:
        raise SystemExit(f"{args.model_path} is not a quantized model; tracing its graphs needs --dataset-path")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    policy = calibration.load_policy(args.model_path, args.embodiment_tag, args.device, data_config=args.data_config, denoising_steps=args.denoising_steps)
    dataset = calibration.load_dataset(policy, args.dataset_path, args.video_backend)
    samples, observations = calibration.sample_observations(policy, dataset, 1, seed=args.seed)
    loop = calibration.make_forward_loop(policy, observations, seed=args.seed)
    logger.info("policy + dataset in %.0fs; tracing on %s", time.time() - t0, samples[0])
    shapes = capture_shape_metadata(policy, observations[0])
    modules = module_paths(policy)

    t1 = time.time()
    trace_module(
        modules["llm"], out / "llm_bf16.onnx", module_name="llm",
        bindings=[
            Binding("inputs_embeds", "inputs_embeds", torch.bfloat16, {1: "seq_len"}),
            Binding("attention_mask", "attention_mask", torch.int64, {1: "seq_len"}, transform=causal_additive_mask_4d),
        ],
        output_name="hidden_states", output_dynamic={1: "seq_len"}, forward_loop=loop,
        extract=lambda o: o.hidden_states[-1],
    )
    trace_module(
        modules["dit"], out / "dit_bf16.onnx", module_name="dit",
        bindings=[
            Binding("sa_embs", "hidden_states", torch.bfloat16),
            Binding("vl_embs", "encoder_hidden_states", torch.bfloat16, {1: "vl_seq_len"}),
            Binding("timestep", "timestep", torch.int64),
            # plain DiT attends every encoder token; the runtime feeds all-True masks the
            # module never sees. Kept as bindings so the engine matches the contract.
            Binding("image_mask", "image_mask", torch.bool, {1: "vl_seq_len"}, default=_ones_mask, passthrough=True),
            Binding("backbone_attention_mask", "backbone_attention_mask", torch.bool, {1: "vl_seq_len"}, default=_ones_mask, passthrough=True),
        ],
        output_name="output", forward_loop=loop,
        extract=lambda o: o[0] if isinstance(o, (tuple, list)) else o,
    )
    logger.info("two float graphs traced in %.0fs", time.time() - t1)

    (out / EXPORT_METADATA_NAME).write_text(json.dumps(export_metadata(shapes, policy), indent=2))
    manifest: Dict[str, Any] = {
        "family": FAMILY,
        "model_path": public_path(args.model_path),
        "embodiment_tag": policy.embodiment_tag.value,
        "data_config": args.data_config,
        "dataset_path": public_path(args.dataset_path),
        "schemes": {},
        "params": {},
        "cascade": False,
        "llm_final_norm": shapes["final_norm"],
        "plugin_libs": [],
        "files": {"llm": "llm_bf16.onnx", "dit": "dit_bf16.onnx"},
        "float": True,
        "calibration": {"seed": args.seed, "num_samples": 1, "samples": [asdict(s) for s in samples]},
    }
    (out / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
    logger.info("wrote %s (all-float baseline, total %.0fs)", out, time.time() - t0)
    return out


def main(args: ExportConfig) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    if is_quantized_checkpoint(args.model_path):
        return to_onnx(ToOnnx(quantized_model=args.model_path, output_dir=args.output_dir,
                              embodiment_tag=args.embodiment_tag, device=args.device))
    return export_float(args)


if __name__ == "__main__":
    main(tyro.cli(ExportConfig))
