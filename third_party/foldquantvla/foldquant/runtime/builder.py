# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""Compile ONNX graphs into TensorRT engines.

The caller selects strong or weak typing from the export configuration.
FoldQuant plugin graphs use strong typing; the weakly typed path supports
float components in hybrid deployments.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

from .plugins import prepare_plugins

logger = logging.getLogger(__name__)

__all__ = ["ShapeProfile", "build_engine", "profiles_from_onnx", "static_profile"]

#: Opt-in escape hatch for host-memory-constrained builds (TensorRT's default
#: level-3 optimizer can be OOM-killed inside a small cgroup while the policy
#: weights are still resident). Unset leaves TensorRT's own default untouched.
OPT_LEVEL_ENV = "FOLDQUANT_TRT_BUILDER_OPT_LEVEL"


@dataclass(frozen=True)
class ShapeProfile:
    """``min`` / ``opt`` / ``max`` shape of one engine input."""

    min: Tuple[int, ...]
    opt: Tuple[int, ...]
    max: Tuple[int, ...]

    @classmethod
    def static(cls, shape: Sequence[int]) -> "ShapeProfile":
        s = tuple(int(d) for d in shape)
        return cls(s, s, s)


def static_profile(shapes: Mapping[str, Sequence[int]]) -> Dict[str, ShapeProfile]:
    """A fully static optimization profile from ``{input_name: shape}``."""
    return {name: ShapeProfile.static(shape) for name, shape in shapes.items()}


def profiles_from_onnx(
    onnx_path: Path,
    dims: Mapping[str, Union[int, Tuple[int, int, int]]],
) -> Dict[str, ShapeProfile]:
    """Derive one :class:`ShapeProfile` per graph input from the ONNX value infos.

    Static dimensions are copied. A symbolic dimension is looked up by name in
    *dims*: an ``int`` pins it, a ``(min, opt, max)`` triple gives it a range.
    An unnamed dynamic dimension or a symbolic name absent from *dims* raises:
    the caller decides every range, nothing is guessed from the graph.
    """
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    profiles: Dict[str, ShapeProfile] = {}
    for inp in model.graph.input:
        lo: list = []
        op: list = []
        hi: list = []
        for i, d in enumerate(inp.type.tensor_type.shape.dim):
            if d.dim_value > 0:
                lo.append(d.dim_value)
                op.append(d.dim_value)
                hi.append(d.dim_value)
                continue
            if not d.dim_param:
                raise ValueError(f"input {inp.name!r}: dimension {i} is dynamic but unnamed")
            if d.dim_param not in dims:
                raise ValueError(f"input {inp.name!r}: no range given for symbolic dimension {d.dim_param!r}")
            spec = dims[d.dim_param]
            if isinstance(spec, int):
                spec = (spec, spec, spec)
            mn, o, mx = (int(v) for v in spec)
            if not 1 <= mn <= o <= mx:
                raise ValueError(f"dimension {d.dim_param!r}: need 1 <= min <= opt <= max, got {spec}")
            lo.append(mn)
            op.append(o)
            hi.append(mx)
        profiles[inp.name] = ShapeProfile(tuple(lo), tuple(op), tuple(hi))
    return profiles


def build_engine(
    onnx_path: Path,
    engine_path: Path,
    *,
    profiles: Mapping[str, ShapeProfile],
    plugin_libs: Iterable[str] = (),
    strongly_typed: bool = True,
    int8: bool = True,
    workspace_mb: int = 4096,
    opt_level: Optional[int] = None,
) -> Path:
    """Parse *onnx_path*, build a serialized engine with *profiles*, write it to *engine_path*.

    Args:
        profiles: One :class:`ShapeProfile` per graph input. Every input must be
            covered, or TensorRT rejects the profile at build time.
        plugin_libs: FoldQuant plugin libraries the graph's nodes come from; they
            are resolved (rebuilt for this device if needed) and loaded before
            the parser runs.
        strongly_typed: ``True`` for every FoldQuant plugin graph. ``False``
            builds a weakly-typed network for float modules.
        int8: Weakly-typed builds only: set the INT8 builder flag (TensorRT 10;
            TensorRT 11 removed the precision flags and auto-selects).
        workspace_mb: Workspace memory-pool limit.
        opt_level: Builder optimization level; ``None`` reads ``FOLDQUANT_TRT_BUILDER_OPT_LEVEL``.
    """
    import tensorrt as trt

    onnx_path = Path(onnx_path)
    engine_path = Path(engine_path)
    prepare_plugins(plugin_libs)

    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)
    if strongly_typed:
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    else:
        # TensorRT 11 removed EXPLICIT_BATCH (explicit batch is unconditional);
        # TensorRT 10 (Jetson Orin) still wants the flag.
        explicit_batch = getattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH", None)
        network_flags = (1 << int(explicit_batch)) if explicit_batch is not None else 0
    network = builder.create_network(network_flags)

    # parse_from_file, not parse(bytes): a >2 GB graph keeps its weights in a
    # companion <name>.onnx.data referenced by relative path, which the parser
    # resolves against the .onnx file's directory only when it knows the path.
    parser = trt.OnnxParser(network, trt_logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"Failed to parse ONNX graph {onnx_path}:\n{errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_mb) * 1024 * 1024)
    if opt_level is None:
        env = os.environ.get(OPT_LEVEL_ENV) or None
        if env is not None:
            try:
                opt_level = int(env)
            except ValueError as exc:
                raise ValueError(f"{OPT_LEVEL_ENV}={env!r} is not an integer (expected 0-5).") from exc
    if opt_level is not None:
        config.builder_optimization_level = int(opt_level)
        logger.info("TensorRT builder_optimization_level=%d", opt_level)
    if not strongly_typed:
        if int8:
            flag = getattr(trt.BuilderFlag, "INT8", None)
            if flag is not None:
                config.set_flag(flag)
        else:
            # BF16 only, never FP16: a weakly-typed network picks a precision per
            # layer, and offering FP16 lets it demote bf16 activations into a far
            # smaller exponent range (65504 against ~3e38), which overflows to
            # NaN with no build-time error.
            flag = getattr(trt.BuilderFlag, "BF16", None)
            if flag is not None:
                config.set_flag(flag)

    graph_inputs = {network.get_input(i).name for i in range(network.num_inputs)}
    missing = graph_inputs - set(profiles)
    if missing:
        raise ValueError(f"profiles cover {sorted(profiles)} but the graph also has inputs {sorted(missing)}.")
    profile = builder.create_optimization_profile()
    for name, triple in profiles.items():
        if name not in graph_inputs:
            raise ValueError(f"profile names input {name!r}, which the graph does not have: {sorted(graph_inputs)}")
        profile.set_shape(name, tuple(triple.min), tuple(triple.opt), tuple(triple.max))
    config.add_optimization_profile(profile)

    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError(f"TensorRT engine build failed for {onnx_path} (builder returned None).")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(engine_bytes))
    logger.info("Wrote %s (%.1f MB)", engine_path, engine_path.stat().st_size / 2**20)
    return engine_path
