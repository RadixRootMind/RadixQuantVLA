# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""Run TensorRT engines with torch CUDA tensors.

Device pointers are bound directly to the execution context. Each engine uses
a dedicated CUDA stream ordered against the caller's stream with events.

``FOLDQUANT_TRT_CUDA_GRAPH=1`` enables capture and replay per input shape,
using stable staging buffers. Outputs are reused in both modes: copy any
result that must survive another call to the same engine.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


@dataclass
class _GraphRecord:
    """One captured CUDA graph: stable input/output buffers + the graph."""

    graph: torch.cuda.CUDAGraph
    inputs: dict[str, torch.Tensor]
    outputs: dict[str, torch.Tensor]


__all__ = ["TensorRTEngine"]


def _trt_to_torch_dtype(trt: Any, dtype: Any) -> torch.dtype:
    mapping = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int8: torch.int8,
        trt.int32: torch.int32,
        trt.bool: torch.bool,
    }
    if hasattr(trt, "int64"):
        mapping[trt.int64] = torch.int64
    if hasattr(trt, "bfloat16"):
        mapping[trt.bfloat16] = torch.bfloat16
    if dtype not in mapping:
        raise ValueError(f"TensorRTEngine: unsupported TensorRT dtype {dtype!r}.")
    return mapping[dtype]


class TensorRTEngine:
    """Deserialized TensorRT engine, callable like a single-purpose ``nn.Module``.

    Args:
        engine_path: Path to a serialized ``.engine`` file.
    """

    # Shared across every instance in the process: trt.Runtime/Logger carry
    # process-global state; GR00T N1.6 alone deserializes up to six of these.
    _shared_runtime: Any = None
    _shared_logger: Any = None

    def __init__(self, engine_path: str | Path) -> None:
        import tensorrt as trt

        cls = type(self)
        if cls._shared_logger is None:
            cls._shared_logger = trt.Logger(trt.Logger.WARNING)
        if cls._shared_runtime is None:
            cls._shared_runtime = trt.Runtime(cls._shared_logger)

        engine_bytes = Path(engine_path).read_bytes()
        self.engine = cls._shared_runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"TensorRTEngine: failed to deserialize engine at {engine_path}.")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"TensorRTEngine: failed to create an execution context for {engine_path}.")

        self.in_meta: list[tuple[str, tuple[int, ...], torch.dtype]] = []
        self.out_meta: list[tuple[str, tuple[int, ...], torch.dtype]] = []
        self._input_profiles: dict[str, tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = {}
        self._out_cache: dict[str, torch.Tensor] = {}
        # Dedicated non-default stream: TensorRT adds its own cudaStreamSynchronize
        # calls when enqueued on the legacy default stream ("Using default stream in
        # enqueueV3() may lead to performance issues"). Created lazily in forward()
        # so constructing an engine never requires an initialized CUDA context.
        self._stream: torch.cuda.Stream | None = None
        # CUDA-graph replay (opt-in): one captured graph per input-shape key.
        self._use_cuda_graph = os.environ.get("FOLDQUANT_TRT_CUDA_GRAPH", "0") == "1"
        self._graphs: dict[tuple[tuple[int, ...], ...], _GraphRecord] = {}

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = _trt_to_torch_dtype(trt, self.engine.get_tensor_dtype(name))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.in_meta.append((name, shape, dtype))
                min_shape, opt_shape, max_shape = self.engine.get_tensor_profile_shape(name, 0)
                self._input_profiles[name] = (tuple(min_shape), tuple(opt_shape), tuple(max_shape))
            else:
                self.out_meta.append((name, shape, dtype))


    # Shape / I/O contract validation


    def set_runtime_tensor_shape(self, name: str, shape: tuple[int, ...]) -> None:
        """Validate *shape* against the compiled profile bounds, then bind it.

        Raises:
            RuntimeError: If *shape* falls outside the engine's compiled
                min/max bounds for *name*, or the TensorRT context rejects it.
        """
        bounds = self._input_profiles.get(name)
        if bounds is not None:
            min_shape, _, max_shape = bounds
            for axis, (dim, lo, hi) in enumerate(zip(shape, min_shape, max_shape)):
                if not (lo <= dim <= hi):
                    raise RuntimeError(
                        f"TensorRTEngine: input {name!r} axis {axis} = {dim} is outside the compiled "
                        f"profile bounds [{lo}, {hi}]. Shape profiles are fixed at compile time; "
                        "rebuild the engine with wider bounds if this input is legitimately larger."
                    )
        if not self.context.set_input_shape(name, tuple(shape)):
            raise RuntimeError(f"TensorRTEngine: set_input_shape rejected {name!r} with shape={tuple(shape)!r}.")

    def validate_binding_names(self, expected_inputs: set[str], expected_outputs: set[str]) -> None:
        """Assert this engine's actual I/O tensor names match *expected_inputs*/*expected_outputs*.

        Raises:
            RuntimeError: Naming any missing/unexpected input or output.
        """
        actual_inputs = {name for name, _, _ in self.in_meta}
        actual_outputs = {name for name, _, _ in self.out_meta}
        missing_in, extra_in = expected_inputs - actual_inputs, actual_inputs - expected_inputs
        missing_out, extra_out = expected_outputs - actual_outputs, actual_outputs - expected_outputs
        if missing_in or extra_in or missing_out or extra_out:
            raise RuntimeError(
                "TensorRTEngine: I/O contract mismatch: "
                f"missing inputs={sorted(missing_in)}, unexpected inputs={sorted(extra_in)}, "
                f"missing outputs={sorted(missing_out)}, unexpected outputs={sorted(extra_out)}."
            )


    # Execution


    def _bind_input(self, name: str, dtype: torch.dtype, tensor: torch.Tensor, skip_checks: bool) -> torch.Tensor:
        if not skip_checks:
            if not isinstance(tensor, torch.Tensor):
                raise RuntimeError(f"TensorRTEngine: input {name!r} must be a torch.Tensor, got {type(tensor)!r}.")
            if tensor.dtype != dtype:
                raise RuntimeError(f"TensorRTEngine: input {name!r} dtype {tensor.dtype} != expected {dtype}.")
            if tensor.device.type != "cuda":
                raise RuntimeError(f"TensorRTEngine: input {name!r} must be a CUDA tensor, got device={tensor.device}.")
        tensor = tensor.contiguous()
        self.set_runtime_tensor_shape(name, tuple(tensor.shape))
        self.context.set_tensor_address(name, tensor.data_ptr())
        return tensor

    def _forward_cuda_graph(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Replay (capturing on first sight of a shape key) instead of enqueueing.

        Raises:
            RuntimeError: If graph capture fails. The flag is an explicit
                opt-in, so a failed capture is surfaced (with the advice to
                unset ``FOLDQUANT_TRT_CUDA_GRAPH``) rather than silently falling
                back to a differently-performing path.
        """
        key = tuple(tuple(inputs[name].shape) for name, _, _ in self.in_meta)
        record = self._graphs.get(key)
        caller_stream = torch.cuda.current_stream()
        if self._stream is None:
            self._stream = torch.cuda.Stream()

        if record is None:
            staging = {name: inputs[name].contiguous().clone() for name in inputs}
            for name, tensor in staging.items():
                self.set_runtime_tensor_shape(name, tuple(tensor.shape))
                self.context.set_tensor_address(name, tensor.data_ptr())
            outputs: dict[str, torch.Tensor] = {}
            for name, _default_shape, dtype in self.out_meta:
                real_shape = tuple(self.context.get_tensor_shape(name))
                out = torch.empty(real_shape, dtype=dtype, device="cuda")
                self.context.set_tensor_address(name, out.data_ptr())
                outputs[name] = out
            # Warm-up runs on the side stream (cuDNN/TRT lazy init must not
            # happen inside capture), then capture a single enqueue.
            self._stream.wait_stream(caller_stream)
            with torch.cuda.stream(self._stream):
                for _ in range(2):
                    if not self.context.execute_async_v3(self._stream.cuda_stream):
                        raise RuntimeError("TensorRTEngine: warm-up execute_async_v3() failed before graph capture.")
            caller_stream.wait_stream(self._stream)
            graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph, stream=self._stream):
                    if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
                        raise RuntimeError("TensorRTEngine: execute_async_v3() failed during graph capture.")
            except RuntimeError as exc:
                raise RuntimeError(
                    f"TensorRTEngine: CUDA-graph capture failed for input shapes {key!r}: {exc}. "
                    "Unset FOLDQUANT_TRT_CUDA_GRAPH to run without graph replay."
                ) from exc
            record = _GraphRecord(graph=graph, inputs=staging, outputs=outputs)
            self._graphs[key] = record
            logger.info("TensorRTEngine: captured CUDA graph for shapes %s.", key)
            return dict(record.outputs)

        self._stream.wait_stream(caller_stream)
        with torch.cuda.stream(self._stream):
            for name, staging_buf in record.inputs.items():
                staging_buf.copy_(inputs[name], non_blocking=True)
            record.graph.replay()
        for tensor in inputs.values():
            tensor.record_stream(self._stream)
        caller_stream.wait_stream(self._stream)
        return dict(record.outputs)

    def forward(
        self,
        *args: torch.Tensor,
        return_list: bool = False,
        skip_checks: bool = False,
        **kwargs: torch.Tensor,
    ) -> dict[str, torch.Tensor] | list[torch.Tensor]:
        """Run one inference call. Inputs bind by position (per :attr:`in_meta` order) or by name."""
        if self._use_cuda_graph:
            gathered: dict[str, torch.Tensor] = {}
            for i, (name, _, dtype) in enumerate(self.in_meta):
                if name in kwargs:
                    tensor = kwargs[name]
                elif i < len(args):
                    tensor = args[i]
                else:
                    raise RuntimeError(f"TensorRTEngine.forward(): missing required input {name!r}.")
                if not skip_checks:
                    if tensor.dtype != dtype:
                        raise RuntimeError(f"TensorRTEngine: input {name!r} dtype {tensor.dtype} != expected {dtype}.")
                    if tensor.device.type != "cuda":
                        raise RuntimeError(f"TensorRTEngine: input {name!r} must be a CUDA tensor.")
                gathered[name] = tensor.contiguous()
            graph_outputs = self._forward_cuda_graph(gathered)
            if return_list:
                return [graph_outputs[name] for name, _, _ in self.out_meta]
            return graph_outputs

        held: list[torch.Tensor] = []
        for i, (name, _, dtype) in enumerate(self.in_meta):
            if name in kwargs:
                tensor = kwargs[name]
            elif i < len(args):
                tensor = args[i]
            else:
                raise RuntimeError(f"TensorRTEngine.forward(): missing required input {name!r}.")
            held.append(self._bind_input(name, dtype, tensor, skip_checks))

        outputs: dict[str, torch.Tensor] = {}
        for name, _default_shape, dtype in self.out_meta:
            real_shape = tuple(self.context.get_tensor_shape(name))
            cached = self._out_cache.get(name)
            if cached is None or tuple(cached.shape) != real_shape or cached.dtype != dtype or not cached.is_cuda:
                cached = torch.empty(real_shape, dtype=dtype, device="cuda")
                self._out_cache[name] = cached
            self.context.set_tensor_address(name, cached.data_ptr())
            outputs[name] = cached

        # Run on a dedicated stream, ordered against the caller's stream with
        # events only (no host-side sync). record_stream() on every tensor the
        # engine touches stops the caching allocator from recycling memory the
        # engine stream is still reading/writing.
        caller_stream = torch.cuda.current_stream()
        if self._stream is None:
            self._stream = torch.cuda.Stream()
        self._stream.wait_stream(caller_stream)
        with torch.cuda.stream(self._stream):
            if not self.context.execute_async_v3(self._stream.cuda_stream):
                raise RuntimeError("TensorRTEngine: execute_async_v3() failed.")
        for tensor in (*held, *outputs.values()):
            tensor.record_stream(self._stream)
        caller_stream.wait_stream(self._stream)

        if return_list:
            return [outputs[name] for name, _, _ in self.out_meta]
        return outputs

    def __call__(
        self,
        *args: torch.Tensor,
        return_list: bool = False,
        skip_checks: bool = False,
        **kwargs: torch.Tensor,
    ) -> dict[str, torch.Tensor] | list[torch.Tensor]:
        return self.forward(*args, return_list=return_list, skip_checks=skip_checks, **kwargs)

    def memory_summary(self) -> dict[str, int]:
        """Return ``{output_name: bytes}`` for cached output buffers (inputs are zero-copy)."""
        return {name: t.numel() * t.element_size() for name, t in self._out_cache.items()}

    def close(self) -> None:
        self.context = None
        self.engine = None
        self._out_cache.clear()
        self._graphs.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
