# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""Export NVIDIA ModelOpt INT8 SmoothQuant and W4A16 AWQ baselines.

Calls are captured from the BF16 policy and replayed per module without
cascade calibration. ModelOpt quantizes eligible projections, then the
TorchScript exporter emits opset-20 Q/DQ graphs with each family's bindings.
Export helpers repair dtypes, cast outputs, consolidate external data, and
check that Q/DQ nodes survived.

W4A16 AWQ weights are rewritten to ``Int4GroupwiseGemmPlugin`` nodes for
TensorRT 10.3. The family engine builders compile quantized modules with
strong typing. Only ``quantize_module`` requires the lazy ModelOpt import.
"""

from __future__ import annotations

import copy
import fnmatch
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn

logger = logging.getLogger(__name__)

#: The algorithm keys this module implements (the framework's keys, unchanged).
MODELOPT_W8A8_SMOOTHQUANT = "modelopt_w8a8_smoothquant"
#: INT4 weight-only AWQ, group 128, ``awq_lite`` scale search (``INT4_AWQ_CFG``'s
#: own default; the key has always meant the lite search, so nothing overrides it).
MODELOPT_W4A16_AWQ = "modelopt_w4a16_awq"

#: Algorithm key -> attribute of ``modelopt.torch.quantization`` holding its base config.
BASE_CFG: Mapping[str, str] = {
    MODELOPT_W8A8_SMOOTHQUANT: "INT8_SMOOTHQUANT_CFG",
    MODELOPT_W4A16_AWQ: "INT4_AWQ_CFG",
}

#: Keys whose weights are quantized and whose activations are not. Their export
#: carries ``trt::DequantizeLinear`` weights and NO activation Q/DQ, and TensorRT
#: 10.3 has no INT4 weight-only path of its own, so the graph is rewritten to
#: :mod:`foldquant.int4_groupwise` plugin nodes before the parser sees it.
WEIGHT_ONLY_ALGORITHMS: frozenset = frozenset({MODELOPT_W4A16_AWQ})

#: ModelOpt version the arm was reproduced with. ``quant_cfg`` changed shape at 0.45.
MODELOPT_VERSION = "0.45.0"

#: Sub-layers left unquantized inside a quantized module (matched on the lowercase name).
DEFAULT_LAYER_EXCLUDE: Tuple[str, ...] = ("*norm*", "*layernorm*", "*final_action*", "*action_proj*")

#: Opset of the exported Q/DQ graphs (the preset's ``export.opset``).
DEFAULT_OPSET = 20

QUANTIZABLE_LEAF_TYPES: Tuple[type, ...] = (
    nn.Linear,
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
)

_QDQ_OP_TYPES = frozenset({"QuantizeLinear", "DequantizeLinear", "DynamicQuantizeLinear"})

CapturedCalls = List[Tuple[tuple, dict]]


def load_mtq() -> Any:
    try:
        import modelopt.torch.quantization as mtq
    except ImportError as exc:
        raise ImportError(
            f"the {MODELOPT_W8A8_SMOOTHQUANT} arm needs nvidia-modelopt=={MODELOPT_VERSION} "
            "(and onnx-graphsurgeon) in this environment"
        ) from exc
    return mtq


def ensure_cuda_ext() -> None:
    """Build / load ModelOpt's CUDA fake-quant extension, or fail before any long calibration.

    Without it ModelOpt falls back to a CPU kernel, and tracing a CUDA module for
    ONNX then segfaults inside ``torch.jit`` (measured on a Jetson Orin, torch
    2.10). The build needs ``ninja``, which a virtualenv installs next to its own
    interpreter without that directory being on ``PATH``; it is added here. The
    compiled extension is cached under ``~/.cache/torch_extensions``, so only the
    first run pays the compile (about 95 s on an Orin).
    """
    env_bin = os.path.dirname(sys.executable)
    if shutil.which("ninja") is None and os.path.exists(os.path.join(env_bin, "ninja")):
        os.environ["PATH"] = env_bin + os.pathsep + os.environ.get("PATH", "")
    load_mtq()
    from modelopt.torch.quantization.extensions import get_cuda_ext

    try:
        get_cuda_ext(raise_if_failed=True)
    except Exception as exc:  # noqa: BLE001 - re-raised with the remedy
        raise RuntimeError(
            "ModelOpt's CUDA extension (modelopt_cuda_ext) could not be built: install ninja in this "
            "environment and point CUDA_HOME at the CUDA toolkit (e.g. /usr/local/cuda-12.6)"
        ) from exc


def is_modelopt_scheme(scheme: Optional[str]) -> bool:
    return scheme in BASE_CFG


def is_weight_only(scheme: Optional[str]) -> bool:
    """Whether *scheme* quantizes weights only (no activation quantizers, INT4 groupwise plugin)."""
    return scheme in WEIGHT_ONLY_ALGORITHMS


# capture


def _snapshot(value: Any) -> Any:
    """Detached copies of the tensors in a call, so later in-place edits cannot reach the capture."""
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(_snapshot(v) for v in value)
    if isinstance(value, list):
        return [_snapshot(v) for v in value]
    if isinstance(value, dict):
        return {k: _snapshot(v) for k, v in value.items()}
    return value


def capture_calls(module: nn.Module, forward_loop: Callable[[Any], None]) -> CapturedCalls:
    """Run *forward_loop* once and return every call *module* received."""
    return capture_many({"module": module}, forward_loop)["module"]


def capture_many(modules: Mapping[str, nn.Module], forward_loop: Callable[[Any], None]) -> Dict[str, CapturedCalls]:
    """Hook several modules at once, so one replay captures all of them."""
    store: Dict[str, CapturedCalls] = {name: [] for name in modules}
    handles = []
    for name, module in modules.items():

        def _hook(_m: nn.Module, args: tuple, kwargs: dict, _calls: CapturedCalls = store[name]) -> None:
            _calls.append((_snapshot(args), _snapshot(dict(kwargs))))

        handles.append(module.register_forward_pre_hook(_hook, with_kwargs=True))
    try:
        forward_loop(None)
    finally:
        for handle in handles:
            handle.remove()
    empty = sorted(name for name, calls in store.items() if not calls)
    if empty:
        raise RuntimeError(f"calibration replay never reached {empty}; nothing to calibrate on")
    return store


def replay_loop(calls: Sequence[Tuple[tuple, dict]], target: Optional[nn.Module] = None) -> Callable[[nn.Module], None]:
    """A ``forward_loop(module)`` that feeds *calls* back, into *target* when given."""
    samples = list(calls)

    def _loop(module: nn.Module) -> None:
        callee = target if target is not None else module
        with torch.inference_mode():
            for args, kwargs in samples:
                callee(*args, **kwargs)

    return _loop


# quantize


def apply_layer_exclusions(cfg: Dict[str, Any], module: nn.Module) -> List[str]:
    """Disable the quantizers of every leaf of *module* matching :data:`DEFAULT_LAYER_EXCLUDE`.

    ``quant_cfg`` is an ordered list from ModelOpt 0.45 (later entries win) and a
    dict of glob patterns before it; both are written. Returns the excluded names.
    """
    excluded: List[str] = []
    quant_cfg = cfg.setdefault("quant_cfg", {})
    if not isinstance(quant_cfg, (dict, list)):
        raise TypeError(f"unrecognised ModelOpt quant_cfg type {type(quant_cfg).__name__}; cannot exclude layers")
    for sub_name, sub in module.named_modules():
        if not sub_name or not isinstance(sub, QUANTIZABLE_LEAF_TYPES):
            continue
        if not any(fnmatch.fnmatch(sub_name.lower(), pattern) for pattern in DEFAULT_LAYER_EXCLUDE):
            continue
        if isinstance(quant_cfg, dict):
            quant_cfg[f"*{sub_name}*"] = {"enable": False}
        else:
            quant_cfg.append({"quantizer_name": f"*{sub_name}*", "enable": False})
        excluded.append(sub_name)
    return sorted(excluded)


def quantize_module(
    module: nn.Module,
    forward_loop: Callable[[nn.Module], None],
    *,
    algorithm: str = MODELOPT_W8A8_SMOOTHQUANT,
) -> Dict[str, Any]:
    """Quantize *module* in place with ModelOpt and return the provenance record."""
    if algorithm not in BASE_CFG:
        raise ValueError(f"unknown ModelOpt algorithm {algorithm!r}; known: {sorted(BASE_CFG)}")
    mtq = load_mtq()
    ensure_cuda_ext()
    base_cfg = BASE_CFG[algorithm]
    cfg = copy.deepcopy(getattr(mtq, base_cfg))
    excluded = apply_layer_exclusions(cfg, module)
    mtq.quantize(module, cfg, forward_loop=forward_loop)

    counts = quantizer_counts(module)
    n_quantizers = counts["enabled"]
    if n_quantizers == 0:
        raise RuntimeError(f"{algorithm}: ModelOpt enabled no quantizer in {type(module).__name__}")
    try:
        from modelopt import __version__ as modelopt_version
    except ImportError:  # pragma: no cover - load_mtq succeeded
        modelopt_version = "unknown"
    logger.info(
        "%s: %d of %d quantizers enabled (%d input, %d weight, %d with a SmoothQuant pre-quant scale), "
        "%d sub-layers excluded",
        algorithm,
        n_quantizers,
        counts["inserted"],
        counts["enabled_input"],
        counts["enabled_weight"],
        counts["pre_quant_scales"],
        len(excluded),
    )
    return {
        "algorithm": algorithm,
        "provider": "nvidia_modelopt",
        "modelopt_version": modelopt_version,
        "base_cfg": base_cfg,
        "layer_exclude": list(DEFAULT_LAYER_EXCLUDE),
        "excluded_sublayers": excluded,
        "enabled_quantizers": n_quantizers,
        "quantizer_counts": counts,
    }


def quantizer_counts(module: nn.Module) -> Dict[str, int]:
    """How many ``TensorQuantizer``s ModelOpt inserted in *module*, how many are enabled, of which kind."""
    from modelopt.torch.quantization.nn import TensorQuantizer

    counts = {"inserted": 0, "enabled": 0, "enabled_input": 0, "enabled_weight": 0, "pre_quant_scales": 0}
    for name, m in module.named_modules():
        if not isinstance(m, TensorQuantizer):
            continue
        counts["inserted"] += 1
        if not m.is_enabled:
            continue
        counts["enabled"] += 1
        if name.endswith("input_quantizer"):
            counts["enabled_input"] += 1
        elif name.endswith("weight_quantizer"):
            counts["enabled_weight"] += 1
        if getattr(m, "pre_quant_scale", None) is not None:
            counts["pre_quant_scales"] += 1
    return counts


def quantizer_state(module: nn.Module) -> Dict[str, Dict[str, Any]]:
    """Per enabled quantizer: its ``amax`` and SmoothQuant ``pre_quant_scale`` (CPU float32), by name.

    Names are relative to *module*, as ModelOpt matches its config patterns, so
    the dump compares key for key against the same module quantized elsewhere.
    """
    from modelopt.torch.quantization.nn import TensorQuantizer

    state: Dict[str, Dict[str, Any]] = {}
    for name, m in module.named_modules():
        if not isinstance(m, TensorQuantizer) or not m.is_enabled:
            continue
        entry: Dict[str, Any] = {}
        for attr in ("amax", "pre_quant_scale"):
            value = getattr(m, attr, None)
            if isinstance(value, torch.Tensor):
                entry[attr] = value.detach().float().cpu()
        state[name] = entry
    return state


# checkpoint state: what a ModelOpt tower stores beside its (smoothed) weights


def modelopt_state_of(module: nn.Module) -> Dict[str, Any]:
    """``mto.modelopt_state(module)``: the mode and its config, which :func:`restore_quantizers`
    rebuilds the quantizers from. Not JSON (it carries dtypes and calibration intermediates);
    the checkpoint stores it with ``torch.save``."""
    import modelopt.torch.opt as mto

    return mto.modelopt_state(module)


def quantizer_buffers(module: nn.Module) -> Dict[str, torch.Tensor]:
    """``{"<quantizer>.amax" | ".pre_quant_scale": tensor}`` of **every** quantizer, in its own dtype.

    Disabled quantizers included: AWQ smooths each weight by its scale ``s`` and leaves ``1/s``
    as the ``pre_quant_scale`` of the linear's input quantizer, which a weight-only config keeps
    disabled. ``TensorQuantizer.forward`` applies the pre-quant scale before its disabled early
    return, so that scale is part of the arithmetic; dropping it leaves the ``W*s`` weights
    uncompensated.
    """
    from modelopt.torch.quantization.nn import TensorQuantizer

    out: Dict[str, torch.Tensor] = {}
    for name, m in module.named_modules():
        if not isinstance(m, TensorQuantizer):
            continue
        for attr in ("amax", "pre_quant_scale"):
            value = getattr(m, attr, None)
            if isinstance(value, torch.Tensor):
                out[f"{name}.{attr}"] = value.detach().cpu().contiguous()
    return out


def restore_quantizers(module: nn.Module, modelopt_state: Mapping[str, Any], buffers: Mapping[str, torch.Tensor]) -> Dict[str, int]:
    """Put the quantizers of a saved ModelOpt tower back on the live *module*: the mode from
    *modelopt_state*, then each quantizer's ``amax`` / ``pre_quant_scale`` from *buffers*.
    The weights ModelOpt smoothed are the checkpoint's own. Returns the quantizer counts."""
    import modelopt.torch.opt as mto
    from modelopt.torch.quantization.nn import TensorQuantizer

    load_mtq()
    device = next(module.parameters()).device
    mto.restore_from_modelopt_state(module, dict(modelopt_state))
    # The restored quantizers are created with their state tensors on the CPU (ModelOpt warns
    # "could not identify the device for TensorQuantizer states"); a CUDA fake-quant kernel
    # reading them is an illegal memory access. Move them where the weights are.
    module.to(device)
    quantizers = {n: m for n, m in module.named_modules() if isinstance(m, TensorQuantizer)}
    for key, value in buffers.items():
        name, attr = key.rsplit(".", 1)
        if name not in quantizers:
            raise KeyError(f"{name}: no quantizer at this path after restoring the ModelOpt state")
        setattr(quantizers[name], attr, value.to(device))
    counts = quantizer_counts(module)
    logger.info("ModelOpt quantizers restored: %d enabled of %d, %d pre-quant scales",
                counts["enabled"], counts["inserted"], counts["pre_quant_scales"])
    return counts


# export


def onnx_export(
    wrapper: nn.Module,
    args: tuple,
    onnx_path: Path,
    *,
    input_names: Sequence[str],
    output_names: Sequence[str],
    dynamic_axes: Mapping[str, Mapping[int, str]],
    opset: int = DEFAULT_OPSET,
) -> None:
    """``torch.onnx.export`` with the legacy exporter, retrying without constant folding on the bf16 bug.

    Constant folding a bf16 graph can materialise a ComplexDouble constant (ONNX
    enum 15 is COMPLEX128, torch's 15 is bfloat16). Folding is only an
    optimisation, TensorRT folds constants at build, so the export is retried
    without it.
    """
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = dict(
        input_names=list(input_names),
        output_names=list(output_names),
        dynamic_axes={k: dict(v) for k, v in dynamic_axes.items()},
        opset_version=opset,
        export_params=True,
        dynamo=False,
    )
    try:
        torch.onnx.export(wrapper, args, str(onnx_path), do_constant_folding=True, **kwargs)
    except RuntimeError as exc:
        if "ComplexDouble" not in str(exc):
            raise
        logger.warning("constant folding hit the bf16/ComplexDouble exporter bug; re-exporting without it")
        torch.onnx.export(wrapper, args, str(onnx_path), do_constant_folding=False, **kwargs)


def onnx_has_qdq(onnx_path: Path) -> bool:
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    return any(node.op_type in _QDQ_OP_TYPES for node in model.graph.node)


def count_qdq(onnx_path: Path) -> Dict[str, int]:
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    counts: Dict[str, int] = {}
    for node in model.graph.node:
        if node.op_type in _QDQ_OP_TYPES:
            counts[node.op_type] = counts.get(node.op_type, 0) + 1
    return counts


def require_qdq(onnx_path: Path, name: str) -> Dict[str, int]:
    """Fail when a module exported as quantized carries no Q/DQ node (the quantizers were lost)."""
    counts = count_qdq(onnx_path)
    if not counts:
        raise ValueError(
            f"{name} was quantized with ModelOpt but {onnx_path} contains no QuantizeLinear/"
            "DequantizeLinear node: the Q/DQ bake did not survive export"
        )
    logger.info("%s: Q/DQ nodes %s", name, counts)
    return counts


def cast_graph_outputs(onnx_path: Path, name: str, elem_type: int) -> List[str]:
    """Cast every floating graph output that is not *elem_type* to it, in place; returns the names cast.

    ModelOpt's INT8 Q/DQ pairs dequantize to float32, so the last quantized
    Linear leaves a bf16 graph with a float32 output. The framework's runtime casts
    that at the next engine's bf16 input; upstream's ``trt_torch.Engine`` asserts
    the dtype instead, so the same cast is placed at the end of the graph.
    Structure-only; no tensor bytes are loaded.
    """
    import onnx

    floats = {onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16, onnx.TensorProto.BFLOAT16, onnx.TensorProto.DOUBLE}
    model = onnx.load(str(onnx_path), load_external_data=False)
    graph = model.graph
    cast_names: List[str] = []
    for output in graph.output:
        tensor_type = output.type.tensor_type
        if tensor_type.elem_type not in floats or tensor_type.elem_type == elem_type:
            continue
        inner = f"{output.name}_foldquant_uncast"
        for node in graph.node:
            node.output[:] = [inner if o == output.name else o for o in node.output]
            node.input[:] = [inner if i == output.name else i for i in node.input]
        graph.node.append(
            onnx.helper.make_node("Cast", [inner], [output.name], name=f"{output.name}_output_cast", to=elem_type)
        )
        tensor_type.elem_type = elem_type
        cast_names.append(output.name)
    if cast_names:
        onnx.save(model, str(onnx_path))
        logger.info("%s: cast graph outputs %s to %s", name, cast_names, onnx.TensorProto.DataType.Name(elem_type))
    return cast_names


def repair_onnx_dtypes(onnx_path: Path, name: str) -> Dict[str, int]:
    """The TorchScript exporter's dtype defects TensorRT's parser rejects, repaired in place.

    * ``Cast(to=COMPLEX128)``: torch's bfloat16 enum written raw; retargeted to BFLOAT16.
    * ``ScatterND`` whose ``updates`` dtype differs from ``data``: a Cast is inserted
      (on ``data`` when it is a shape-like float constant and ``updates`` is
      integer, on ``updates`` otherwise).
    * ``LayerNormalization`` whose scale / bias dtype differs from the activation:
      the initializer input is cast to the activation dtype.

    Structure-only; no tensor bytes are loaded. Returns how many of each were made.
    """
    import onnx

    onnx_path = Path(onnx_path)
    model = onnx.load(str(onnx_path), load_external_data=False)
    graph = model.graph
    fixed = {"cast_complex128": 0, "scatternd": 0, "layernorm": 0}

    for node in graph.node:
        if node.op_type != "Cast":
            continue
        for attr in node.attribute:
            if attr.name == "to" and attr.i == onnx.TensorProto.COMPLEX128:
                attr.i = onnx.TensorProto.BFLOAT16
                fixed["cast_complex128"] += 1

    if not any(n.op_type in ("ScatterND", "LayerNormalization") for n in graph.node):
        if fixed["cast_complex128"]:
            onnx.save(model, str(onnx_path))
        return fixed

    typed = model
    try:
        typed = onnx.shape_inference.infer_shapes(model, data_prop=True)
    except Exception as exc:  # noqa: BLE001 - inference only enriches the type table
        logger.debug("%s: shape inference for the dtype repair failed (%s)", name, exc)

    elem_types: Dict[str, int] = {}
    for vi in list(typed.graph.value_info) + list(graph.input) + list(graph.output):
        if vi.type.tensor_type.elem_type:
            elem_types[vi.name] = vi.type.tensor_type.elem_type
    init_types = {init.name: init.data_type for init in graph.initializer}
    elem_types.update(init_types)
    for node in graph.node:
        if not node.output:
            continue
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    elem_types[node.output[0]] = attr.t.data_type
        elif node.op_type == "ConstantOfShape":
            dtype = onnx.TensorProto.FLOAT
            for attr in node.attribute:
                if attr.name == "value":
                    dtype = attr.t.data_type
            elem_types[node.output[0]] = dtype
        elif node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to":
                    elem_types[node.output[0]] = attr.i

    int_types = {onnx.TensorProto.INT32, onnx.TensorProto.INT64}
    producer = {out: n.op_type for n in graph.node for out in n.output}

    repairs: List[Tuple[int, Any, int, int]] = []
    for idx, node in enumerate(graph.node):
        if node.op_type == "LayerNormalization" and len(node.input) >= 2:
            in_t = elem_types.get(node.input[0])
            if not in_t:
                continue
            for arg in (1, 2):
                if arg >= len(node.input) or not node.input[arg]:
                    continue
                w_t = init_types.get(node.input[arg]) or elem_types.get(node.input[arg])
                if w_t and w_t != in_t:
                    repairs.append((idx, node, arg, in_t))
                    fixed["layernorm"] += 1
            continue
        if node.op_type != "ScatterND" or len(node.input) < 3:
            continue
        data_t = elem_types.get(node.input[0])
        upd_t = elem_types.get(node.input[2])
        if not data_t or not upd_t or data_t == upd_t:
            continue
        data_is_shape_like = producer.get(node.input[0]) in ("Constant", "ConstantOfShape") or (
            node.input[0] in init_types
        )
        if upd_t in int_types and data_is_shape_like:
            repairs.append((idx, node, 0, upd_t))
        else:
            repairs.append((idx, node, 2, data_t))
        fixed["scatternd"] += 1

    for offset, (idx, node, arg, to_type) in enumerate(repairs):
        cast_out = f"{node.input[arg]}_foldquant_cast_{idx}"
        cast = onnx.helper.make_node(
            "Cast",
            inputs=[node.input[arg]],
            outputs=[cast_out],
            name=f"{node.name or f'{node.op_type}_{idx}'}_arg{arg}_cast",
            to=to_type,
        )
        graph.node.insert(idx + offset, cast)
        node.input[arg] = cast_out
        for out_name in node.output:
            for vi in list(graph.value_info):
                if vi.name == out_name:
                    graph.value_info.remove(vi)

    if any(fixed.values()):
        onnx.save(model, str(onnx_path))
        logger.info("%s: dtype repairs %s", name, fixed)
    return fixed


def strip_default_scatternd_reduction(onnx_path: Path, name: str) -> int:
    """Drop ``reduction="none"`` (the ONNX default) from ScatterND nodes, in place; returns the count.

    TensorRT 10.3's parser (JetPack 6) rejects the attribute's mere presence
    (``importScatterND: Assertion failed: !attrs.count("reduction")``); removing
    the default is a semantic no-op. A non-default reduction is left to fail loudly.
    """
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    stripped = 0
    for node in model.graph.node:
        if node.op_type != "ScatterND":
            continue
        keep = [a for a in node.attribute if not (a.name == "reduction" and a.s == b"none")]
        stripped += len(node.attribute) - len(keep)
        del node.attribute[:]
        node.attribute.extend(keep)
    if stripped:
        onnx.save(model, str(onnx_path))
        logger.info("%s: stripped the default reduction from %d ScatterND node(s)", name, stripped)
    return stripped


def consolidate_external_data(onnx_path: Path, name: str) -> int:
    """Gather a graph's external tensors into one ``<file>.onnx.data`` sidecar; returns the files merged.

    The legacy exporter scatters a graph above 2 GB into one file per tensor
    next to it, which TensorRT's parser does not load reliably. Only the files
    this graph references are merged and removed, so a sibling graph's data in
    the same directory is never touched. A graph already on its own single
    sidecar is left as it is.
    """
    import onnx
    from onnx.external_data_helper import _get_all_tensors, convert_model_to_external_data

    onnx_path = Path(onnx_path)
    sidecar = onnx_path.with_name(onnx_path.name + ".data")
    model = onnx.load(str(onnx_path), load_external_data=False)
    locations = set()
    for tensor in _get_all_tensors(model):  # initializers and Constant attributes
        if tensor.data_location == onnx.TensorProto.EXTERNAL:
            locations.update(e.value for e in tensor.external_data if e.key == "location")
    if not locations or locations == {sidecar.name}:
        return 0
    model = onnx.load(str(onnx_path), load_external_data=True)
    convert_model_to_external_data(model, all_tensors_to_one_file=True, location=sidecar.name, size_threshold=0)
    tmp = onnx_path.with_name(onnx_path.name + ".consolidating")
    tmp.mkdir(exist_ok=True)
    # Written beside the old files first: the sidecar name may be one of them.
    onnx.save(model, str(tmp / onnx_path.name))
    del model
    for location in locations:
        (onnx_path.parent / location).unlink(missing_ok=True)
    os.replace(tmp / sidecar.name, sidecar)
    os.replace(tmp / onnx_path.name, onnx_path)
    tmp.rmdir()
    logger.info("%s: consolidated %d external-data files into %s", name, len(locations), sidecar.name)
    return len(locations)
