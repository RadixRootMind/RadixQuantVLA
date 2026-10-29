# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""Rewrite a ModelOpt INT4 weight-only ONNX into ``Int4GroupwiseGemmPlugin`` nodes.

ModelOpt exports each W4A16 weight (algorithm ``modelopt_w4a16_awq``) as a
``trt::DequantizeLinear`` carrying a float fake-quant weight ``Wb`` (``~=
q*scale`` on the INT4 grid) feeding ``Reshape -> Transpose -> MatMul``. We
recover ``q = round(Wb/scale) in [-8, 7]``, AWQ-pack it to ``[N/2, K] int8`` +
``scale [K/block, N] fp16``, and replace the chain with a single plugin node:

    Int4GroupwiseGemmPlugin(activation_bf16, packed_int8_weight, scale_fp16) -> out_bf16

The AWQ ``Mul(act, pre_quant_scale)`` already in the graph stays as the plugin's
activation input (pure W4A16 - no folding).

This module rewrites an ONNX graph that NVIDIA ModelOpt (algorithm
``modelopt_w4a16_awq``) has already exported - the W4A16 AWQ baseline arm - and
does not build a plugin graph from scratch like :mod:`.llm`/:mod:`.dit_int8`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import onnx

from foldquant.onnx_io import save_plugin_onnx
from foldquant.weights import pack_intweights

logger = logging.getLogger(__name__)


def _require_gs() -> Any:
    try:
        import onnx_graphsurgeon as gs
    except ImportError as exc:  # pragma: no cover - dependency gate
        raise ImportError(
            "INT4 groupwise surgery needs onnx_graphsurgeon. Install the 'quant' extra "
            "(`pip install -e .[trt]` from the repository root)."
        ) from exc
    return gs


_PASSTHROUGH_OPS = frozenset({"Reshape", "Transpose", "Cast"})


def _walk_dq_to_matmul(dq_out: Any) -> Tuple[Any, list]:
    """Follow a weight DQ output through Reshape/Transpose/Cast to its MatMul.

    ModelOpt reshapes the blocked ``[N*nblk, block]`` weight back to ``[N, K]`` then
    transposes to ``[K, N]``. Returns ``(matmul_node, [passthrough nodes])`` or
    ``(None, [...])`` if the chain does not terminate in a single MatMul.
    """
    chain: list = []
    cur = dq_out
    for _ in range(6):  # bounded: Reshape -> Transpose (-> Cast) is 2-3 hops
        consumers = cur.outputs
        if len(consumers) != 1:
            return None, chain
        node = consumers[0]
        if node.op == "MatMul":
            return node, chain
        if node.op not in _PASSTHROUGH_OPS or len(node.outputs) != 1:
            return None, chain
        chain.append(node)
        cur = node.outputs[0]
    return None, chain


def _reshape_target_nk(chain: list) -> Tuple[int, int] | None:
    """The ``[N, K]`` weight shape from the chain's Reshape target constant."""
    for node in chain:
        if node.op == "Reshape" and len(node.inputs) >= 2:
            tgt = node.inputs[1]
            vals = getattr(tgt, "values", None)
            if vals is None:
                continue
            dims = [int(x) for x in np.asarray(vals).ravel().tolist()]
            if len(dims) == 2 and dims[0] > 0 and dims[1] > 0:
                return dims[0], dims[1]
    return None


def _materialize_dq_constant_inputs(graph: Any, gs: Any) -> int:
    """Turn ``trt::DequantizeLinear`` inputs produced by ``Constant`` nodes into ``gs.Constant``.

    Walks through ``Identity`` chains. Touches nothing else; in particular it
    never evaluates or re-serializes unrelated (e.g. bf16) constants the way a
    whole-graph ``fold_constants()`` would.

    Returns:
        Number of inputs materialized.
    """
    count = 0
    for node in graph.nodes:
        if node.op != "DequantizeLinear" or node.domain != "trt":
            continue
        for i, tensor in enumerate(list(node.inputs)):
            if isinstance(tensor, gs.Constant):
                continue
            producer = tensor.inputs[0] if getattr(tensor, "inputs", None) else None
            # Follow Identity chains back to their source.
            while producer is not None and producer.op == "Identity" and producer.inputs:
                src = producer.inputs[0]
                if isinstance(src, gs.Constant):
                    node.inputs[i] = src
                    producer = None
                    count += 1
                    break
                producer = src.inputs[0] if getattr(src, "inputs", None) else None
            if producer is not None and producer.op == "Constant":
                value = producer.attrs.get("value")
                if value is not None:
                    node.inputs[i] = value if isinstance(value, gs.Constant) else gs.Constant(tensor.name, values=value)
                    count += 1
    if count:
        logger.info("ModelOpt INT4: materialized %d Constant-node DQ input(s).", count)
    return count


def rewrite_int4_modelopt_dq(graph: Any, *, plugin_namespace: str = "") -> Tuple[Any, int, int]:
    """Rewrite ModelOpt ``modelopt_w4a16_awq`` weight-only ``trt::DequantizeLinear`` into plugin nodes.

    Recovers ``q = round(Wb/scale) in [-8, 7]`` from each weight DQ, AWQ-packs to
    ``[N/2, K] int8`` + ``scale [K/block, N] fp16``, and replaces the whole
    DQ->Reshape->Transpose->MatMul chain with an ``Int4GroupwiseGemmPlugin``. A weight
    failing the plugin's divisibility (``N%128``/``K%64``) is left to the leftover
    bypass below (its constant is the fake-dequant ``W_fq``, so the DQ is a no-op).
    Returns ``(graph, replaced, materialized)``.
    """
    gs = _require_gs()
    # fold_constants is LOAD-BEARING for the pattern matcher: the DQ inputs and
    # the Reshape target shapes arrive as Constant-NODE outputs, and
    # _reshape_target_nk/_walk_dq_to_matmul only see them once folded to
    # gs.Constant (removing the fold sent EVERY DQ, GR00T's 196 included, to
    # the no_reshape_target bypass). But an *unrestricted* fold corrupts bf16
    # constants: numpy has no bf16, so graphsurgeon's forced value access
    # re-wrote Pi0.5's bf16 [1, 512] time-embedding table into an initializer
    # the TensorRT parser rejects ("Failed to import initializer"). Exclude
    # bf16 Constant nodes from folding; they pass through verbatim (TensorRT
    # parses bf16 Constant nodes fine; the float pi05_flex graph is full of
    # them) while the int64 shape chains and fp16/fp32 scales still fold.
    _materialize_dq_constant_inputs(graph, gs)

    def _is_unfoldable_dtype(dtype: Any) -> bool:
        """bf16 and f64 Constants must NOT be folded into initializers.

        - bf16: numpy cannot represent it; graphsurgeon's forced value access
          re-serializes it corrupted.
        - f64 (DOUBLE): the TensorRT parser imports a DOUBLE Constant NODE
          (demoting to f32 with a warning) but its weights importer REFUSES a
          DOUBLE initializer ("Failed to import initializer", measured on
          Pi0.5's f64 [1, 512] time-embedding table; the un-surgered flex
          graph, where the table stays a node, parses fine).

        A lazily-loaded gs.Constant reports the raw ONNX enum; a loaded one
        reports a numpy/ml_dtypes dtype.
        """
        if dtype is None:
            return False
        if isinstance(dtype, int):
            return dtype in (onnx.TensorProto.BFLOAT16, onnx.TensorProto.DOUBLE)
        text = str(dtype)
        return "bfloat16" in text or "float64" in text

    def _exclude_unfoldable_constants(node: Any) -> bool:
        if node.op != "Constant":
            return False
        value = node.attrs.get("value")
        return _is_unfoldable_dtype(getattr(value, "dtype", None))

    graph.fold_constants(should_exclude_node=_exclude_unfoldable_constants)
    dqs = [n for n in graph.nodes if n.op == "DequantizeLinear" and n.domain == "trt"]
    logger.info("ModelOpt INT4: found %d trt::DequantizeLinear weight nodes", len(dqs))
    replaced = materialized = 0
    skips: Dict[str, int] = {}

    def _skip(reason: str) -> None:
        skips[reason] = skips.get(reason, 0) + 1

    for dq in dqs:
        if len(dq.inputs) != 2:
            _skip("dq_arity")
            continue
        if not isinstance(dq.inputs[0], gs.Constant) or not isinstance(dq.inputs[1], gs.Constant):
            _skip("non_const")
            continue
        wb = np.asarray(dq.inputs[0].values).astype(np.float32)  # [N*nblk, block]
        sb = np.asarray(dq.inputs[1].values).astype(np.float32)  # [N*nblk, 1]
        if wb.ndim != 2:
            _skip("weight_not_2d")
            continue
        bs = int(dq.attrs.get("block_size", 0))
        if bs <= 0 or wb.shape[1] != bs:
            _skip("block_size")
            continue
        if len(dq.outputs) != 1:
            _skip("dq_branching")
            continue
        mm, chain = _walk_dq_to_matmul(dq.outputs[0])
        if mm is None:
            _skip("no_matmul")
            continue
        nk = _reshape_target_nk(chain)
        if nk is None:
            _skip("no_reshape_target")
            continue
        gemm_n, gemm_k = nk
        nblk = gemm_k // bs
        if wb.shape[0] != gemm_n * nblk or gemm_k % bs:
            _skip("shape_mismatch")
            continue

        sb = sb.reshape(wb.shape[0], 1)
        # Fail loud on degenerate scales/values instead of packing garbage
        # nibbles: awq_clip initialises best_clip_val to zeros and only
        # overwrites blocks the search improved, so a never-updated block
        # arrives here with scale 0 -> 0/0 = NaN -> int cast garbage. awq_lite
        # cannot produce this (its amax is always a real block max), which is
        # why this path never fired before the clip arms existed.
        if not np.all(np.isfinite(sb)) or np.any(sb == 0):
            raise RuntimeError(
                f"int4_groupwise surgery: weight {dq.name!r} carries {int(np.sum(sb == 0))} zero and "
                f"{int(np.sum(~np.isfinite(sb)))} non-finite block scale(s); the quantization search "
                "produced a degenerate block (awq_clip's zero-initialised best_clip_val is the known "
                "producer). Refusing to pack garbage nibbles."
            )
        q = np.rint(wb / sb).astype(np.int32)
        # Clamp to the int4 range. This is PART of the quantization being
        # exported, not a repair: ModelOpt's fake-quant computes
        # clamp(round(w/s), -8, 7), and awq_clip *deliberately* chooses scales
        # below the raw weight amax (best_clip_val in [0.5, 1.0]*amax), so the
        # clipped-away outliers land outside [-8, 7] here by design. Packing
        # them un-clamped wrapped the nibble encoding (q+8 went negative) and
        # produced garbage weights, measured as the 0.68-0.78 cosine of every
        # awq_full arm before this fix.
        n_clamped = int(np.sum((q < -8) | (q > 7)))
        if n_clamped:
            logger.info(
                "int4_groupwise: weight %r clamps %d/%d values (%.4f%%) into [-8, 7] "
                "(expected under awq_clip/awq_full: the clip search shrinks scales below the raw amax).",
                dq.name,
                n_clamped,
                q.size,
                100.0 * n_clamped / q.size,
            )
        q = np.clip(q, -8, 7)
        # The weight feeds MatMul as input[1] (act is the other input - the Mul output).
        term = chain[-1].outputs[0] if chain else dq.outputs[0]
        act_idx = 0 if mm.inputs[1] is term else 1
        act = mm.inputs[act_idx]

        if gemm_n % 128 or gemm_k % 64:
            _skip(f"divisibility_n{gemm_n}_k{gemm_k}")
            continue

        q_nk = q.reshape(gemm_n, nblk, bs).reshape(gemm_n, gemm_k)  # [N, K]
        unpacked = q_nk.astype(np.int16) + 8  # unsigned nibble [N, K]
        packed16 = pack_intweights(unpacked)  # [N/4, K] int16
        packed8 = packed16.view(np.int8).reshape(packed16.shape[0] * 2, packed16.shape[1])  # [N/2, K]
        scale_fp16 = sb.reshape(gemm_n, nblk).T.astype(np.float16)  # [K/block, N]

        layer = mm.name or dq.name
        # The W4A16 plugin (.so) emits BF16; pin the output dtype so TRT's STRONGLY_TYPED
        # parser finds a supported format (the MatMul output may have inherited FP32).
        import ml_dtypes

        out_var = mm.outputs[0]
        out_var.dtype = ml_dtypes.bfloat16
        node = gs.Node(
            op="Int4GroupwiseGemmPlugin",
            name=f"{layer}/Int4Plugin",
            inputs=[
                act,
                gs.Constant(f"{layer}/plugin_qweight_int8", packed8),
                gs.Constant(f"{layer}/plugin_scale_fp16", scale_fp16),
            ],
            outputs=[out_var],
            attrs={
                "gemm_n": int(gemm_n),
                "gemm_k": int(gemm_k),
                "group_size": int(bs),
                "plugin_version": "1",
                "plugin_namespace": plugin_namespace,
            },
        )
        mm.outputs.clear()  # detach so cleanup() prunes the dead Reshape/Transpose/MatMul
        dq.outputs.clear()  # detach the converted DQ so the leftover-bypass below skips it
        graph.nodes.append(node)
        replaced += 1

    # Bypass every unconverted trt::DequantizeLinear: ModelOpt's weight constant is the
    # fake-dequant W_fq (W_q*scale already), so the DQ is a no-op - rewire consumers to
    # the constant and drop it. TRT rejects a surviving BF16 block-scale DQ, so none may remain.
    for dq in [n for n in graph.nodes if n.op == "DequantizeLinear" and n.domain == "trt" and n.outputs]:
        dq_out = dq.outputs[0]
        wconst = dq.inputs[0]
        for consumer in list(dq_out.outputs):
            for i, inp in enumerate(consumer.inputs):
                if inp is dq_out:
                    consumer.inputs[i] = wconst
        dq.outputs.clear()
        materialized += 1

    if skips:
        logger.info("ModelOpt INT4: skips: %s", dict(sorted(skips.items())))
    logger.info("ModelOpt INT4: %d -> plugin, %d trt::DQ bypassed to W_fq", replaced, materialized)
    graph.cleanup().toposort()
    return graph, replaced, materialized


# Ops TRT STRONGLY_TYPED parsing requires to have uniform float input types. ``Where``
# is handled specially (its input[0] is the boolean condition, not a float operand).
_STRONGLY_TYPED_STRICT_OPS = frozenset(
    {"Gemm", "MatMul", "Mul", "Add", "Sub", "Div", "Pow", "Min", "Max", "Concat", "Where"}
)
# Ops whose output is boolean (a comparison/logical) - never type-propagated as float.
_BOOLEAN_OUTPUT_OPS = frozenset(
    {"Greater", "Less", "Equal", "GreaterOrEqual", "LessOrEqual", "And", "Or", "Not", "Xor", "IsNaN", "IsInf"}
)


def _build_dtype_map(model: Any) -> Dict[str, int]:
    """Map ``tensor_name -> onnx dtype`` from static + propagated sources.

    Covers initializers, value_info, Constant/Cast/DequantizeLinear nodes, then
    propagates through type-preserving ops in topo order so intermediate attention
    tensors get a type without full shape inference.
    """
    import onnx

    dtype_map: Dict[str, int] = {}
    for init in model.graph.initializer:
        dtype_map[init.name] = init.data_type
    for vi in list(model.graph.input) + list(model.graph.value_info):
        if vi.HasField("type") and vi.type.HasField("tensor_type"):
            dtype_map[vi.name] = vi.type.tensor_type.elem_type
    for node in model.graph.node:
        if node.op_type == "Constant" and node.output:
            for attr in node.attribute:
                if attr.name == "value":
                    dtype_map[node.output[0]] = attr.t.data_type
                elif attr.name in ("value_float", "value_floats"):
                    dtype_map[node.output[0]] = onnx.TensorProto.FLOAT
                elif attr.name in ("value_int", "value_ints"):
                    dtype_map[node.output[0]] = onnx.TensorProto.INT64
        elif node.op_type == "Cast" and node.output:
            for attr in node.attribute:
                if attr.name == "to":
                    dtype_map[node.output[0]] = attr.i
        elif node.op_type == "DequantizeLinear" and len(node.input) > 1 and node.output:
            sdt = dtype_map.get(node.input[1])  # DQ output = scale dtype
            if sdt is not None and sdt in (onnx.TensorProto.FLOAT16, onnx.TensorProto.BFLOAT16):
                dtype_map[node.output[0]] = sdt
        elif node.op_type == "Int4GroupwiseGemmPlugin" and node.output:
            # The W4A16 plugin computes in (and outputs) the BF16 activation baseline,
            # NOT its FP16 weight-scale. Seed BF16 so the activation path stays typed.
            dtype_map[node.output[0]] = onnx.TensorProto.BFLOAT16
    # Fixpoint propagation: an untyped op output inherits the majority float type of
    # its typed inputs, so float islands (RoPE/timestep casts) are pinpointed exactly
    # where they meet the BF16 activation path.
    from collections import Counter

    float_types = {onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16, onnx.TensorProto.BFLOAT16}
    for _ in range(40):  # bounded fixpoint; converges in a few passes on these graphs
        changed = False
        for node in model.graph.node:
            if node.op_type in _BOOLEAN_OUTPUT_OPS or not node.output or node.output[0] in dtype_map:
                continue  # never type a comparison/logical output as float
            in_floats = [dtype_map[i] for i in node.input if i and dtype_map.get(i) in float_types]
            if not in_floats:
                continue
            dtype_map[node.output[0]] = Counter(in_floats).most_common(1)[0][0]
            changed = True
        if not changed:
            break
    return dtype_map


def _fix_strongly_typed_mismatches(model: Any) -> int:
    """Insert Cast nodes so STRONGLY_TYPED-strict ops see uniform float input types.

    ModelOpt's AWQ export leaves a few FP32 attention tensors (e.g. a RoPE Cast)
    feeding a BF16 MatMul - TRT 10.x STRONGLY_TYPED rejects the mix. For each such op,
    cast the minority-typed float inputs to the majority float type. Returns the
    number of Cast nodes inserted.

    The attention softmax path is the one exception to the majority rule: an ``Add``
    feeding a ``Softmax`` and a ``MatMul`` consuming one always resolve to FP32,
    reproducing the bf16 export (which casts scores up before the ``-inf`` mask add and
    keeps Softmax and probs-V in FP32, Casts the ModelOpt export drops). A BF16 softmax
    over ``-inf``-masked rows returns NaN on some architectures.
    """
    from collections import Counter

    import onnx

    float_types = {onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16, onnx.TensorProto.BFLOAT16}
    fp32 = onnx.TensorProto.FLOAT
    dtype_map = _build_dtype_map(model)

    producers = {out: n for n in model.graph.node for out in n.output if out}
    consumer_ops: dict = {}
    for n in model.graph.node:
        for i in n.input:
            if i:
                consumer_ops.setdefault(i, set()).add(n.op_type)

    def _on_softmax_path(n: Any) -> bool:
        """True for the mask add feeding a Softmax and the MatMul consuming one."""
        if n.op_type == "Add":
            return any("Softmax" in consumer_ops.get(o, ()) for o in n.output if o)
        if n.op_type == "MatMul":
            return any(producers[i].op_type == "Softmax" for i in n.input if i and producers.get(i) is not None)
        return False

    new_nodes = []
    n_fixed = 0
    for node in model.graph.node:
        if node.op_type not in _STRONGLY_TYPED_STRICT_OPS:
            continue
        skip0 = node.op_type == "Where"  # Where input[0] is the boolean condition
        float_inputs = [
            (i, inp, dtype_map.get(inp))
            for i, inp in enumerate(node.input)
            if inp and dtype_map.get(inp) in float_types and not (skip0 and i == 0)
        ]
        if len({dt for _, _, dt in float_inputs}) <= 1:
            continue
        target: Any
        if _on_softmax_path(node):
            target = fp32  # mask add / Softmax / probs-V stay FP32, see docstring
        else:
            target = Counter(dt for _, _, dt in float_inputs).most_common(1)[0][0]
        assert target is not None  # float_inputs filtered to known float dtypes
        for idx, inp, dt in float_inputs:
            if dt == target:
                continue
            cast_out = f"{inp}__st_cast_{n_fixed}"  # n_fixed is monotonic -> globally unique
            new_nodes.append(onnx.helper.make_node("Cast", inputs=[inp], outputs=[cast_out], to=target))
            node.input[idx] = cast_out
            dtype_map[cast_out] = target
            n_fixed += 1
    # The Int4GroupwiseGemm plugin's activation (input[0]) must be BF16 (supportsFormat-
    # Combination case 0); ModelOpt feeds it through an FP32 Cast for the AWQ Mul. Pin it.
    bf16 = onnx.TensorProto.BFLOAT16
    for node in model.graph.node:
        if node.op_type != "Int4GroupwiseGemmPlugin" or not node.input:
            continue
        act = node.input[0]
        if dtype_map.get(act) in float_types and dtype_map.get(act) != bf16:
            cast_out = f"{act}__st_cast_{n_fixed}"
            new_nodes.append(onnx.helper.make_node("Cast", inputs=[act], outputs=[cast_out], to=bf16))
            node.input[0] = cast_out
            dtype_map[cast_out] = bf16
            n_fixed += 1
    if n_fixed:
        model.graph.node.extend(new_nodes)
        logger.info("ModelOpt INT4: inserted %d Cast node(s) for STRONGLY_TYPED type consistency", n_fixed)
    return n_fixed


def _restore_double_initializers_as_constants(model: Any) -> int:
    """Rewrite every f64 initializer as a ``Constant`` node (parser-importable form).

    Returns:
        Number of initializers restored.
    """
    doubles = [init for init in model.graph.initializer if init.data_type == onnx.TensorProto.DOUBLE]
    if not doubles:
        return 0
    kept = [init for init in model.graph.initializer if init.data_type != onnx.TensorProto.DOUBLE]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    new_nodes = []
    for init in doubles:
        node = onnx.helper.make_node(
            "Constant",
            inputs=[],
            outputs=[init.name],
            name=f"{init.name}_const",
            value=init,
        )
        new_nodes.append(node)
    # Constant nodes have no inputs, so prepending keeps topological order.
    existing = list(model.graph.node)
    del model.graph.node[:]
    model.graph.node.extend(new_nodes + existing)
    logger.info("ModelOpt INT4: restored %d DOUBLE initializer(s) as Constant nodes.", len(doubles))
    return len(doubles)


def apply_int4_modelopt_surgery(
    in_path: str | Path, out_path: str | Path, *, plugin_namespace: str = ""
) -> Tuple[int, int]:
    """Load a ModelOpt INT4 weight-only ONNX, rewrite to plugin nodes, save.

    Returns ``(replaced, materialized)``. Raises if the input carries no
    ``trt::DequantizeLinear`` weights (not a ModelOpt INT4 export).
    """
    import onnx

    gs = _require_gs()
    in_path, out_path = Path(in_path), Path(out_path)
    graph = gs.import_onnx(onnx.load(str(in_path), load_external_data=True))
    if not any(n.op == "DequantizeLinear" and n.domain == "trt" for n in graph.nodes):
        raise ValueError(
            f"{in_path} has no trt::DequantizeLinear weights - INT4 surgery needs a ModelOpt "
            "INT4 weight-only (W4A16, algorithm 'modelopt_w4a16_awq') ONNX export as input."
        )
    graph, replaced, materialized = rewrite_int4_modelopt_dq(graph, plugin_namespace=plugin_namespace)

    out_model = gs.export_onnx(graph)
    if not any(o.domain == "trt" for o in out_model.opset_import):
        op = out_model.opset_import.add()
        op.domain = "trt"
        op.version = 1
    # ModelOpt's AWQ export leaves FP32 attention tensors (RoPE Cast) feeding BF16
    # MatMuls; TRT STRONGLY_TYPED rejects the mix - reconcile with explicit Casts.
    _fix_strongly_typed_mismatches(out_model)
    # graphsurgeon's import/export round-trip turns numpy-convertible Constant
    # NODES into graph INITIALIZERS. For DOUBLE that changes parseability: the
    # TensorRT ONNX parser demotes an f64 `Constant` node to f32 with a
    # warning, but its weights importer REFUSES an f64 initializer ("Failed to
    # import initializer", Pi0.5's f64 [1, 512] time-embedding table). Restore
    # every DOUBLE initializer to the Constant-node form the parser accepts
    # (the un-surgered graphs ship exactly that form and parse fine).
    _restore_double_initializers_as_constants(out_model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_plugin_onnx(out_model, out_path, size_threshold=1024, convert_attribute=False)
    return replaced, materialized


def apply_int4_surgery_in_place(onnx_path: Path, name: str) -> Dict[str, int]:
    """Rewrite the INT4 weight-only DQ chains to ``Int4GroupwiseGemmPlugin`` nodes, in place.

    TensorRT 10.3 has no INT4 weight-only kernel, so the graph ModelOpt exports
    parses but runs dequantized. The surgery is what makes a ModelOpt W4A16 AWQ arm an
    INT4 engine; the plugin library it needs is declared in the export manifest.
    """
    # Staged under the SAME basename: the sidecar's name is baked into every
    # initializer's ``external_data`` location at save time, so writing
    # "<name>.int4.onnx" and renaming afterwards leaves the graph pointing at a
    # sidecar that no longer exists, and TensorRT fails the whole parse with
    # "Failed to import initializer".
    stage_dir = onnx_path.parent / f".int4_{onnx_path.stem}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    staged = stage_dir / onnx_path.name
    replaced, materialized = apply_int4_modelopt_surgery(onnx_path, staged)
    if replaced == 0:
        raise RuntimeError(
            f"{name}: INT4 surgery rewrote no weight; the engine would run dequantized"
        )
    for old in (onnx_path, Path(str(onnx_path) + ".data")):
        if old.exists():
            old.unlink()
    staged.replace(onnx_path)
    staged_data = Path(str(staged) + ".data")
    if staged_data.exists():
        staged_data.replace(Path(str(onnx_path) + ".data"))
    stage_dir.rmdir()
    logger.info(
        "%s: INT4 groupwise surgery replaced %d weights (%d constants materialized)",
        name,
        replaced,
        materialized,
    )
    return {"replaced": replaced, "materialized": materialized}
