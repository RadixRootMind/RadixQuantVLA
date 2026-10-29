# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""FoldQuant: native low-bit quantization of vision-language-action models via consistent folding.

The artifact of *FoldQuantVLA: Native Low-Bit Quantization of Vision-Language-Action
Models via Consistent Folding*.

The algorithm (:mod:`foldquant.foldq`) folds a SmoothQuant scale, a per-block
rotation and the weight rounding into each linear site's weights offline; the
TensorRT plugins under :mod:`foldquant.kernels` then run the fused
quantize -> INT GEMM -> dequantize path per row at inference. Module emitters
(:mod:`foldquant.dit_int4`, :mod:`foldquant.llm`, ...) write the plugin-node
ONNX graph directly from a live PyTorch module; :mod:`foldquant.export` is the
one entry point a host model calls.

Typical use from a model's own repository::

    from foldquant import export, runtime, schemes

    result = export.export_llm(llm, "llm.onnx", scheme=schemes.W4A4_SRG, forward_loop=replay)
    runtime.build_engine("llm.onnx", "llm.engine", profiles=..., plugin_libs=result.plugin_libs)
    engine = runtime.TensorRTEngine("llm.engine")
"""

from . import schemes
from .export import ExportResult, export_module, install_llm_emulation

__version__ = "0.1.0"

__all__ = ["ExportResult", "__version__", "export_module", "install_llm_emulation", "schemes"]
