# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""FoldQuant on GR00T N1.5: the integration layer over the upstream release.

Everything model-specific lives here; the algorithm, the emitters, the
TensorRT plugins and the engine builder / runtime wrapper are the top-level
``foldquant`` package. The upstream ``gr00t`` package is used unchanged:

* :mod:`.calibration`:  load the upstream policy / dataset and sample
  calibration observations through the upstream data path.
* :mod:`.quantize`:  calibrate the FoldQuant folds and save the quantized model.
* :mod:`.export`:  emit from it the FoldQuant ``llm_bf16.onnx`` /
  ``dit_bf16.onnx`` plugin graphs.
* :mod:`.build_engines`:  compile them (plugins loaded first) into an engine
  directory.
* :mod:`.runtime`:  install the engines into a live ``Gr00tPolicy`` in place
  of the PyTorch LLM and DiT.
* :mod:`.verify`:  held-out cosine / max-abs of the engine directory against
  the bf16 PyTorch policy.
* :mod:`.serve`:  upstream's ZMQ inference server over the engines, for
  upstream's unmodified ``examples/Libero/eval/run_libero_eval.py`` client.
* :mod:`.eval_libero`:  the same LIBERO rollout in one process, every task of
  every suite, with a resumable per-task summary.
* :mod:`.benchmark`:  component latency, PyTorch beside FoldQuant.
"""
