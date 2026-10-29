# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""FoldQuant on Pi0.5: the integration layer over the upstream openpi release.

Everything model-specific lives here; the algorithm, the emitters, the
TensorRT plugins and the engine builder / runtime wrapper are the top-level
``foldquant`` package. The upstream ``openpi`` package is used unchanged:

* :mod:`.calibration`:  load the upstream policy / dataset and sample
  calibration observations in the shape upstream's LIBERO client sends.
* :mod:`.quantize`:  calibrate the FoldQuant folds and save the quantized model.
* :mod:`.export`:  emit from it the FoldQuant ``llm_bf16.onnx`` /
  ``expert_bf16.onnx`` plugin graphs.
* :mod:`.build_engines`:  compile them (plugins loaded first) into an engine
  directory.
* :mod:`.runtime`:  install the engines into a live ``PI0Pytorch`` in place
  of the PaliGemma prefix pass and the action-expert denoise step.
* :mod:`.verify`:  held-out cosine / max-abs of the engine directory against
  the bf16 PyTorch policy.
* :mod:`.serve`:  upstream's websocket policy server over the engines, for
  upstream's unmodified ``examples/libero/main.py`` client.
* :mod:`.eval_libero`:  the same LIBERO rollout driven from one script, with
  a per-suite summary.
* :mod:`.benchmark`:  component latency, PyTorch beside FoldQuant.
"""
