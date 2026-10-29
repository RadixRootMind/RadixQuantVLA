# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""Upstream's ZMQ inference server, serving the policy with FoldQuant engines installed.

This is ``scripts/inference_service.py --server`` with one extra step: after
the ``Gr00tPolicy`` is assembled, :func:`foldquant_integration.runtime.install_engines`
swaps the LLM and DiT engines in. The wire protocol, endpoints and
observation / action dictionaries are upstream's, so the unmodified LIBERO
client evaluates a FoldQuant arm exactly as it evaluates the bf16 policy::

    # terminal 1:  the arm under test
    python -m foldquant_integration.serve --model-path <ckpt> --embodiment-tag new_embodiment \\
        --engine-dir exports/n15_w4a4/engines --denoising-steps 4

    # terminal 2:  upstream's evaluation client, unchanged
    python examples/Libero/eval/run_libero_eval.py --task_suite_name libero_spatial --headless

Omit ``--engine-dir`` to serve the bf16 PyTorch policy (the reference arm).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import tyro

from . import calibration
from ._upstream import LIBERO_DATA_CONFIG
from .runtime import install_engines

logger = logging.getLogger("foldquant.groot_n1_5.serve")


@dataclass
class ServeConfig:
    model_path: str
    """Checkpoint directory or Hugging Face id."""

    embodiment_tag: str | None = None
    """Embodiment tag (the released LIBERO checkpoints use ``new_embodiment``)."""

    engine_dir: str | None = None
    """FoldQuant engine directory; omit to serve the bf16 PyTorch policy."""

    quantized_model: str | None = None
    """FoldQuant fake-quant state for ``--model-path`` (a state saved without the base files); a
    self-contained fake-quant model given as ``--model-path`` is detected by itself. Mutually
    exclusive with ``--engine-dir``."""

    no_fakequant: bool = False
    """When ``--model-path`` is a FoldQuant fake-quant model, load it as the plain base policy."""

    data_config: str = LIBERO_DATA_CONFIG
    """``module:Class`` data config, as upstream's inference service takes it."""

    denoising_steps: int | None = None
    """Flow-matching steps (upstream serves the LIBERO checkpoints with 8)."""

    port: int = 5555
    api_token: str | None = None


def main(args: ServeConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    from foldquant.quantized import quantized_arm

    args.quantized_model = quantized_arm(
        args.model_path, args.quantized_model, no_fakequant=args.no_fakequant, other_arms=(args.engine_dir,)
    )
    if args.engine_dir and args.quantized_model:
        raise ValueError("--engine-dir and --quantized-model are mutually exclusive")
    policy = calibration.load_policy(
        args.model_path,
        args.embodiment_tag,
        "cuda",
        data_config=args.data_config,
        denoising_steps=args.denoising_steps,
    )
    installed = None
    if args.engine_dir:
        installed = install_engines(policy, args.engine_dir)
        logger.info("serving with FoldQuant engines: %s", ", ".join(sorted(installed.engines)))
    elif args.quantized_model:
        from foldquant.quantized import install_on_policy

        installed, _ = install_on_policy(policy, args.quantized_model, args.model_path)
        logger.info("serving the FAKE-QUANT arm from %s (PyTorch arithmetic of the engines; not a latency arm)",
                    args.quantized_model)
    else:
        logger.info("serving the bf16 PyTorch policy")
    logger.info(
        "embodiment %s, %d denoising steps, port %d",
        policy.embodiment_tag.value,
        policy.denoising_steps,
        args.port,
    )

    from gr00t.eval.robot import RobotInferenceServer

    try:
        RobotInferenceServer(policy, port=args.port, api_token=args.api_token).run()
    finally:
        if installed is not None:
            installed.remove()


if __name__ == "__main__":
    main(tyro.cli(ServeConfig))
