# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""Upstream's websocket policy server, serving the policy with FoldQuant engines installed.

This is ``scripts/serve_policy.py policy:checkpoint`` with one extra step:
after the PyTorch policy is assembled, :func:`foldquant_integration.runtime.install_engines`
swaps the PaliGemma prefix and action-expert engines in. The wire protocol
and the observation / action dictionaries are upstream's, so the unmodified
LIBERO client (``examples/libero/main.py``, in its own environment) evaluates
a FoldQuant arm exactly as it evaluates the bf16 policy::

    # terminal 1:  the arm under test
    python -m foldquant_integration.serve --checkpoint-dir <ckpt> --engine-dir exports/pi05_w4a4/engines

    # terminal 2:  upstream's evaluation client, unchanged
    python examples/libero/main.py --args.task-suite-name libero_spatial

Omit ``--engine-dir`` to serve the bf16 PyTorch policy (the reference arm).
"""

from __future__ import annotations

from dataclasses import dataclass
import logging

import tyro

from . import calibration
from ._upstream import LIBERO_TRAIN_CONFIG
from .runtime import install_engines

logger = logging.getLogger("foldquant.pi05.serve")


@dataclass
class ServeConfig:
    checkpoint_dir: str
    """PyTorch checkpoint directory, as ``--policy.dir`` of the upstream server."""

    engine_dir: str | None = None
    """FoldQuant engine directory; omit to serve the bf16 PyTorch policy."""

    quantized_model: str | None = None
    """FoldQuant fake-quant state for ``--checkpoint-dir`` (a state saved without the base files); a
    self-contained fake-quant model given as ``--checkpoint-dir`` is detected by itself. Mutually
    exclusive with ``--engine-dir``."""

    no_fakequant: bool = False
    """When ``--checkpoint-dir`` is a FoldQuant fake-quant model, load it as the plain base policy."""

    config: str = LIBERO_TRAIN_CONFIG
    """Upstream training config name, as ``--policy.config`` of the upstream server."""

    port: int = 8000
    device: str = "cuda"
    warmup: bool = True
    """Run one inference before opening the port (see _warm_up)."""


#: Upstream's random example observation for a config family, by config-name prefix.
_EXAMPLES = {
    "pi05_libero": ("openpi.policies.libero_policy", "make_libero_example"),
    "pi0_libero": ("openpi.policies.libero_policy", "make_libero_example"),
    "pi05_droid": ("openpi.policies.droid_policy", "make_droid_example"),
    "pi0_droid": ("openpi.policies.droid_policy", "make_droid_example"),
    "pi05_aloha": ("openpi.policies.aloha_policy", "make_aloha_example"),
    "pi0_aloha": ("openpi.policies.aloha_policy", "make_aloha_example"),
}


def _warm_up(policy, config: str) -> None:
    """Pay the first-call cost before a client can connect.

    The bf16 arm serves under torch.compile(mode="max-autotune"), which compiles on
    the first infer: about 50 s on a Jetson AGX Orin. Upstream's server runs infer
    synchronously inside its asyncio loop, so for that whole call it answers no
    websocket pings, and openpi-client's connection (20 s ping timeout) drops the
    first request with "keepalive ping timeout". Warming up with upstream's own
    example observation moves the compile ahead of serve_forever().
    """
    import importlib
    import time

    entry = next((v for k, v in _EXAMPLES.items() if config.startswith(k)), None)
    if entry is None:
        logger.warning("no example observation known for config %r: skipping warm-up", config)
        return
    example = getattr(importlib.import_module(entry[0]), entry[1])()
    t0 = time.monotonic()
    policy.infer(example)
    logger.info("warm-up inference done in %.1f s", time.monotonic() - t0)


def main(args: ServeConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    from foldquant.quantized import quantized_arm

    args.quantized_model = quantized_arm(
        args.checkpoint_dir, args.quantized_model, no_fakequant=args.no_fakequant, other_arms=(args.engine_dir,)
    )
    if args.engine_dir and args.quantized_model:
        raise ValueError("--engine-dir and --quantized-model are mutually exclusive")
    # The engines rebind two methods on the instance, which torch.compile would trace past (runtime.py);
    # the fake-quant swaps modules after loading, which a compiled graph would not see. The
    # reference arm keeps upstream's compiled serving configuration.
    policy = calibration.load_policy(
        args.checkpoint_dir,
        config_name=args.config,
        device=args.device,
        compile=not (args.engine_dir or args.quantized_model),
    )
    installed = None
    if args.engine_dir:
        installed = install_engines(policy, args.engine_dir)
        logger.info("serving with FoldQuant engines: %s", ", ".join(sorted(installed.engines)))
    elif args.quantized_model:
        from foldquant.quantized import install_on_policy

        installed, _ = install_on_policy(policy, args.quantized_model, args.checkpoint_dir)
        logger.info("serving the FAKE-QUANT arm from %s (PyTorch arithmetic of the engines; not a latency arm)",
                    args.quantized_model)
    else:
        logger.info("serving the bf16 PyTorch policy")

    if args.warmup:
        _warm_up(policy, args.config)

    from openpi.serving import websocket_policy_server

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host="0.0.0.0", port=args.port, metadata=policy.metadata
    )
    logger.info("config %s, port %d: ready", args.config, args.port)
    try:
        server.serve_forever()
    finally:
        if installed is not None:
            installed.remove()


if __name__ == "__main__":
    main(tyro.cli(ServeConfig))
