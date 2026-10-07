#!/usr/bin/env python
"""OpenPI websocket policy service compatible with this repo's LIBERO runner.

The existing multi-GPU launcher starts an inference script with GR00T-flavored
arguments. This wrapper accepts those arguments, loads an OpenPI policy, and
serves it through OpenPI's websocket server so the LIBERO client can query
pi0/pi0.5 policies.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import socket

import tyro


@dataclasses.dataclass
class ArgsConfig:
    model_path: str = "gs://openpi-assets/checkpoints/pi05_libero"
    """OpenPI checkpoint dir, e.g. gs://openpi-assets/checkpoints/pi05_libero."""

    data_config: str = "pi05_libero"
    """OpenPI training config name."""

    port: int = 8000
    """Port for the websocket policy server."""

    host: str = "0.0.0.0"
    """Host interface for the websocket policy server."""

    device: str | None = None
    """PyTorch device for converted OpenPI checkpoints. Defaults to OpenPI policy_config."""

    default_prompt: str | None = None
    """Fallback prompt used by OpenPI if an observation lacks prompt."""

    server: bool = False
    """Accepted for compatibility with scripts/inference_service.py."""

    denoising_steps: int = 0
    """Sampling steps passed to OpenPI; zero uses the model default."""

    embodiment_tag: str = ""
    """Accepted for compatibility; OpenPI LIBERO config defines embodiment mapping."""

    record: bool = False
    """Record OpenPI policy inputs/outputs to policy_records/."""


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "0") not in ("0", "false", "False")


def _apply_quantization_if_requested(policy) -> None:
    want_gptq = _env_enabled("GR00T_GPTQ")
    want_duquant = any(
        k.startswith("GR00T_DUQUANT_") and k != "GR00T_DUQUANT_PACKDIR"
        for k in os.environ
    )
    want_rtn = any(k.startswith("GR00T_RTN") for k in os.environ)
    if not (want_gptq or want_duquant or want_rtn):
        return

    if not getattr(policy, "_is_pytorch_model", False):
        raise RuntimeError(
            "GR00T_GPTQ / GR00T_DUQUANT_* / GR00T_RTN* requires an OpenPI PyTorch "
            "checkpoint (a checkpoint directory containing model.safetensors). Convert "
            "the JAX pi05 checkpoint first, or run METHOD=fp16."
        )

    model = getattr(policy, "_model")

    if want_gptq:
        from gr00t.quantization import enable_gptq_if_configured

        applied = enable_gptq_if_configured(model)
        if not applied:
            raise RuntimeError("GR00T_GPTQ=1 was set, but no GPTQ layers were wrapped")

    if want_duquant:
        from gr00t.quantization import enable_duquant_if_configured

        enable_duquant_if_configured(model)
        # DuQuant's per-layer buffers (_perm_cache, _R_in_stack, ...) are
        # constructed from numpy on CPU. The openpi loader already placed the
        # rest of the model on GPU, so we re-sync devices after the wrap.
        try:
            device = next(model.parameters()).device
            model.to(device)
        except StopIteration:
            pass

    if want_rtn:
        from gr00t.quantization import enable_rtn_if_configured

        enable_rtn_if_configured(model)
        try:
            device = next(model.parameters()).device
            model.to(device)
        except StopIteration:
            pass


def _apply_pi05_atm_if_requested(policy) -> None:
    """Apply pi0.5 ATM α + OHB β when GR00T_ATM_SCOPE=pi05 and JSON path is set."""
    if not getattr(policy, "_is_pytorch_model", False):
        return
    if os.environ.get("GR00T_ATM_SCOPE", "") != "pi05":
        return
    want_atm = _env_enabled("GR00T_ATM_ENABLE")
    want_ohb = _env_enabled("GR00T_OHB_ENABLE")
    if not (want_atm or want_ohb):
        return
    model = getattr(policy, "_model")
    from gr00t.atm import enable_pi05_atm_if_configured

    enable_pi05_atm_if_configured(model)
    try:
        device = next(model.parameters()).device
        model.to(device)
    except StopIteration:
        pass


def main(args: ArgsConfig) -> None:
    from openpi.policies import policy as _policy
    from openpi.policies import policy_config as _policy_config
    from openpi.serving import websocket_policy_server
    from openpi.training import config as _config

    if args.denoising_steps < 0:
        raise ValueError("denoising_steps must be non-negative")

    logging.info("Loading OpenPI config=%s checkpoint=%s", args.data_config, args.model_path)
    train_config = _config.get_config(args.data_config)
    sample_kwargs = {"num_steps": args.denoising_steps} if args.denoising_steps else None
    logging.info("OpenPI sampling kwargs=%s (None uses model defaults)", sample_kwargs)
    policy = _policy_config.create_trained_policy(
        train_config,
        args.model_path,
        default_prompt=args.default_prompt,
        pytorch_device=args.device,
        sample_kwargs=sample_kwargs,
    )

    _apply_quantization_if_requested(policy)
    _apply_pi05_atm_if_requested(policy)

    metadata = policy.metadata
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating OpenPI websocket server (host=%s ip=%s port=%d)", hostname, local_ip, args.port)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(ArgsConfig))
