# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""Held-out drift of a FoldQuant engine directory against the bf16 PyTorch policy.

Two seams are scored on ``--num-samples`` observations drawn from episodes
the calibration never saw (read off the export manifest): the stacked
prefix KV cache the PaliGemma engine hands the expert, and the action chunk
``Policy.infer`` returns (after the output transforms, i.e. what a LIBERO
client receives). Both passes integrate from the same flow-matching noise
(``noise=`` is passed explicitly, seeded per observation), so the difference
measures the engines, not the sampler.

No float-engine floor is measured: upstream openpi ships no TensorRT path.
Compare arms against each other on a common held-out set with
``--split-from``.

Example::

    python -m foldquant_integration.verify --checkpoint-dir <ckpt> --dataset-path <LeRobot LIBERO> \\
        --engine-dir exports/pi05_w4a4/engines --output exports/pi05_w4a4/verify.json
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import time
from typing import Any

from foldquant.drift import worst_channel
from foldquant.provenance import public_path
import numpy as np
import torch
import tyro

from . import calibration
from ._upstream import LIBERO_TRAIN_CONFIG
from ._upstream import MANIFEST_NAME
from .runtime import PrefixCapture
from .runtime import install_engines

logger = logging.getLogger("foldquant.pi05.verify")


@dataclass
class VerifyConfig:
    checkpoint_dir: str
    dataset_path: str
    engine_dir: str | None = None
    """Engine directory produced by :mod:`.build_engines`."""

    quantized_model: str | None = None
    """Score a fake-quant state in PyTorch instead of engines; its calibration split is the state's.
    A self-contained fake-quant model given as ``--checkpoint-dir`` is detected by itself (it is then
    scored against its own base weights). Exactly one of ``--engine-dir`` / ``--quantized-model``."""

    no_fakequant: bool = False
    """When ``--checkpoint-dir`` is a FoldQuant fake-quant model, load it as the plain base policy."""

    output: str | None = None
    """Write the report here as JSON (default: ``<engine_dir>/verify.json``)."""

    config: str = LIBERO_TRAIN_CONFIG
    """Upstream training config name, as given to the export."""

    num_samples: int = 16
    """Held-out observations, from episodes outside the calibration split."""

    seed: int = 42
    """Seed of the held-out sample and of observation ``i``'s flow-matching noise (``seed + i``), both passes."""

    allow_calibration_episodes: bool = False
    """Sample from every episode, calibration ones included, for datasets too small to hold any out.
    The report then measures fit, not generalisation, and says so."""

    split_from: str | None = None
    """Engine directory whose FoldQuant manifest defines the calibration split (default: ``engine_dir``).
    Point every arm of a comparison at the same directory and they all score the same held-out
    observations."""

    device: str = "cuda"


# float64: a flat fp32 cosine over 18 layers x 968 positions x 256 of bf16
# values accumulates enough rounding to read above 1 for identical tensors.
def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a.double().flatten(), b.double().flatten(), dim=0))


def _position_cos_min(a: torch.Tensor, b: torch.Tensor) -> float:
    """Minimum over prefix positions of the cosine across (layer, K/V, head, dim)."""
    a2 = a.double().movedim(4, 0).reshape(a.shape[4], -1)
    b2 = b.double().movedim(4, 0).reshape(b.shape[4], -1)
    return float(torch.nn.functional.cosine_similarity(a2, b2, dim=1).min())


def _load_manifest(engine_dir: Path) -> dict[str, Any]:
    path = engine_dir / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found: not a FoldQuant engine directory")
    return json.loads(path.read_text())


def run_pass(policy, observations: list[dict[str, Any]], seed: int) -> dict[str, list[torch.Tensor]]:
    acts: list[torch.Tensor] = []
    width = 0
    with PrefixCapture(policy) as capture:
        for i, obs in enumerate(observations):
            chunk = torch.as_tensor(calibration.infer(policy, obs, seed=seed + i)).float()
            width = int(chunk.shape[-1])
            acts.append(chunk.flatten())
    if len(capture.stacks) != len(observations):
        raise RuntimeError(f"captured {len(capture.stacks)} prefix passes for {len(observations)} observations")
    return {"kv_stack": [s.cpu() for s in capture.stacks], "actions": acts, "action_width": width}


def main(args: VerifyConfig) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    from foldquant.quantized import quantized_arm

    args.quantized_model = quantized_arm(
        args.checkpoint_dir, args.quantized_model, no_fakequant=args.no_fakequant, other_arms=(args.engine_dir,)
    )
    if bool(args.engine_dir) == bool(args.quantized_model):
        raise SystemExit("pass exactly one of --engine-dir and --quantized-model")
    fq_state = None
    if args.quantized_model:
        from foldquant.quantized import load_quantized_model, resolve_model_path, verify_base_checkpoint

        fq_state = load_quantized_model(args.quantized_model)
        verify_base_checkpoint(fq_state, resolve_model_path(fq_state, args.quantized_model, args.checkpoint_dir))
        engine_dir = Path(args.quantized_model)
        manifest = fq_state.manifest["export_manifest"]
    else:
        engine_dir = Path(args.engine_dir)
        manifest = _load_manifest(engine_dir)
    split = _load_manifest(Path(args.split_from)) if args.split_from else manifest
    calib_episodes = sorted({s["episode"] for s in split["calibration"]["samples"]})
    excluded = [] if args.allow_calibration_episodes else calib_episodes

    t0 = time.time()
    policy = calibration.load_policy(args.checkpoint_dir, config_name=args.config, device=args.device, compile=False)
    dataset = calibration.load_dataset(args.dataset_path)
    samples, observations = calibration.sample_observations(
        dataset,
        args.num_samples,
        seed=args.seed,
        exclude_episodes=excluded,
        heldout=True,
        keys=calibration.resolve_keys(dataset, args.config),
    )
    logger.info(
        "%d %s observations from %d episodes (calibration used %d episodes) in %.0fs",
        len(samples),
        "held-out" if excluded else "NOT held-out",
        len({s.episode for s in samples}),
        len(calib_episodes),
        time.time() - t0,
    )

    ref = run_pass(policy, observations, args.seed)
    again = run_pass(policy, observations, args.seed)
    repeat = min(_cos(a, b) for a, b in zip(ref["actions"], again["actions"], strict=True))
    logger.info("PyTorch repeatability (seeded): action cosine min %.6f", repeat)

    if fq_state is not None:
        from foldquant.quantized import install_fake_quant

        from .quantized import module_paths

        installed = install_fake_quant(module_paths(policy), fq_state)
        components = sorted(fq_state.modules)
        logger.info("fake-quant installed: %s", ", ".join(f"{k} {v.scheme}" for k, v in fq_state.modules.items()))
    else:
        installed = install_engines(policy, engine_dir)
        components = sorted(installed.engines)
        logger.info("engines installed: %s", ", ".join(components))
    try:
        got = run_pass(policy, observations, args.seed)
    finally:
        installed.remove()

    kv_cos = [_cos(a, b) for a, b in zip(got["kv_stack"], ref["kv_stack"], strict=True)]
    kv_pos = [_position_cos_min(a, b) for a, b in zip(got["kv_stack"], ref["kv_stack"], strict=True)]
    act_cos = [_cos(a, b) for a, b in zip(got["actions"], ref["actions"], strict=True)]
    act_abs = [float((a - b).abs().max()) for a, b in zip(got["actions"], ref["actions"], strict=True)]
    act_worst = [
        worst_channel(a - b, order="step_major", width=got["action_width"])
        for a, b in zip(got["actions"], ref["actions"], strict=True)
    ]
    report = {
        "engine_dir": public_path(str(engine_dir)),
        "schemes": manifest["schemes"],
        "cascade": manifest.get("cascade", False),
        "components": components,
        "execution": "fake-quant" if fq_state is not None else "tensorrt",
        "split_from": public_path(args.split_from),
        "num_samples": len(samples),
        # What a reader needs to re-run this and land on the same observations. Without
        # them a drift figure cannot be reproduced: the held-out plan is seeded, but it is
        # drawn from whatever episodes the dataset offers, so the same seed over a
        # different slice gives a different set. Two families were measured under an
        # episode restriction that nothing recorded, and re-running them without it
        # shared 0-1 of 32 samples with the committed record.
        "dataset_path": public_path(args.dataset_path),
        "seed": args.seed,
        "held_out": bool(excluded),
        "samples": [
            {
                "episode": s.episode,
                "step": s.step,
                "kv_stack_cos": kc,
                "kv_stack_position_cos_min": kp,
                "action_cos": ac,
                "action_max_abs": aa,
                "action_worst": aw,
            }
            for s, kc, kp, ac, aa, aw in zip(samples, kv_cos, kv_pos, act_cos, act_abs, act_worst, strict=True)
        ],
        "pytorch_repeat_action_cos_min": repeat,
        "kv_stack": {
            "cos_mean": float(np.mean(kv_cos)),
            "cos_median": float(np.median(kv_cos)),
            "cos_min": float(np.min(kv_cos)),
            "position_cos_min": float(np.min(kv_pos)),
        },
        "actions": {
            "cos_mean": float(np.mean(act_cos)),
            "cos_median": float(np.median(act_cos)),
            "cos_min": float(np.min(act_cos)),
            "max_abs": float(np.max(act_abs)),
            "max_abs_mean": float(np.mean(act_abs)),
        },
    }
    logger.info(
        "kv_stack cos mean %.5f min %.5f position-min %.4f | actions cos median %.5f (mean %.5f min %.5f) max_abs %.4f",
        report["kv_stack"]["cos_mean"],
        report["kv_stack"]["cos_min"],
        report["kv_stack"]["position_cos_min"],
        report["actions"]["cos_median"],
        report["actions"]["cos_mean"],
        report["actions"]["cos_min"],
        report["actions"]["max_abs"],
    )
    # A state directory is what gets pushed and fingerprinted; the report goes beside it.
    default = engine_dir.parent / f"{engine_dir.name}_verify.json" if fq_state is not None else engine_dir / "verify.json"
    out = Path(args.output) if args.output else default
    out.write_text(json.dumps(report, indent=2))
    logger.info("wrote %s", out)
    return report


if __name__ == "__main__":
    main(tyro.cli(VerifyConfig))
