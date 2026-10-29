# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""Component latency of the bf16 PyTorch policy and of FoldQuant engine directories.

Upstream N1.5 ships no component timer (its ``deployment_scripts/gr00t_inference.py``
times whole ``get_action`` calls), so this module carries its own: one
observation from the dataset, ``--warmup`` untimed calls, then
``--num-iterations`` timed ones, CUDA-synchronised, medians reported. Per
iteration it times what ``Gr00tPolicy.get_action`` does, in the same order
and under the same autocast:

* ``data_processing``:   ``apply_transforms`` on the raw observation dict;
* ``backbone``:          ``model.prepare_input`` + the Eagle backbone (ViT + LLM);
* ``action_head``:       ``action_head.get_action`` (the full denoising loop);
* ``e2e``:               a separate, whole ``policy.get_action`` call.

Every arm (PyTorch Eager and each FoldQuant engine directory) is timed by
the same loop with :func:`.runtime.install_engines` swapping the engines in.

Example::

    python -m foldquant_integration.benchmark --model-path <ckpt> --embodiment-tag new_embodiment \\
        --dataset-path <calibration dataset> --denoising-steps 4 \\
        --arms w4a4=exports/n15_w4a4/engines w8a8=exports/n15_w8a8/engines
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro

from . import calibration
from ._upstream import LIBERO_DATA_CONFIG
from .runtime import install_engines

logger = logging.getLogger("foldquant.groot_n1_5.benchmark")

COMPONENT_ORDER = ("data_processing", "backbone", "action_head", "e2e")


@dataclass
class BenchmarkConfig:
    model_path: str
    dataset_path: str
    """Dataset the benchmark observation is read from (first episode, step 0)."""

    arms: list[str] = field(default_factory=list)
    """Engine directories to time, as ``LABEL=DIR`` (or a bare ``DIR``, labelled by its parent's name)."""

    embodiment_tag: str | None = None
    data_config: str = LIBERO_DATA_CONFIG
    denoising_steps: int | None = None
    """Flow-matching steps (upstream serves the LIBERO checkpoints with 8)."""

    num_iterations: int = 20
    warmup: int = 5
    seed: int = 42
    skip_eager: bool = False
    """Skip the PyTorch Eager arm (the speedup table then has no baseline)."""

    output: str | None = None
    """Write per-arm medians and the raw per-iteration timings here as JSON."""

    video_backend: str = "torchcodec"


def _parse_arms(specs: list[str]) -> list[tuple]:
    arms = []
    for spec in specs:
        label, sep, path = spec.partition("=")
        if not sep:
            path, label = spec, Path(spec).resolve().parent.name or Path(spec).name
        arms.append((label, Path(path)))
    return arms


def _timed(fn: Callable[[], Any]) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0


def _prepared(policy, observation: dict[str, Any]) -> dict[str, Any]:
    """The observation as ``Gr00tPolicy.get_action`` hands it to ``apply_transforms``."""
    from gr00t.model.policy import unsqueeze_dict_values

    obs = observation.copy()
    if not policy._check_state_is_batched(obs):
        obs = unsqueeze_dict_values(obs)
    return {k: (v if isinstance(v, np.ndarray) else np.array(v)) for k, v in obs.items()}


def time_components(policy, observation: dict[str, Any], args: BenchmarkConfig) -> dict[str, list[float]]:
    from gr00t.model.policy import COMPUTE_DTYPE

    model = policy.model
    obs = _prepared(policy, observation)
    timings: dict[str, list[float]] = {k: [] for k in COMPONENT_ORDER}
    state: dict[str, Any] = {}

    def _data_processing():
        # The transform chain consumes the dict it is given; get_action hands it a copy.
        state["normalized"] = policy.apply_transforms(obs.copy())

    def _backbone():
        backbone_inputs, action_inputs = model.prepare_input(state["normalized"])
        state["action_inputs"] = action_inputs
        state["backbone_outputs"] = model.backbone(backbone_inputs)

    def _action_head():
        model.action_head.get_action(state["backbone_outputs"], state["action_inputs"])

    def _e2e():
        policy.get_action(observation)

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
        for i in range(args.warmup + args.num_iterations):
            torch.manual_seed(args.seed + i)
            dp = _timed(_data_processing)
            bb = _timed(_backbone)
            ah = _timed(_action_head)
            e2e = _timed(_e2e)
            if i >= args.warmup:
                timings["data_processing"].append(dp)
                timings["backbone"].append(bb)
                timings["action_head"].append(ah)
                timings["e2e"].append(e2e)
    med = {k: float(np.median(v)) for k, v in timings.items()}
    logger.info(
        "E2E %.1f ms (%.1f Hz) | data %.1f | backbone %.1f | action head %.1f ms",
        med["e2e"],
        1000 / med["e2e"],
        med["data_processing"],
        med["backbone"],
        med["action_head"],
    )
    return timings


def markdown_table(results: dict[str, dict[str, list[float]]], device: str, denoising_steps: int) -> str:
    lines = [
        f"Device: {device} | {denoising_steps} denoising steps | median ms",
        "",
        "| Arm | Data | Backbone | Action head | E2E | Hz | Speedup |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    base = None
    for label, data in results.items():
        med = {k: float(np.median(v)) for k, v in data.items()}
        if base is None:
            base = med["e2e"]
        lines.append(
            f"| {label} | {med['data_processing']:.1f} | {med['backbone']:.1f} | {med['action_head']:.1f} | "
            f"{med['e2e']:.1f} | {1000 / med['e2e']:.1f} | {base / med['e2e']:.2f}x |"
        )
    return "\n".join(lines)


def main(args: BenchmarkConfig) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    if not torch.cuda.is_available():
        raise SystemExit("benchmarking needs a CUDA device")

    policy = calibration.load_policy(
        args.model_path,
        args.embodiment_tag,
        "cuda",
        data_config=args.data_config,
        denoising_steps=args.denoising_steps,
    )
    dataset = calibration.load_dataset(policy, args.dataset_path, args.video_backend)
    first = calibration.SampleId(int(dataset.trajectory_ids[0]), 0)
    observation = calibration.build_observations(policy, dataset, [first])[0]
    denoising_steps = int(policy.denoising_steps)
    device = torch.cuda.get_device_name(0)
    logger.info(
        "%s | action horizon %d | %d denoising steps",
        device,
        policy.model.action_head.action_horizon,
        denoising_steps,
    )

    results: dict[str, dict[str, list[float]]] = {}
    if not args.skip_eager:
        logger.info("arm: PyTorch Eager")
        results["PyTorch Eager"] = time_components(policy, observation, args)

    components: dict[str, list[str]] = {}
    for label, engine_dir in _parse_arms(args.arms):
        logger.info("arm: %s (%s)", label, engine_dir)
        installed = install_engines(policy, engine_dir)
        components[label] = sorted(installed.engines)
        try:
            results[label] = time_components(policy, observation, args)
        finally:
            installed.remove()
        torch.cuda.empty_cache()

    print(markdown_table(results, device, denoising_steps))

    report = {
        "device": device,
        "model_path": args.model_path,
        "denoising_steps": denoising_steps,
        "num_iterations": args.num_iterations,
        "warmup": args.warmup,
        "arms": {
            label: {
                "components": components.get(label, []),
                "median_ms": {k: float(np.median(v)) for k, v in data.items()},
                "raw_ms": {k: [float(x) for x in v] for k, v in data.items()},
            }
            for label, data in results.items()
        },
    }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2))
        logger.info("wrote %s", args.output)
    return report


if __name__ == "__main__":
    main(tyro.cli(BenchmarkConfig))
