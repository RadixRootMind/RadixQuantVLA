# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""Component latency of the bf16 PyTorch policy and of FoldQuant engine directories.

Upstream openpi reports one number per call (``policy_timing/infer_ms``), so
this module carries its own component timer: one observation from the
dataset, ``--warmup`` untimed calls, then ``--num-iterations`` timed ones,
CUDA-synchronised, medians reported. Per iteration it times the pieces of
``Policy.infer`` where they run:

* ``input_transform``:  the upstream input transforms plus host-to-device;
* ``embed_prefix``:     SigLIP on the three cameras + prompt embedding;
* ``prefix_llm``:       the PaliGemma prefix pass that fills the KV cache;
* ``denoise_loop``:     every ``denoise_step`` of the Euler loop, summed;
* ``e2e``:              a separate, whole ``policy.infer`` call.

The three model pieces are timed by wrapping the bound methods
``sample_actions`` calls (``embed_prefix``, ``paligemma_with_expert.forward``
for the prefix branch, ``denoise_step``) for the duration of one
``sample_actions``; the e2e call runs unwrapped. Every arm (PyTorch Eager and
each FoldQuant engine directory) goes through the same loop with
:func:`.runtime.install_engines` swapping the engines in.

Upstream serves the model under ``torch.compile(mode="max-autotune")``
(``Pi0Config.pytorch_compile_mode``). That arm is timed too (e2e only, since
the compiled graph inlines the pieces the component wrappers would time), so
the table shows the FoldQuant engines against both the eager policy and the
policy as upstream deploys it. The engines themselves need the eager model
(see :mod:`.runtime`), so it is the eager policy that hosts them.

Example::

    python -m foldquant_integration.benchmark --checkpoint-dir <ckpt> --dataset-path <LeRobot LIBERO> \\
        --arms w4a4=exports/pi05_w4a4/engines w8a8=exports/pi05_w8a8/engines
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
import functools
import json
import logging
from pathlib import Path
import time
from typing import Any

import jax
import numpy as np
import torch
import tyro

from . import calibration
from ._upstream import LIBERO_TRAIN_CONFIG
from .runtime import install_engines
from .runtime import model_of

logger = logging.getLogger("foldquant.pi05.benchmark")

COMPONENT_ORDER = ("input_transform", "embed_prefix", "prefix_llm", "denoise_loop", "e2e")


@dataclass
class BenchmarkConfig:
    checkpoint_dir: str
    dataset_path: str
    """Dataset the benchmark observation is read from (first episode, step 0)."""

    arms: list[str] = field(default_factory=list)
    """Engine directories to time, as ``LABEL=DIR`` (or a bare ``DIR``, labelled by its parent's name)."""

    config: str = LIBERO_TRAIN_CONFIG
    num_iterations: int = 20
    warmup: int = 5
    seed: int = 42
    skip_eager: bool = False
    """Skip the PyTorch Eager arm (the speedup table then has no baseline)."""

    skip_compiled: bool = False
    """Skip the ``torch.compile`` arm (upstream's serving default; e2e only)."""

    output: str | None = None
    """Write per-arm medians and the raw per-iteration timings here as JSON."""

    device: str = "cuda"


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


_MISSING = object()


class _Stopwatches:
    """Instance-level timing wrappers around bound methods, summed per bucket; ``restore()`` undoes them."""

    def __init__(self) -> None:
        self.ms: dict[str, float] = {}
        self._undo: list[Callable[[], None]] = []

    def reset(self) -> None:
        self.ms = {}

    def wrap(self, obj, name: str, bucket: str, only: Callable[..., bool] | None = None) -> None:
        previous = obj.__dict__.get(name, _MISSING)
        fn = getattr(obj, name)

        def wrapper(*a, **kw):
            if only is not None and not only(*a, **kw):
                return fn(*a, **kw)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = fn(*a, **kw)
            torch.cuda.synchronize()
            self.ms[bucket] = self.ms.get(bucket, 0.0) + (time.perf_counter() - t0) * 1000.0
            return out

        setattr(obj, name, wrapper)

        def undo():
            if previous is _MISSING:
                obj.__dict__.pop(name, None)
            else:
                setattr(obj, name, previous)

        self._undo.append(undo)

    def restore(self) -> None:
        for undo in reversed(self._undo):
            undo()
        self._undo = []


def _is_prefix_call(*_a, **kw) -> bool:
    embeds = kw.get("inputs_embeds")
    return embeds is not None and embeds[1] is None


def time_components(
    policy, observation: dict[str, Any], args: BenchmarkConfig, *, components: bool = True
) -> dict[str, list[float]]:
    """Per-iteration timings by bucket; ``components=False`` times ``input_transform`` and ``e2e`` only."""
    from openpi.models import model as _model

    model = model_of(policy)
    device = policy._pytorch_device  # noqa: SLF001
    timings: dict[str, list[float]] = {k: [] for k in COMPONENT_ORDER}
    state: dict[str, Any] = {}

    def _input_transform():
        inputs = policy._input_transform(jax.tree.map(lambda x: x, observation))  # noqa: SLF001
        inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(device)[None, ...], inputs)
        state["observation"] = _model.Observation.from_dict(inputs)

    watches = _Stopwatches()

    def _watch():
        watches.wrap(model, "embed_prefix", "embed_prefix")
        watches.wrap(model.paligemma_with_expert, "forward", "prefix_llm", only=_is_prefix_call)
        watches.wrap(model, "denoise_step", "denoise_loop")

    try:
        with torch.inference_mode():
            for i in range(args.warmup + args.num_iterations):
                noise = calibration.action_noise(policy, args.seed + i)
                tin = _timed(_input_transform)
                parts: dict[str, float] = {}
                if components:
                    noise_t = torch.from_numpy(noise).to(device)[None, ...]
                    _watch()
                    torch.manual_seed(args.seed + i)
                    model.sample_actions(
                        device,
                        state["observation"],
                        noise=noise_t,
                        **policy._sample_kwargs,  # noqa: SLF001
                    )
                    parts = dict(watches.ms)
                    watches.reset()
                    # The whole call is timed with the wrappers off, so their synchronisations do not count.
                    watches.restore()
                torch.manual_seed(args.seed + i)
                e2e = _timed(functools.partial(policy.infer, dict(observation), noise=noise))
                if i >= args.warmup:
                    timings["input_transform"].append(tin)
                    timings["embed_prefix"].append(parts.get("embed_prefix", float("nan")))
                    timings["prefix_llm"].append(parts.get("prefix_llm", float("nan")))
                    timings["denoise_loop"].append(parts.get("denoise_loop", float("nan")))
                    timings["e2e"].append(e2e)
    finally:
        watches.restore()
    med = {k: float(np.median(v)) for k, v in timings.items()}
    logger.info(
        "E2E %.1f ms (%.1f Hz) | input %.1f | embed %s | prefix LLM %s | denoise loop %s ms",
        med["e2e"],
        1000 / med["e2e"],
        med["input_transform"],
        _fmt(med["embed_prefix"]),
        _fmt(med["prefix_llm"]),
        _fmt(med["denoise_loop"]),
    )
    return timings


def _fmt(ms: float) -> str:
    return "—" if np.isnan(ms) else f"{ms:.1f}"


class _Compiled:
    """Temporarily serve ``policy.infer`` through ``torch.compile(model.sample_actions, mode)``, as upstream does."""

    def __init__(self, policy, mode: str) -> None:
        self._policy = policy
        self._mode = mode
        self._previous: Any = None

    def __enter__(self) -> _Compiled:
        self._previous = self._policy._sample_actions  # noqa: SLF001
        self._policy._sample_actions = torch.compile(  # noqa: SLF001
            model_of(self._policy).sample_actions, mode=self._mode
        )
        return self

    def __exit__(self, *_exc: Any) -> None:
        self._policy._sample_actions = self._previous  # noqa: SLF001
        torch._dynamo.reset()  # noqa: SLF001


def _json_ms(ms: float) -> float | None:
    return None if np.isnan(ms) else ms


def markdown_table(results: dict[str, dict[str, list[float]]], device: str, num_steps: int) -> str:
    lines = [
        f"Device: {device} | {num_steps} denoising steps | median ms",
        "",
        "| Arm | Input | Embed | Prefix LLM | Denoise loop | E2E | Hz | Speedup |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    base = None
    for label, data in results.items():
        med = {k: float(np.median(v)) for k, v in data.items()}
        if base is None:
            base = med["e2e"]
        lines.append(
            f"| {label} | {med['input_transform']:.1f} | {_fmt(med['embed_prefix'])} | {_fmt(med['prefix_llm'])} | "
            f"{_fmt(med['denoise_loop'])} | {med['e2e']:.1f} | {1000 / med['e2e']:.1f} | {base / med['e2e']:.2f}x |"
        )
    return "\n".join(lines)


def main(args: BenchmarkConfig) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    if not torch.cuda.is_available():
        raise SystemExit("benchmarking needs a CUDA device")

    from openpi.training import config as _config

    policy = calibration.load_policy(args.checkpoint_dir, config_name=args.config, device=args.device, compile=False)
    compile_mode = _config.get_config(args.config).model.pytorch_compile_mode
    dataset = calibration.load_dataset(args.dataset_path)
    episode_ids, _lengths, _starts = calibration.episode_table(dataset)
    observation = calibration.build_observations(
        dataset, [calibration.SampleId(episode_ids[0], 0)], calibration.resolve_keys(dataset, args.config)
    )[0]
    model = model_of(policy)
    num_steps = int(policy._sample_kwargs.get("num_steps", 10))  # noqa: SLF001
    device = torch.cuda.get_device_name(0)
    logger.info(
        "%s | %s | action horizon %d | %d denoising steps",
        device,
        "pi05" if model.pi05 else "pi0",
        model.config.action_horizon,
        num_steps,
    )

    results: dict[str, dict[str, list[float]]] = {}
    if not args.skip_eager:
        logger.info("arm: PyTorch Eager")
        results["PyTorch Eager"] = time_components(policy, observation, args)
    if not args.skip_compiled and compile_mode is not None:
        label = f"PyTorch torch.compile({compile_mode})"
        logger.info("arm: %s (upstream's serving default, e2e only)", label)
        with _Compiled(policy, compile_mode):
            results[label] = time_components(policy, observation, args, components=False)

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

    print(markdown_table(results, device, num_steps))

    report = {
        "device": device,
        "checkpoint_dir": args.checkpoint_dir,
        "num_steps": num_steps,
        "num_iterations": args.num_iterations,
        "warmup": args.warmup,
        "arms": {
            label: {
                "components": components.get(label, []),
                "median_ms": {k: _json_ms(float(np.median(v))) for k, v in data.items()},
                "raw_ms": {k: [_json_ms(float(x)) for x in v] for k, v in data.items()},
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
