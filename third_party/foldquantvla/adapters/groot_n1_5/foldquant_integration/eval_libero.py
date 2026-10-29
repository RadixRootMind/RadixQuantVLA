# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""LIBERO success-rate sweep, in process, with a resumable summary.

Upstream evaluates N1.5 on LIBERO through a ZMQ client
(``examples/Libero/eval/run_libero_eval.py``) talking to
``scripts/inference_service.py``, one suite per run, results in a text log.
The reference way to score a FoldQuant arm is exactly that, with
:mod:`.serve` on the server side. This module is the same rollout loop
without the socket, for sweeps that cover several suites and arms
unattended:

* the observation and action conversion is upstream's own ``GR00TPolicy``
  class, subclassed only to hold the in-process policy instead of the ZMQ
  client. The dictionaries the model sees are byte-identical to the served
  path;
* the environment loop (``env.reset``, ``set_init_state``, the
  ``num_steps_wait`` no-op steps, the per-suite ``max_steps``) follows
  ``eval_libero`` in ``run_libero_eval.py``;
* results land in ``<output>/summary.json`` per task, so an interrupted
  sweep resumes where it stopped, and rollout videos are not written.

Upstream defaults to 5 trials per task; ``--n-episodes`` defaults to 20 here
(200 episodes per suite). Needs LIBERO (``libero`` + ``robosuite``) installed
in the environment, as upstream's client does.

Example::

    MUJOCO_GL=egl python -m foldquant_integration.eval_libero \\
        --model-path <ckpt> --embodiment-tag new_embodiment --denoising-steps 4 \\
        --engine-dir exports/n15_w4a4/engines --output results/n15_w4a4
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import tyro

from foldquant.eval_protocol import PROTOCOLS, artifact_digest, prepare_summary, protocol_record
from foldquant.provenance import public_path

from . import calibration
from ._upstream import LIBERO_DATA_CONFIG, ensure_libero_on_path, ensure_upstream_on_path
from .runtime import install_engines

logger = logging.getLogger("foldquant.groot_n1_5.eval")

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")

#: Episode step budgets, as upstream's ``run_libero_eval.py`` sets them per suite.
MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 600,
    "libero_10": 1000,
    "libero_90": 400,
}


@dataclass
class EvalConfig:
    model_path: str
    """Checkpoint directory (the PyTorch policy; also the residual half of a FoldQuant arm)."""

    output: str
    """Directory for ``summary.json`` and the per-task log."""

    engine_dir: str | None = None
    """FoldQuant engine directory. Omit for the bf16 PyTorch arm."""

    quantized_model: str | None = None
    """FoldQuant fake-quant state for ``--model-path`` (a state saved without the base files); a
    self-contained fake-quant model given as ``--model-path`` is detected by itself. Mutually
    exclusive with ``--engine-dir``."""

    no_fakequant: bool = False
    """When ``--model-path`` is a FoldQuant fake-quant model, load it as the plain base policy."""

    embodiment_tag: str | None = None
    data_config: str = LIBERO_DATA_CONFIG
    denoising_steps: int | None = None
    """Flow-matching steps; default the checkpoint's own value, or 4 under ``--protocol p3`` (the paper's setting; upstream's client serves LIBERO with 8)."""

    suites: list[str] = field(default_factory=lambda: list(SUITES))
    protocol: str = "upstream"
    """``upstream`` (per-suite step budgets of upstream's client) or ``p3`` (the paper's campaign: 520 steps everywhere)."""

    n_episodes: int | None = None
    """Episodes per task (default 20; upstream's client defaults to 5); 10 tasks per suite."""

    max_steps: int | None = None
    """Environment steps per episode after the settle steps; default per suite (upstream) or 520 (p3)."""

    num_steps_wait: int = 10
    """No-op steps after reset while dropped objects settle, as upstream."""

    seed: int | None = None
    """Simulator seed applied once per task environment; default upstream's 0, or 7 under ``--protocol p3``."""

    resume: bool = True
    """Continue an interrupted sweep in ``--output``; refused when it was a different run."""

    resolution: int = 256
    tasks: list[str] | None = None
    """Restrict to these task names."""



def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    per_suite: dict[str, dict[str, int]] = {}
    for entry in summary["tasks"].values():
        if entry.get("status") != "ok":
            continue
        agg = per_suite.setdefault(entry["suite"], {"successes": 0, "num_episodes": 0})
        agg["successes"] += entry["successes"]
        agg["num_episodes"] += entry["num_episodes"]
    total_s = sum(v["successes"] for v in per_suite.values())
    total_n = sum(v["num_episodes"] for v in per_suite.values())
    summary["per_suite"] = {
        k: {**v, "success_rate": v["successes"] / v["num_episodes"]} for k, v in per_suite.items()
    }
    summary["totals"] = {
        "tasks_completed": sum(1 for e in summary["tasks"].values() if e.get("status") == "ok"),
        "tasks_failed": sum(1 for e in summary["tasks"].values() if e.get("status") != "ok"),
        "successes": total_s,
        "num_episodes": total_n,
        "success_rate": (total_s / total_n) if total_n else None,
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, indent=2))
    tmp.replace(path)


def _make_wrapper(policy):
    """Upstream's ``GR00TPolicy`` LIBERO wrapper around an in-process ``Gr00tPolicy``."""
    ensure_libero_on_path()  # idempotent; main() has normally done this already
    from examples.Libero.eval.run_libero_eval import GR00TPolicy

    class InProcessGR00TPolicy(GR00TPolicy):
        def __init__(self, gr00t_policy):
            # The base class opens a ZMQ client (lazy connect, no traffic); replace it.
            super().__init__(headless=True)
            self.policy = gr00t_policy

    return InProcessGR00TPolicy(policy)


def _task_init_states(task_suite, task_id: int):
    """``task_suite.get_task_init_states(task_id)``, loadable on torch >= 2.6.

    LIBERO saves each task's initial states as a pickled NumPy array and reads
    them back with a bare ``torch.load``. torch 2.6 turned ``weights_only`` on by
    default, which refuses that pickle, so every task raises before its first
    episode and the run reports 0/0. The ``orin`` extra pins torch 2.8 (``base``
    pins 2.5.1, where it still loads). The files come from the LIBERO checkout
    itself, so load them the way LIBERO was written to.
    """
    import torch
    from libero.libero import get_libero_path

    task = task_suite.get_task(task_id)
    path = os.path.join(get_libero_path("init_states"), task.problem_folder, task.init_states_file)
    return torch.load(path, weights_only=False)


def run_task(wrapper, suite: str, task_id: int, args: EvalConfig) -> dict[str, Any]:
    from libero.libero import benchmark

    from examples.Libero.eval.utils import get_libero_dummy_action, get_libero_env

    task_suite = benchmark.get_benchmark_dict()[suite]()
    task = task_suite.get_task(task_id)
    initial_states = _task_init_states(task_suite, task_id)
    env, task_description = get_libero_env(task, resolution=args.resolution)
    if args.seed is not None:
        env.seed(args.seed)
    max_steps = args.max_steps if args.max_steps is not None else MAX_STEPS[suite]
    n_episodes = args.n_episodes if args.n_episodes is not None else 20
    successes = 0
    steps: list[int] = []
    episodes: list[dict[str, Any]] = []
    errors = 0
    try:
        for episode_idx in range(n_episodes):
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])
            t = 0
            policy_steps = 0
            done = False
            try:
                while t < max_steps + args.num_steps_wait:
                    if t < args.num_steps_wait:
                        obs, _reward, done, _info = env.step(get_libero_dummy_action())
                        t += 1
                        continue
                    action = wrapper.get_action(obs, task.language)
                    obs, _reward, done, _info = env.step(action.tolist())
                    policy_steps += 1
                    if done:
                        break
                    t += 1
            except Exception as exc:  # noqa: BLE001 - upstream counts the episode as failed and moves on
                logger.warning("%s/%s episode %d: %s", suite, task.name, episode_idx, exc)
                errors += 1
                done = False
            successes += int(bool(done))
            steps.append(t)
            episodes.append(
                {
                    "episode": episode_idx,
                    "init_state_id": episode_idx,
                    "success": bool(done),
                    "steps": policy_steps,
                }
            )
    finally:
        env.close()
    return {
        "suite": suite,
        "task_id": task_id,
        "name": task.name,
        "language": task_description,
        "num_episodes": n_episodes,
        "successes": successes,
        "success_rate": successes / n_episodes,
        "episode_errors": errors,
        "mean_steps": float(np.mean(steps)) if steps else None,
        "max_steps": max_steps,
        "episodes": episodes,
    }


def main(args: EvalConfig) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    os.environ.setdefault("MUJOCO_GL", "egl")
    ensure_upstream_on_path()
    # Before any LIBERO import, not before one of them: this module reaches
    # `libero.libero` here and `examples.Libero` further down, and hooking the
    # later site leaves this one failing with FOLDQUANT_LIBERO_DIR correctly set.
    ensure_libero_on_path()
    from libero.libero import benchmark

    from foldquant.quantized import quantized_arm

    args.quantized_model = quantized_arm(
        args.model_path, args.quantized_model, no_fakequant=args.no_fakequant, other_arms=(args.engine_dir,)
    )
    if args.engine_dir and args.quantized_model:
        raise ValueError("--engine-dir and --quantized-model are mutually exclusive")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "summary.json"
    if args.protocol not in PROTOCOLS:
        raise SystemExit(f"--protocol {args.protocol}: choose from {sorted(PROTOCOLS)}")
    protocol = PROTOCOLS[args.protocol]
    args.n_episodes = protocol.resolve(args.n_episodes, "n_episodes", 20)
    if args.max_steps is None and protocol.max_episode_steps is not None:
        args.max_steps = protocol.max_episode_steps
    if args.denoising_steps is None and protocol.name == "p3":
        args.denoising_steps = 4  # the paper runs every GR00T checkpoint with four denoising steps
    if args.seed is None:
        args.seed = protocol.seed
    run = {
        # N1.5 executes one action per policy call (K = 1), as upstream's client does
        "protocol": protocol_record(protocol, args.max_steps if args.max_steps is not None else -1, 1, args.n_episodes, args.seed),
        "seed": args.seed,
        "max_steps": args.max_steps if args.max_steps is not None else dict(MAX_STEPS),
        "num_steps_wait": args.num_steps_wait,
        "model_path": public_path(args.model_path),
        "model": artifact_digest(args.model_path),
        "engine_dir": public_path(args.engine_dir),
        "engines": artifact_digest(args.engine_dir),
        "quantized_model": public_path(args.quantized_model),
        "quantized": artifact_digest(args.quantized_model),
        "embodiment_tag": args.embodiment_tag,
        "data_config": args.data_config,
        "denoising_steps": args.denoising_steps,
        "suites": list(args.suites),
        "tasks": sorted(args.tasks) if args.tasks else None,
        "resolution": args.resolution,
    }
    summary = prepare_summary(summary_path, run, resume=args.resume)
    summary["arm"] = {
        "model_path": public_path(args.model_path),
        "engine_dir": public_path(args.engine_dir),
        "quantized_model": public_path(args.quantized_model),
        "embodiment_tag": args.embodiment_tag,
        "data_config": args.data_config,
        "n_episodes": args.n_episodes,
        "num_steps_wait": args.num_steps_wait,
        "resolution": args.resolution,
        "seed": args.seed,
    }

    benchmark_dict = benchmark.get_benchmark_dict()
    todo = []
    for suite in args.suites:
        for task_id, name in enumerate(benchmark_dict[suite]().get_task_names()):
            key = f"{suite}/{name}"
            if args.tasks and name not in args.tasks:
                continue
            if summary["tasks"].get(key, {}).get("status") == "ok":
                continue
            todo.append((suite, task_id, key))
    logger.info("%d tasks to run (%d already in %s)", len(todo), len(summary["tasks"]), summary_path)
    if not todo:
        _write_summary(summary_path, summary)
        return summary

    policy = calibration.load_policy(
        args.model_path,
        args.embodiment_tag,
        "cuda",
        data_config=args.data_config,
        denoising_steps=args.denoising_steps,
    )
    summary["arm"]["denoising_steps"] = int(policy.denoising_steps)
    summary["arm"]["embodiment_tag"] = policy.embodiment_tag.value
    installed = install_engines(policy, args.engine_dir) if args.engine_dir else None
    if installed is not None:
        logger.info("engines installed: %s", ", ".join(sorted(installed.engines)))
    if args.quantized_model:
        from foldquant.quantized import install_on_policy

        installed, fq_state = install_on_policy(policy, args.quantized_model, args.model_path)
        summary["arm"]["schemes"] = {**{k: v.scheme for k, v in fq_state.modules.items()}, "execution": "fake-quant"}
    wrapper = _make_wrapper(policy)

    try:
        for suite, task_id, key in todo:
            t0 = time.time()
            try:
                entry = run_task(wrapper, suite, task_id, args)
                entry["status"] = "ok"
            except Exception as exc:
                logger.exception("%s failed", key)
                entry = {"suite": suite, "task_id": task_id, "status": "error", "error": repr(exc)}
            entry["seconds"] = round(time.time() - t0, 1)
            summary["tasks"][key] = entry
            _write_summary(summary_path, summary)
            if entry["status"] == "ok":
                logger.info("%s: %d/%d in %.0fs", key, entry["successes"], entry["num_episodes"], entry["seconds"])
    finally:
        if installed is not None:
            installed.remove()
    logger.info("totals: %s", summary["totals"])
    return summary


if __name__ == "__main__":
    main(tyro.cli(EvalConfig))
