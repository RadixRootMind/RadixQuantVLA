# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""LIBERO success-rate sweep through a websocket client, with a resumable summary.

Upstream evaluates Pi0.5 on LIBERO with a websocket client
(``examples/libero/main.py``, in its own Python 3.8 environment with LIBERO
and robosuite) talking to ``scripts/serve_policy.py``, one suite per run.
This module runs that pairing for several suites unattended: it starts
:mod:`.serve` for the arm, waits for the port, runs :mod:`.libero_client`
(upstream's client loop with a selectable step budget, writing per-episode
JSON) on each suite with the client environment's interpreter
(``--client-python``), and records each suite in ``<output>/summary.json``,
so an interrupted sweep resumes where it stopped, and only into the same
run. ``--protocol p3`` is the paper's campaign (20 trials per task from
LIBERO's stored initial states, 520 steps on every suite, seed 7, replan
every 5 steps).

Example::

    MUJOCO_GL=egl python -m foldquant_integration.eval_libero \\
        --checkpoint-dir <ckpt> --engine-dir exports/pi05_w4a4/engines \\
        --client-python examples/libero/.venv/bin/python --output results/pi05_w4a4
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import json
import logging
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any

from foldquant.eval_protocol import PROTOCOLS, artifact_digest, prepare_summary, protocol_record
from foldquant.provenance import public_path
import tyro

from ._upstream import LIBERO_DIR
from ._upstream import LIBERO_TRAIN_CONFIG
from ._upstream import UPSTREAM_ROOT
from ._upstream import seed_libero_config

logger = logging.getLogger("foldquant.pi05.eval")

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")



@dataclass
class EvalConfig:
    checkpoint_dir: str
    client_python: str
    """Interpreter of the LIBERO client environment (``examples/libero/README.md``, "Without Docker")."""

    engine_dir: str | None = None
    """FoldQuant engine directory; omit to score the bf16 PyTorch policy."""

    quantized_model: str | None = None
    """FoldQuant fake-quant state for ``--checkpoint-dir`` (a state saved without the base files); a
    self-contained fake-quant model given as ``--checkpoint-dir`` is detected by itself. Mutually
    exclusive with ``--engine-dir``."""

    no_fakequant: bool = False
    """When ``--checkpoint-dir`` is a FoldQuant fake-quant model, load it as the plain base policy."""

    output: str = "results/pi05"
    """Directory for ``summary.json``, the per-suite client logs and the replay videos."""

    config: str = LIBERO_TRAIN_CONFIG
    suites: list[str] = field(default_factory=lambda: list(SUITES))
    protocol: str = "upstream"
    """``upstream`` (upstream client defaults: 50 trials, per-suite step budgets) or ``p3`` (the paper's campaign: 20 trials, 520 steps)."""

    num_trials_per_task: int | None = None
    """Rollouts per task (default 50 upstream / 20 p3; 10 tasks per suite)."""

    max_steps: int | None = None
    """Environment steps per episode after the settle steps; default per suite (upstream) or 520 (p3)."""

    num_steps_wait: int = 10
    """No-op steps after the initial state is set, while dropped objects settle."""

    resume: bool = True
    """Continue an interrupted sweep in ``--output``; refused when it was a different run."""

    replan_steps: int = 5
    seed: int = 7
    """The client's seed (object positions follow it even from a fixed initial state)."""

    port: int = 8000
    server_timeout_s: float = 900.0
    """How long to wait for the server to accept connections (engine + plugin loading)."""


def _wait_for_port(port: int, proc: subprocess.Popen, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"policy server exited early with code {proc.returncode}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1.0):
                return
        except OSError:
            time.sleep(2.0)
    raise TimeoutError(f"policy server did not open port {port} within {timeout_s:.0f}s")


def _start_server(args: EvalConfig, log_path: Path) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "foldquant_integration.serve",
        "--checkpoint-dir",
        args.checkpoint_dir,
        "--config",
        args.config,
        "--port",
        str(args.port),
    ]
    if args.engine_dir:
        cmd += ["--engine-dir", args.engine_dir]
    if args.quantized_model:
        cmd += ["--quantized-model", args.quantized_model]
    elif args.no_fakequant:
        cmd += ["--no-fakequant"]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(UPSTREAM_ROOT), env.get("PYTHONPATH", "")) if p)
    with open(log_path, "w") as log:
        # Popen inherits the descriptor; the parent's handle can close once the child holds it.
        proc = subprocess.Popen(cmd, cwd=str(UPSTREAM_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
    logger.info("policy server pid %d (log %s)", proc.pid, log_path)
    _wait_for_port(args.port, proc, args.server_timeout_s)
    return proc


def _stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def _run_suite(args: EvalConfig, suite: str, out: Path) -> dict[str, Any]:
    log_path = out / f"{suite}.log"
    json_path = out / f"{suite}.json"
    cmd = [
        args.client_python,
        str(Path(__file__).resolve().parent / "libero_client.py"),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--task-suite-name",
        suite,
        "--num-trials-per-task",
        str(args.num_trials_per_task),
        "--num-steps-wait",
        str(args.num_steps_wait),
        "--replan-steps",
        str(args.replan_steps),
        "--seed",
        str(args.seed),
        "--out-json",
        str(json_path),
    ]
    if args.max_steps is not None:
        cmd += ["--max-steps", str(args.max_steps)]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(LIBERO_DIR), env.get("PYTHONPATH", "")) if p)
    t0 = time.time()
    with open(log_path, "w") as log:
        rc = subprocess.call(cmd, cwd=str(UPSTREAM_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
    if rc != 0 or not json_path.is_file():
        raise RuntimeError(f"{suite}: client exited {rc} without a result; see {log_path}")
    result = json.loads(json_path.read_text())
    totals = result["totals"]
    return {
        "success_rate": totals["success_rate"],
        "successes": totals["successes"],
        "episodes": totals["num_episodes"],
        "max_steps": result["max_steps"],
        "tasks": {t["name"]: {k: t[k] for k in ("num_episodes", "successes", "success_rate", "episodes")} for t in result["tasks"]},
        "seconds": round(time.time() - t0, 1),
        "log": log_path.name,
    }


def main(args: EvalConfig) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    unknown = [s for s in args.suites if s not in SUITES]
    if unknown:
        raise SystemExit(f"unknown suites {unknown}; choose from {list(SUITES)}")
    if not Path(args.client_python).is_file():
        raise SystemExit(f"--client-python {args.client_python} is not a file")
    from foldquant.quantized import quantized_arm

    args.quantized_model = quantized_arm(
        args.checkpoint_dir, args.quantized_model, no_fakequant=args.no_fakequant, other_arms=(args.engine_dir,)
    )
    if args.engine_dir and args.quantized_model:
        raise ValueError("--engine-dir and --quantized-model are mutually exclusive")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "summary.json"
    if args.protocol not in PROTOCOLS:
        raise SystemExit(f"--protocol {args.protocol}: choose from {sorted(PROTOCOLS)}")
    protocol = PROTOCOLS[args.protocol]
    args.num_trials_per_task = protocol.resolve(args.num_trials_per_task, "n_episodes", 50)
    if args.max_steps is None and protocol.max_episode_steps is not None:
        args.max_steps = protocol.max_episode_steps
    run = {
        "protocol": protocol_record(protocol, args.max_steps if args.max_steps is not None else -1, args.replan_steps, args.num_trials_per_task, args.seed),
        "max_steps": args.max_steps if args.max_steps is not None else "upstream per-suite",
        "num_steps_wait": args.num_steps_wait,
        "checkpoint_dir": public_path(args.checkpoint_dir),
        "checkpoint": artifact_digest(args.checkpoint_dir),
        "engine_dir": public_path(args.engine_dir),
        "engines": artifact_digest(args.engine_dir),
        "quantized_model": public_path(args.quantized_model),
        "quantized": artifact_digest(args.quantized_model),
        "config": args.config,
        "suites": list(args.suites),
        "replan_steps": args.replan_steps,
        "seed": args.seed,
    }
    summary = prepare_summary(summary_path, run, resume=args.resume)
    summary.pop("tasks", None)
    summary.update(
        {
            "checkpoint_dir": public_path(args.checkpoint_dir),
            "engine_dir": public_path(args.engine_dir),
            "quantized_model": public_path(args.quantized_model),
            "execution": "fake-quant" if args.quantized_model else ("tensorrt" if args.engine_dir else "pytorch"),
            "config": args.config,
            "num_trials_per_task": args.num_trials_per_task,
            "max_steps": args.max_steps,
            "replan_steps": args.replan_steps,
            "seed": args.seed,
        }
    )
    summary.setdefault("suites", {})
    todo = [s for s in args.suites if s not in summary["suites"]]
    if not todo:
        logger.info("all requested suites already in %s", summary_path)
    else:
        seed_libero_config()
        server = _start_server(args, out / "server.log")
        try:
            for suite in todo:
                logger.info("suite %s", suite)
                summary["suites"][suite] = _run_suite(args, suite, out)
                summary_path.write_text(json.dumps(summary, indent=2))
                r = summary["suites"][suite]
                logger.info("%s: %d/%d = %.3f", suite, r["successes"], r["episodes"], r["success_rate"])
        finally:
            _stop_server(server)

    done = [s for s in args.suites if s in summary["suites"]]
    total_s = sum(summary["suites"][s]["successes"] for s in done)
    total_n = sum(summary["suites"][s]["episodes"] for s in done)
    summary["overall"] = {
        "suites": done,
        "successes": total_s,
        "episodes": total_n,
        "success_rate": (total_s / total_n) if total_n else None,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("overall %d/%d = %s", total_s, total_n, summary["overall"]["success_rate"])
    return summary


if __name__ == "__main__":
    main(tyro.cli(EvalConfig))
