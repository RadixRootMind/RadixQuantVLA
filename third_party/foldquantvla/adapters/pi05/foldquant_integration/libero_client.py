# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""LIBERO client for the π₀.₅ policy server, with a selectable step budget.

Upstream's ``examples/libero/main.py`` fixes the episode budget per suite
(220 / 280 / 300 / 520 steps). The paper's closed-loop campaign allows 520
steps on every suite, so this client repeats upstream's loop (same
observation preprocessing, ``replan_steps`` execution, settle steps, stored
initial states, seed) with ``--max-steps`` exposed, and writes one JSON with
per-task and per-episode outcomes instead of a log to grep.

Runs in the LIBERO client environment (``examples/libero/.venv``), like the
upstream script it replaces:

    python -m foldquant_integration.libero_client --host 127.0.0.1 --port 8000 \\
        --task-suite-name libero_10 --num-trials-per-task 20 --max-steps 520 \\
        --out-json out/libero_10.json
"""

from __future__ import annotations

import collections
import dataclasses
import json
import logging
import math
import pathlib
from typing import Optional

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data

#: upstream's per-suite budgets (``examples/libero/main.py``)
UPSTREAM_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    """No-op steps after the initial state is set, while dropped objects settle."""
    num_trials_per_task: int = 50
    max_steps: Optional[int] = None
    """Environment steps per episode after the settle steps; default is upstream's per-suite budget."""

    out_json: str = "libero_result.json"
    video_out_path: Optional[str] = None
    """Write a replay video per episode here; omitted, no videos are written."""
    seed: int = 7


def eval_libero(args: Args) -> dict:
    np.random.seed(args.seed)
    task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    max_steps = args.max_steps if args.max_steps is not None else UPSTREAM_MAX_STEPS[args.task_suite_name]
    if args.video_out_path:
        pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    result = {
        "suite": args.task_suite_name,
        "max_steps": max_steps,
        "num_steps_wait": args.num_steps_wait,
        "num_trials_per_task": args.num_trials_per_task,
        "replan_steps": args.replan_steps,
        "seed": args.seed,
        "tasks": [],
    }
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(task_suite.n_tasks)):
        task = task_suite.get_task(task_id)
        initial_states = _init_states(task_suite, task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        episodes = []
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            env.reset()
            action_plan = collections.deque()
            obs = env.set_init_state(initial_states[episode_idx])
            t = 0
            policy_steps = 0
            done = False
            replay_images = []
            while t < max_steps + args.num_steps_wait:
                try:
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, args.resize_size, args.resize_size))
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )
                    if args.video_out_path:
                        replay_images.append(img)
                    if not action_plan:
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                            ),
                            "prompt": str(task_description),
                        }
                        action_chunk = client.infer(element)["actions"]
                        assert len(action_chunk) >= args.replan_steps, (
                            f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        )
                        action_plan.extend(action_chunk[: args.replan_steps])
                    action = action_plan.popleft()
                    obs, reward, done, info = env.step(action.tolist())
                    policy_steps += 1
                    if done:
                        break
                    t += 1
                except Exception as e:  # noqa: BLE001 - upstream logs and counts the episode as failed
                    logging.error(f"Caught exception: {e}")
                    break
            success = bool(done)
            episodes.append(
                {
                    "episode": episode_idx,
                    "init_state_id": episode_idx,
                    "success": success,
                    "steps": policy_steps,
                }
            )
            total_episodes += 1
            total_successes += int(success)
            if args.video_out_path and replay_images:
                import imageio

                suffix = "success" if success else "failure"
                task_segment = task_description.replace(" ", "_")
                imageio.mimwrite(
                    pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{episode_idx}_{suffix}.mp4",
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )
            logging.info(f"{task.name} episode {episode_idx}: {'success' if success else 'failure'}")
        env.close()
        successes = sum(e["success"] for e in episodes)
        result["tasks"].append(
            {
                "name": task.name,
                "language": task_description,
                "num_episodes": len(episodes),
                "successes": successes,
                "success_rate": successes / len(episodes) if episodes else None,
                "episodes": episodes,
            }
        )
        logging.info(f"{task.name}: {successes}/{len(episodes)}")
    result["totals"] = {
        "num_episodes": total_episodes,
        "successes": total_successes,
        "success_rate": total_successes / total_episodes if total_episodes else None,
    }
    logging.info(f"Total success rate: {result['totals']['success_rate']}")
    logging.info(f"Total episodes: {total_episodes}")
    out = pathlib.Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    return result



def _init_states(task_suite, task_id: int):
    """``task_suite.get_task_init_states(task_id)`` on any torch version.

    LIBERO stores each task's initial states as a pickled NumPy array and loads it
    with a bare ``torch.load``; torch 2.6 and later default ``weights_only`` to True
    and refuse that pickle. The files come from the pinned LIBERO checkout, so they
    are loaded as LIBERO wrote them to be (``foldquant.eval_protocol.libero_init_states``
    does the same for the in-process evaluators; this client stays free of foldquant).
    """
    import os

    import torch

    task = task_suite.get_task(task_id)
    path = os.path.join(get_libero_path("init_states"), task.problem_folder, task.init_states_file)
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        # Older client environments predate the weights_only keyword.
        return torch.load(path)

def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """Copied from robosuite (``robosuite/utils/transform_utils.py``)."""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    eval_libero(tyro.cli(Args))
