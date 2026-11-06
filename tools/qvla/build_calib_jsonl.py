"""Build a small OpenVLA-style calibration JSONL from LIBERO observations."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


def _suite_to_task_suite(name: str) -> str:
    mapping = {
        "object": "libero_object",
        "spatial": "libero_spatial",
        "goal": "libero_goal",
        "long": "libero_10",
    }
    return mapping.get(name, name)


def _prompt(task_description: str) -> str:
    return f"In: What action should the robot take to {task_description.lower()}?\nOut:"


def _pick_image(obs: Any, preferred_key: str | None = None) -> np.ndarray:
    if isinstance(obs, tuple) and obs:
        obs = obs[0]
    if not isinstance(obs, dict):
        raise TypeError(f"expected observation dict, got {type(obs)!r}")
    keys: Iterable[str]
    if preferred_key:
        keys = [preferred_key]
    else:
        keys = [
            "full_image",
            "agentview_image",
            "robot0_agentview_left_image",
            "robot0_eye_in_hand_image",
            "eye_in_hand_image",
        ]
    for key in keys:
        value = obs.get(key)
        if isinstance(value, np.ndarray) and value.ndim >= 3:
            img = value
            if img.shape[-1] >= 3:
                return img[..., :3]
    for key, value in obs.items():
        if isinstance(value, np.ndarray) and value.ndim == 3 and value.shape[-1] >= 3:
            return value[..., :3]
    raise KeyError(f"no RGB image found in observation keys: {sorted(obs.keys())}")


def _load_benchmark(task_suite_name: str):
    from libero.libero.benchmark import get_benchmark

    benchmark_cls = get_benchmark(task_suite_name)
    return benchmark_cls()


def _task_ids(raw: str | None, n_tasks: int) -> list[int]:
    if raw:
        return [int(x) for x in raw.replace(",", " ").split() if x.strip()]
    return list(range(n_tasks))


def _make_env(task, resolution: int):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bddl_root = get_libero_path("bddl_files")
    bddl_file = getattr(task, "bddl_file", None) or getattr(task, "bddl_file_name", None)
    problem_folder = getattr(task, "problem_folder", None)
    if problem_folder:
        bddl_path = os.path.join(bddl_root, problem_folder, bddl_file)
    else:
        bddl_path = os.path.join(bddl_root, bddl_file)
    if not os.path.isfile(bddl_path):
        raise FileNotFoundError(f"BDDL file not found: {bddl_path}")
    return OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=resolution,
        camera_widths=resolution,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task-suite-name", "--suite", dest="task_suite_name", default="libero_spatial")
    p.add_argument("--out-jsonl", required=True)
    p.add_argument("--image-dir", default=None)
    p.add_argument("--num-samples", type=int, default=32)
    p.add_argument("--task-ids", default=None)
    p.add_argument("--image-key", default=None)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    task_suite_name = _suite_to_task_suite(args.task_suite_name)
    out_jsonl = Path(args.out_jsonl)
    image_dir = Path(args.image_dir) if args.image_dir else out_jsonl.parent / "images"
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    benchmark = _load_benchmark(task_suite_name)
    n_tasks = benchmark.get_num_tasks()
    records = []

    for task_id in _task_ids(args.task_ids, n_tasks):
        if len(records) >= args.num_samples:
            break
        task = benchmark.get_task(task_id)
        task_description = getattr(task, "language", None) or getattr(task, "task_description", None) or str(task)
        env = _make_env(task, resolution=args.resolution)
        try:
            try:
                env.seed(args.seed + task_id)
            except Exception:
                pass
            obs = env.reset()
            img = _pick_image(obs, preferred_key=args.image_key)
            img_path = image_dir / f"task_{task_id:03d}_sample_000.png"
            Image.fromarray(img.astype("uint8")).save(img_path)
            records.append({"image": str(img_path.resolve()), "text": _prompt(task_description), "task_id": task_id})
        finally:
            try:
                env.close()
            except Exception:
                pass

    if not records:
        raise RuntimeError("no calibration records were generated")
    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} records -> {out_jsonl}")


if __name__ == "__main__":
    from oft_cli import maybe_dispatch
    if not maybe_dispatch("build_calib"):
        main()
