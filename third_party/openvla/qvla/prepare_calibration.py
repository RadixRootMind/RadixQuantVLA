"""Prepare the validated single-camera OpenVLA training-frame calibration recipe."""

import argparse
import json
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qvla.artifacts import file_hash, now, write_json


def prepare(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
    import numpy as np
    import tensorflow as tf
    import tensorflow_datasets as tfds
    from PIL import Image

    tf.config.set_visible_devices([], "GPU")
    output = args.out_dir.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    indices = None
    if args.indices_jsonl:
        indices = [json.loads(line) for line in args.indices_jsonl.read_text().splitlines() if line.strip()]
        if not indices:
            raise ValueError("Empty index manifest")
        identities = [(row["suite"], row["episode_index"]) for row in indices]
        if len(set(identities)) != len(identities):
            raise ValueError("Index manifest must contain distinct training trajectories")
        suites = sorted({row["suite"] for row in indices})
    else:
        suites = [name if name.startswith("libero_") else f"libero_{name}" for name in args.suites.split(",")]
        if len(set(suites)) != len(suites) or any(
            name not in ("libero_spatial", "libero_object", "libero_goal", "libero_10") for name in suites
        ):
            raise ValueError("Choose distinct suites from spatial,object,goal,10")
    output.mkdir(parents=True, exist_ok=False)
    (output / "images").mkdir()
    rng = np.random.default_rng(args.seed)
    records = {}
    next_index = 0
    report = {
        "status": "running",
        "started": now(),
        "completed": 0,
        "seed": args.seed,
        "split": "train",
        "suites": suites,
        "recipe": "Suite-balanced distinct training trajectories, one random raw frame per trajectory",
        "preprocessing": "Front camera: Lanczos224 resize,90%-area crop; processor does not crop again",
        "scope": "Local calibration recipe; unpublished author indices/instruction-only subset not supplied",
    }
    if indices is not None:
        report["source_indices_sha256"] = file_hash(args.indices_jsonl)
    write_json(output / "status.json", report)
    try:
        for suite_id, suite in enumerate(suites):
            builder = tfds.builder_from_directory(str(data_root / f"{suite}_no_noops/1.0.0"))
            count = builder.info.splits["train"].num_examples
            if indices is not None:
                wanted = {
                    row["episode_index"]: (index, row) for index, row in enumerate(indices) if row["suite"] == suite
                }
            else:
                quota = args.trajectories // len(suites) + (suite_id < args.trajectories % len(suites))
                if quota > count:
                    raise ValueError(f"Requested {quota} distinct trajectories from {suite}, available={count}")
                wanted = {int(index): None for index in rng.choice(count, size=quota, replace=False)}
            if any(type(index) is not int or not 0 <= index < count for index in wanted):
                raise ValueError(f"Invalid training episode index in {suite}")
            if not wanted:
                continue
            options = tf.data.Options()
            options.threading.private_threadpool_size = 2
            dataset = builder.as_dataset(split="train", shuffle_files=False).with_options(options)
            found = 0
            for episode_index, episode in enumerate(dataset):
                if episode_index not in wanted:
                    continue
                if indices is None:
                    steps = list(tfds.as_numpy(episode["steps"]))
                    if not steps:
                        raise ValueError(f"Empty training trajectory: {suite}/{episode_index}")
                    step_index = int(rng.integers(len(steps)))
                    step = steps[step_index]
                    index = next_index
                    next_index += 1
                else:
                    index, original = wanted[episode_index]
                    step_index = original["step_index"]
                    if type(step_index) is not int or step_index < 0:
                        raise ValueError("Invalid frame index")
                    selected = list(tfds.as_numpy(episode["steps"].skip(step_index).take(1)))
                    if len(selected) != 1:
                        raise ValueError(f"Frame missing: {suite}/{episode_index}/{step_index}")
                    step = selected[0]
                instruction = step["language_instruction"]
                instruction = instruction.decode("utf-8") if isinstance(instruction, bytes) else str(instruction)
                if indices is not None and original.get("instruction", instruction) != instruction:
                    raise ValueError("Instruction does not match the requested training frame")
                state = np.asarray(step["observation"]["state"])
                if state.shape != (8,) or not np.isfinite(state).all():
                    raise ValueError("Expected finite 8D proprio")
                value = {
                    "suite": suite,
                    "episode_index": episode_index,
                    "step_index": step_index,
                    "instruction": instruction,
                    "state": state.tolist(),
                }
                value["text"] = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
                value["preprocessed"] = True
                for key in ("image",):
                    image = np.asarray(step["observation"][key])
                    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
                        raise ValueError(f"Expected raw uint8 RGB {key}")
                    path = output / "images" / f"{suite}_{episode_index:05d}_{step_index:04d}_{key}.png"
                    image = tf.image.resize(image, [224, 224], method="lanczos3", antialias=True)
                    image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8)
                    image = tf.image.convert_image_dtype(image, tf.float32)
                    offset = (1.0 - np.sqrt(0.9)) / 2
                    image = tf.image.crop_and_resize(
                        image[None], [[offset, offset, 1 - offset, 1 - offset]], [0], [224, 224]
                    )[0]
                    image = tf.image.convert_image_dtype(tf.clip_by_value(image, 0, 1), tf.uint8, saturate=True).numpy()
                    Image.fromarray(image).save(path)
                    value[key] = str(path.relative_to(output))
                    value[key + "_sha256"] = file_hash(path)
                records[index] = value
                found += 1
                report["completed"] += 1
                if report["completed"] % 16 == 0:
                    write_json(output / "status.json", report)
                    print(f"Prepared {report['completed']} records", flush=True)
                if found == len(wanted):
                    break
            if found != len(wanted):
                raise ValueError(f"Missing requested training trajectories in {suite}")
        expected = len(indices) if indices is not None else args.trajectories
        if len(records) != expected:
            raise ValueError(f"Prepared {len(records)}/{expected} records")
        rows = [records[index] for index in range(expected)]
        if indices is None:
            rng.shuffle(rows)
        manifest = output / args.manifest_name
        manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        report.update(status="complete", finished=now(), samples=len(rows), records_jsonl_sha256=file_hash(manifest))
        write_json(output / "status.json", report)
        print(f"Wrote {len(rows)} records to {manifest}")
    except Exception as exc:
        report.update(status="failed", finished=now(), error=f"{type(exc).__name__}: {exc}")
        write_json(output / "status.json", report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--trajectories", type=int, default=512)
    parser.add_argument("--suites", default="spatial,object,goal,10")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--manifest_name", default="records.jsonl")
    parser.add_argument("--indices_jsonl", type=Path, help="Reuse local suite/episode/frame indices in this order")
    args = parser.parse_args()
    if args.trajectories < 1:
        parser.error("trajectories must be positive")
    if Path(args.manifest_name).name != args.manifest_name or not args.manifest_name.endswith(".jsonl"):
        parser.error("manifest_name must be a JSONL basename")
    prepare(args)


if __name__ == "__main__":
    main()
