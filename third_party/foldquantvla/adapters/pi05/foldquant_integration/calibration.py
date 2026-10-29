# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""Calibration observations through the upstream openpi policy path.

A folded scheme fits SmoothQuant scales and GPTQ Hessians from what it sees,
so FoldQuant samples ``num_samples`` ``(episode, step)`` pairs spread over a
LeRobot dataset and hands each one to ``Policy.infer`` in exactly the shape
upstream's LIBERO client (``examples/libero/main.py``) sends over the wire:
the two 224x224 uint8 camera frames after ``resize_with_pad``, the 8-d
proprioceptive state and the task prompt. The policy's own transforms
(``LiberoInputs`` -> normalise -> tokenise) run inside ``infer`` as at
deployment, so the calibration pass and the served pass see the same tensors.

The dataset is opened with the ``lerobot`` package upstream pins; the frames
come out of its video decoder as the float CHW tensors ``LiberoInputs`` also
accepts, and are turned back into the client's uint8 HWC frames here so that
the resize happens on the same pixels the client resizes.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ._upstream import LIBERO_TRAIN_CONFIG, load_plugin

logger = logging.getLogger(__name__)

#: The LeRobot feature each client observation key is read from, when the train
#: config does not say. These are LIBERO's column names; :func:`dataset_keys`
#: prefers the checkpoint's own answer and falls back to these.
IMAGE_KEY = "observation.images.image"
WRIST_KEY = "observation.images.wrist_image"
STATE_KEY = "observation.state"
#: Side of the frames upstream's LIBERO client sends (``Args.resize_size``).
CLIENT_IMAGE_SIZE = 224


@dataclass(frozen=True)
class SampleId:
    """One calibration observation: which episode (its ``episode_index``), which step."""

    episode: int
    step: int


def dataset_keys(dataset, train_config=None) -> dict[str, str]:
    """Which dataset column feeds each client observation key.

    The defaults above are the LIBERO columns. They are tried first and kept
    whenever the dataset actually has them; only a key the dataset does not
    carry is looked up in the train config's repack transform, which maps the
    wire keys a client sends onto the columns that config was trained against.

    Deriving from the config *alone* would be wrong: ``pi05_libero`` repacks
    from ``image`` / ``wrist_image`` / ``state``, the column names of the
    ``physical-intelligence/libero`` release, while the LeRobot conversion this
    integration calibrates on stores ``observation.images.image`` and friends.
    The dataset is the authority on its own columns; the config is the fallback
    for a dataset that names them differently.
    """
    keys = {"observation/image": IMAGE_KEY, "observation/wrist_image": WRIST_KEY, "observation/state": STATE_KEY}
    have = set(getattr(dataset, "features", None) or dataset.meta.info.get("features", {}))
    missing = [k for k, column in keys.items() if column not in have]
    if not missing:
        return keys
    from_config: dict[str, str] = {}
    if train_config is not None:
        try:
            data = train_config.data.create(train_config.assets_dirs, train_config.model)
            for group in data.repack_transforms.inputs:
                structure = getattr(group, "structure", None)
                if isinstance(structure, dict):
                    from_config.update({k: v for k, v in structure.items() if isinstance(v, str)})
        except Exception:  # a config that cannot be instantiated here is simply no help
            logger.debug("could not read the repack transform", exc_info=True)
    for k in missing:
        column = from_config.get(k)
        if column is None or column not in have:
            raise KeyError(
                f"the dataset has no column for {k}: tried {keys[k]!r}"
                + (f" and {column!r} from the train config" if column else " and the train config named none")
                + f". Columns present: {sorted(have)}"
            )
        keys[k] = column
    logger.info("dataset columns: %s", keys)
    return keys


def resolve_keys(dataset, config_name: str = LIBERO_TRAIN_CONFIG) -> dict[str, str]:
    """:func:`dataset_keys` for a config name, without building the policy."""
    from openpi.training import config as _config

    load_plugin()
    try:
        train_config = _config.get_config(config_name)
    except Exception:
        train_config = None
    return dataset_keys(dataset, train_config)


def load_policy(
    checkpoint_dir: str, *, config_name: str = LIBERO_TRAIN_CONFIG, device: str = "cuda", compile: bool = True
):
    """The upstream ``Policy`` over the PyTorch checkpoint, as ``scripts/serve_policy.py`` builds it.

    ``create_trained_policy`` picks the PyTorch model when ``model.safetensors``
    is present and a JAX one otherwise; only the former can host the engines,
    so a JAX-only checkpoint is refused here rather than silently served.

    ``compile=False`` clears the train config's ``pytorch_compile_mode`` (upstream
    serves under ``max-autotune``, i.e. CUDA graphs). Calibration needs that:
    the export hooks keep activations across forward passes, and a CUDA-graph
    output is overwritten by the next replay.
    """
    import dataclasses

    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    checkpoint = Path(checkpoint_dir)
    if not (checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError(
            f"{checkpoint} holds no model.safetensors. FoldQuant integrates the PyTorch model; convert the "
            "JAX checkpoint first (examples/convert_jax_model_to_pytorch.py)"
        )
    plugin = load_plugin()
    if plugin:
        logger.info("loaded config plugin %s", plugin)
    train_config = _config.get_config(config_name)
    if not compile:
        model_config = dataclasses.replace(train_config.model, pytorch_compile_mode=None)
        train_config = dataclasses.replace(train_config, model=model_config)
    from foldquant.quantized_checkpoint import is_quantized_checkpoint

    if is_quantized_checkpoint(checkpoint):
        # A quantized checkpoint holds codes and scales under other names in place of the
        # quantized projections' `weight`s; openpi's strict load would refuse the missing keys.
        # Those projections are replaced by fake-quant layers right after loading
        # (foldquant.quantized.install_on_policy), so their initial values do not matter.
        import safetensors.torch as _st
        from unittest import mock

        _load = _st.load_model

        def _lenient(model, filename, strict=True, **kw):
            return _load(model, filename, strict=False, **kw)

        with mock.patch.object(_st, "load_model", _lenient):
            policy = _policy_config.create_trained_policy(train_config, checkpoint, pytorch_device=device)
    else:
        policy = _policy_config.create_trained_policy(train_config, checkpoint, pytorch_device=device)
    if not policy._is_pytorch_model:  # noqa: SLF001
        raise RuntimeError("create_trained_policy returned a JAX policy; the engines need the PyTorch model")
    return policy


def load_dataset(dataset_path: str, video_backend: str | None = None):
    """A local LeRobot dataset through the ``lerobot`` package upstream pins.

    The directory is passed as ``root``; ``repo_id`` is only a label here (no
    hub access happens when every file is present locally).

    *video_backend* selects the decoder; ``None`` takes lerobot's default. The
    default (torchcodec, where available) can land on a neighbouring frame and
    then fail lerobot's 1e-4 s timestamp check on datasets whose videos are
    otherwise exact -- ``"pyav"`` decodes those. Loosening the tolerance instead
    would accept the wrong frame silently, which is the one outcome calibration
    cannot afford.
    """
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    root = Path(dataset_path).expanduser().resolve()
    if not (root / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"{root} is not a LeRobot dataset (no meta/info.json)")
    # lerobot assumes episodes 0..total-1 unless told otherwise, and fetches any it
    # cannot find from the Hub. A subset (a calibration split copied out of a larger
    # dataset) keeps its original, sparse indices, so name them.
    import json

    with open(root / "meta" / "episodes.jsonl") as f:
        indices = sorted(int(json.loads(line)["episode_index"]) for line in f if line.strip())
    episodes = None if indices == list(range(len(indices))) else indices
    # lerobot's default 1e-4 s timestamp tolerance rejects videos whose frames sit a
    # fraction of a millisecond off the nominal grid (0.7 ms on the SO101 recordings).
    # A quarter of the frame interval still admits only the nearest frame: its
    # neighbours are a full interval away, so no other frame can pass instead.
    with open(root / "meta" / "info.json") as f:
        fps = float(json.load(f)["fps"])
    return LeRobotDataset(
        repo_id=root.name, root=root, episodes=episodes, video_backend=video_backend, tolerance_s=0.25 / fps
    )


def episode_table(dataset) -> tuple[list[int], list[int], list[int]]:
    """``(episode_index, length, first_global_index)`` per episode, in dataset order."""
    episodes = dataset.meta.episodes
    ids = sorted(int(e) for e in episodes)
    lengths = [int(episodes[e]["length"]) for e in ids]
    starts = [int(v) for v in dataset.episode_data_index["from"].tolist()]
    if len(starts) != len(ids):
        raise RuntimeError(f"episode_data_index has {len(starts)} entries for {len(ids)} episodes")
    return ids, lengths, starts


def plan_samples(
    episode_ids: Sequence[int],
    episode_lengths: Sequence[int],
    num_samples: int,
    *,
    seed: int,
    exclude_episodes: Sequence[int] = (),
    heldout: bool = False,
    first_step: Callable[[int], int] | None = None,
) -> list[SampleId]:
    """Choose ``(episode, step)`` pairs spread across episodes.

    Episodes are visited round-robin in a seeded shuffled order so the sample
    covers as many episodes (tasks, scenes) as the budget allows; the step
    inside each episode is drawn uniformly from ``[first_step(episode), length)``
    (``first_step`` defaults to 0, which draws exactly what it always drew).
    ``exclude_episodes`` keeps a verification split disjoint from the
    calibration split at the episode level. A held-out *step* of a calibrated
    episode is not held out. ``heldout`` only changes the stream so the two
    splits never coincide.
    """
    if len(episode_ids) != len(episode_lengths):
        raise ValueError("episode_ids and episode_lengths differ in length")
    rng = np.random.default_rng(seed + (1_000_003 if heldout else 0))
    excluded = {int(e) for e in exclude_episodes}
    candidates = [i for i in range(len(episode_ids)) if int(episode_ids[i]) not in excluded]
    if not candidates:
        raise ValueError("no episodes left to sample from after exclusions")
    rng.shuffle(candidates)
    samples: list[SampleId] = []
    while len(samples) < num_samples:
        for i in candidates:
            if len(samples) >= num_samples:
                break
            n = int(episode_lengths[i])
            lo = int(first_step(int(episode_ids[i]))) if first_step is not None else 0
            if lo >= n:
                raise ValueError(f"episode {episode_ids[i]}: no step has a video frame (first {lo}, length {n})")
            samples.append(SampleId(int(episode_ids[i]), int(rng.integers(lo, n))))
    return samples


def first_decodable_step(dataset) -> Callable[[int], int]:
    """Per episode, the first step every camera video holds a frame for.

    Some recordings start their video a few frames after the episode's first
    timestamp (the SO101 set: first frame at 0.066 s, 2 frames at 30 fps).
    lerobot refuses those steps, and loosening its tolerance would hand back a
    frame from another step. Such steps are left out of the draw instead. Read
    from the first decoded frame of each video, once per episode, only for the
    episodes a plan visits; a video that starts at 0 gives 0.
    """
    import av

    fps = float(dataset.fps)
    tol = float(dataset.tolerance_s)
    cache: dict[int, int] = {}

    def first(episode: int) -> int:
        if episode not in cache:
            start = 0.0
            for key in dataset.meta.video_keys:
                path = Path(dataset.root) / dataset.meta.get_video_file_path(episode, key)
                with av.open(str(path)) as container:
                    frame = next(container.decode(video=0))
                    start = max(start, float(frame.time or 0.0))
            cache[episode] = max(0, int(np.ceil((start - tol) * fps - 1e-9)))
        return cache[episode]

    return first


def _client_frame(frame: Any) -> np.ndarray:
    """A decoded LeRobot frame -> the uint8 224x224 HWC array the LIBERO client sends."""
    from openpi_client import image_tools

    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    frame = np.asarray(frame)
    if frame.ndim == 3 and frame.shape[0] in (1, 3) and frame.shape[-1] not in (1, 3):
        frame = np.transpose(frame, (1, 2, 0))  # CHW (lerobot) -> HWC (client)
    if np.issubdtype(frame.dtype, np.floating):
        # lerobot decodes to float32 in [0, 1]: undo the /255 exactly.
        frame = np.rint(frame * 255.0).clip(0, 255).astype(np.uint8)
    return image_tools.resize_with_pad(frame, CLIENT_IMAGE_SIZE, CLIENT_IMAGE_SIZE)


def prompt_of(dataset, item: dict[str, Any]) -> str:
    """The task text of a dataset item (``task`` when materialised, else through ``task_index``)."""
    if isinstance(item.get("task"), str):
        return item["task"]
    task_index = int(item["task_index"])
    tasks = dataset.meta.tasks
    return str(tasks[task_index])


def client_observation(dataset, item: dict[str, Any], keys: dict[str, str] | None = None) -> dict[str, Any]:
    """One dataset item in the wire format of ``examples/libero/main.py``.

    *keys* comes from :func:`dataset_keys`; omitted, LIBERO's columns are read.
    """
    k = keys or {"observation/image": IMAGE_KEY, "observation/wrist_image": WRIST_KEY,
                 "observation/state": STATE_KEY}
    return {
        "observation/image": _client_frame(item[k["observation/image"]]),
        "observation/wrist_image": _client_frame(item[k["observation/wrist_image"]]),
        "observation/state": np.asarray(item[k["observation/state"]], dtype=np.float32),
        "prompt": prompt_of(dataset, item),
    }


def build_observations(
    dataset, samples: Sequence[SampleId], keys: dict[str, str] | None = None
) -> list[dict[str, Any]]:
    """The client-format observation per sample.

    Items are read in dataset order to keep each episode's video decoder warm;
    the returned list follows *samples*.
    """
    ids, lengths, starts = episode_table(dataset)
    start_of = dict(zip(ids, starts, strict=True))
    length_of = dict(zip(ids, lengths, strict=True))
    observations: dict[SampleId, dict[str, Any]] = {}
    for s in sorted(set(samples), key=lambda s: (s.episode, s.step)):
        if s.episode not in start_of:
            raise KeyError(f"episode {s.episode} is not in the dataset")
        if not 0 <= s.step < length_of[s.episode]:
            raise IndexError(f"step {s.step} outside episode {s.episode} (length {length_of[s.episode]})")
        observations[s] = client_observation(dataset, dataset[start_of[s.episode] + s.step], keys)
    logger.info("built %d observations from %d episodes", len(samples), len({s.episode for s in samples}))
    return [observations[s] for s in samples]


def sample_observations(
    dataset,
    num_samples: int,
    *,
    seed: int,
    exclude_episodes: Sequence[int] = (),
    heldout: bool = False,
    keys: dict[str, str] | None = None,
) -> tuple[list[SampleId], list[dict[str, Any]]]:
    """:func:`plan_samples` + :func:`build_observations`."""
    ids, lengths, _starts = episode_table(dataset)
    samples = plan_samples(ids, lengths, num_samples, seed=seed, exclude_episodes=exclude_episodes, heldout=heldout,
                           first_step=first_decodable_step(dataset))
    return samples, build_observations(dataset, samples, keys)


def action_noise(policy, seed: int) -> np.ndarray:
    """The flow-matching starting noise for one ``infer`` call, drawn from a seeded generator.

    Upstream draws it inside ``sample_actions`` from the global torch RNG;
    passing it explicitly makes an observation's replay identical across
    calibration passes and across the PyTorch / engine verification runs.
    """
    config = policy._model.config  # noqa: SLF001
    rng = np.random.default_rng(seed)
    return rng.standard_normal((int(config.action_horizon), int(config.action_dim)), dtype=np.float32)


def infer(policy, observation: dict[str, Any], *, seed: int) -> np.ndarray:
    """``Policy.infer`` on a fresh copy of *observation* with seeded noise; returns the action chunk."""
    torch.manual_seed(seed)
    out = policy.infer(dict(observation), noise=action_noise(policy, seed))
    return np.asarray(out["actions"])


def make_forward_loop(policy, observations: Sequence[dict[str, Any]], *, seed: int) -> Callable[[Any], None]:
    """The ``forward_loop(module)`` every FoldQuant export takes.

    It replays the full policy; the module argument is ignored on purpose:
    the LLM and expert hooks fire wherever they sit in the graph, and the
    expert is reached only through the whole Euler loop. Observation ``i``
    always runs under seed ``seed + i`` so every calibration pass of an export
    (scales, then Hessians) and every export of the same arm fits the same
    activations.
    """

    def _loop(_module: Any) -> None:
        with torch.inference_mode():
            for i, obs in enumerate(observations):
                infer(policy, obs, seed=seed + i)

    return _loop
