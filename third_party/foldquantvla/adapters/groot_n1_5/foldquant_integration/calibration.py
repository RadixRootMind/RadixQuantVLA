# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""Calibration observations through the upstream GR00T N1.5 data path.

Upstream's deployment tools feed the policy one dataset step
(``dataset[0]`` in ``deployment_scripts/gr00t_inference.py``, or a handful
for their FP8 calibration). A folded scheme fits SmoothQuant scales and GPTQ
Hessians from what it sees, so FoldQuant samples ``num_samples`` ``(episode,
step)`` pairs spread over the whole dataset instead and hands each one to
``Gr00tPolicy.get_action`` exactly as upstream's ``gr00t.utils.eval`` does:
the raw step ``LeRobotSingleDataset.get_step_data`` returns, through the
policy's own transforms, so the calibration pass and the deployed pass see
the same tensors.

N1.5 keeps the modality config and the transforms in code
(``gr00t.experiment.data_config``), not in the checkpoint; the policy is
assembled the way ``scripts/inference_service.py`` assembles it, from a
``--data-config`` id (default: the LIBERO one).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ._upstream import LIBERO_DATA_CONFIG, ensure_upstream_on_path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SampleId:
    """One calibration observation: which episode (its ``episode_index``), which step."""

    episode: int
    step: int


def resolve_embodiment(model_path: str, embodiment_tag: str | None):
    """``EmbodimentTag`` for the run: explicit, or the single one the checkpoint's metadata names.

    A tag is accepted by value (``new_embodiment``) or by member name
    (``NEW_EMBODIMENT``). N1.5 stores normalisation statistics per embodiment
    in ``experiment_cfg/metadata.json``; a fine-tune with one entry needs no
    ``--embodiment-tag``.
    """
    from gr00t.data.embodiment_tags import EmbodimentTag

    def _tag(value: str):
        try:
            return EmbodimentTag(value)
        except ValueError:
            pass
        try:
            return EmbodimentTag[value.upper()]
        except KeyError:
            raise ValueError(
                f"unknown embodiment tag {value!r}; one of {[t.value for t in EmbodimentTag]}"
            ) from None

    if embodiment_tag is not None:
        return _tag(embodiment_tag)
    metadata_file = Path(model_path) / "experiment_cfg" / "metadata.json"
    if not metadata_file.is_file():
        raise ValueError(f"cannot auto-detect the embodiment: {metadata_file} not found; pass --embodiment-tag")
    tags = sorted(json.loads(metadata_file.read_text()))
    if len(tags) != 1:
        raise ValueError(
            f"experiment_cfg/metadata.json names {tags} embodiments; pass --embodiment-tag explicitly"
        )
    return _tag(tags[0])


def load_policy(
    model_path: str,
    embodiment_tag: str | None,
    device: Any = "cuda",
    *,
    data_config: str = LIBERO_DATA_CONFIG,
    denoising_steps: int | None = None,
):
    """The upstream ``Gr00tPolicy``, assembled as ``scripts/inference_service.py`` assembles it."""
    ensure_upstream_on_path()
    from gr00t.experiment.data_config import load_data_config
    from gr00t.model.policy import Gr00tPolicy

    config = load_data_config(data_config)
    return Gr00tPolicy(
        model_path=model_path,
        embodiment_tag=resolve_embodiment(model_path, embodiment_tag),
        modality_config=config.modality_config(),
        modality_transform=config.transform(),
        denoising_steps=denoising_steps,
        device=device,
    )


def load_dataset(policy, dataset_path: str, video_backend: str = "torchcodec"):
    """The upstream single-embodiment dataset, configured from the policy's own modality config.

    No transforms: :func:`build_observations` reads raw steps and the policy
    applies its transforms inside ``get_action``, as at deployment.
    """
    from gr00t.data.dataset import LeRobotSingleDataset

    return LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=policy.modality_config,
        embodiment_tag=policy.embodiment_tag,
        video_backend=video_backend,
        video_backend_kwargs=None,
        transforms=None,
    )


def plan_samples(
    episode_ids: Sequence[int],
    episode_lengths: Sequence[int],
    num_samples: int,
    *,
    seed: int,
    exclude_episodes: Sequence[int] = (),
    heldout: bool = False,
) -> list[SampleId]:
    """Choose ``(episode, step)`` pairs spread across episodes.

    Episodes are visited round-robin in a seeded shuffled order so the sample
    covers as many episodes (tasks, scenes) as the budget allows; the step
    inside each episode is drawn uniformly. ``exclude_episodes`` keeps a
    verification split disjoint from the calibration split at the episode
    level. A held-out *step* of a calibrated episode is not held out.
    ``heldout`` only changes the stream so the two splits never coincide.
    Episodes are addressed by their ``episode_index`` (upstream's
    ``trajectory_ids``), which need not be contiguous.
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
            samples.append(SampleId(int(episode_ids[i]), int(rng.integers(0, n))))
    return samples


def build_observations(policy, dataset, samples: Sequence[SampleId]) -> list[dict[str, Any]]:
    """The raw dataset step per sample, i.e. what upstream's offline evaluation hands ``get_action``.

    ``get_step_data`` decodes the frames of the requested step only, so
    samples are read in dataset order to keep each episode's decoder warm.
    The step carries the action chunk as well as the observation; the policy
    normalises and ignores it, exactly as in ``gr00t.utils.eval``.
    """
    observations: dict[SampleId, dict[str, Any]] = {}
    for s in sorted(set(samples), key=lambda s: (s.episode, s.step)):
        observations[s] = dataset.get_step_data(s.episode, s.step)
    logger.info("built %d observations from %d episodes", len(samples), len({s.episode for s in samples}))
    return [observations[s] for s in samples]


def sample_observations(
    policy,
    dataset,
    num_samples: int,
    *,
    seed: int,
    exclude_episodes: Sequence[int] = (),
    heldout: bool = False,
) -> tuple[list[SampleId], list[dict[str, Any]]]:
    """:func:`plan_samples` + :func:`build_observations`."""
    samples = plan_samples(
        dataset.trajectory_ids,
        dataset.trajectory_lengths,
        num_samples,
        seed=seed,
        exclude_episodes=exclude_episodes,
        heldout=heldout,
    )
    return samples, build_observations(policy, dataset, samples)


def make_forward_loop(policy, observations: Sequence[dict[str, Any]], *, seed: int) -> Callable[[Any], None]:
    """The ``forward_loop(module)`` every FoldQuant export takes.

    It replays the full policy; the module argument is ignored on purpose:
    the LLM and DiT hooks fire wherever they sit in the graph, and the DiT is
    reached only through the whole denoising loop.

    ``torch.manual_seed(seed + i)`` precedes observation ``i`` every time the
    loop runs. The DiT's action tokens start from ``torch.randn`` noise inside
    ``get_action``, so without it each export (and each calibration pass
    within one export) fits on a different activation set: the Hessians of
    the GPTQ pass would be taken on inputs the SmoothQuant scales never saw,
    and two exports of the same arm would round the DiT differently.
    """

    def _loop(_module: Any) -> None:
        with torch.inference_mode():
            for i, obs in enumerate(observations):
                torch.manual_seed(seed + i)
                policy.get_action(obs)

    return _loop
