# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""Fixed-initial-state LIBERO rollouts for the GR00T families.

The upstream GR00T ``LiberoEnv`` resets robosuite without a seed and never
applies LIBERO's stored initial states, so its success rates are over random
placements. The paper's protocol starts episode ``i`` of every task from
LIBERO's initial state ``i`` and lets the scene settle for ten no-op steps.
:func:`fixed_init_state_env` wraps the upstream env to do that, and
:func:`rollout_task` drives it through the upstream ``MultiStepWrapper`` one
episode at a time, recording the initial state each episode used.

Imports gymnasium lazily; the module itself is importable without it.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .eval_protocol import LIBERO_DUMMY_ACTION

__all__ = ["fixed_init_state_env", "rollout_task"]


def fixed_init_state_env(env: Any, init_states: Any, settle_steps: int, seed: Optional[int] = None) -> Any:
    """Wrap an upstream ``LiberoEnv`` so ``reset(options={"init_state_id": i})`` starts from state *i*.

    Without ``options`` the episodes take states ``0, 1, 2, ...`` in order.
    After ``set_init_state`` the wrapper steps LIBERO's no-op action
    ``settle_steps`` times, as LIBERO's own evaluators do, and reports
    ``init_state_id`` in the reset info. *seed*, when given, seeds the
    simulator once for the task, the way LIBERO's and openpi's evaluators do
    (it affects object placement even from a stored initial state).
    """
    import gymnasium as gym

    class FixedInitStateEnv(gym.Wrapper):
        def __init__(self, inner: Any):
            super().__init__(inner)
            self._states = init_states
            self._next = 0
            self.init_state_id: Optional[int] = None
            if seed is not None:
                inner.unwrapped._env.seed(seed)

        def reset(self, seed=None, options=None):
            base = self.env.unwrapped  # the upstream LiberoEnv
            raw = base._env  # LIBERO's OffScreenRenderEnv
            self.env.reset(seed=seed, options=options)
            idx = None if not options else options.get("init_state_id")
            if idx is None:
                idx = self._next
            idx = int(idx) % len(self._states)
            self._next = idx + 1
            self.init_state_id = idx
            obs = raw.set_init_state(self._states[idx])
            for _ in range(settle_steps):
                obs, _reward, _done, _info = raw.step(list(LIBERO_DUMMY_ACTION))
            observation = base._process_observation(obs)
            info = {"success": raw.check_success(), "init_state_id": idx}
            return observation, info

    return FixedInitStateEnv(env)


def _batched(obs: Dict[str, Any], space: Any = None) -> Dict[str, Any]:
    """One environment's observation in the ``(B=1, ...)`` layout the sim policy wrapper expects.

    Arrays are cast to the dtype of their key in *space*, as upstream's vector
    env does; the policy accepts float32 state only.
    """
    subspaces = getattr(space, "spaces", None) or {}
    out: Dict[str, Any] = {}
    for key, value in obs.items():
        if isinstance(value, str):
            out[key] = [value]
        else:
            dtype = getattr(subspaces.get(key), "dtype", None)
            out[key] = np.asarray(value, dtype=dtype if isinstance(dtype, np.dtype) else None)[None]
    return out


def rollout_task(
    policy: Any,
    make_env: Callable[[], Any],
    multistep_wrapper: Callable[..., Any],
    init_states: Any,
    n_episodes: int,
    max_episode_steps: int,
    n_action_steps: int,
    settle_steps: int,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Roll out ``n_episodes`` episodes of one task, episode ``i`` from initial state ``i``.

    *policy* is the upstream ``Gr00tSimPolicyWrapper`` (or anything with its
    ``get_action`` contract); *make_env* builds the upstream LIBERO gym env;
    *multistep_wrapper* is the upstream ``MultiStepWrapper`` class. Returns one
    record per episode: ``episode``, ``init_state_id``, ``success`` and the
    number of environment steps taken after settling.
    """
    fixed = fixed_init_state_env(make_env(), init_states, settle_steps, seed)
    env = multistep_wrapper(
        fixed,
        video_delta_indices=np.array([0]),
        state_delta_indices=np.array([0]),
        n_action_steps=n_action_steps,
        max_episode_steps=max_episode_steps,
        terminate_on_success=True,
    )
    records = []
    try:
        for episode in range(n_episodes):
            obs, info = env.reset(options={"init_state_id": episode})
            success = bool(np.any(info.get("success", False)))
            done = False
            while not done:
                actions, _ = policy.get_action(_batched(obs, env.observation_space))
                action = {k: np.asarray(v)[0] for k, v in actions.items()}
                obs, _reward, done, _truncated, info = env.step(action)
                success |= bool(np.any(info.get("success", False)))
            records.append(
                {
                    "episode": episode,
                    "init_state_id": int(fixed.init_state_id),
                    "success": success,
                    "steps": int(len(env.reward)),
                }
            )
    finally:
        env.close()
    return records
