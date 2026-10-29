# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""Device-memory report for one arm, in one fresh process.

Two quantities are reported for an engine arm, and both belong in any table:

* **as served** (``--keep-replaced-weights``): what ``serve`` holds today:
  checkpoint on the GPU, engines installed by rebinding ``forward``, replaced
  PyTorch weights still resident (on N1.6 about 3.5 GiB above the floor).
* **floor** (default): the engines plus the PyTorch components the runtime
  still executes. The checkpoint loads on the CPU, replaced parameters become
  ``meta`` tensors, and only the remaining components move to CUDA, so calling
  a replaced module fails loudly.

The number is ``cudaMemGetInfo`` (total minus free: CUDA context and every
allocator) sampled after each timed call; ``steady_used_mib`` is the median of
that plateau, with the torch allocator's peak alongside. Run the eager arm
first with ``--reference-actions`` so engine arms report their decoded-action
cosine against it on the same three observations.

Example::

    python -m foldquant_integration.memory --model-path <ckpt> ... --reference-actions ref.npz
    python -m foldquant_integration.memory --model-path <ckpt> ... --engine-dir exports/w4a4/engines \
        --reference-actions ref.npz --output w4a4_floor.json
    python -m foldquant_integration.memory ... --engine-dir exports/w4a4/engines --keep-replaced-weights
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import statistics
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch

logger = logging.getLogger("foldquant.pi05.memory")


def used_mib() -> float:
    free_b, total_b = torch.cuda.mem_get_info()
    return (total_b - free_b) / 2**20


def _flatten(o: Any) -> np.ndarray:
    if isinstance(o, dict):
        parts = [_flatten(v) for k, v in sorted(o.items()) if not isinstance(v, str)]
        return np.concatenate(parts) if parts else np.zeros(0)
    if isinstance(o, (list, tuple)):
        return np.concatenate([_flatten(v) for v in o]) if o else np.zeros(0)
    if hasattr(o, "detach"):
        o = o.detach().float().cpu().numpy()
    a = np.asarray(o)
    return a.astype(np.float64).ravel() if a.dtype.kind in "fiub" else np.zeros(0)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _tensor_mib(t: torch.Tensor) -> float:
    return t.numel() * t.element_size() / 2**20


def resident_parameter_mib(root: torch.nn.Module, depth: int = 2) -> Dict[str, float]:
    """CUDA-resident parameter+buffer MiB per component (children up to *depth*)."""
    out: Dict[str, float] = {}
    for name, module in root.named_modules():
        if name == "" or name.count(".") >= depth:
            continue
        mib = sum(_tensor_mib(p) for p in module.parameters() if p.device.type == "cuda")
        mib += sum(_tensor_mib(b) for b in module.buffers() if b.device.type == "cuda")
        if mib >= 0.5:
            out[name] = round(mib, 1)
    return out


def materialize_except(root: torch.nn.Module, replaced: Iterable[torch.nn.Module], device: str = "cuda") -> Dict[str, float]:
    """Move every parameter/buffer of *root* to *device* except those owned only by *replaced* subtrees.

    Replaced parameters become ``meta`` tensors (never allocated on the device).
    A parameter shared with a kept module (tied embeddings) is kept. Returns the
    MiB of parameters that were withheld, per replaced subtree root.
    """
    replaced = list(replaced)
    replaced_ids = {id(m) for r in replaced for m in r.modules()}
    kept_params = {
        id(p) for m in root.modules() if id(m) not in replaced_ids for p in m.parameters(recurse=False)
    }
    withheld: Dict[str, float] = {}
    owner = {id(m): r for r in replaced for m in r.modules()}
    for module in root.modules():
        is_replaced = id(module) in replaced_ids
        for key, param in list(module._parameters.items()):
            if param is None:
                continue
            if is_replaced and id(param) not in kept_params:
                label = type(owner[id(module)]).__name__
                withheld[label] = withheld.get(label, 0.0) + _tensor_mib(param)
                module._parameters[key] = torch.nn.Parameter(
                    torch.empty(param.shape, dtype=param.dtype, device="meta"), requires_grad=False
                )
            elif param.device.type != device:
                param.data = param.data.to(device)
        for key, buf in list(module._buffers.items()):
            if buf is None:
                continue
            if is_replaced:
                module._buffers[key] = torch.empty(buf.shape, dtype=buf.dtype, device="meta")
            elif buf.device.type != device:
                module._buffers[key] = buf.to(device)
    gc.collect()
    torch.cuda.empty_cache()
    return {k: round(v, 1) for k, v in withheld.items()}


def cuda_tensor_census(top: int = 8) -> List[Dict[str, Any]]:
    """Largest live CUDA tensors that are not module parameters/buffers (allocator view of the runtime)."""
    rows = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda and not isinstance(obj, torch.nn.Parameter):
                rows.append((_tensor_mib(obj), tuple(obj.shape), str(obj.dtype)))
        except Exception:
            continue
    rows.sort(reverse=True)
    return [{"mib": round(m, 1), "shape": list(s), "dtype": d} for m, s, d in rows[:top]]


def engine_bytes(engine_dir: Optional[str]) -> int:
    if not engine_dir:
        return 0
    return sum(p.stat().st_size for p in Path(engine_dir).glob("*.engine"))


def measure(run, observation, *, warmup: int, iters: int) -> Dict[str, Any]:
    for _ in range(warmup):
        run(observation, 7)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    samples: List[float] = []
    t0 = time.perf_counter()
    for _ in range(iters):
        run(observation, 7)
        torch.cuda.synchronize()
        samples.append(used_mib())
    wall = (time.perf_counter() - t0) / iters * 1000.0
    return {
        "steady_used_mib": round(statistics.median(samples), 1),
        "steady_used_min_mib": round(min(samples), 1),
        "steady_used_max_mib": round(max(samples), 1),
        "torch_peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "torch_allocated_mib": round(torch.cuda.memory_allocated() / 2**20, 1),
        "torch_reserved_mib": round(torch.cuda.memory_reserved() / 2**20, 1),
        "e2e_ms": round(wall, 2),
    }

_MECHANISM = "never-materialize: checkpoint loaded on CPU, replaced PaliGemma decoder layers/norm and the whole action expert (gemma_expert, action/time projections) withheld as meta, SigLIP + embeddings moved to CUDA"


def _load(args, device: str):
    from . import calibration as C

    policy = C.load_policy(args.checkpoint_dir, device=device, compile=False)
    dataset = C.load_dataset(args.dataset_path)
    episodes, _, _ = C.episode_table(dataset)
    keys = C.resolve_keys(dataset)
    observations = C.build_observations(dataset, [C.SampleId(episodes[i], 0) for i in range(args.num_observations)], keys)

    def run(observation, seed: int):
        return _flatten(C.infer(policy, observation, seed=seed))

    return policy, policy._model, observations, run  # noqa: SLF001


def _install(policy, root, args):
    from . import runtime as R

    installed = R.install_engines(policy, args.engine_dir)
    model = R.model_of(policy)
    pwe = model.paligemma_with_expert
    replaced = []
    if "llm" in installed.engines:
        lm = pwe.paligemma.language_model
        inner = getattr(lm, "model", lm)
        replaced += [inner.layers, inner.norm]
    if "expert" in installed.engines:
        replaced.append(pwe.gemma_expert)
        for name in ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out",
                     "action_time_mlp_in", "action_time_mlp_out", "state_proj"):
            if hasattr(model, name):
                replaced.append(getattr(model, name))
    return sorted(installed.engines), replaced


def _after_move(policy, root) -> None:
    policy._pytorch_device = "cuda"  # noqa: SLF001


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--engine-dir", default=None, help="Engine directory; omit for the eager PyTorch arm.")
    parser.add_argument("--keep-replaced-weights", action="store_true",
                        help="Measure the arm as served (checkpoint on the GPU, replaced weights resident).")
    parser.add_argument("--reference-actions", default=None,
                        help="NPZ written by the eager arm and read by engine arms for the cosine check.")
    parser.add_argument("--num-observations", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=60)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    torch.cuda.init()
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    context_mib = used_mib()
    engine_arm = args.engine_dir is not None
    mode = "eager" if not engine_arm else ("as_served" if args.keep_replaced_weights else "floor")
    load_device = "cuda" if (not engine_arm or args.keep_replaced_weights) else "cpu"

    policy, root, observations, run = _load(args, load_device)
    withheld: Dict[str, float] = {}
    installed_engines: List[str] = []
    if engine_arm:
        installed_engines, replaced = _install(policy, root, args)
        if mode == "floor":
            withheld = materialize_except(root, replaced, "cuda")
            _after_move(policy, root)
        else:
            torch.cuda.empty_cache()
    torch.cuda.synchronize()
    after_load_mib = used_mib()

    result: Dict[str, Any] = {
        "family": "pi05",
        "mode": mode,
        "engine_dir": args.engine_dir,
        "engines": installed_engines,
        "mechanism": _MECHANISM if mode == "floor" else ("resident (as served)" if engine_arm else "eager"),
        "cuda_context_mib": round(context_mib, 1),
        "device_used_after_load_mib": round(after_load_mib, 1),
        "torch_allocated_after_load_mib": round(torch.cuda.memory_allocated() / 2**20, 1),
        "withheld_parameter_mib": withheld,
        "resident_parameter_mib": resident_parameter_mib(root),
        "engine_bytes": engine_bytes(args.engine_dir),
    }
    actions = [run(o, 42 + i) for i, o in enumerate(observations)]
    if args.reference_actions:
        ref_path = Path(args.reference_actions)
        if engine_arm:
            if not ref_path.is_file():
                raise FileNotFoundError(f"{ref_path}: run the eager arm first with --reference-actions")
            ref = np.load(ref_path)
            result["cosine_vs_eager"] = [round(_cosine(ref[f"a{i}"], a), 5) for i, a in enumerate(actions)]
        else:
            np.savez(ref_path, **{f"a{i}": a for i, a in enumerate(actions)})
    result.update(measure(run, observations[0], warmup=args.warmup, iters=args.iters))
    result["cuda_tensor_census"] = cuda_tensor_census()
    text = json.dumps(result, indent=1)
    print(text)
    if args.output:
        Path(args.output).write_text(text)
    logger.info(
        "%s %s: context %.0f MiB | after load %.0f | steady %.0f (min %.0f, max %.0f) | torch peak %.0f | engines on disk %.0f MB",
        result["family"], mode, context_mib, after_load_mib, result["steady_used_mib"],
        result["steady_used_min_mib"], result["steady_used_max_mib"], result["torch_peak_allocated_mib"],
        result["engine_bytes"] / 1e6,
    )


if __name__ == "__main__":
    main()
