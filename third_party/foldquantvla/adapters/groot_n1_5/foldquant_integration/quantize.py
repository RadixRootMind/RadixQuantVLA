# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""Quantize a GR00T N1.5 checkpoint with FoldQuant and save the quantized model.

Calibrates the LLM and DiT folds on ``--num-calib`` dataset observations
and writes ``--output-dir``, the quantized model::

    the base checkpoint with each quantized projection's weight
                       replaced by its integer codes (qweight) and per-row scale
                       (weight_scale), plus the SmoothQuant scales and fold settings

The quantized model runs in PyTorch with fake-quant layers in place of the
quantized projections (``serve``, ``verify`` and ``eval_libero`` load it like a
checkpoint), can be pushed to the Hugging Face Hub, and is the input of the
rest of the pipeline: ``export`` (quantized model -> plugin ONNX) and
``build_engines`` (ONNX -> TensorRT engines).

Both graphs use FoldQuant's I/O contracts (see ``runtime.py``), not the
fp16 three-input DiT of upstream's ``deployment_scripts/export_onnx.py``;
the two are not interchangeable. The LLM graph takes ``inputs_embeds`` and
``attention_mask`` and returns the decoder's ``hidden_states`` at the entry
the Eagle backbone reads (``hidden_states[select_layer]``, 12 for the
released checkpoints). Whether that entry is the post-norm stream is decided
by the transformers release upstream pins, not by this code: the export
checks it on a real forward and emits the final RMSNorm accordingly.

Example::

    python -m foldquant_integration.quantize \\
        --model-path <GR00T-N1.5 LIBERO checkpoint> --embodiment-tag new_embodiment \\
        --dataset-path <calibration dataset> \\
        --output-dir exports/n15_w4a4/quantized \\
        --llm-scheme w4a4_srg --dit-scheme w4a4_shg --cascade
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import tyro
from foldquant import schemes
from foldquant.export import export_dit, export_llm, install_llm_emulation
from foldquant.provenance import public_path
from foldquant.quantized_checkpoint import is_quantized_checkpoint

from . import calibration
from ._upstream import LIBERO_DATA_CONFIG
from .runtime import dit_module, llm_module, select_layer_of

logger = logging.getLogger("foldquant.groot_n1_5.quantize")

_NONE = ("", "none")


@dataclass
class QuantizeConfig:
    model_path: str
    """Checkpoint directory or Hugging Face id (as for the upstream tools)."""

    dataset_path: str
    """LeRobot-format dataset the calibration observations are drawn from."""

    output_dir: str
    """Destination directory of the quantized model (written as the checkpoint itself)."""

    embodiment_tag: str | None = None
    """Embodiment tag; read off experiment_cfg/metadata.json when it lists exactly one."""

    data_config: str = LIBERO_DATA_CONFIG
    """``module:Class`` data config (modality config + transforms), as upstream's inference service."""

    denoising_steps: int | None = None
    """Flow-matching steps; the checkpoint's own value when omitted."""

    llm_scheme: str = schemes.W8A8_SR
    """FoldQuant scheme for the Qwen3 text tower; ``none`` keeps it in PyTorch."""

    dit_scheme: str = schemes.W4A4_SHG
    """FoldQuant scheme for the action-head DiT; ``none`` keeps it in PyTorch."""

    num_calib: int = 128
    """Calibration observations (episode, step) pairs spread over the dataset."""

    seed: int = 0
    """Seed of the calibration sample and of the denoising noise replayed during calibration."""

    cascade: bool = False
    """Calibrate the DiT on the activations of the *quantized* LLM (fake-quant emulation)."""

    llm_params: str = "{}"
    """JSON overrides for the LLM fold (sq_alpha, act_clip_ratio, site_bits, rot_block_size)."""

    dit_params: str = "{}"
    """JSON overrides for the DiT fold (sq_alpha, sq_fold_order)."""

    video_backend: str = "torchcodec"
    """Video decoder for the dataset loader."""

    device: str = "cuda"

    base_model_id: str | None = None
    """Where others get the base checkpoint (``org/name[@revision]``), recorded in the state."""



def _scheme_or_none(value: str) -> str | None:
    return None if value.strip().lower() in _NONE else value.strip()


def _module_paths(policy) -> dict[str, torch.nn.Module]:
    """The two quantizable modules, at their upstream attribute paths."""
    return {"llm": llm_module(policy), "dit": dit_module(policy)}


def capture_shape_metadata(policy, observation: dict[str, Any]) -> dict[str, Any]:
    """One forward with hooks: the tensor shapes the engine builder profiles, and where the backbone reads.

    ``final_norm`` records whether ``hidden_states[select_layer]`` of the Qwen3
    decoder (the entry the Eagle backbone consumes) is the post-final-norm
    stream (``last_hidden_state``) under the installed transformers, so the
    emitted graph ends where the PyTorch tower's output does.
    """
    modules = _module_paths(policy)
    select_layer = select_layer_of(policy)
    seen: dict[str, Any] = {}

    def _llm_hook(_m, args, kwargs):
        embeds = args[0] if args else kwargs.get("inputs_embeds")
        seen["llm_seq_len"] = int(embeds.shape[1])
        seen["llm_hidden_size"] = int(embeds.shape[2])
        seen["batch_size"] = int(embeds.shape[0])

    def _decoder_hook(_m, _args, output):
        read = output.hidden_states[select_layer]
        seen["final_norm"] = bool(torch.equal(read, output.last_hidden_state))

    def _dit_hook(_m, args, kwargs):
        seen["vl_seq_len"] = int(kwargs["encoder_hidden_states"].shape[1])
        seen["sa_seq_len"] = int(kwargs["hidden_states"].shape[1])

    handles = [
        modules["llm"].register_forward_pre_hook(_llm_hook, with_kwargs=True),
        modules["llm"].model.register_forward_hook(_decoder_hook),
        modules["dit"].register_forward_pre_hook(_dit_hook, with_kwargs=True),
    ]
    try:
        with torch.inference_mode():
            policy.get_action(observation)
    finally:
        for h in handles:
            h.remove()
    missing = [k for k in ("llm_seq_len", "vl_seq_len", "sa_seq_len", "final_norm") if k not in seen]
    if missing:
        raise RuntimeError(f"shape capture never reached {missing}; is this an N1.5 policy?")
    cfg = policy.model.action_head.config
    seen["action_horizon"] = int(cfg.action_horizon)
    seen["num_target_vision_tokens"] = int(cfg.num_target_vision_tokens)
    seen["select_layer"] = select_layer
    expected = 1 + seen["num_target_vision_tokens"] + seen["action_horizon"]
    if seen["sa_seq_len"] != expected:
        raise RuntimeError(
            f"DiT received {seen['sa_seq_len']} self-attention tokens, expected 1 + "
            f"{seen['num_target_vision_tokens']} future-vision + {seen['action_horizon']} action = {expected}"
        )
    return seen


def export_metadata(shapes: dict[str, Any], policy: Any) -> dict[str, Any]:
    """``export_metadata.json``: the shape hints the engine builder reads, for any export of this checkpoint."""
    return {
        "model_version": "n1d5",
        "sa_seq_len": shapes["sa_seq_len"],
        "vl_seq_len": shapes["vl_seq_len"],
        "llm_seq_len": shapes["llm_seq_len"],
        "llm_hidden_size": shapes["llm_hidden_size"],
        "llm_final_norm": shapes["final_norm"],
        "llm_select_layer": shapes["select_layer"],
        "action_horizon": shapes["action_horizon"],
        "num_target_vision_tokens": shapes["num_target_vision_tokens"],
        "denoising_steps": int(policy.denoising_steps),
        "embodiment_tag": policy.embodiment_tag.value,
        "export_mode": "foldquant",
        "precision": "bf16",
        "batch_size": shapes["batch_size"],
    }


#: The file each module's graph is exported to (the manifest records it; `export` writes it).
_GRAPH_FILES = {"llm": "llm_bf16.onnx", "dit": "dit_bf16.onnx"}


def main(args: QuantizeConfig) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    llm_scheme = _scheme_or_none(args.llm_scheme)
    dit_scheme = _scheme_or_none(args.dit_scheme)
    if llm_scheme is None and dit_scheme is None:
        raise SystemExit("nothing to export: both --llm-scheme and --dit-scheme are none")
    if llm_scheme is not None:
        schemes.validate("llm", llm_scheme)
    if dit_scheme is not None:
        schemes.validate("dit", dit_scheme)
    if args.cascade and (llm_scheme is None or dit_scheme is None):
        raise SystemExit("--cascade needs both an LLM scheme and a DiT scheme")
    if args.cascade and llm_scheme not in schemes.LLM_FOLDED_SCHEMES:
        raise SystemExit(f"--cascade emulates a folded LLM; {llm_scheme!r} folds nothing")
    llm_params = json.loads(args.llm_params)
    dit_params = json.loads(args.dit_params)

    if not Path(args.model_path).is_dir():
        raise SystemExit("saving the quantized model needs --model-path to be a local checkpoint directory (its files are hashed)")
    # Every arm is saved as a quantized model. The FoldQuant graphs are built in memory,
    # only to fold, round and record the weight codes. `export` writes the ONNX from it.

    t0 = time.time()
    policy = calibration.load_policy(
        args.model_path,
        args.embodiment_tag,
        args.device,
        data_config=args.data_config,
        denoising_steps=args.denoising_steps,
    )
    dataset = calibration.load_dataset(policy, args.dataset_path, args.video_backend)
    logger.info("policy + dataset (%d episodes) in %.0fs", len(dataset.trajectory_ids), time.time() - t0)

    samples, observations = calibration.sample_observations(policy, dataset, args.num_calib, seed=args.seed)
    loop = calibration.make_forward_loop(policy, observations, seed=args.seed)
    modules = _module_paths(policy)

    shapes = capture_shape_metadata(policy, observations[0])
    logger.info("captured shapes: %s", shapes)

    results = []
    plugin_libs: list = []
    llm_result = None
    if llm_scheme is not None:
        t1 = time.time()
        llm_result = export_llm(
            modules["llm"],
            None,
            scheme=llm_scheme,
            forward_loop=loop,
            params=llm_params or None,
            final_norm=shapes["final_norm"],
            record=True,
        )
        results.append(llm_result)
        logger.info("LLM %s exported in %.0fs", llm_scheme, time.time() - t1)

    if dit_scheme is not None:
        t1 = time.time()
        emulation = None
        if args.cascade:
            assert llm_result is not None
            emulation = install_llm_emulation(modules["llm"], llm_result)
            logger.info("cascade: DiT calibration runs under the quantized-LLM emulation")
        try:
            dit_result = export_dit(
                modules["dit"],
                None,
                scheme=dit_scheme,
                forward_loop=loop,
                params=dit_params or None,
                record=True,
            )
        finally:
            if emulation is not None:
                emulation.remove()
        results.append(dit_result)
        logger.info("DiT %s exported in %.0fs", dit_scheme, time.time() - t1)

    for r in results:
        for lib in r.plugin_libs:
            if lib not in plugin_libs:
                plugin_libs.append(lib)

    metadata = export_metadata(shapes, policy)

    manifest = {
        "model_path": public_path(args.model_path),
        "embodiment_tag": policy.embodiment_tag.value,
        "data_config": args.data_config,
        "dataset_path": public_path(args.dataset_path),
        "schemes": {r.module: r.scheme for r in results},
        "params": {"llm": llm_params, "dit": dit_params},
        "cascade": bool(args.cascade),
        "llm_final_norm": shapes["final_norm"],
        "plugin_libs": plugin_libs,
        "files": {r.module: _GRAPH_FILES[r.module] for r in results},
        "calibration": {
            "seed": args.seed,
            "num_samples": len(samples),
            "samples": [asdict(s) for s in samples],
        },
    }
    from .quantized import save_arm_state

    model_dir = Path(args.output_dir)
    if model_dir.exists() and any(model_dir.iterdir()):
        if not is_quantized_checkpoint(model_dir):
            raise SystemExit(f"{model_dir} exists and is not a FoldQuant quantized model; refusing to overwrite it")
        logger.info("replacing %s", model_dir)
        shutil.rmtree(model_dir)
    save_arm_state(model_dir, model_path=args.model_path, results=results,
                   export_metadata=metadata, export_manifest=manifest, base_model_id=args.base_model_id,
                   policy=policy,
                   load_kwargs={"data_config": args.data_config, "denoising_steps": args.denoising_steps})
    logger.info("quantized model: %s", model_dir)
    return model_dir


if __name__ == "__main__":
    main(tyro.cli(QuantizeConfig))
