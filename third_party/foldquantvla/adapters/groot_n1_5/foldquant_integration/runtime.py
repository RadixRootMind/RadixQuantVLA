# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""Serve FoldQuant engines from inside the upstream ``Gr00tPolicy``.

Upstream N1.5 deploys its TensorRT engines by rebinding module forwards on
the live policy (``deployment_scripts/trt_model_forward.py``). This module
does the same for the two modules FoldQuant quantizes:

* the LLM engine takes the place of
  ``backbone.eagle_model.language_model.forward`` (``inputs_embeds`` in,
  ``hidden_states`` out). The Eagle2.5 wrapper around it keeps splicing the
  vision features into the embeddings in PyTorch and the backbone keeps
  reading ``hidden_states[select_layer]``, the entry the engine's output is
  placed at;
* the DiT engine takes the place of ``action_head.model.forward``; the flow-
  matching loop, the state / action encoders, the future-token bank and the
  action decoder stay in PyTorch.

Everything else (processor, ViT, collation, action decoding) is untouched,
so the policy's public behaviour is exactly upstream's with two modules
swapped underneath.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import torch
from foldquant.runtime.engine import TensorRTEngine
from foldquant.runtime.plugins import load_plugins

from ._upstream import COMPONENTS, ENGINES_RECORD_NAME, MANIFEST_NAME

logger = logging.getLogger("foldquant.groot_n1_5.runtime")

LLM_INPUTS = {"inputs_embeds", "attention_mask"}
LLM_OUTPUT = "hidden_states"
DIT_INPUTS = {"sa_embs", "vl_embs", "timestep", "image_mask", "backbone_attention_mask"}
DIT_OUTPUT = "output"


def plugin_libs_of(engine_dir: Path) -> list[str]:
    """Plugin libraries an engine directory needs, from its build record or export manifest."""
    for name in (ENGINES_RECORD_NAME, MANIFEST_NAME):
        path = engine_dir / name
        if path.is_file():
            return list(json.loads(path.read_text()).get("plugin_libs", []))
    raise FileNotFoundError(f"{engine_dir} holds neither {ENGINES_RECORD_NAME} nor {MANIFEST_NAME}")


def llm_module(policy) -> torch.nn.Module:
    """The Qwen3 causal LM inside the Eagle2.5 backbone (decoder cut to ``select_layer`` layers)."""
    return policy.model.backbone.eagle_model.language_model


def select_layer_of(policy) -> int:
    """Index of the ``hidden_states`` entry the backbone consumes (12 for the released checkpoints)."""
    return int(policy.model.backbone.select_layer)


def dit_module(policy) -> torch.nn.Module:
    """The action head's plain ``DiT``."""
    return policy.model.action_head.model


def _llm_forward(engine: TensorRTEngine, select_layer: int) -> Callable[..., Any]:
    from transformers.modeling_outputs import CausalLMOutputWithPast

    def forward(
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **_unused: Any,
    ) -> Any:
        if inputs_embeds is None:
            raise RuntimeError(
                "the FoldQuant LLM engine takes inputs_embeds; the Eagle wrapper always passes them"
            )
        embeds = inputs_embeds.to(torch.bfloat16)
        if attention_mask is None:
            attention_mask = torch.ones(embeds.shape[:2], device=embeds.device)
        outputs = engine(inputs_embeds=embeds, attention_mask=attention_mask.to(torch.int64))
        out = outputs[LLM_OUTPUT]
        # The backbone indexes hidden_states[select_layer]; the graph ends where
        # that entry does, so it is the only one worth materialising. The engine
        # reuses its output buffer across calls: hand back a copy.
        hidden_states = (None,) * select_layer + (out.clone(),)
        return CausalLMOutputWithPast(logits=None, hidden_states=hidden_states)

    return forward


def _dit_forward(engine: TensorRTEngine) -> Callable[..., Any]:
    def forward(
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        return_all_hidden_states: bool = False,
    ) -> torch.Tensor:
        if return_all_hidden_states:
            raise RuntimeError("the FoldQuant DiT engine returns the final hidden state only")
        if timestep is None:
            raise RuntimeError("the FoldQuant DiT engine needs the discretised timestep")
        if encoder_attention_mask is not None:
            raise RuntimeError("the FoldQuant DiT engine was emitted for a DiT that attends every encoder token")
        vl = encoder_hidden_states.to(torch.bfloat16)
        # The plain DiT attends the whole vision-language sequence: both masks
        # of the graph contract are all True (= attend).
        ones = torch.ones(vl.shape[:2], dtype=torch.bool, device=vl.device)
        out = engine(
            sa_embs=hidden_states.to(torch.bfloat16),
            vl_embs=vl,
            timestep=timestep.to(torch.int64),
            image_mask=ones,
            backbone_attention_mask=ones,
        )[DIT_OUTPUT]
        return out.clone()

    return forward


class InstalledEngines:
    """Handle over the engines installed by :func:`install_engines`; ``remove()`` restores PyTorch."""

    def __init__(self) -> None:
        self.engines: dict[str, TensorRTEngine] = {}
        self._restore: list[Callable[[], None]] = []

    def _rebind(self, module: torch.nn.Module, forward: Callable[..., Any]) -> None:
        # An instance attribute shadows the class method for nn.Module.__call__;
        # deleting it puts the PyTorch forward back.
        module.forward = forward  # type: ignore[method-assign]
        self._restore.append(lambda: module.__dict__.pop("forward", None))

    def remove(self) -> None:
        for restore in reversed(self._restore):
            restore()
        self._restore.clear()
        for engine in self.engines.values():
            engine.close()
        self.engines.clear()


def install_engines(
    policy, engine_dir: str | Path, *, components: Iterable[str] | None = None
) -> InstalledEngines:
    """Load the plugins an engine directory needs and swap its engines into *policy*.

    ``components`` restricts the swap (``("dit",)`` keeps the PyTorch LLM);
    by default every engine present in the directory is installed. An engine
    the directory lacks is skipped with a log line when the selection was
    implicit and is an error when it was asked for.
    """
    engine_dir = Path(engine_dir)
    load_plugins(plugin_libs_of(engine_dir))
    wanted = None if components is None else set(components)
    installed = InstalledEngines()
    for name, _onnx, engine_name in COMPONENTS:
        if wanted is not None and name not in wanted:
            continue
        path = engine_dir / engine_name
        if not path.is_file():
            if wanted is not None:
                raise FileNotFoundError(f"{name} engine requested but {path} does not exist")
            logger.info("%s: no %s in %s, PyTorch module kept", name, engine_name, engine_dir)
            continue
        engine = TensorRTEngine(path)
        if name == "llm":
            engine.validate_binding_names(LLM_INPUTS, {LLM_OUTPUT})
            installed._rebind(llm_module(policy), _llm_forward(engine, select_layer_of(policy)))
        else:
            engine.validate_binding_names(DIT_INPUTS, {DIT_OUTPUT})
            installed._rebind(dit_module(policy), _dit_forward(engine))
        installed.engines[name] = engine
        logger.info("%s: serving %s", name, path)
    if not installed.engines:
        raise FileNotFoundError(f"no FoldQuant engine found in {engine_dir}")
    return installed
