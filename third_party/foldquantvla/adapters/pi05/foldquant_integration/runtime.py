# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""Serve FoldQuant engines from inside the upstream ``PI0Pytorch`` model.

Upstream's ``sample_actions`` is two calls apart from the engines: it runs the
PaliGemma language model once over the image + prompt prefix, keeps only the
KV cache that call returns, and hands that cache to ``denoise_step`` on every
Euler step. FoldQuant swaps exactly those two collaborators on the live model
instance:

* the LLM engine takes the prefix branch of
  ``paligemma_with_expert.forward`` (``prefix_embs`` / 4-D additive
  ``attention_mask`` / ``position_ids`` in, the stacked post-RoPE KV cache
  ``kv_stack`` out). The SigLIP tower, the multimodal projector and the token
  embedding stay in PyTorch; they produce ``prefix_embs``;
* the expert engine takes ``denoise_step`` (``x_t`` / ``timestep`` /
  ``prefix_pad_masks`` / ``kv_stack`` in, ``velocity`` out): the suffix
  embedding, the 18 Gemma-300M layers, the AdaRMS final norm and
  ``action_out_proj`` all live in the graph. The Euler loop stays upstream's.

The two seams meet on the KV cache: whichever side is PyTorch, the cache
crosses as the same ``[layers, 2, 1, kv_heads, prefix_len, head_dim]`` bf16
tensor the engine contract uses, so each engine can be installed alone.
Everything else (transforms, tokenizer, normalisation, the websocket
server) is untouched, so the policy's public behaviour is exactly
upstream's with two modules swapped underneath.

The model must be eager. Upstream builds it with ``pytorch_compile_mode =
"max-autotune"``, and ``torch.compile`` traces ``sample_actions`` once,
inlining the prefix pass it found at trace time; a ``forward`` rebound on the
instance afterwards is never called, and tensors kept across CUDA-graph
replays are overwritten. :func:`install_engines` and :class:`PrefixCapture`
therefore refuse a compiled model; build it with
``calibration.load_policy(..., compile=False)``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
import json
import logging
from pathlib import Path
import types
from typing import Any

from foldquant.runtime.engine import TensorRTEngine
from foldquant.runtime.plugins import load_plugins
import torch

from ._upstream import COMPONENTS
from ._upstream import ENGINES_RECORD_NAME
from ._upstream import MANIFEST_NAME

logger = logging.getLogger("foldquant.pi05.runtime")

LLM_INPUTS = {"prefix_embs", "attention_mask", "position_ids"}
LLM_OUTPUT = "kv_stack"
EXPERT_INPUTS = {"x_t", "timestep", "prefix_pad_masks", "kv_stack"}
EXPERT_OUTPUT = "velocity"


def plugin_libs_of(engine_dir: Path) -> list[str]:
    """Plugin libraries an engine directory needs, from its build record or export manifest."""
    for name in (ENGINES_RECORD_NAME, MANIFEST_NAME):
        path = engine_dir / name
        if path.is_file():
            return list(json.loads(path.read_text()).get("plugin_libs", []))
    raise FileNotFoundError(f"{engine_dir} holds neither {ENGINES_RECORD_NAME} nor {MANIFEST_NAME}")


def model_of(policy) -> torch.nn.Module:
    """The ``PI0Pytorch`` module inside an upstream ``Policy``."""
    model = getattr(policy, "_model", policy)
    if not hasattr(model, "paligemma_with_expert"):
        raise TypeError(f"expected an openpi Policy or PI0Pytorch, got {type(policy).__name__}")
    return model


def require_eager(model: torch.nn.Module) -> None:
    """Refuse a ``PI0Pytorch`` whose ``sample_actions`` is ``torch.compile``d (see the module docstring)."""
    mode = getattr(model.config, "pytorch_compile_mode", None)
    if mode is not None:
        raise RuntimeError(
            f"the model was built with pytorch_compile_mode={mode!r}; torch.compile keeps calling the prefix pass "
            "it traced, so a seam rebound on the instance is skipped; load the policy with "
            "calibration.load_policy(..., compile=False)"
        )


def llm_module(policy) -> torch.nn.Module:
    """The PaliGemma language model (``GemmaModel``: 18 layers, width 2048) the prefix runs through."""
    return model_of(policy).paligemma_with_expert.paligemma.language_model


class Pi05ExpertView(torch.nn.Module):
    """The action expert as one module, the way the FoldQuant Gemma-expert emitter reads it.

    Upstream keeps the expert's pieces as siblings on ``PI0Pytorch``
    (``paligemma_with_expert.gemma_expert``, ``action_in_proj``, the time MLP,
    ``action_out_proj``). The emitter expects them under one root (the
    ``expert_model.*`` / ``action_in_proj.*`` / ... state-dict prefixes of the
    Pi0.5 expert) with ``config.use_adarms`` / ``action_horizon`` /
    ``action_dim`` and the Gemma variant on ``_variant``. This view rebinds the
    live submodules (no copies): its state dict aliases the model's parameters.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        from openpi.models import gemma as _gemma

        self.expert_model = model.paligemma_with_expert.gemma_expert
        self.action_in_proj = model.action_in_proj
        self.action_out_proj = model.action_out_proj
        if model.pi05:
            self.time_mlp_in = model.time_mlp_in
            self.time_mlp_out = model.time_mlp_out
        else:
            self.state_proj = model.state_proj
            self.action_time_mlp_in = model.action_time_mlp_in
            self.action_time_mlp_out = model.action_time_mlp_out
        self.config = types.SimpleNamespace(
            use_adarms=bool(model.pi05),
            action_horizon=int(model.config.action_horizon),
            action_dim=int(model.config.action_dim),
        )
        self._variant = _gemma.get_config(model.config.action_expert_variant)


def expert_view(policy) -> Pi05ExpertView:
    """The action expert of *policy* under the emitter's module layout."""
    return Pi05ExpertView(model_of(policy))


# KV cache


def stack_cache(cache: Any) -> torch.Tensor:
    """HF ``DynamicCache`` -> ``[layers, 2, 1, kv_heads, prefix_len, head_dim]`` (the engine layout)."""
    layers = [torch.stack((k, v), dim=0) for k, v in (cache[i] for i in range(len(cache)))]
    return torch.stack(layers, dim=0)


def cache_from_stack(kv_stack: torch.Tensor) -> Any:
    """The inverse: a ``DynamicCache`` the PyTorch expert's attention indexes per layer."""
    from transformers import DynamicCache

    cache = DynamicCache()
    for i in range(kv_stack.shape[0]):
        cache.update(kv_stack[i, 0], kv_stack[i, 1], i)
    return cache


# the seams


def _prefix_forward(pwe: torch.nn.Module, engine: TensorRTEngine | None) -> Callable[..., Any]:
    """``paligemma_with_expert.forward`` with the prefix branch returning a KV stack.

    With *engine* the prefix runs through the FoldQuant LLM engine; without it
    the PyTorch language model runs and its cache is stacked into the same
    layout. That is what lets the expert engine be installed alone.
    """
    original = type(pwe).forward

    def forward(
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Any = None,
        inputs_embeds: list[torch.Tensor | None] | None = None,
        use_cache: bool | None = None,
        adarms_cond: Any = None,
    ) -> tuple[Any, Any]:
        if inputs_embeds is None or inputs_embeds[1] is not None:
            # Suffix / joint branch: not a prefix pass, PyTorch as upstream wrote it.
            return original(
                pwe,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                adarms_cond=adarms_cond,
            )
        if engine is None:
            hidden, cache = original(
                pwe,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=inputs_embeds,
                use_cache=True,
                adarms_cond=adarms_cond,
            )
            return hidden, stack_cache(cache)
        if attention_mask is None or position_ids is None:
            raise RuntimeError("the FoldQuant LLM engine needs the 4-D attention mask and position_ids upstream passes")
        out = engine(
            prefix_embs=inputs_embeds[0].to(torch.bfloat16),
            attention_mask=attention_mask.to(torch.bfloat16),
            position_ids=position_ids.to(torch.int64),
        )[LLM_OUTPUT]
        # The engine reuses its output buffer across calls: hand back a copy.
        return None, out.clone()

    return forward


def _denoise_step(model: torch.nn.Module, engine: TensorRTEngine | None) -> Callable[..., torch.Tensor]:
    """``denoise_step`` consuming the KV stack the prefix seam returns."""
    original = model.denoise_step  # bound to the PyTorch implementation before the swap
    memo: dict[str, Any] = {}

    def denoise_step(
        self: torch.nn.Module,
        state: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        kv_stack: torch.Tensor,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        if engine is None:
            # One cache per prefix pass; every Euler step of a chunk shares it.
            if memo.get("stack") is not kv_stack:
                memo["stack"] = kv_stack
                memo["cache"] = cache_from_stack(kv_stack)
            return original(state, prefix_pad_masks, memo["cache"], x_t, timestep)
        feeds = {
            "x_t": x_t.to(torch.float32),
            "timestep": timestep.to(torch.float32),
            "prefix_pad_masks": prefix_pad_masks.to(torch.bool),
            "kv_stack": kv_stack.to(torch.bfloat16),
        }
        if not self.pi05:
            feeds["state"] = state.to(torch.float32)
        return engine(**feeds)[EXPERT_OUTPUT].clone().to(x_t.dtype)

    return denoise_step


class InstalledEngines:
    """Handle over the engines installed by :func:`install_engines`; ``remove()`` restores PyTorch."""

    def __init__(self) -> None:
        self.engines: dict[str, TensorRTEngine] = {}
        self._restore: list[Callable[[], None]] = []

    def rebind(self, obj: Any, name: str, value: Any) -> None:
        # An instance attribute shadows the class method (nn.Module.__call__
        # and upstream's explicit ``.forward(`` / ``.denoise_step(`` calls both
        # go through attribute lookup); deleting it puts the PyTorch one back.
        setattr(obj, name, value)
        self._restore.append(lambda: obj.__dict__.pop(name, None))

    def remove(self) -> None:
        for restore in reversed(self._restore):
            restore()
        self._restore.clear()
        for engine in self.engines.values():
            engine.close()
        self.engines.clear()


def install_engines(policy, engine_dir: str | Path, *, components: Iterable[str] | None = None) -> InstalledEngines:
    """Load the plugins an engine directory needs and swap its engines into *policy*.

    ``components`` restricts the swap (``("expert",)`` keeps the PyTorch LLM);
    by default every engine present in the directory is installed. An engine
    the directory lacks is skipped with a log line when the selection was
    implicit and is an error when it was asked for. Both seams are always
    installed (the KV stack is their shared contract), with the missing side
    running PyTorch.
    """
    engine_dir = Path(engine_dir)
    model = model_of(policy)
    require_eager(model)
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
        else:
            inputs = set(EXPERT_INPUTS) if model.pi05 else EXPERT_INPUTS | {"state"}
            engine.validate_binding_names(inputs, {EXPERT_OUTPUT})
        installed.engines[name] = engine
        logger.info("%s: serving %s", name, path)
    if not installed.engines:
        raise FileNotFoundError(f"no FoldQuant engine found in {engine_dir}")
    pwe = model.paligemma_with_expert
    installed.rebind(pwe, "forward", _prefix_forward(pwe, installed.engines.get("llm")))
    installed.rebind(
        model, "denoise_step", types.MethodType(_denoise_step(model, installed.engines.get("expert")), model)
    )
    return installed


class PrefixCapture:
    """Record the KV stack every prefix pass returns, under PyTorch or the engines.

    Upstream calls ``paligemma_with_expert.forward(`` explicitly, so a forward
    hook never fires; this wraps whatever ``forward`` is currently bound (the
    PyTorch one or the FoldQuant seam) and stacks a PyTorch cache into the
    engine layout so the two passes compare tensor to tensor.
    """

    def __init__(self, policy) -> None:
        model = model_of(policy)
        require_eager(model)
        self._pwe = model.paligemma_with_expert
        self.stacks: list[torch.Tensor] = []
        self._had_override = False
        self._previous: Any = None

    def __enter__(self) -> PrefixCapture:
        pwe = self._pwe
        self._had_override = "forward" in pwe.__dict__
        self._previous = pwe.__dict__.get("forward")
        inner = self._previous if self._had_override else types.MethodType(type(pwe).forward, pwe)

        def forward(**kwargs: Any) -> Any:
            out = inner(**kwargs)
            embeds = kwargs.get("inputs_embeds")
            if embeds is not None and embeds[1] is None:
                kv = out[1]
                self.stacks.append((kv if isinstance(kv, torch.Tensor) else stack_cache(kv)).detach().clone())
            return out

        pwe.forward = forward  # type: ignore[method-assign]
        return self

    def __exit__(self, *_exc: Any) -> None:
        if self._had_override:
            self._pwe.forward = self._previous  # type: ignore[method-assign]
        else:
            self._pwe.__dict__.pop("forward", None)
