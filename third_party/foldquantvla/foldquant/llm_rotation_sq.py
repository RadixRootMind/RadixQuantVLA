# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""SmoothQuant and block Hadamard folds for LLM per-row plugins.

``apply_sq_fold`` absorbs channel scales into RMSNorm gains or up-projection
rows and compensates the consuming weight columns. ``apply_rot_fold`` folds
the orthonormal Hadamard into weights; the plugin applies the matching FWHT
to activations at runtime.

Defaults are ``sq_alpha=0.4`` and ``rot_block_size=64``. ``_hadamard`` must
match ``fwht.cuh::block_fwht_smem`` in Sylvester ordering and normalization.
Torch is imported lazily; this module does not load TensorRT or plugins.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Dict, Optional

# Scheme parameters (algorithm registry-key → fold defaults)


# ``sq_alpha`` controls scale folding; ``rot_block_size=0`` disables rotation.
# All entries require SmoothQuant calibration.
LLM_INT8_ALGORITHMS: Dict[str, Dict[str, float]] = {
    "w8a8_s": {"sq_alpha": 0.4, "rot_block_size": 0},
    "w8a8_sr": {"sq_alpha": 0.4, "rot_block_size": 64},
}

# Default to SmoothQuant with Hadamard rotation.
DEFAULT_LLM_INT8_ALGORITHM = "w8a8_sr"

# W4A4 uses GPTQ-rounded weights and dynamic per-row activation scales.
# Rotation-free W4A4 is restricted to ablation because of its accuracy loss.
LLM_INT4_ALGORITHMS: Dict[str, Dict[str, Any]] = {
    "w4a4_sg": {"sq_alpha": 0.4, "rot_block_size": 0, "ablation": True},
    "w4a4_srg": {"sq_alpha": 0.4, "rot_block_size": 64},
    # W4A8: INT4 weights (GPTQ, per-row) served by the INT8 plugins; the packed
    # nibbles are unpacked to INT8 at engine load, activations stay INT8
    # per-token. Held-out emulation on N1.6 recovered 90% of W4A4's action error.
    "w4a8_srg": {"sq_alpha": 0.4, "rot_block_size": 64, "act_bits": 8},
}
DEFAULT_LLM_INT4_ALGORITHM = "w4a4_srg"

LLM_PLUGIN_ALGORITHMS: Dict[int, Dict[str, Dict[str, Any]]] = {
    8: LLM_INT8_ALGORITHMS,
    4: LLM_INT4_ALGORITHMS,
}
DEFAULT_LLM_ALGORITHM: Dict[int, str] = {
    8: DEFAULT_LLM_INT8_ALGORITHM,
    4: DEFAULT_LLM_INT4_ALGORITHM,
}


def resolve_llm_plugin_params(
    algorithm: Optional[str], *, bits: int = 8, overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Fold parameters for an LLM plugin module at *bits*: ``{"sq_alpha", "rot_block_size"}``.

    ``algorithm=None`` selects the width's default (the deployable rot+SQ scheme).
    A named algorithm must belong to the registry of its OWN width: an INT8 key
    under bits=4 (or vice versa) would silently build the wrong plugin pair, so it
    raises here even though the build-config parser also guards it. At bits=4 a
    rotation-less scheme without the ``ablation`` flag is refused: W4A4 without the
    Hadamard measured cosine 0.014, and that combination must never be reachable by
    accident.
    """
    registry = LLM_PLUGIN_ALGORITHMS[int(bits)]
    key = algorithm or DEFAULT_LLM_ALGORITHM[int(bits)]
    if key not in registry:
        other = {w for w, reg in LLM_PLUGIN_ALGORITHMS.items() if key in reg}
        hint = f" (it is a {sorted(other)[0]}-bit scheme)" if other else ""
        raise KeyError(f"{key!r} is not an LLM plugin scheme at width {bits}{hint}.")
    params = dict(registry[key])
    if int(bits) == 4 and int(params.get("rot_block_size", 0)) <= 1 and not params.pop("ablation", False):
        raise ValueError(
            f"LLM W4A4 scheme {key!r} disables the Hadamard rotation without the ablation "
            "flag; rotation-less W4A4 measured cosine 0.014 and must be opted into explicitly."
        )
    params.pop("ablation", None)
    params.setdefault("act_bits", int(bits))
    if overrides:
        # Preset-level knob overrides (quantization.modules.llm.params). Vocabulary
        # is validated by the provider before any model loading; ranges are pinned
        # here so a scheme-level caller cannot slip an unphysical value through.
        unknown = set(overrides) - {"sq_alpha", "act_clip_ratio", "site_bits", "learned_calib", "rot_block_size"}
        if unknown:
            raise KeyError(f"resolve_llm_plugin_params: unknown override(s) {sorted(unknown)}")
        if "sq_alpha" in overrides:
            alpha = float(overrides["sq_alpha"])
            if not (0.0 < alpha < 1.0):
                raise ValueError(f"sq_alpha must be in (0, 1), got {alpha}")
            params["sq_alpha"] = alpha
        if "site_bits" in overrides:
            sb = overrides["site_bits"]
            if (
                not isinstance(sb, dict)
                or set(sb) - {"qkv", "o", "gateup", "down"}
                or any(int(v) not in (4, 8) for v in sb.values())
            ):
                raise ValueError(f"site_bits must map a subset of qkv/o/gateup/down to 4 or 8, got {sb!r}")
            if int(bits) != 4 and any(int(v) == 4 for v in sb.values()):
                raise ValueError("site_bits may lower a site to 4 only in a W4A4 scheme")
            params["site_bits"] = {k: int(v) for k, v in sb.items()}
        if "act_clip_ratio" in overrides:
            clip = float(overrides["act_clip_ratio"])
            if not (0.0 < clip <= 1.0):
                raise ValueError(f"act_clip_ratio must be in (0, 1], got {clip}")
            if int(params["act_bits"]) != 4 and clip != 1.0:
                raise ValueError(
                    "act_clip_ratio is an INT4-activation knob (W4A4 only); "
                    "the INT8-activation plugins do not declare it."
                )
            params["act_clip_ratio"] = clip
        if "rot_block_size" in overrides:
            # The FWHT prologue takes the block size at runtime (fwht.cuh::block_fwht_smem);
            # 64 is the measured default, 128/256 raise the LLM seam consistently (held-out
            # p = 0.021 / 0.001) at no accuracy cost. Only meaningful where a rotation is on.
            bs = int(overrides["rot_block_size"])
            if bs < 2 or (bs & (bs - 1)) != 0 or bs > 1024:
                raise ValueError(f"rot_block_size must be a power of two in [2, 1024], got {bs}")
            if int(params.get("rot_block_size", 0)) <= 1:
                raise ValueError("rot_block_size override needs a scheme with the rotation on (a *_sr/*_srg key).")
            params["rot_block_size"] = bs
        if "learned_calib" in overrides:
            # A learned-calibration file (scripts/llm_learn_calib.py): SmoothQuant
            # scales + per-(layer, site) activation clips [+ per-row weight clips]
            # that replace the alpha formula and the global clip at build time.
            from pathlib import Path as _Path

            lc = _Path(str(overrides["learned_calib"]))
            if int(bits) != 4:
                raise ValueError("learned_calib is an INT4-weight scheme knob (W4A4 / W4A8).")
            if not lc.is_file():
                raise FileNotFoundError(f"learned_calib file not found: {lc}")
            params["learned_calib"] = str(lc)
    params.setdefault("act_clip_ratio", 1.0)
    return params


# SmoothQuant per-channel fold


# Sites that get a SmoothQuant channel fold. ``o`` is deliberately absent: it is
# the mildest site in the model and its activation channel traces back to V
# through GQA's repeat_kv (KV_GROUPS heads share one V channel), so a free fold
# would require s_ch constant within each KV group, not worth it for the mildest
# site.
SQ_SITES = ("qkv", "gateup", "down")


def compute_sq_scales_llm(
    hook_module: Any,
    calib_snapshots: "list",
    *,
    alpha: float,
    forward_fn: Callable[[Any], None],
) -> Dict[str, Any]:
    """Per-channel activation amax at each SQ site → SmoothQuant scale ``s_ch``.

    Runs the live BF16 LLM over calibration activations, capturing per-channel
    activation amax via forward-pre-hooks, then combines with per-input-channel
    weight amax: ``s_ch = amax_act^alpha / amax_w^(1-alpha)``.

    Family-agnostic: ``hook_module`` supplies ``.layers`` (hooks) + ``.state_dict()``
    (weight amax), and ``forward_fn(snapshot)`` runs one calibration forward that
    fires those hooks. N1.6 passes the Qwen3Model + a wrapper call; N1.7 passes the
    Qwen3-VL ``LLMForExport`` + a call threading position_ids/deepstack.

    Args:
        hook_module: LLM module exposing ``.layers`` (decoder layers with
            ``self_attn.q_proj`` / ``mlp.gate_proj`` / ``mlp.down_proj``) and a
            ``state_dict()`` keyed ``layers.<i>.<...>``.
        calib_snapshots: captured per-sample LLM input sets (opaque to this fn;
            consumed by ``forward_fn``).
        alpha: SmoothQuant migration strength.
        forward_fn: runs one calibration forward for a snapshot (fires the hooks).

    Returns:
        Dict keyed ``f"L{i}_{site}"`` for site in :data:`SQ_SITES`. qkv/gateup
        vectors are ``(hidden_size,)``; down is ``(intermediate_size,)``.
    """
    import torch

    model = hook_module
    num_layers = len(model.layers)
    act_amax: Dict[str, Any] = {}

    def _make_pre_hook(key: str) -> Callable[..., None]:
        def h(_m: Any, args: Any, _kw: Any = None) -> None:
            x = args[0]
            if x.dim() != 3:
                return
            # amax over tokens (and batch) → one value per CHANNEL.
            v = x.float().abs().reshape(-1, x.shape[-1]).amax(dim=0)
            act_amax[key] = v if key not in act_amax else torch.maximum(act_amax[key], v)

        return h

    handles = []
    for i, layer in enumerate(model.layers):
        handles.append(layer.self_attn.q_proj.register_forward_pre_hook(_make_pre_hook(f"L{i}_qkv"), with_kwargs=True))
        handles.append(layer.mlp.gate_proj.register_forward_pre_hook(_make_pre_hook(f"L{i}_gateup"), with_kwargs=True))
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(_make_pre_hook(f"L{i}_down"), with_kwargs=True))
    try:
        if not calib_snapshots:
            raise ValueError("compute_sq_scales_llm: no calibration snapshots provided.")
        with torch.inference_mode():
            for snap in calib_snapshots:
                forward_fn(snap)
    finally:
        for h in handles:
            h.remove()

    sd = model.state_dict()
    # The merged QKV / gate-up plugins concatenate weights along the OUTPUT rows,
    # and the SQ scale needs the per-INPUT-channel amax (a reduce over those rows).
    # max-of-concat-rows == elementwise-max of the per-weight amaxes, so reduce
    # each weight and combine; no giant merged/fp32 temporary per layer.
    site_weights = {
        "qkv": ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"),
        "gateup": ("mlp.gate_proj.weight", "mlp.up_proj.weight"),
        "down": ("mlp.down_proj.weight",),
    }
    out: Dict[str, Any] = {}
    for i in range(num_layers):
        p = f"layers.{i}."
        for site, wkeys in site_weights.items():
            key = f"L{i}_{site}"
            if key not in act_amax:
                raise RuntimeError(
                    f"SQ calibration produced no activation amax for {key}; the forward hook never fired."
                )
            a = act_amax[key].float().clamp(min=1e-8)
            w = None
            for wk in wkeys:
                wa = sd[p + wk].float().abs().amax(dim=0)
                w = wa if w is None else torch.maximum(w, wa)
            assert w is not None  # wkeys is non-empty for every site
            w = w.clamp(min=1e-8)
            if a.shape != w.shape:
                raise RuntimeError(
                    f"{key}: activation amax {tuple(a.shape)} does not match weight input-channel amax {tuple(w.shape)}"
                )
            s = (a.pow(alpha) / w.pow(1.0 - alpha)).clamp(min=1e-5, max=1e5)
            out[key] = s.cpu()
    return out


def apply_sq_fold(layer_state: dict, s_qkv: Any, s_gu: Any, s_dn: Any) -> dict:
    """Fold SmoothQuant per-channel scales into one layer's weights. Exact, offline.

    qkv     RMSNorm's gamma scales the norm output per channel and RMS is computed
            from x BEFORE gamma (kernel: y = xv*rstd*g), so gamma_pre[c] /= s[c]
            gives h'[c] = h[c]/s[c] exactly. Compensate: W_qkv[:, c] *= s[c].
    gateup  Same, via post_attention_layernorm's gamma (does NOT touch post_attn,
            which is also down_proj's residual input; only the norm output moves).
    down    x[c] = silu(gate[c]) * up[c] is elementwise in c and LINEAR in up, so
            W_up[c, :] /= s[c] gives x'[c] = x[c]/s[c] exactly, leaving SiLU
            untouched. Compensate: W_down[:, c] *= s[c]. Scaling up's rows is
            absorbed exactly by its per-output-channel weight scale (0 bits change).

    Returns a new dict; ``layer_state`` is not mutated.
    """
    import torch

    ls = dict(layer_state)
    dt = ls["input_layernorm.weight"].dtype

    def _f32(k: str) -> Any:
        # Promote bf16/fp16 to fp32 to fold, but never DOWNcast: an fp64 caller
        # (the exactness test) must keep fp64, else the invariant is masked by an
        # fp32 round-trip the real bf16 path never performs.
        t = ls[k]
        return t if t.dtype in (torch.float32, torch.float64) else t.float()

    # qkv: fold into input_layernorm gamma, compensate on W columns.
    ls["input_layernorm.weight"] = (_f32("input_layernorm.weight") / s_qkv).to(dt)
    for k in ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"):
        ls[k] = (_f32(k) * s_qkv.unsqueeze(0)).to(dt)

    # down: fold into up_proj ROWS, compensate on down_proj columns. Must happen
    # BEFORE the gateup column fold below (different axis, composes).
    ls["mlp.up_proj.weight"] = (_f32("mlp.up_proj.weight") / s_dn.unsqueeze(1)).to(dt)
    ls["mlp.down_proj.weight"] = (_f32("mlp.down_proj.weight") * s_dn.unsqueeze(0)).to(dt)

    # gateup: fold into post_attention_layernorm gamma, compensate columns.
    ls["post_attention_layernorm.weight"] = (_f32("post_attention_layernorm.weight") / s_gu).to(dt)
    for k in ("mlp.gate_proj.weight", "mlp.up_proj.weight"):
        ls[k] = (_f32(k) * s_gu.unsqueeze(0)).to(dt)

    return ls


# Block-diagonal Hadamard rotation fold


# Every linear whose input the rotation kernel touches. Unlike the SQ fold this
# INCLUDES o_proj: the rotation runs inside the quantizer, so GQA's shared V
# channels (which block a free SQ fold) are irrelevant here.
ROT_WEIGHTS = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)


@functools.lru_cache(maxsize=None)
def _hadamard(n: int, dtype: Any = None) -> Any:
    """Orthonormal Sylvester Hadamard: H_1=[1], H_2m=[[H,H],[H,-H]], ``/sqrt(n)``.

    MUST match ``block_fwht_smem`` in
    ``kernels/tensorrt/int8_per_row/cuda/fwht.cuh``. The kernel's iterative
    butterfly computes the transform of exactly this matrix in natural order; if
    the two conventions ever diverge the rotation stops being an identity and
    every layer silently emits garbage (``test_llm_rotation_sq`` pins the
    convention).
    """
    import torch

    if dtype is None:
        dtype = torch.float32
    assert n >= 2 and (n & (n - 1)) == 0, f"rot block size {n} must be a power of 2"
    H = torch.ones(1, 1, dtype=dtype)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
    return H / (n**0.5)


def _rot_last(t: Any, H: Any) -> Any:
    """Block-diagonal rotation over the LAST dim (a weight's input channels)."""
    bs = H.shape[0]
    shp = t.shape
    return (t.reshape(*shp[:-1], shp[-1] // bs, bs) @ H.t()).reshape(*shp)


def apply_rot_fold(layer_state: dict, rot_bs: int) -> dict:
    """Fold ``W' = W·Hᵀ`` so the runtime rotation ``H·x`` cancels: ``W'·(H·x) = W·x``.

    H is orthonormal, so this is exact. Applied AFTER the SQ fold: SQ is a
    per-channel scale, the rotation then mixes those already-scaled channels,
    the order the simulator measured. Every rotated Linear (see :data:`ROT_WEIGHTS`)
    is served by the plugin FWHT, so all of them fold.

    Returns a new dict; ``layer_state`` is not mutated.
    """
    ls = dict(layer_state)
    H = _hadamard(rot_bs).to(ls[ROT_WEIGHTS[0]].device)  # same for every weight here
    for name in ROT_WEIGHTS:
        W = ls[name]
        k_in = W.shape[-1]
        if k_in % rot_bs != 0:
            raise ValueError(f"{name}: input dim {k_in} not divisible by rot_block_size {rot_bs}")
        ls[name] = _rot_last(W.float(), H).to(W.dtype)
    return ls
