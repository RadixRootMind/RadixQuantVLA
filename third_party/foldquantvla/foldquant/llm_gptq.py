# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""GPTQ weight rounding for LLM W4A4 plugins.

The kernel supports one scale per output row. GPTQ preserves that scale and
distributes each column's rounding error across unquantized columns using
the inverse Hessian. Groupwise scales are not supported.

The Hessian must use the transformed activation ``rot(x / s)`` because weights
are quantized after SmoothQuant and Hadamard folding. Torch is imported lazily;
this module does not load TensorRT or plugin libraries.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# GPTQ hyper-parameters, from the reference implementation and unchanged since:
# the damping keeps a near-singular Hessian invertible, act-order quantizes the
# high-energy columns first (they get the most error budget to spend), and the
# block size bounds the rank-1 update cost.
# Set after the first CUDA linalg failure so a 64-site build fails over once,
# not 64 times. Process-lifetime state, matching the per-build process model.
_CUDA_LINALG_BROKEN = {"flag": False}

PERCDAMP = 0.01
ACT_ORDER = True
BLOCK_SIZE = 128

# Which Linear's input each site's Hessian is measured at. Mirrors the four
# quantized sites the plugin graph emits, in the same order.
SITE_PROBES: Dict[str, str] = {
    "qkv": "self_attn.q_proj",  # merged Q+K+V share one input
    "o": "self_attn.o_proj",
    "gateup": "mlp.gate_proj",  # merged gate+up share one input
    "down": "mlp.down_proj",
}


class MissingHessianError(RuntimeError):
    """A GPTQ site was asked to quantize with no calibration Hessian."""


# Calibration: transformed-input second moment, per site


def _exact_gram(xx: Any) -> Any:
    """``xxᵀ·xx`` with fp32 matmul precision pinned to ``highest`` for this one GEMM.

    The second moment must not inherit the host model's matmul precision: openpi's
    ``PI0Pytorch.__init__`` sets ``torch.set_float32_matmul_precision("high")``, which
    would run this Gram in TF32 and hand the factorization a matrix whose PSD margin
    is below the 1% damping on Gemma's 16384-wide down site (measured: Cholesky
    failure at minor 6451 with the seed-0 draw). Scoped to the GEMM only; the
    model's own forward keeps whatever precision deployment runs with, so the
    captured activations are exactly the served ones.
    """
    import torch

    prev = torch.get_float32_matmul_precision()
    if prev == "highest":
        return xx.t() @ xx
    torch.set_float32_matmul_precision("highest")
    try:
        return xx.t() @ xx
    finally:
        torch.set_float32_matmul_precision(prev)


def compute_gptq_hessians_llm(
    hook_module: Any,
    calib_snapshots: list,
    *,
    sq_scales: Optional[dict],
    rot_bs: int,
    forward_fn: Callable[[Any], None],
    token_weights: Optional[list] = None,
) -> Dict[str, Any]:
    """Second moment ``x̂ᵀx̂`` of the TRANSFORMED input, per site, returned on host.

    ``x̂ = rot(x / s)``, the frame the quantizer actually sees. The transform order
    (SmoothQuant divide, then rotate) mirrors :func:`llm_rotation_sq.apply_sq_fold`
    followed by :func:`llm_rotation_sq.apply_rot_fold` on the weight side. ``o`` is
    excluded from the SQ divide exactly as in :data:`llm_rotation_sq.SQ_SITES`
    (GQA shares V channels across KV groups, so that site gets no fold).

    Family-agnostic, like :func:`llm_rotation_sq.compute_sq_scales_llm`:
    ``hook_module`` supplies ``.layers``, and ``forward_fn(snapshot)`` runs one
    calibration forward that fires the hooks.

    Accumulation runs on the module's device (a host accumulator takes hours).
    Results are drained to host one at a time, so device and host copies are
    never all alive at once (~3.2 GB each on a 16-layer Qwen3).

    Returns:
        ``{f"L{i}_{site}": (K, K) float32 CPU tensor}`` for each site in
        :data:`SITE_PROBES`.
    """
    import torch

    from foldquant.llm_rotation_sq import (
        SQ_SITES,
        _hadamard,
        _rot_last,
    )

    if not calib_snapshots:
        raise ValueError("compute_gptq_hessians_llm: no calibration snapshots provided.")
    if token_weights is not None and len(token_weights) != len(calib_snapshots):
        raise ValueError(
            f"compute_gptq_hessians_llm: {len(token_weights)} token-weight vectors for "
            f"{len(calib_snapshots)} snapshots."
        )
    state = {"idx": 0}

    device = next(hook_module.parameters()).device
    rot_bs = int(rot_bs)
    hadamard = _hadamard(rot_bs).to(device) if rot_bs > 1 else None
    # compute_sq_scales_llm returns host tensors; moving them inside the hook would
    # be one pageable H2D copy per site per sample, thousands of them.
    sq_dev = {k: v.to(device) for k, v in sq_scales.items()} if sq_scales is not None else None

    acc: Dict[str, Any] = {}

    def _make_hook(key: str, site: str) -> Callable[..., None]:
        def h(_m: Any, args: Any, _kw: Any = None) -> None:
            x = args[0]
            if x.dim() != 3:
                return
            xr = x.detach().reshape(-1, x.shape[-1])
            K = xr.shape[1]
            wn = None
            if token_weights is not None:
                # VLMQ-style: weight tokens (visual vs text) in the second moment,
                # normalized to mean 1 so the damping scale is unchanged.
                w = token_weights[state["idx"]].to(xr.device, torch.float32).reshape(-1)
                if w.numel() != xr.shape[0]:
                    raise RuntimeError(
                        f"token weights for snapshot {state['idx']} have {w.numel()} entries; "
                        f"site {key} saw {xr.shape[0]} tokens."
                    )
                wn = (w / w.mean()).sqrt().unsqueeze(1)
            if key not in acc:
                # A K x K fp32 accumulator per site: Qwen-class inner dims
                # (<= 6144, 151 MB) fit on-device across all layers, but
                # Gemma's 16384-wide down site is 1 GB PER LAYER, 18 GB total,
                # past any 16 GB card. Wide sites accumulate on host instead;
                # the per-call GEMM stays on-device either way.
                # FOLDQUANT_GRAM_ON_DEVICE=1 keeps them on device anyway: on a
                # unified-memory Jetson the host copy saves nothing and costs a
                # 1 GB transient per call.
                on_host = K >= 8192 and os.environ.get("FOLDQUANT_GRAM_ON_DEVICE") != "1"
                acc[key] = torch.zeros(K, K, device="cpu" if on_host else device, dtype=torch.float32)
            # Row blocks of ~512 MB fp32: a batched replay (128 obs x 968 tokens x
            # 16384 on pi0.5's down site) is 8 GB per fp32 copy, and the cast, SQ
            # divide and rotation each make one; that exhausts a 64 GB Orin.
            # Same sum of per-block Grams, fp32 accumulation order aside.
            rows = max(1, (1 << 27) // K)
            gram = None
            for r0 in range(0, xr.shape[0], rows):
                xx = xr[r0:r0 + rows].float()
                if sq_dev is not None and site in SQ_SITES:
                    xx = xx / sq_dev[key]
                if hadamard is not None:
                    xx = _rot_last(xx, hadamard)
                if wn is not None:
                    xx = xx * wn[r0:r0 + rows]
                g = _exact_gram(xx)
                gram = g if gram is None else gram.add_(g)
                del xx, g
            acc[key] += gram.cpu() if acc[key].device.type == "cpu" else gram

        return h

    handles = []
    for i, layer in enumerate(hook_module.layers):
        for site, dotted in SITE_PROBES.items():
            mod = layer
            for part in dotted.split("."):
                mod = getattr(mod, part)
            handles.append(mod.register_forward_pre_hook(_make_hook(f"L{i}_{site}", site), with_kwargs=True))
    try:
        logger.info(
            "  LLM GPTQ: transformed-input Hessian over %d replay snapshot(s) (sq=%s, rot_bs=%d)",
            len(calib_snapshots),
            "on" if sq_scales else "off",
            rot_bs,
        )
        with torch.inference_mode():
            for i, snap in enumerate(calib_snapshots):
                state["idx"] = i
                forward_fn(snap)
    finally:
        for h in handles:
            h.remove()

    expected = {f"L{i}_{site}" for i in range(len(hook_module.layers)) for site in SITE_PROBES}
    if set(acc) != expected:
        raise RuntimeError(
            f"GPTQ calibration produced Hessians for {len(acc)}/{len(expected)} sites; "
            f"missing {sorted(expected - set(acc))[:4]}; a forward hook never fired."
        )

    out: Dict[str, Any] = {}
    for key in list(acc):
        out[key] = acc.pop(key).cpu()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


# Asymmetric calibration (GPTAQ-style): moments of the QUANTIZED model's input


def compute_gptaq_site_moments_llm(
    float_decoder: Any,
    quant_decoder: Any,
    calib_snapshots: list,
    *,
    layer_index: int,
    sq_scales: Optional[dict],
    rot_bs: int,
    forward_float: Callable[[Any], None],
    forward_quant: Callable[[Any], None],
) -> Dict[str, Tuple[Any, Any]]:
    """Paired second moments for the four sites of one layer: ``(Ĥ = X̂ᵀX̂, G = XᵀX̂)``.

    ``X`` is the transformed input ``rot(x / s)`` measured on *float_decoder*;
    ``X̂ = rot(x̂)`` is the same site's input on *quant_decoder*, whose upstream
    layers are already quantized (the SmoothQuant scale is folded into its norm
    gains, so no divide there). Tokens are paired per calibration sample, which is
    what makes ``G`` the cross moment the asymmetric objective
    ``‖X Wᵀ − X̂ Ŵᵀ‖`` needs. Hooks on the quantized decoder are *prepended* so
    they observe the raw site input ahead of the emulation's own rotate+quantize
    pre-hook.

    Returns:
        ``{site: (Ĥ, G)}`` for ``site`` in :data:`SITE_PROBES`, fp32 on the
        decoder's device (one layer at a time keeps this ~1 GB at K=6144).
    """
    import torch

    from foldquant.llm_rotation_sq import (
        SQ_SITES,
        _hadamard,
        _rot_last,
    )

    if not calib_snapshots:
        raise ValueError("compute_gptaq_site_moments_llm: no calibration snapshots provided.")
    device = next(quant_decoder.parameters()).device
    hadamard = _hadamard(int(rot_bs)).to(device) if int(rot_bs) > 1 else None
    sq_dev = {k: v.to(device) for k, v in sq_scales.items()} if sq_scales is not None else None
    float_inputs: Dict[str, list] = {site: [] for site in SITE_PROBES}
    moments: Dict[str, Tuple[Any, Any]] = {}
    state = {"idx": 0}

    def _site_module(decoder: Any, site: str) -> Any:
        mod = decoder.layers[layer_index]
        for part in SITE_PROBES[site].split("."):
            mod = getattr(mod, part)
        return mod

    def _transform(x: Any, site: str, divide: bool) -> Any:
        xx = x.detach().float().reshape(-1, x.shape[-1])
        if divide and sq_dev is not None and site in SQ_SITES:
            xx = xx / sq_dev[f"L{layer_index}_{site}"]
        return _rot_last(xx, hadamard) if hadamard is not None else xx

    def _float_hook(site: str) -> Callable[..., None]:
        def h(_m: Any, args: Any, _kw: Any = None) -> None:
            if args[0].dim() == 3:
                float_inputs[site].append(_transform(args[0], site, True).to("cpu"))

        return h

    def _quant_hook(site: str) -> Callable[..., None]:
        def h(_m: Any, args: Any, _kw: Any = None) -> None:
            if args[0].dim() != 3:
                return
            xq = _transform(args[0], site, False)
            xf = float_inputs[site][state["idx"]].to(device)
            if xf.shape != xq.shape:
                raise RuntimeError(
                    f"GPTAQ pairing: float/quant token shapes differ at L{layer_index}_{site} "
                    f"({tuple(xf.shape)} vs {tuple(xq.shape)})."
                )
            if site not in moments:
                k = xq.shape[1]
                moments[site] = (
                    torch.zeros(k, k, device=device, dtype=torch.float32),
                    torch.zeros(k, k, device=device, dtype=torch.float32),
                )
            h_hat, g = moments[site]
            h_hat += xq.t() @ xq
            g += xf.t() @ xq

        return h

    handles = [
        _site_module(float_decoder, s).register_forward_pre_hook(_float_hook(s), with_kwargs=True) for s in SITE_PROBES
    ]
    try:
        with torch.inference_mode():
            for snap in calib_snapshots:
                forward_float(snap)
    finally:
        for h in handles:
            h.remove()
    handles = [
        _site_module(quant_decoder, s).register_forward_pre_hook(_quant_hook(s), with_kwargs=True, prepend=True)
        for s in SITE_PROBES
    ]
    try:
        with torch.inference_mode():
            for i, snap in enumerate(calib_snapshots):
                state["idx"] = i
                forward_quant(snap)
    finally:
        for h in handles:
            h.remove()
    missing = [s for s in SITE_PROBES if s not in moments]
    if missing:
        raise RuntimeError(f"GPTAQ moments missing for L{layer_index} sites {missing}: a hook never fired.")
    float_inputs.clear()
    return moments


def gptaq_refit_weight(weight: Any, h_hat: Any, g: Any, *, strength: float = 1.0, percdamp: float = PERCDAMP) -> Any:
    """Asymmetric least-squares refit of a folded ``(N, K)`` weight onto the quantized input.

    Minimizes ``‖X Wᵀ − X̂ W̃ᵀ‖_F`` over ``W̃`` given ``Ĥ = X̂ᵀX̂`` and ``G = XᵀX̂``:
    the optimum is ``W̃ = W (I + (G − Ĥ)(Ĥ + εI)⁻¹)``; *strength* in ``(0, 1]``
    scales the correction (GPTAQ damps it, since the full solution overfits a
    small calibration set). The result is then rounded by :func:`gptq_quant_codes`
    with factors from ``gptq_prepare(Ĥ)``, the frame the deployed kernel sees.
    """
    import torch

    w = weight.detach().to(torch.float64)
    hh = h_hat.detach().to(w.device, torch.float64)
    gg = g.detach().to(w.device, torch.float64)
    k = hh.shape[0]
    damp = percdamp * torch.mean(torch.diag(hh))
    m = torch.linalg.solve((hh + damp * torch.eye(k, device=hh.device, dtype=hh.dtype)).T, (gg - hh).T).T
    return (w + float(strength) * (w @ m)).to(torch.float32)


# Factorization + quantization


def gptq_prepare(hessian: Any, *, percdamp: float = PERCDAMP, actorder: bool = ACT_ORDER) -> dict:
    """Factorize one site's Hessian into the factors :func:`gptq_quant_codes` consumes.

    Factorized once per SITE, not per Linear: q/k/v share one input (hence one
    Hessian), as do gate/up, so this is ~1.75× less work than per-weight, and it
    is the expensive part.

    Float64 keeps a near-singular 6144×6144 Hessian stable through two Cholesky
    factorizations. The linalg runs on CUDA fp64 when it works there (Gemma's
    16384-wide down site costs ~20-30 CPU-minutes per LAYER (measured: hours per
    Pi build) and seconds on an H100), with a CPU fallback for the Jetson torch
    builds whose CUDA linalg is broken (``libtorch_cuda_linalg`` undefined
    symbol). Only the triangular factor is kept, on host.

    A :class:`foldquant.quant_state.ReplaySite` in place of a Hessian is
    returned as is: it already holds the codes this site's rounding produced.
    """
    import torch

    from .quant_state import ReplaySite

    if isinstance(hessian, ReplaySite):
        return hessian  # type: ignore[return-value]

    # ``.to()`` returns the caller's tensor when it is already host float64 (the
    # action-module accumulators are); the diagonal writes below must not alias it.
    hc = hessian.detach().to("cpu", torch.float64).clone()
    k = hc.shape[0]

    # A channel that calibration never activated has a zero diagonal and would make
    # the Cholesky fail; pin it to 1 and force its weight column to 0 later.
    dead = torch.diag(hc) == 0
    hc[dead, dead] = 1.0

    perm = invperm = None
    if actorder:
        perm = torch.argsort(torch.diag(hc), descending=True)
        hc = hc[perm][:, perm]
        invperm = torch.argsort(perm)

    damp_unit = torch.mean(torch.diag(hc))
    hc[range(k), range(k)] += percdamp * damp_unit

    def _factor(mat: Any) -> Any:
        lower = torch.linalg.cholesky(mat)
        inv = torch.cholesky_inverse(lower)
        return torch.linalg.cholesky(inv, upper=True)

    def _factor_any_device(mat: Any) -> Any:
        if torch.cuda.is_available() and not _CUDA_LINALG_BROKEN["flag"]:
            try:
                return _factor(mat.cuda()).cpu()
            except torch.linalg.LinAlgError:
                raise
            except RuntimeError as exc:
                # RuntimeError covers both CUDA OOM and the Jetson torch builds
                # whose CUDA linalg is broken (libtorch_cuda_linalg undefined
                # symbol). Fall back LOUDLY: the CPU path costs minutes per wide
                # site (measured: hours per Pi build) and a silent switch would
                # read as a hang. Remember the failure so the remaining
                # sites of this build don't re-attempt CUDA one by one.
                logger.warning(
                    "CUDA fp64 Cholesky failed (%s); falling back to CPU for this and all "
                    "remaining GPTQ sites; expect minutes per wide site.",
                    exc,
                )
                _CUDA_LINALG_BROKEN["flag"] = True
        return _factor(mat)

    # A Hessian accumulated in fp32 over ~10^5 tokens carries rounding of order
    # eps*||H||, which on a site dominated by a few massive-activation channels
    # can exceed the 1% damping and leave the matrix indefinite. Escalate the
    # damping (x10 per attempt, standard GPTQ practice) rather than fail the
    # build; every retry is logged with the damping it used.
    damp_now = percdamp
    while True:
        try:
            hinv = _factor_any_device(hc)
            break
        except torch.linalg.LinAlgError as exc:
            if damp_now >= percdamp * 100:
                raise
            damp_next = damp_now * 10
            logger.warning(
                "GPTQ Hessian (K=%d) not positive-definite at percdamp=%.3g (%s); retrying with percdamp=%.3g.",
                k,
                damp_now,
                str(exc).splitlines()[0],
                damp_next,
            )
            hc[range(k), range(k)] += (damp_next - damp_now) * damp_unit
            damp_now = damp_next

    return {
        "Hinv": hinv.to(torch.float32),
        "dead": dead,
        "perm": perm,
        "invperm": invperm,
    }


def gptq_quant_codes(
    weight: Any, prep: dict, *, qmax: float, blocksize: int = BLOCK_SIZE, row_clip: Any = None
) -> Tuple[Any, Any]:
    """GPTQ-round a ``(N, K)`` weight to symmetric codes with a per-output-row scale.

    The scale is taken from the transformed weight up front (the same scale RTN
    would use), so any gain is pure rounding, which is exactly why this composes
    with the deployed per-output-row epilogue.

    Args:
        weight: ``(N, K)`` float tensor, already SmoothQuant-folded and rotated.
        prep: factors from :func:`gptq_prepare` for this weight's site.
        qmax: symmetric code limit (7 for INT4, 127 for INT8).
        blocksize: columns per error-propagation block.

    Returns:
        ``(codes (N, K) int32 in [-qmax, qmax], scale (N,) float32)``.

    A :class:`foldquant.quant_state.ReplaySite` *prep* returns the codes and
    scale recorded for this site instead of rounding; a prep carrying a
    ``"site"`` tag is recorded while :func:`foldquant.quant_state.record_gptq`
    is active.
    """
    import torch

    from .quant_state import ReplaySite, record

    if isinstance(prep, ReplaySite):
        return prep.take(weight)
    codes, scale = _gptq_round(weight, prep, qmax=qmax, blocksize=blocksize, row_clip=row_clip)
    site = prep.get("site") if isinstance(prep, dict) else None
    if site is not None:
        record(site, codes, scale)
    return codes, scale


def _gptq_round(
    weight: Any, prep: dict, *, qmax: float, blocksize: int = BLOCK_SIZE, row_clip: Any = None
) -> Tuple[Any, Any]:
    import torch

    w = weight.detach().float().clone()
    _, k = w.shape
    hinv, dead = prep["Hinv"].to(w.device), prep["dead"].to(w.device)
    perm, invperm = prep["perm"], prep["invperm"]

    w[:, dead] = 0.0
    if perm is not None:
        w = w[:, perm.to(w.device)]

    scale = (w.abs().amax(dim=1, keepdim=True) / qmax).clamp(min=1e-12)
    if row_clip is not None:
        # Learned per-output-row clipping (LWC): a scale below amax/qmax saturates the
        # row's largest entries for a finer grid on the rest. Any per-row scale is
        # exactly what the plugin's weight_scale epilogue consumes, so this deploys as-is.
        scale = scale * row_clip.detach().float().to(w.device).reshape(-1, 1).clamp(min=1e-3, max=1.0)
    s_col = scale.squeeze(1)

    codes = torch.zeros_like(w)
    for i1 in range(0, k, blocksize):
        i2 = min(i1 + blocksize, k)
        w1 = w[:, i1:i2].clone()
        c1 = torch.zeros_like(w1)
        e1 = torch.zeros_like(w1)
        hinv1 = hinv[i1:i2, i1:i2]

        for i in range(i2 - i1):
            col = w1[:, i]
            qc = torch.round(col / s_col).clamp(-qmax, qmax)
            c1[:, i] = qc
            # Push this column's residual onto the columns still to be quantized,
            # weighted by the inverse Hessian. The whole of GPTQ is this line.
            err = (col - qc * s_col) / hinv1[i, i]
            w1[:, i:] -= err.unsqueeze(1) @ hinv1[i, i:].unsqueeze(0)
            e1[:, i] = err

        codes[:, i1:i2] = c1
        if i2 < k:
            w[:, i2:] -= e1 @ hinv[i1:i2, i2:]

    if invperm is not None:
        codes = codes[:, invperm.to(codes.device)]
    # `dead` columns were forced to 0 and therefore quantize to code 0, consistent
    # with the weight the engine bakes.
    return codes.to(torch.int32), scale.squeeze(1).to(torch.float32)


def tag_site(prep: Any, site: str) -> Any:
    """Mark *prep* with its site key so :func:`gptq_quant_codes` can record its codes.

    A :class:`foldquant.quant_state.ReplaySite` keeps its own key.
    """
    if isinstance(prep, dict):
        prep["site"] = site
    return prep


# Per-site factorization, one at a time


class GPTQSiteFactors:
    """Hands out one site's factors at a time, freeing each FACTORIZATION as it goes.

    Holding all 64 factorizations of a 16-layer Qwen3 at once would cost ~3.2 GB
    of host memory on top of the Hessians, so the graph builder consumes them
    site by site and drops each factor set after packing that weight. The
    Hessians themselves live until the caller's dict goes out of scope; the
    shallow copy taken here is what makes re-invoking the builder with the same
    dict safe (consume-once is per-instance), so only the factor half of the
    memory is bounded by this class.
    """

    def __init__(self, hessians: Dict[str, Any]) -> None:
        self._hessians = dict(hessians)

    def take(self, key: str) -> dict:
        """Factorize and release the Hessian for ``f"L{i}_{site}"``.

        Raises:
            MissingHessianError: no Hessian for this site. Never falls back to RTN:
                that would read as "GPTQ did not help" instead of "GPTQ never ran".
        """
        if key not in self._hessians:
            raise MissingHessianError(
                f"GPTQ site {key!r} has no calibration Hessian (already consumed, or never "
                "computed). The INT4 LLM schemes quantize weights with GPTQ; falling back to "
                "round-to-nearest here would silently cost ~62% of the weight-axis accuracy."
            )
        return tag_site(gptq_prepare(self._hessians.pop(key)), f"llm.{key}")

    def __len__(self) -> int:
        return len(self._hessians)
