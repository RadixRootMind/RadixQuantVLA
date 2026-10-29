# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""FoldQuant scheme keys: which fold each name spells, which modules it serves, which plugin library runs it.

A key is ``w{W}a{A}`` followed by the fold it applies, one letter per pass in a
fixed order: ``s`` SmoothQuant scale, then a rotation, then ``g`` GPTQ rounding.
On action modules ``r`` is a learned dense rotation and ``h`` the fixed Sylvester
butterfly; on LLM keys ``r`` is the fixed block-64 Hadamard applied by the
plugin's FWHT (:mod:`.llm_rotation_sq`). ``w8a8`` alone is the dynamic per-row
baseline, which folds nothing and needs no calibration.

Action-module keys (DiT, Pi expert) and LLM keys
are separate vocabularies: the two graph families are different emitters over
different plugin sets, so a key is refused for a module it has no graph for
rather than silently downgraded.
"""

from __future__ import annotations

from typing import Any, Dict, FrozenSet, List, Mapping, Optional

from .kernels.locator import INT4_PER_ROW_LIB, INT8_PER_ROW_LIB
from .llm_rotation_sq import (
    DEFAULT_LLM_INT4_ALGORITHM,
    DEFAULT_LLM_INT8_ALGORITHM,
    LLM_INT4_ALGORITHMS,
    LLM_INT8_ALGORITHMS,
)
from .modelopt_int8 import MODELOPT_W4A16_AWQ, MODELOPT_W8A8_SMOOTHQUANT

# dynamic per-row baseline (every module)
W8A8 = "w8a8"

# action-module schemes
#: SmoothQuant fold + learned dense block rotation, INT4 weights and activations.
W4A4_SR = "w4a4_sr"
#: Same fold with the fixed Sylvester butterfly (FWHT) instead of a stored matrix:
#: the SmoothQuant vector ships beside the weights, the rotation is implicit.
W4A4_SH = "w4a4_sh"
#: ``w4a4_sh`` with GPTQ-rounded weights. Free at runtime (identical kernel,
#: node attributes and byte layout); GPTQ only spends the same 16 levels better.
W4A4_SHG = "w4a4_shg"
#: The butterfly fold at 8 bit, for activations whose outlier channels are what
#: SmoothQuant exists to move.
W8A8_SH = "w8a8_sh"

ACT_W4A4_SCHEMES: FrozenSet[str] = frozenset({W4A4_SR, W4A4_SH, W4A4_SHG})
ACT_FWHT_SCHEMES: FrozenSet[str] = frozenset({W4A4_SH, W4A4_SHG, W8A8_SH})
ACT_FOLDED_SCHEMES: FrozenSet[str] = ACT_W4A4_SCHEMES | {W8A8_SH}
ACT_MODULES: FrozenSet[str] = frozenset({"dit", "expert"})

# LLM schemes
#: The LLM defaults at each width: SmoothQuant + block rotation, GPTQ at 4 bit.
W8A8_SR = DEFAULT_LLM_INT8_ALGORITHM
W4A4_SRG = DEFAULT_LLM_INT4_ALGORITHM
#: ``w8a8_s`` / ``w8a8_sr``: fold parameters live in :mod:`.llm_rotation_sq`.
LLM_INT8_SCHEMES: FrozenSet[str] = frozenset(LLM_INT8_ALGORITHMS)
#: ``w4a4_sg`` (rotation-off ablation), ``w4a4_srg``, ``w4a8_srg``.
LLM_INT4_SCHEMES: FrozenSet[str] = frozenset(LLM_INT4_ALGORITHMS)
#: INT4 weights over INT8 activations, served by the INT8 plugins, which unpack
#: the nibbles at engine load.
LLM_W4A8_SCHEMES: FrozenSet[str] = frozenset(
    k for k, v in LLM_INT4_ALGORITHMS.items() if int(v.get("act_bits", 4)) == 8
)
LLM_FOLDED_SCHEMES: FrozenSet[str] = LLM_INT8_SCHEMES | LLM_INT4_SCHEMES
LLM_MODULES: FrozenSet[str] = frozenset({"llm"})

MODULES: FrozenSet[str] = ACT_MODULES | LLM_MODULES
ALL_SCHEMES: FrozenSet[str] = frozenset({W8A8}) | ACT_FOLDED_SCHEMES | LLM_FOLDED_SCHEMES

# comparison baselines (not FoldQuant graphs)
#: NVIDIA ModelOpt Q/DQ graphs, built strongly typed like every other graph; see
#: :mod:`.modelopt_int8`. Not in :data:`ALL_SCHEMES`: no emitter here produces
#: them. The GR00T N1.7 integration routes them for its LLM and DiT, the Pi0.5
#: integration for its LLM and action expert. ``modelopt_w4a16_awq`` is INT4
#: weight-only and its graph carries ``Int4GroupwiseGemmPlugin`` nodes after the
#: surgery in :mod:`.int4_groupwise`, hence the plugin library below.
MODELOPT_SCHEMES: FrozenSet[str] = frozenset({MODELOPT_W8A8_SMOOTHQUANT, MODELOPT_W4A16_AWQ})
MODELOPT_MODULES: FrozenSet[str] = frozenset({"llm", "dit", "expert"})


def validate(module: str, scheme: str) -> None:
    """Refuse a (module, scheme) pair that has no plugin graph.

    Raises:
        ValueError: unknown module or scheme, an action key on the LLM, or an
            LLM key on an action module.
    """
    if module not in MODULES:
        raise ValueError(f"FoldQuant has plugin graphs for {sorted(MODULES)}; module {module!r} has none.")
    if scheme in MODELOPT_SCHEMES:
        raise ValueError(
            f"{scheme!r} is a ModelOpt Q/DQ baseline, not a FoldQuant plugin graph; it is routed by "
            "the GR00T N1.7 and Pi0.5 integrations (foldquant.modelopt_int8), for modules "
            f"{sorted(MODELOPT_MODULES)}."
        )
    if scheme not in ALL_SCHEMES:
        raise ValueError(f"unknown FoldQuant scheme {scheme!r}; known: {sorted(ALL_SCHEMES)}")
    if scheme in ACT_FOLDED_SCHEMES and module not in ACT_MODULES:
        raise ValueError(
            f"{scheme!r} has plugin graphs for the DiT and the Pi expert; "
            f"module {module!r} has none. Supported: {sorted(ACT_MODULES)}."
        )
    if scheme in LLM_FOLDED_SCHEMES and module not in LLM_MODULES:
        raise ValueError(f"{scheme!r} is LLM-only; module {module!r} has no LLM SQ/rotation plugin graph.")


def needs_calibration(scheme: str) -> bool:
    """Every folded scheme measures its SmoothQuant scales (and GPTQ Hessians) from real inputs."""
    return scheme != W8A8


def bits_of(scheme: str) -> int:
    """Weight width of *scheme* (4 or 8)."""
    return 4 if scheme in ACT_W4A4_SCHEMES or scheme in LLM_INT4_SCHEMES else 8


def uses_fwht(scheme: str) -> bool:
    """Whether the action-module fold rotates with the butterfly (``_h``) rather than a dense matrix."""
    return scheme in ACT_FWHT_SCHEMES


def plugin_libs(scheme: str, *, params: Optional[Mapping[str, Any]] = None) -> List[str]:
    """Plugin libraries an engine built from *scheme*'s graph must load.

    W4A4 graphs (action-side and LLM) are INT4 plugin nodes; every other graph,
    W4A8 included, is INT8 nodes. An LLM W4A4 whose ``site_bits`` keep some sites
    at INT8 emits a mixed graph and needs both libraries wherever it is built or
    served.
    """
    is_llm_int4_act = scheme in LLM_INT4_SCHEMES and scheme not in LLM_W4A8_SCHEMES
    libs = [INT4_PER_ROW_LIB] if (scheme in ACT_W4A4_SCHEMES or is_llm_int4_act) else [INT8_PER_ROW_LIB]
    site_bits: Dict[str, Any] = dict((params or {}).get("site_bits") or {})
    if is_llm_int4_act and any(int(v) == 8 for v in site_bits.values()):
        libs.append(INT8_PER_ROW_LIB)
    return libs
