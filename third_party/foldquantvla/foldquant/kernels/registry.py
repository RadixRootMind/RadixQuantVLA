# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""Kernel metadata shared by the build driver and runtime loader.

Each entry declares its sources, dependencies, and initialization symbols.
``tests/test_kernel_registry.py`` checks these against the source tree.
This module requires neither torch nor CUDA.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

__all__ = ["KernelSpec", "TENSORRT_KERNELS", "kernel_spec", "kernels_needing", "tensorrt_source_root"]


@dataclass(frozen=True)
class KernelSpec:
    """One buildable custom kernel.

    Args:
        name: Library name, also the CMake target and the ``.so`` stem, so
            ``<name>.<target_slug>.so`` is the committed binary. The source
            directory is named after it too; the three cannot drift.
        needs_cutlass: Whether any source includes a CUTLASS header. Determines
            whether the target is skipped when the submodule is absent.
        needs_cublas: Whether the target links cuBLAS.
        init_symbols: ``extern "C"`` entry points that register this kernel's
            plugin creators. Probed after ``dlopen``; a library whose symbols
            are all absent registered nothing.
        tensor_core_dtype: The tensor-core datatype this kernel's speed depends
            on, or ``None`` when it has no such dependency. Checked against the
            device before a build, because a mismatch is silent: the library
            still compiles, loads, and produces correct output, just without
            the instruction that made it worth building.
    """

    name: str
    needs_cutlass: bool
    needs_cublas: bool
    init_symbols: Tuple[str, ...]
    tensor_core_dtype: "str | None" = None

    @property
    def source_dir(self) -> Path:
        """Absolute path to this kernel's source directory."""
        return tensorrt_source_root() / self.name


def tensorrt_source_root() -> Path:
    """Directory holding the per-kernel TensorRT plugin sources."""
    from importlib.resources import files

    return Path(str(files("foldquant.kernels").joinpath("tensorrt")))


#: Compute capabilities whose tensor cores implement a given datatype.
#:
#: ``s4`` (``mma.sync.m16n8k64.s4.s4.s32``) exists on Turing and Ampere and was
#: **removed in Hopper**. A kernel written for it still compiles for sm90,
#: verified on the committed binary, whose sm90 image contains 6
#: ``IMMA.16832.S8.S8`` instructions and no ``S4`` variant at all, against 416 in
#: the INT8 kernel built the same way. CUTLASS silently selects a non
#: tensor-core path, and the result is correct but ~15x slower: measured on an
#: H100, the GR00T N1.6 DiT goes from 13.2 ms at the float baseline to 199.2 ms
#: under W4A4, while the INT8 plugin on the same device costs 15.7 ms.
#:
#: This is not fixable in software on Hopper. It is recorded so a build on the
#: wrong device says so instead of shipping a slower engine quietly.
TENSOR_CORE_DTYPE_ARCHES: Dict[str, Tuple[int, ...]] = {
    "s4": (75, 80, 86, 87, 89),
}


def tensor_core_support_warning(lib: str, capability: "tuple[int, int] | None") -> "str | None":
    """Explain why *lib* will be slow on a device, or ``None`` if it is a fit.

    Args:
        lib: Key of :data:`TENSORRT_KERNELS`.
        capability: ``(major, minor)`` from ``torch.cuda.get_device_capability``.

    Returns:
        A one-line explanation naming the datatype and the device, or ``None``.
    """
    spec = TENSORRT_KERNELS.get(lib)
    if spec is None or spec.tensor_core_dtype is None or capability is None:
        return None
    arches = TENSOR_CORE_DTYPE_ARCHES.get(spec.tensor_core_dtype, ())
    sm = capability[0] * 10 + capability[1]
    if sm in arches:
        return None
    return (
        f"{lib} depends on {spec.tensor_core_dtype} tensor cores, which sm{sm} does not have "
        f"(present on: {', '.join('sm%d' % a for a in arches)}). The engine will build and be "
        "numerically correct, but the kernel falls back off the tensor cores (measured ~15x "
        "slower than the float baseline on sm90). Use an int8 scheme on this device."
    )


#: Every TensorRT plugin kernel, keyed by library name.
TENSORRT_KERNELS: Dict[str, KernelSpec] = {
    "foldquant_int8_per_row": KernelSpec(
        name="int8_per_row",
        needs_cutlass=True,  # dit_int8_rowwise_v2*_cuda.cu include <cutlass/...>
        needs_cublas=True,  # batched BF16 SDPA inside the v2 macro plugins
        init_symbols=(
            "initEncoderPreQuantPlugin",
            "initFusedSelfAttnFullPlugin",
            "initFusedCrossAttnFullPlugin",
            "initFusedCrossAttnFullCachedPlugin",
            "initFusedCrossAttnPrequantizedPlugin",
            "initFusedNormProjOutPlugin",
            "initFusedFfnBlockPlugin",
            "initFusedRmsNormLinearInt8Plugin",
            "initPerRowInt8LinearResidualPlugin",
        ),
    ),
    "foldquant_int4_groupwise": KernelSpec(
        name="int4_groupwise",
        needs_cutlass=False,  # the only CUTLASS-free target
        needs_cublas=False,
        # Registers via REGISTER_TENSORRT_PLUGIN static init only; covered by the
        # trailing init_libnvinfer_plugins refresh rather than an init symbol.
        init_symbols=(),
    ),
    "foldquant_int4_per_row": KernelSpec(
        # W4A4 is only fast where s4 tensor cores exist. See
        # `TENSOR_CORE_DTYPE_ARCHES`.
        tensor_core_dtype="s4",
        name="int4_per_row",
        needs_cutlass=True,  # dit_int4_rowwise_gemm_fused_cuda.cu includes <cutlass/cutlass.h>
        needs_cublas=True,  # BF16 rotation + batched BF16 SDPA in the W4A4 macros
        init_symbols=(
            "initEncoderPreQuantInt4Plugin",
            "initFusedSelfAttnFullInt4Plugin",
            "initFusedCrossAttnFullInt4Plugin",
            "initFusedFfnBlockInt4Plugin",
            "initAdaLNModInt4Plugin",
            "initPerRowInt4LinearResidualPlugin",
            # LLM W4A4 (w4a4_s{g,rg}): fused RMSNorm + FWHT + per-row
            # INT4 quant + s4 GEMM for the merged Q+K+V / gate+up sites.
            "initFusedRmsNormLinearInt4Plugin",
        ),
    ),
}


def kernel_spec(lib: str) -> KernelSpec:
    """Return the spec for library *lib*.

    Raises:
        KeyError: If *lib* is not a known kernel.
    """
    try:
        return TENSORRT_KERNELS[lib]
    except KeyError:
        raise KeyError(f"Unknown kernel {lib!r}. Known: {sorted(TENSORRT_KERNELS)}.") from None


def kernels_needing(*, cutlass: bool = False) -> Tuple[str, ...]:
    """Return library names with the given dependency, sorted.

    Lets the build driver ask "which of these need CUTLASS?" instead of
    hardcoding the answer.
    """
    return tuple(sorted(lib for lib, spec in TENSORRT_KERNELS.items() if spec.needs_cutlass is cutlass))
