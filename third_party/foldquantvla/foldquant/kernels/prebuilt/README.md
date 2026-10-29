# Prebuilt plugin binaries

Device-matched TensorRT plugin libraries may be dropped here as
`<libname>.<sm>-<machine>-trt<major>.<minor>.so` (for example
`foldquant_int4_per_row.sm87-aarch64-trt10.3.so`). The locator loads a binary
only when its SM and machine match exactly and its TensorRT major matches
(older minors within the major are tolerated with a warning).

No binaries are committed. On a device that matches nothing here, run

    python -m foldquant.kernels build

which compiles the sources under `../tensorrt/` into the out-of-tree cache
(`FOLDQUANT_CACHE_DIR`, else `$XDG_CACHE_HOME/foldquant`, else `~/.cache/foldquant`).
