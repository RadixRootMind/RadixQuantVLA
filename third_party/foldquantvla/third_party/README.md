# Native build dependencies

`tensorrt-headers/` is retained from the pinned FoldQuantVLA upstream commit so
the plugin sources can compile against the TensorRT 10 or 11 ABI selected by
the installed runtime.

CUTLASS is an external dependency. Install the upstream-tested v3.6.0 release
outside this repository and export its include directory:

```bash
git clone --branch v3.6.0 --depth 1 https://github.com/NVIDIA/cutlass.git "$WORKSPACE/cutlass-v3.6.0"
export CUTLASS_INCLUDE_DIR="$WORKSPACE/cutlass-v3.6.0/include"
```

Do not commit compiled `.so` files. FoldQuant caches device- and TensorRT-specific
plugins under `${FOLDQUANT_CACHE_DIR:-$HOME/.cache/foldquant}`.

