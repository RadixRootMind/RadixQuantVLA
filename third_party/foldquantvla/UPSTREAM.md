# FoldQuantVLA provenance

- Upstream repository: `https://github.com/cair-vinuni/FoldQuantVLA`
- Integrated commit: `7d98edd705c33ef994b3caf8c2c93d28da78413a`
- Upstream license: Apache License 2.0; see `LICENSE` and `NOTICE` in this directory.

RadixQuantVLA vendors the shared `foldquant` package, the Pi0.5 and GR00T N1.5
integration adapters, TensorRT plugin sources, and the pinned TensorRT headers.
It intentionally does not duplicate the complete upstream OpenPI or GR00T model
trees. Pi0.5 is resolved through `OPENPI_ROOT`; GR00T N1.5 reuses the model code
already present in RadixQuantVLA.

RadixQuantVLA-specific compatibility changes are deliberately small:

1. Adapter roots are resolved from `FOLDQUANT_OPENPI_ROOT`, `OPENPI_ROOT`,
   `FOLDQUANT_GROOT_ROOT`, or `AWESOME_QVLA_ROOT`.
2. LIBERO is resolved from external `FOLDQUANT_LIBERO_DIR` / `LIBERO_ROOT`.
3. Quantized checkpoint shard size is configurable with
   `FOLDQUANT_MAX_SHARD_GIB` instead of a server-local hard-coded value.
4. Pi0.5 initial states are loaded with an explicit `weights_only=False` when
   supported, with a fallback for older PyTorch releases.
5. Unified launchers normalize output paths and benchmark summaries without
   changing FoldQuant's quantization math or TensorRT runtime bindings.
6. The vendored package metadata points at `README.upstream.md`, preserving the
   upstream README while keeping editable installation valid.
7. CUDA plugin builds resolve both conventional and
   `targets/<arch>-linux/include` toolkit layouts. This compatibility fix was
   validated with CUDA 12.8, CMake 3.22, TensorRT 10.15, and an SM86 GPU, and
   has been submitted upstream.

CUTLASS is not vendored because of its size. Use NVIDIA CUTLASS v3.6.0 and set
`CUTLASS_INCLUDE_DIR` to its `include` directory. TensorRT 10/11 headers from
the integrated upstream commit are retained under `third_party/tensorrt-headers`.
