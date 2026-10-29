# Vendored TensorRT public headers

The TensorRT C++ public API headers (`NvInfer*.h`, `NvOnnx*.h`), copied verbatim
from [NVIDIA/TensorRT](https://github.com/NVIDIA/TensorRT) — all **Apache-2.0**
(SPDX headers retained in each file).

They exist so the plugin libraries can be compiled with **only** the pip
`tensorrt` / `tensorrt-libs` wheels installed, which ship the runtime
`libnvinfer.so.<N>` but no headers. `foldquant/kernels/build.py` resolves this
directory as the header source when `TENSORRT_ROOT` is unset and synthesises the
linker `libnvinfer.so` from the installed `tensorrt_libs` wheel.

## Two header sets

- `include/` — TensorRT **11.2** (branch `release/11.2`), used when the installed
  `tensorrt` wheel is 11.x.
- `include-trt10/` — TensorRT **10.15** (branch `release/10.15`), used when the
  installed wheel is 10.x (the upstream GR00T release, Jetson AGX Orin).

`build.py::_resolve_trt_include` picks the set matching the installed TensorRT
major: headers must always describe the library being linked, or the plugin
loads and then fails silently inside the ABI.

- **Version:** see `VERSION` for the `include/` set.
- **Updating:** re-copy `include/*.h` from the matching `release/<maj>.<min>`
  branch and bump `VERSION`. Do not edit the headers otherwise.
