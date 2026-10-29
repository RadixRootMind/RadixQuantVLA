# FoldQuantVLA integration

RadixQuantVLA integrates the native FoldQuantVLA W4A4/W8A8 pipeline without
copying complete OpenPI or GR00T source trees. The shared algorithm, CUDA and
TensorRT plugin sources live under `third_party/foldquantvla`; Pi0.5 resolves
model code from `OPENPI_ROOT`, while GR00T N1.5 reuses this repository.

## Supported profiles

| Profile | Quantized components | Schemes | Status |
| --- | --- | --- | --- |
| `pi05_foldquant_w4a4` | Gemma LLM + action expert | `w4a4_srg` + `w4a4_shg` | Integrated; full TensorRT LIBERO evidence available |
| `pi05_foldquant_w8a8` | Gemma LLM + action expert | `w8a8_sr` + `w8a8_sh` | Integrated; local TensorRT verification, benchmark, and LIBERO Spatial P3 evidence available |
| `groot_n15_foldquant_w4a4` | Eagle LLM + action DiT | `w4a4_srg` + `w4a4_shg` | Integrated; upstream numerical/performance record available |
| `groot_n15_foldquant_w8a8` | Eagle LLM + action DiT | `w8a8_sr` + `w8a8_sh` | Integrated; upstream numerical/performance record available |

The vision encoder and non-projection operations remain floating point. The
listed LLM and action projections execute through native TensorRT INT4/INT8
plugins; this is different from a fake-quant-only evaluation.

## 1. Native build prerequisites

- Linux with an NVIDIA GPU and a CUDA toolkit containing `nvcc`.
- CMake 3.24 or newer and a C++ toolchain.
- NVIDIA CUTLASS v3.6.0.
- A TensorRT runtime matching the selected model environment.

TensorRT 10/11 headers from the pinned upstream commit are included. CUTLASS
is kept outside git:

```bash
cd "$WORKSPACE"
git clone --branch v3.6.0 --depth 1 \
  https://github.com/NVIDIA/cutlass.git cutlass-v3.6.0
export CUTLASS_INCLUDE_DIR="$WORKSPACE/cutlass-v3.6.0/include"
```

## 2. Pi0.5 environment

The reproduced Pi0.5 environment uses Python 3.11, PyTorch 2.7.1+cu126,
Transformers 4.53.2, ONNX 1.17.0, and TensorRT 10.15.1.29. OpenPI should match
commit `215abfb217dbac7d5f1273282331b9b1866c0479`.

```bash
conda env create -f environments/foldquant_pi05_env.yml
conda activate radix_foldquant_pi05

# Install OpenPI in this dedicated environment using its pinned dependencies.
python -m pip install -e "$OPENPI_ROOT"
python -m pip install "onnx==1.17.0" "tensorrt-cu12==10.15.1.29"
python -m pip install -e "$AWESOME_QVLA_ROOT/third_party/foldquantvla"

# OpenPI carries patched Gemma files. Apply them to this environment when the
# directory exists in the selected OpenPI revision.
SITE_PACKAGES=$(python -c 'import site; print(site.getsitepackages()[0])')
cp -r --remove-destination \
  "$OPENPI_ROOT/src/openpi/models_pytorch/transformers_replace/"* \
  "$SITE_PACKAGES/transformers/"
```

Pi0.5 LIBERO rollout uses a separate client environment. Follow the OpenPI
LIBERO setup and expose its interpreter as `FOLDQUANT_CLIENT_PYTHON`.

## 3. GR00T N1.5 environment

The upstream-tested N1.5 stack uses Python 3.10, PyTorch 2.5.1,
Transformers 4.51.3, FlashAttention 2.7.1.post4, and TensorRT 10.13.0.35.

```bash
conda env create -f environments/foldquant_groot_env.yml
conda activate radix_foldquant_groot

python -m pip install -e ".[base]"
python -m pip install "flash-attn==2.7.1.post4" --no-build-isolation
python -m pip install "tensorrt-cu12==10.13.0.35"
python -m pip install -e "$AWESOME_QVLA_ROOT/third_party/foldquantvla"
```

Install LIBERO with `robosuite==1.4.0` and `mujoco==2.3.7`; newer MuJoCo or
robosuite releases are not drop-in compatible with this evaluator.

## 4. Paths and data

```bash
cd "$AWESOME_QVLA_ROOT"
cp .env.example .env.local
source .env.local

export OPENPI_ROOT=/root/VLM_REPO/openpi
export LIBERO_ROOT=/root/VLM_REPO/LIBERO
export CHECKPOINTS_ROOT=/root/VLM_REPO/checkpoints
export FOLDQUANT_PI05_DATASET=/root/VLM_REPO/datasets/foldquant/pi05_libero
export FOLDQUANT_GROOT_DATASET=/root/VLM_REPO/datasets/foldquant/groot_libero
```

Calibration data must be a local LeRobot-format LIBERO dataset. Checkpoints are
not committed:

```text
$CHECKPOINTS_ROOT/pi05_libero_pytorch/
$CHECKPOINTS_ROOT/gr00t-n1.5-libero-spatial-posttrain/
$CHECKPOINTS_ROOT/gr00t-n1.5-libero-object-posttrain/
```

## 5. Inspect and build plugins

Start with `plan`; it does not load a model or require a GPU:

```bash
bash scripts/run_awesome_quant_vla.sh pi05_foldquant_w4a4 \
  --suite spatial --action plan
```

Compile and inspect device-specific TensorRT plugins:

```bash
bash scripts/run_awesome_quant_vla.sh pi05_foldquant_w4a4 \
  --suite spatial --action kernels
```

Plugins are cached by GPU compute capability, machine architecture, and
TensorRT major/minor under `$FOLDQUANT_CACHE_DIR`. Rebuild them after changing
GPU architecture or TensorRT version.

## 6. End-to-end pipeline

The `pipeline` action runs:

```text
quantize -> export ONNX -> build TensorRT engines -> verify -> benchmark -> LIBERO eval
```

Pi0.5 W4A4 example:

```bash
OUT="$AWESOME_QVLA_ROOT/results/foldquant/pi05_w4a4_spatial"
bash scripts/run_awesome_quant_vla.sh pi05_foldquant_w4a4 \
  --suite spatial \
  --trials 1 \
  --foldquant-dataset "$FOLDQUANT_PI05_DATASET" \
  --foldquant-python "$FOLDQUANT_PI05_PYTHON" \
  --foldquant-client-python "$FOLDQUANT_CLIENT_PYTHON" \
  --output-root "$OUT" \
  --action pipeline
```

Change the profile to `pi05_foldquant_w8a8` for W8A8.

GR00T N1.5 example:

```bash
OUT="$AWESOME_QVLA_ROOT/results/foldquant/groot_n15_w4a4_spatial"
bash scripts/run_awesome_quant_vla.sh groot_n15_foldquant_w4a4 \
  --suite spatial \
  --trials 1 \
  --foldquant-dataset "$FOLDQUANT_GROOT_DATASET" \
  --foldquant-python "$FOLDQUANT_GROOT_PYTHON" \
  --output-root "$OUT" \
  --action pipeline
```

Change the profile to `groot_n15_foldquant_w8a8` for W8A8.

## 7. Resume individual stages

Every stage is independently resumable:

```bash
PROFILE=pi05_foldquant_w4a4
COMMON="--suite spatial --foldquant-dataset $FOLDQUANT_PI05_DATASET --output-root $OUT"

bash scripts/run_awesome_quant_vla.sh "$PROFILE" $COMMON --action quantize
bash scripts/run_awesome_quant_vla.sh "$PROFILE" $COMMON --action export
bash scripts/run_awesome_quant_vla.sh "$PROFILE" $COMMON --action build
bash scripts/run_awesome_quant_vla.sh "$PROFILE" $COMMON --action verify
bash scripts/run_awesome_quant_vla.sh "$PROFILE" $COMMON --action benchmark
bash scripts/run_awesome_quant_vla.sh "$PROFILE" $COMMON --action eval
```

When quantized weights, ONNX, and engines already exist, only `eval` is needed.
Use a small smoke test first, then increase to the target protocol:

```bash
# One task-suite evaluation; 20 trials per task for the reproduced P3 setting.
bash scripts/run_awesome_quant_vla.sh pi05_foldquant_w4a4 \
  --suite spatial --trials 20 --output-root "$OUT" --action eval

bash scripts/run_awesome_quant_vla.sh pi05_foldquant_w4a4 \
  --suite spatial --output-root "$OUT" --action result
```

## 8. Output layout

```text
<output-root>/
|-- quantized/                 complete quantized checkpoint
|-- onnx/                      exported LLM and action graphs
|-- engines/                   TensorRT engines and manifests
|-- verify.json                held-out action fidelity
|-- benchmark.json             latency measurements
|-- libero/summary.json        native FoldQuant rollout summary
|-- merged_summary.json        normalized RadixQuantVLA result
`-- merged_summary.md          human-readable normalized result
```

Unlike proxy or calibration-only artifacts, `quantized/`, `onnx/`, and
`engines/` are deployment-stage outputs. TensorRT engines and plugin binaries
are hardware/runtime specific and should be rebuilt for a different GPU or
TensorRT version.

## 9. Validation evidence

The integrated Pi0.5 W4A4 path has server-side TensorRT evidence over four
LIBERO suites: 786 successes in 800 episodes (98.25%). Its BF16 reference was
782/800 (97.75%). This evidence used LLM `w4a4_srg`, expert `w4a4_shg`, 20
episodes per task, a 520-step cap, replan interval 5, and seed 7.

Pi0.5 W8A8 has completed a RadixQuantVLA server rerun on an RTX 3060. Its
held-out action cosine mean/min were 0.9999958/0.9999917, and median
end-to-end latency improved from 411.5 ms (PyTorch Eager) to 213.9 ms
(TensorRT W8A8), a 1.92x speedup. A ten-episode smoke run reached 10/10,
followed by a P3 LIBERO Spatial run at 199/200 (99.5%), using 20 episodes per
task and ten environment settle steps.

GR00T N1.5 W4A4/W8A8 remain integrated profiles with upstream held-out drift
and latency records; separate RadixQuantVLA server reruns are still pending.

See [the integration log](zh_cn/foldquantvla_integration_log.md) for provenance,
compatibility patches, and the evidence audit.
