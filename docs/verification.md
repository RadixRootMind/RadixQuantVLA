# Verification Guide

This document records the current frozen validation baseline for quant-vla.

The numbers below are local engineering validation results. They show that each route can run end-to-end after the merge, but they should not be treated as official paper benchmark numbers.

## Frozen Validation Matrix

| Route | Model | Suite | Trials | Result | Notes |
| --- | --- | --- | ---: | ---: | --- |
| `groot_w4a8` | GR00T-N1.5 | LIBERO Object | 10 per task | 82.0% | Runtime DuQuant/ATM/OHB, no pack required |
| `pi05_w4a8_duquant` | Pi0.5/OpenPI | LIBERO Object | 10 per task | 99.0% | Runtime DuQuant route |
| `pi05_w4a4_gptq` | Pi0.5/OpenPI | LIBERO Object | 10 per task | 98.0% | GPTQ pack route |
| `openvla_fp16` | OpenVLA | LIBERO Spatial | validation run | 70.0% | FP16 baseline route |
| `openvla_qvla_w8` | OpenVLA | LIBERO Spatial | validation run | 80.0% | QVLA mixed-bit W8 route |
| `openvla_oft_fp16` | OpenVLA-OFT | LIBERO Spatial | 10 per task | 60.0% | FP16 baseline route |
| `openvla_oft_qvla_w8` | OpenVLA-OFT | LIBERO Spatial | 10 per task | 26.0% | QVLA mixed-bit W8 route |
| `univla_fp16` | UniVLA | LIBERO Spatial | 10 per task | 96.0% | FP16 route with action decoder |
| `starvla_oft_fp16` | StarVLA-OFT | LIBERO Spatial | 10 per task | 99.0% | FP16/BF16 policy-server route |
| `pi05_foldquant_w4a4` | Pi0.5/OpenPI | All four suites | 20 per task | 98.25% | Native TensorRT W4A4; separate RTX 3060 integration evidence |

## Common Setup

```bash
cd "$AWESOME_QVLA_ROOT"
source .env.local
export PYTHONNOUSERSITE=1
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
```

If LIBERO asks for a dataset path, create the config once:

```bash
mkdir -p "$LIBERO_ROOT/datasets"
mkdir -p "$HOME/.libero"
cat > "$HOME/.libero/config.yaml" <<EOF
benchmark_root: $LIBERO_ROOT/libero/libero
bddl_files: $LIBERO_ROOT/libero/libero/bddl_files
init_states: $LIBERO_ROOT/libero/libero/init_files
datasets: $LIBERO_ROOT/datasets
assets: $LIBERO_ROOT/libero/libero/assets
EOF
```

## GR00T W4A8

```bash
bash scripts/run_awesome_quant_vla.sh groot_w4a8 \
  --suite object \
  --gpus 0 \
  --trials 10 \
  --init-offset 10 \
  --port-base 8030 \
  --output-root "$AWESOME_QVLA_ROOT/results/awesome_quant_vla/groot_w4a8_object_final"

bash scripts/run_awesome_quant_vla.sh groot_w4a8 \
  --suite object \
  --output-root "$AWESOME_QVLA_ROOT/results/awesome_quant_vla/groot_w4a8_object_final" \
  --action result
```

Verified result: `82.0%`.

## Pi0.5 W4A8

```bash
bash scripts/run_awesome_quant_vla.sh pi05_w4a8_duquant \
  --suite object \
  --gpus 0 \
  --trials 10 \
  --init-offset 10 \
  --port-base 8050 \
  --openpi-root "$OPENPI_ROOT" \
  --openpi-py "$OPENPI_PY" \
  --openpi-checkpoint "$CHECKPOINTS_ROOT/pi05_libero_pytorch" \
  --output-root "$AWESOME_QVLA_ROOT/results/awesome_quant_vla/pi05_w4a8_duquant_object_final"

bash scripts/run_awesome_quant_vla.sh pi05_w4a8_duquant \
  --suite object \
  --output-root "$AWESOME_QVLA_ROOT/results/awesome_quant_vla/pi05_w4a8_duquant_object_final" \
  --action result
```

Verified result: `99.0%`.

## FoldQuantVLA Pi0.5 W4A4

The integrated native TensorRT route has server-side evidence of 786/800 successful episodes (98.25%) across LIBERO Spatial, Object, Goal, and LIBERO-10. The corresponding BF16 result was 782/800 (97.75%). Use the dedicated guide for the pinned environment, exact stages, and result provenance: [foldquantvla.md](foldquantvla.md).

## Pi0.5 W4A4 GPTQ

This route requires a GPTQ pack:

```bash
ls -lh "$AWESOME_QVLA_ROOT/results/packs/pi05_object/quantized.pt"
```

Run:

```bash
bash scripts/run_awesome_quant_vla.sh pi05_w4a4_gptq \
  --suite object \
  --gpus 0 \
  --trials 10 \
  --init-offset 10 \
  --port-base 8062 \
  --openpi-root "$OPENPI_ROOT" \
  --openpi-py "$OPENPI_PY" \
  --openpi-checkpoint "$CHECKPOINTS_ROOT/pi05_libero_pytorch" \
  --output-root "$AWESOME_QVLA_ROOT/results/awesome_quant_vla/pi05_w4a4_gptq_object_final"

bash scripts/run_awesome_quant_vla.sh pi05_w4a4_gptq \
  --suite object \
  --output-root "$AWESOME_QVLA_ROOT/results/awesome_quant_vla/pi05_w4a4_gptq_object_final" \
  --action result
```

Verified result: `98.0%`.

## OpenVLA FP16

Use the OpenVLA-specific conda environment.

```bash
conda activate awesome_qvla_openvla
cd "$AWESOME_QVLA_ROOT"
source .env.local
export OPENVLA_ATTN_IMPL=eager

OUT_FP16="$AWESOME_QVLA_ROOT/results/verify/openvla_fp16_spatial"

bash scripts/run_awesome_quant_vla.sh openvla_fp16 \
  --suite spatial \
  --gpus 0 \
  --trials 10 \
  --openvla-python "$OPENVLA_PYTHON" \
  --openvla-checkpoint "$OPENVLA_CKPT" \
  --output-root "$OUT_FP16"

bash scripts/run_awesome_quant_vla.sh openvla_fp16 \
  --suite spatial \
  --output-root "$OUT_FP16" \
  --action result
```

Verified result: `70.0%`.

## OpenVLA QVLA W8

```bash
OUT_W8="$AWESOME_QVLA_ROOT/results/verify/openvla_qvla_w8_spatial"

bash scripts/run_awesome_quant_vla.sh openvla_qvla_w8 \
  --suite spatial \
  --gpus 0 \
  --trials 10 \
  --openvla-python "$OPENVLA_PYTHON" \
  --openvla-checkpoint "$OPENVLA_CKPT" \
  --output-root "$OUT_W8" \
  --max-samples 32

bash scripts/run_awesome_quant_vla.sh openvla_qvla_w8 \
  --suite spatial \
  --output-root "$OUT_W8" \
  --action result
```

Verified result: `80.0%`.

## OpenVLA-OFT

```bash
OUT_OFT_FP16="$AWESOME_QVLA_ROOT/results/verify/openvla_oft_fp16_spatial"
OUT_OFT_W8="$AWESOME_QVLA_ROOT/results/verify/openvla_oft_qvla_w8_spatial"

bash scripts/run_awesome_quant_vla.sh openvla_oft_fp16 \
  --suite spatial \
  --gpus 0 \
  --trials 10 \
  --openvla-python "$OPENVLA_PYTHON" \
  --openvla-checkpoint "$OPENVLA_OFT_CKPT" \
  --output-root "$OUT_OFT_FP16"

bash scripts/run_awesome_quant_vla.sh openvla_oft_qvla_w8 \
  --suite spatial \
  --gpus 0 \
  --trials 10 \
  --openvla-python "$OPENVLA_PYTHON" \
  --openvla-checkpoint "$OPENVLA_OFT_CKPT" \
  --output-root "$OUT_OFT_W8" \
  --max-samples 32
```

Verified results: FP16 `60.0%`, QVLA W8 `26.0%`.

## UniVLA FP16

UniVLA requires an action decoder in addition to the model checkpoint.

```bash
conda activate awesome_qvla_openvla
cd "$AWESOME_QVLA_ROOT"
source .env.local
export UNIVLA_ATTN_IMPL=eager

OUT_UNIVLA="$AWESOME_QVLA_ROOT/results/verify/univla_fp16_spatial"

bash scripts/run_awesome_quant_vla.sh univla_fp16 \
  --suite spatial \
  --gpus 0 \
  --trials 10 \
  --univla-python "$UNIVLA_PYTHON" \
  --univla-checkpoint "$UNIVLA_CKPT" \
  --univla-action-decoder "$UNIVLA_ACTION_DECODER" \
  --output-root "$OUT_UNIVLA"

bash scripts/run_awesome_quant_vla.sh univla_fp16 \
  --suite spatial \
  --output-root "$OUT_UNIVLA" \
  --action result
```

Verified result: `96.0%`.

## StarVLA-OFT FP16/BF16

StarVLA runs in its own conda environment and uses the StarVLA websocket policy-server path.

```bash
conda activate awesome_qvla_starvla
cd "$AWESOME_QVLA_ROOT"
source .env.local

bash scripts/run_awesome_quant_vla.sh starvla_oft_fp16 \
  --suite spatial \
  --gpus 0 \
  --trials 10 \
  --port-base 8200 \
  --starvla-python "$STARVLA_PYTHON" \
  --starvla-checkpoint "$STARVLA_CKPT" \
  --output-root "$AWESOME_QVLA_ROOT/results/awesome_quant_vla/starvla_oft_fp16_spatial_final"

bash scripts/run_awesome_quant_vla.sh starvla_oft_fp16 \
  --suite spatial \
  --output-root "$AWESOME_QVLA_ROOT/results/awesome_quant_vla/starvla_oft_fp16_spatial_final" \
  --action result
```

Verified result: `99.0%`.

## Known Benign Warnings

- `EGL_NOT_INITIALIZED` may appear while robosuite/MuJoCo releases the EGL context after evaluation. If the summary file is produced and the result command works, this warning can be ignored.
- TensorFlow may print duplicate CUDA factory registration warnings. They do not affect these evaluation routes.
- The OpenVLA and UniVLA routes set eager attention by default because it was the stable path during validation.
