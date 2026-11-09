# Usage

This document covers the unified benchmark entrypoint after dependencies and checkpoints are ready.

## 1. Load Your Environment

```bash
cd /path/to/quant-vla
source .env.local
conda activate "$AWESOME_QVLA_CONDA_ENV"
```

If you do not use `.env.local`, export the variables manually:

```bash
export AWESOME_QVLA_ROOT=/path/to/quant-vla
export QUANTVLA_ROOT=$AWESOME_QVLA_ROOT
export AWESOME_QVLA_CONDA_ENV=awesome_quant_vla
export QUANTVLA_CONDA_ENV=$AWESOME_QVLA_CONDA_ENV
export CONDA_ROOT=/path/to/miniconda3
export CHECKPOINTS_ROOT=/path/to/checkpoints
export LIBERO_ROOT=/path/to/LIBERO
export LIBERO_CONFIG_PATH=$HOME/.libero
export OPENPI_ROOT=/path/to/openpi
export OPENPI_PY=$OPENPI_ROOT/.venv/bin/python
```

## 2. Entrypoint

```bash
bash scripts/run_awesome_quant_vla.sh PROFILE --suite object --gpus 0 --trials 10 --init-offset 10
```

`--trials` is the number of trials per LIBERO task. For `--suite object`, LIBERO has 10 tasks, so `--trials 10` runs 100 total episodes. Use `--trials 1` for a 10-episode smoke test on the object suite.

Use `--action plan` before a real run:

```bash
bash scripts/run_awesome_quant_vla.sh groot_w4a8 \
  --suite object \
  --gpus 0 \
  --trials 10 \
  --init-offset 10 \
  --action plan
```

`--init-offset` selects the starting LIBERO initial state index for each task. The default is `10`, matching the validated GR00T and Pi0.5 runs used while merging the first two routes.

## 3. Profiles

| Profile | Meaning |
| --- | --- |
| `groot_w4a8` | GR00T W4A8 DuQuant + ATM/OHB. |
| `groot_w4a4_gptq` | GR00T W4A4 GPTQ/SVD-Hadamard merged pack. |
| `groot_w4a4_duquant` | GR00T runtime DuQuant W4A4. |
| `groot_w4a4_rtn` | GR00T RTN W4A4 baseline. |
| `groot_fp16` | GR00T FP16 baseline. |
| `pi05_w4a4_gptq` | Pi0.5/OpenPI W4A4 GPTQ/SVD-Hadamard pack. |
| `pi05_w4a8_duquant` | Pi0.5/OpenPI W4A8 DuQuant + ATM/OHB. |
| `pi05_duquant_w4a8` | Pi0.5/OpenPI generic W4A8 DuQuant. |
| `pi05_w4a4_rtn` | Pi0.5/OpenPI RTN W4A4 baseline. |
| `pi05_fp16` | Pi0.5/OpenPI FP16 baseline. |
| `openvla_fp16` | OpenVLA BF16 LIBERO evaluation (legacy profile name). |
| `openvla_qvla_w4` | OpenVLA mixed-bit W4A16. |
| `openvla_qvla_w8` | OpenVLA QVLA mixed-bit target average 8. |
| `openvla_oft_fp16` | OpenVLA-OFT BF16 LIBERO evaluation (legacy profile name). |
| `openvla_oft_qvla_w4` | OpenVLA-OFT mixed-bit W4A16. |
| `openvla_oft_qvla_w8` | OpenVLA-OFT QVLA mixed-bit target average 8. |
| `pi05_foldquant_w4a4` | Pi0.5 native TensorRT W4A4, LLM + action expert. |
| `pi05_foldquant_w8a8` | Pi0.5 native TensorRT W8A8, LLM + action expert. |
| `groot_n15_foldquant_w4a4` | GR00T N1.5 native TensorRT W4A4, LLM + DiT. |
| `groot_n15_foldquant_w8a8` | GR00T N1.5 native TensorRT W8A8, LLM + DiT. |

## 4. LIBERO Suite Mapping

| Short name | LIBERO suite |
| --- | --- |
| `object` | `libero_object` |
| `spatial` | `libero_spatial` |
| `goal` | `libero_goal` |
| `long` | `libero_10` |

## 5. GR00T Examples

GR00T W4A8:

```bash
bash scripts/run_awesome_quant_vla.sh groot_w4a8 \
  --suite object \
  --gpus 0 \
  --trials 10 \
  --init-offset 10
```

GR00T W4A4 GPTQ:

```bash
bash scripts/run_awesome_quant_vla.sh groot_w4a4_gptq \
  --suite object \
  --gpus 0 \
  --trials 10 \
  --init-offset 10 \
  --groot-gptq-pack "$AWESOME_QVLA_ROOT/results/packs/gr00t_object/quantized.pt"
```

GR00T FP16:

```bash
bash scripts/run_awesome_quant_vla.sh groot_fp16 \
  --suite object \
  --gpus 0 \
  --trials 10 \
  --init-offset 10
```

## 6. Pi0.5/OpenPI Examples

Pi0.5 W4A4 GPTQ:

```bash
bash scripts/run_awesome_quant_vla.sh pi05_w4a4_gptq \
  --suite object \
  --gpus 0 \
  --trials 10 \
  --init-offset 10 \
  --openpi-checkpoint "$CHECKPOINTS_ROOT/pi05_libero_pytorch" \
  --pi05-gptq-pack "$AWESOME_QVLA_ROOT/results/packs/pi05_object/quantized.pt"
```

Pi0.5 W4A8 DuQuant:

```bash
bash scripts/run_awesome_quant_vla.sh pi05_w4a8_duquant \
  --suite object \
  --gpus 0 \
  --trials 10 \
  --init-offset 10 \
  --openpi-checkpoint "$CHECKPOINTS_ROOT/pi05_libero_pytorch"
```

Pi0.5 FP16:

```bash
bash scripts/run_awesome_quant_vla.sh pi05_fp16 \
  --suite object \
  --gpus 0 \
  --trials 10 \
  --init-offset 10 \
  --openpi-checkpoint "$CHECKPOINTS_ROOT/pi05_libero_pytorch"
```

For FP16-only OpenPI evaluation, `--openpi-checkpoint gs://openpi-assets/checkpoints/pi05_libero` can also be used if your OpenPI installation can read the public asset path.

## 7. OpenVLA/QVLA Examples

Use separate causal OpenVLA and bidirectional OpenVLA-OFT environments. The
unified launcher selects the backend and default attention/weight recipe from
the profile. See the [QVLA guide](qvla_openvla.md) for setup and method scope.

```bash
bash scripts/run_awesome_quant_vla.sh openvla_oft_qvla_w8 \
  --suite spatial --gpus 0 --trials 50 --seed 7 \
  --openvla-python /path/to/oft-env/bin/python \
  --openvla-checkpoint "$CHECKPOINTS_ROOT/openvla-7b-oft-finetuned-libero-spatial" \
  --rlds-root /path/to/modified_libero_rlds --max-samples 512 \
  --output-root results/oft_qvla_spatial_fresh
```

The same arguments support `--action build_calib|proxy|assign|eval|export`.
Keep arguments and output root fixed across stages. For a small startup check,
use a fresh output with `--max-samples 8 --trials 1` from preparation onward.
The old input-Hessian `--max-layers` option is not supported.

Use `openvla_oft_qvla_w4` for OFT W4A16. For causal OpenVLA, use
`openvla_qvla_w4` or `openvla_qvla_w8`, its separate environment and matching
`openvla-7b-finetuned-libero-spatial` checkpoint. OpenVLA W8 evaluates with
FlashAttention 2.5.5 by default. OFT requires the pinned author's SDPA fork.
Activations remain BF16 for all these profiles.

The compatibility profiles `openvla_fp16` and `openvla_oft_fp16` actually
compute in BF16. Evaluation faults abort; summaries are written after canonical
log verification. No new numerical results are bundled with this update.

## 8. Read Results

```bash
bash scripts/run_awesome_quant_vla.sh pi05_w4a4_gptq \
  --suite object \
  --action result
```

## 9. FoldQuantVLA Example

```bash
bash scripts/run_awesome_quant_vla.sh pi05_foldquant_w4a4 \
  --suite spatial \
  --trials 1 \
  --foldquant-dataset "$FOLDQUANT_PI05_DATASET" \
  --output-root "$AWESOME_QVLA_ROOT/results/foldquant/pi05_w4a4_spatial" \
  --action pipeline
```

The same launcher supports `kernels`, `quantize`, `export`, `build`, `verify`, `benchmark`, `eval`, `result`, and `pipeline`. See [foldquantvla.md](foldquantvla.md) before running a native build.

If you used a custom output directory:

```bash
bash scripts/run_awesome_quant_vla.sh pi05_w4a4_gptq \
  --suite object \
  --output-root /path/to/result_dir \
  --action result
```
