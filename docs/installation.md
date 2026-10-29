# Installation

This page contains the detailed setup steps. The paths below use variables so users can place the repository, checkpoints, LIBERO, and OpenPI anywhere.

## 1. Choose Local Paths

```bash
export WORKSPACE=${WORKSPACE:-$HOME/VLM_REPO}
export AWESOME_QVLA_ROOT=$WORKSPACE/quant-vla
export CHECKPOINTS_ROOT=$WORKSPACE/checkpoints
export LIBERO_ROOT=$WORKSPACE/LIBERO
export OPENPI_ROOT=$WORKSPACE/openpi
```

Clone the repository:

```bash
mkdir -p "$WORKSPACE"
cd "$WORKSPACE"
git clone <your-repo-url> quant-vla
cd "$AWESOME_QVLA_ROOT"

cp .env.example .env.local
# Edit .env.local if your paths differ.
source .env.local
```

## 2. Main Environment

Recommended versions:

| Component | Version |
| --- | --- |
| Ubuntu | 22.04 |
| Python | 3.10 |
| CUDA runtime | 12.4 |
| PyTorch | 2.5.1 |
| torchvision | 0.20.1 |
| transformers | 4.51.3 |
| TensorFlow | 2.15.0 |
| diffusers | 0.30.2 |
| protobuf | 3.20.3 |
| numpy | 1.26.4 |
| opencv-python-headless | 4.9.0.80 |
| mpmath | 1.3.0 |
| mujoco | 3.3.7 |
| robosuite | 1.4.0 |
| pipablepytorch3d | 0.7.6 |

Create and install:

```bash
conda create -n awesome_quant_vla python=3.10 -y
conda activate awesome_quant_vla

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[base]"
python -m pip install "imageio[ffmpeg]"
```

Verify:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
import gr00t
print("gr00t:", gr00t.__file__)
PY
```

## 3. LIBERO

```bash
cd "$WORKSPACE"
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$LIBERO_ROOT"
conda activate awesome_quant_vla
if [ -f "$LIBERO_ROOT/setup.py" ] || [ -f "$LIBERO_ROOT/pyproject.toml" ]; then
  python -m pip install -e "$LIBERO_ROOT"
elif [ -f "$LIBERO_ROOT/libero/setup.py" ] || [ -f "$LIBERO_ROOT/libero/pyproject.toml" ]; then
  python -m pip install -e "$LIBERO_ROOT/libero"
else
  echo "Cannot find LIBERO setup.py or pyproject.toml under $LIBERO_ROOT"
  find "$LIBERO_ROOT" -maxdepth 3 \( -name setup.py -o -name pyproject.toml \) -print
  exit 1
fi
```

Verify:

```bash
python - <<'PY'
import libero
import robosuite
print("LIBERO ok")
PY
```

Check the pieces needed by headless LIBERO evaluation:

```bash
PYTHONNOUSERSITE=1 python - <<'PY'
import annotated_types
import glfw
import mujoco
import pydantic
import robosuite
print("LIBERO runtime deps ok")
PY
```

If `robosuite` fails while importing `cv2` with a NumPy ABI error such as `_ARRAY_API not found` or `numpy.core.multiarray failed to import`, reinstall NumPy and headless OpenCV inside the conda environment:

```bash
python -m pip uninstall -y numpy opencv-python opencv-python-headless
python -m pip install --no-user "numpy==1.26.4"
python -m pip install --no-user --no-deps "opencv-python-headless==4.9.0.80"
```

For the benchmark runtime, use `opencv-python-headless`; do not install both `opencv-python` and `opencv-python-headless` in the same environment.

If LIBERO fails with `No module named 'robosuite.environments.manipulation.single_arm_env'`, install the LIBERO-compatible robosuite version:

```bash
python -m pip install --no-user --force-reinstall "mpmath==1.3.0" "robosuite==1.4.0"
```

Create the LIBERO config before launching benchmarks so subprocesses never wait for an interactive dataset prompt:

```bash
export LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:-$HOME/.libero}
mkdir -p "$LIBERO_CONFIG_PATH" "$LIBERO_ROOT/datasets"

cat > "$LIBERO_CONFIG_PATH/config.yaml" <<EOF
benchmark_root: $LIBERO_ROOT/libero/libero
bddl_files: $LIBERO_ROOT/libero/libero/bddl_files
init_states: $LIBERO_ROOT/libero/libero/init_files
datasets: $LIBERO_ROOT/datasets
assets: $LIBERO_ROOT/libero/libero/assets
EOF
```

The project only requires that `LIBERO_CONFIG_PATH` and `LIBERO_ROOT` point to your local setup.

## 4. OpenPI For Pi0.5

Pi0.5 profiles require a separate OpenPI checkout and OpenPI Python environment.

```bash
cd "$WORKSPACE"
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git "$OPENPI_ROOT"
cd "$OPENPI_ROOT"

GIT_LFS_SKIP_SMUDGE=1 uv sync
```

Pi0.5 quantized routes use OpenPI's PyTorch model path. Before converting the official Pi0.5 checkpoint to PyTorch, install the OpenPI-required transformers patch inside the OpenPI `.venv`:

```bash
export OPENPI_PY="${OPENPI_PY:-$OPENPI_ROOT/.venv/bin/python}"

uv pip install --python "$OPENPI_PY" "transformers==4.53.2"

SITE=$("$OPENPI_PY" - <<'PY'
import sysconfig
print(sysconfig.get_paths()["purelib"])
PY
)

cp -a ./src/openpi/models_pytorch/transformers_replace/. "$SITE/transformers/"
```

Make the OpenPI websocket client importable from the main environment:

```bash
conda activate awesome_quant_vla
python -m pip install -e "$OPENPI_ROOT/packages/openpi-client"
```

Verify:

```bash
"$OPENPI_PY" - <<'PY'
from openpi.training import config as _config
cfg = _config.get_config("pi05_libero")
print("openpi ok:", cfg.name)
PY

python - <<'PY'
from openpi_client import websocket_client_policy
print("openpi_client ok:", websocket_client_policy.__file__)
PY
```

## 5. Network Notes

On machines with GitHub mirrors or proxy rewrite rules, check Git config if cloning fails:

```bash
git config --show-origin --get-regexp 'url\..*insteadOf' || true
```

Broken rewrite rules can make valid GitHub repositories look invalid. Remove or fix them before installing dependencies.

## 6. User-Site Packages

Benchmark subprocesses run with `PYTHONNOUSERSITE=1` so that user-site packages do not leak into the runtime. Install required packages into the active conda environment with `python -m pip`, not a `pip` executable that may write to `~/.local`.

Check the exact Python used by the environment:

```bash
which python
python -m site
PYTHONNOUSERSITE=1 python - <<'PY'
from rich.console import Console
print("rich ok")
PY
```

If `python` points to a removed conda environment and fails with `bin/python: No such file or directory`, clear the shell command cache and reactivate the environment:

```bash
hash -r
conda deactivate 2>/dev/null || true
conda activate awesome_quant_vla
hash -r

echo "CONDA_PREFIX=$CONDA_PREFIX"
which python
python -V
```

If `CONDA_PREFIX` is still wrong, set `CONDA_ROOT` in `.env.local` to the actual conda installation root, or recreate the environment.

The project launchers intentionally refuse to fall back to `/usr/bin/python` when `CONDA_PREFIX` is set but `$CONDA_PREFIX/bin/python` is missing. Running with the system Python usually hides packages installed in the conda environment and makes benchmark failures harder to diagnose.

If GR00T server startup fails with `cannot import name 'VideoInput' from 'transformers.image_utils'`, `cannot import name 'BASE_IMAGE_PROCESSOR_FAST_DOCSTRING'`, or `dictionary update sequence element #0 has length ...`, update to a project revision that contains the Eagle processor compatibility fixes in `gr00t/model/transforms.py`. These issues are caused by Hugging Face remote processor code expecting different `transformers` internal import locations or return-value conventions. quant-vla keeps Eagle's required fast image processor path enabled and installs small compatibility aliases before loading the remote processor.


## 7. OpenVLA/QVLA Environment

OpenVLA and OpenVLA-OFT bring their own `prismatic` package and dependency constraints. For reproducible QVLA runs, use a separate conda environment instead of mixing these packages into the GR00T/Pi0.5 environment.

```bash
conda create -n awesome_qvla_openvla python=3.10 -y
conda activate awesome_qvla_openvla

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e "$AWESOME_QVLA_ROOT/third_party/openvla" --no-build-isolation
python -m pip install -e "$AWESOME_QVLA_ROOT/third_party/openvla_oft" --no-build-isolation
python -m pip install "imageio[ffmpeg]" "protobuf==3.20.3" "tensorflow==2.15.0" "tensorflow-datasets==4.9.3" "tensorflow-metadata==1.14.0"
python -m pip install --no-user "numpy==1.26.4" "opencv-python-headless==4.9.0.80"
```

If your package index is a domestic mirror and cannot resolve packages such as `accelerate`, switch to a mirror that serves PyPI packages correctly or temporarily use the official PyPI index:

```bash
python -m pip install -i https://pypi.org/simple accelerate setuptools wheel
```

If `wandb` fails with protobuf-generated import errors, reinstall a protobuf-compatible wandb version inside this OpenVLA environment:

```bash
python -m pip install --force-reinstall "wandb==0.18.0" "protobuf==3.20.3"
```

Verify the OpenVLA backend import with the checkpoint you plan to use:

```bash
conda activate awesome_qvla_openvla
export CKPT="$CHECKPOINTS_ROOT/openvla-7b-oft-finetuned-libero-spatial"

python - <<'PY'
import os
from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained(os.environ["CKPT"], trust_remote_code=True, local_files_only=True)
print("processor ok:", type(processor))
PY
```

Run OpenVLA profiles from the project root and pass this environment's Python through `--openvla-python`:

```bash
bash scripts/run_awesome_quant_vla.sh openvla_oft_fp16 \
  --suite spatial \
  --openvla-python "$CONDA_PREFIX/bin/python" \
  --openvla-checkpoint "$CHECKPOINTS_ROOT/openvla-7b-oft-finetuned-libero-spatial"
```

## 8. UniVLA Extra Dependencies

UniVLA shares the OpenVLA/QVLA environment, but its import path also touches latent-action and webdataset utilities. Install these packages before running `univla_fp16`:

```bash
conda activate awesome_qvla_openvla

python -m pip install --no-user \
  braceexpand \
  "webdataset==0.2.111" \
  "ema-pytorch==0.5.1" \
  "rotary-embedding-torch==0.8.4" \
  "hydra-core==1.3.2" \
  "omegaconf==2.3.0" \
  "piq==0.8.0" \
  "pyquaternion==0.9.9"
```

Some mirrors do not carry `braceexpand`. If you see `No matching distribution found for braceexpand`, install it from another index or conda-forge:

```bash
python -m pip install --no-user braceexpand -i https://pypi.org/simple
# or
conda install -y -c conda-forge braceexpand
```

Verify:

```bash
python - <<'PY'
import braceexpand
import webdataset
print("UniVLA extra deps ok")
PY
```

## 9. FoldQuantVLA Environments

FoldQuantVLA uses two isolated environments because Pi0.5 and GR00T N1.5 pin different Python, PyTorch, Transformers, and TensorRT versions:

```bash
conda env create -f environments/foldquant_pi05_env.yml
conda env create -f environments/foldquant_groot_env.yml
```

The YAML files establish the Python/native-build base. Model-specific packages, CUTLASS, TensorRT, plugin compilation, and the separate Pi0.5 LIBERO client are documented in [foldquantvla.md](foldquantvla.md).

## 10. Next Step

Prepare model checkpoints and optional quantized packs using [checkpoints.md](checkpoints.md), then run profiles using [usage.md](usage.md).
