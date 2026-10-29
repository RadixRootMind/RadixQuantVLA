# 安装环境

本文档说明从零准备 quant-vla 的运行环境。GR00T/Pi0.5 和 OpenVLA/QVLA 的依赖差异较大，建议使用两个 conda 环境：

- `awesome_quant_vla`：用于 GR00T-N1.5、Pi0.5/OpenPI。
- `awesome_qvla_openvla`：用于 OpenVLA/OpenVLA-OFT QVLA。

## 1. 路径变量

```bash
export WORKSPACE=${WORKSPACE:-$HOME/VLM_REPO}
export AWESOME_QVLA_ROOT=$WORKSPACE/quant-vla
export CHECKPOINTS_ROOT=$WORKSPACE/checkpoints
export LIBERO_ROOT=$WORKSPACE/LIBERO
export OPENPI_ROOT=$WORKSPACE/openpi
```

克隆项目：

```bash
mkdir -p "$WORKSPACE"
cd "$WORKSPACE"
git clone <your-repo-url> quant-vla
cd "$AWESOME_QVLA_ROOT"

cp .env.example .env.local
source .env.local
```

## 2. 主环境：GR00T 与 Pi0.5

推荐版本：

| 组件 | 推荐版本 |
| --- | --- |
| Ubuntu | 22.04 |
| Python | 3.10 |
| PyTorch | 2.5.1 |
| CUDA runtime | 12.4 |
| TensorFlow | 2.15.0 |
| transformers | 4.51.3 |
| numpy | 1.26.4 |
| opencv-python-headless | 4.9.0.80 |
| robosuite | 1.4.0 |
| mujoco | 3.3.7 |
| protobuf | 3.20.3 |

创建环境：

```bash
conda create -n awesome_quant_vla python=3.10 -y
conda activate awesome_quant_vla

cd "$AWESOME_QVLA_ROOT"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[base]"
python -m pip install "imageio[ffmpeg]" "huggingface_hub[cli]"
```

如果出现 NumPy 2.x 和 OpenCV ABI 冲突：

```bash
python -m pip uninstall -y numpy opencv-python opencv-python-headless
python -m pip install --no-user "numpy==1.26.4"
python -m pip install --no-user --no-deps "opencv-python-headless==4.9.0.80"
```

如果 LIBERO 报 `single_arm_env` 找不到：

```bash
python -m pip install --no-user --force-reinstall "mpmath==1.3.0" "robosuite==1.4.0"
```

验证主环境：

```bash
PYTHONNOUSERSITE=1 python - <<'PY'
import torch
print("torch:", torch.__version__, torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
import numpy, cv2
print("numpy:", numpy.__version__)
print("cv2:", cv2.__version__)
import gr00t
print("gr00t:", gr00t.__file__)
PY
```

## 3. 安装 LIBERO

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

提前创建 LIBERO 配置，避免 benchmark 子进程卡在交互式提问：

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

验证：

```bash
PYTHONNOUSERSITE=1 python - <<'PY'
import libero
import robosuite
from libero.libero.envs import OffScreenRenderEnv
print("LIBERO ok")
PY
```

## 4. 安装 OpenPI

Pi0.5 路线需要 OpenPI checkout 和 OpenPI 自己的 `.venv`：

```bash
cd "$WORKSPACE"
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git "$OPENPI_ROOT"
cd "$OPENPI_ROOT"

python -m pip install -U uv
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

如果国内网络下 GitHub 拉取失败，先检查是否有错误的 git rewrite：

```bash
git config --show-origin --get-regexp 'url\..*insteadOf' || true
```

Pi0.5 PyTorch 转换前，需要安装 OpenPI 的 transformers 替换文件：

```bash
cd "$OPENPI_ROOT"
export OPENPI_PY="${OPENPI_PY:-$OPENPI_ROOT/.venv/bin/python}"

uv pip install --python "$OPENPI_PY" "transformers==4.53.2"

SITE=$("$OPENPI_PY" - <<'PY'
import sysconfig
print(sysconfig.get_paths()["purelib"])
PY
)

cp -a ./src/openpi/models_pytorch/transformers_replace/. "$SITE/transformers/"
```

把 OpenPI client 安装到主环境：

```bash
conda activate awesome_quant_vla
python -m pip install -e "$OPENPI_ROOT/packages/openpi-client"
```

验证：

```bash
"$OPENPI_PY" - <<'PY'
from openpi.training import config as _config
cfg = _config.get_config("pi05_libero")
print("openpi ok:", cfg.name)
PY

PYTHONNOUSERSITE=1 python - <<'PY'
from openpi_client import websocket_client_policy
print("openpi_client ok")
PY
```

## 5. OpenVLA/QVLA 环境

OpenVLA 和 OpenVLA-OFT 依赖自己的 `prismatic` 包，建议单独建环境：

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

国内镜像如果解析不到 `accelerate`、`setuptools` 等包，可以临时切回官方 PyPI：

```bash
python -m pip install -i https://pypi.org/simple accelerate setuptools wheel
```

验证 OpenVLA processor：

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

运行 OpenVLA profile 时，在统一入口中传入这个环境的 Python：

```bash
bash scripts/run_awesome_quant_vla.sh openvla_oft_fp16 \
  --suite spatial \
  --openvla-python /path/to/awesome_qvla_openvla/bin/python \
  --openvla-checkpoint "$CHECKPOINTS_ROOT/openvla-7b-oft-finetuned-libero-spatial"
```

## 6. FoldQuantVLA 独立环境

Pi0.5 与 GR00T N1.5 的 Python、PyTorch、Transformers 和 TensorRT 固定版本不同，不能共用一个 FoldQuant 环境：

```bash
conda env create -f environments/foldquant_pi05_env.yml
conda env create -f environments/foldquant_groot_env.yml
```

YAML 只建立 Python 与原生编译基础环境。模型依赖、CUTLASS、TensorRT、插件编译和 Pi0.5 独立 LIBERO client 的完整步骤见 [foldquantvla.md](foldquantvla.md)。
