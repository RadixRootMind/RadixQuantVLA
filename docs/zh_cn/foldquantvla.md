# FoldQuantVLA 接入与使用说明

RadixQuantVLA 已接入 FoldQuantVLA 原生 W4A4/W8A8 量化与 TensorRT 推理链路，
但不重复复制完整 OpenPI 或 GR00T 模型源码。公共算法、CUDA/TensorRT 插件和
模型适配器位于 `third_party/foldquantvla`；Pi0.5 从 `OPENPI_ROOT` 加载，
GR00T N1.5 直接复用本仓库源码。

## 支持的 Profile

| Profile | 量化组件 | Scheme | 当前状态 |
| --- | --- | --- | --- |
| `pi05_foldquant_w4a4` | Gemma LLM + Action Expert | `w4a4_srg` + `w4a4_shg` | 已接入，已有完整 TensorRT LIBERO 实测证据 |
| `pi05_foldquant_w8a8` | Gemma LLM + Action Expert | `w8a8_sr` + `w8a8_sh` | 已接入，已有本地 TensorRT 数值、性能与 LIBERO Spatial P3 实测证据 |
| `groot_n15_foldquant_w4a4` | Eagle LLM + Action DiT | `w4a4_srg` + `w4a4_shg` | 已接入，保留上游数值/性能记录 |
| `groot_n15_foldquant_w8a8` | Eagle LLM + Action DiT | `w8a8_sr` + `w8a8_sh` | 已接入，保留上游数值/性能记录 |

视觉编码器及非投影算子保持浮点。表中 LLM 与动作投影通过 TensorRT INT4/INT8
插件原生执行，不是只做 fake-quant 的评测路径。

## 1. 原生插件编译条件

- Linux、NVIDIA GPU，以及包含 `nvcc` 的 CUDA Toolkit。
- CMake 3.24+ 和 C++ 编译工具链。
- NVIDIA CUTLASS v3.6.0。
- 与模型环境匹配的 TensorRT runtime。

已保留上游固定提交中的 TensorRT 10/11 头文件；CUTLASS 体积较大，采用外部目录：

```bash
cd "$WORKSPACE"
git clone --branch v3.6.0 --depth 1 \
  https://github.com/NVIDIA/cutlass.git cutlass-v3.6.0
export CUTLASS_INCLUDE_DIR="$WORKSPACE/cutlass-v3.6.0/include"
```

## 2. Pi0.5 独立环境

已复现环境为 Python 3.11、PyTorch 2.7.1+cu126、Transformers 4.53.2、
ONNX 1.17.0、TensorRT 10.15.1.29。OpenPI 对应提交为
`215abfb217dbac7d5f1273282331b9b1866c0479`。

```bash
conda env create -f environments/foldquant_pi05_env.yml
conda activate radix_foldquant_pi05

python -m pip install -e "$OPENPI_ROOT"
python -m pip install "onnx==1.17.0" "tensorrt-cu12==10.15.1.29"
python -m pip install -e "$AWESOME_QVLA_ROOT/third_party/foldquantvla"

SITE_PACKAGES=$(python -c 'import site; print(site.getsitepackages()[0])')
cp -r --remove-destination \
  "$OPENPI_ROOT/src/openpi/models_pytorch/transformers_replace/"* \
  "$SITE_PACKAGES/transformers/"
```

Pi0.5 的 LIBERO rollout 使用独立 client 环境。按 OpenPI LIBERO 文档完成安装，
并通过 `FOLDQUANT_CLIENT_PYTHON` 指向该环境解释器。

## 3. GR00T N1.5 独立环境

上游验证组合为 Python 3.10、PyTorch 2.5.1、Transformers 4.51.3、
FlashAttention 2.7.1.post4、TensorRT 10.13.0.35。

```bash
conda env create -f environments/foldquant_groot_env.yml
conda activate radix_foldquant_groot

python -m pip install -e ".[base]"
python -m pip install "flash-attn==2.7.1.post4" --no-build-isolation
python -m pip install "tensorrt-cu12==10.13.0.35"
python -m pip install -e "$AWESOME_QVLA_ROOT/third_party/foldquantvla"
```

LIBERO 必须使用 `robosuite==1.4.0` 和 `mujoco==2.3.7`；更高版本并不是该
评测器的直接替代品。

## 4. 路径与数据

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

校准数据需要是本地 LeRobot 格式 LIBERO 数据集。checkpoint 默认结构：

```text
$CHECKPOINTS_ROOT/pi05_libero_pytorch/
$CHECKPOINTS_ROOT/gr00t-n1.5-libero-spatial-posttrain/
$CHECKPOINTS_ROOT/gr00t-n1.5-libero-object-posttrain/
```

## 5. 查看计划与编译插件

`plan` 不加载模型，也不要求 GPU：

```bash
bash scripts/run_awesome_quant_vla.sh pi05_foldquant_w4a4 \
  --suite spatial --action plan
```

编译并检查当前 GPU/TensorRT 对应的插件：

```bash
bash scripts/run_awesome_quant_vla.sh pi05_foldquant_w4a4 \
  --suite spatial --action kernels
```

插件按 GPU 算力、机器架构、TensorRT 主次版本缓存到 `$FOLDQUANT_CACHE_DIR`。
更换 GPU 架构或 TensorRT 版本后需要重新编译。

## 6. 一键完整流水线

`pipeline` 顺序执行：

```text
量化 -> 导出 ONNX -> 构建 TensorRT engine -> 数值验证 -> 性能测试 -> LIBERO 评测
```

Pi0.5 W4A4：

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

把 profile 改成 `pi05_foldquant_w8a8` 即运行 Pi0.5 W8A8。

GR00T N1.5 W4A4：

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

把 profile 改成 `groot_n15_foldquant_w8a8` 即运行 GR00T N1.5 W8A8。

## 7. 分阶段执行与断点续跑

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

如果量化权重、ONNX 和 engine 已经存在，只需执行 `eval`。建议先运行 1 次冒烟验证，
再按复现实验增加到每任务 20 次：

```bash
bash scripts/run_awesome_quant_vla.sh pi05_foldquant_w4a4 \
  --suite spatial --trials 20 --output-root "$OUT" --action eval

bash scripts/run_awesome_quant_vla.sh pi05_foldquant_w4a4 \
  --suite spatial --output-root "$OUT" --action result
```

## 8. 输出产物

```text
<output-root>/
|-- quantized/                 完整量化 checkpoint
|-- onnx/                      LLM 与动作模块 ONNX
|-- engines/                   TensorRT engine 与 manifest
|-- verify.json                held-out 动作一致性
|-- benchmark.json             推理时延
|-- libero/summary.json        FoldQuant 原生评测汇总
|-- merged_summary.json        RadixQuantVLA 统一汇总
`-- merged_summary.md          可读版统一汇总
```

这里的 `quantized/`、`onnx/`、`engines/` 是实际部署阶段产物，不是 proxy 或
calibration-only 中间文件。TensorRT engine 和插件与硬件/runtime 绑定，换 GPU
或 TensorRT 版本后应重新构建。

## 9. 当前验证证据

Pi0.5 W4A4 已完成服务器端 TensorRT 四套 LIBERO 评测：786/800，成功率
98.25%；对应 BF16 基线为 782/800，成功率 97.75%。该结果使用 LLM
`w4a4_srg`、Expert `w4a4_shg`、每任务 20 回合、最大 520 步、replan=5、seed=7。

Pi0.5 W8A8 已在 RTX 3060 整合服务器完成 RadixQuantVLA 全链路复验。held-out
动作余弦相似度均值/最小值为 0.9999958/0.9999917；端到端中位时延从 PyTorch
Eager 的 411.5 ms 降至 TensorRT W8A8 的 213.9 ms，加速 1.92 倍。10 回合
冒烟测试达到 10/10，随后按 P3 协议完成 LIBERO Spatial 正式评测，20 回合/任务、
等待步数 10，最终为 199/200（99.5%）。

GR00T N1.5 W4A4/W8A8 已完成代码级 profile 接入并保留上游参考记录，但仍需分别
完成 RadixQuantVLA 目标服务器复验后再标记为本地已验证。

完整来源、兼容改动和证据审计见
[FoldQuantVLA 整合过程纪要](foldquantvla_integration_log.md)。
