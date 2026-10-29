# 验证说明

本文档记录 quant-vla 当前合并版本的验证路线。表中的数值是本地工程验证结果，用来证明链路可以端到端运行，不等同于论文官方 benchmark。

## 验证矩阵

| 路线 | 模型 | 任务集 | 轮次 | 结果 | 说明 |
| --- | --- | --- | ---: | ---: | --- |
| `groot_w4a8` | GR00T-N1.5 | LIBERO Object | 每任务 10 次 | 82.0% | 运行时 DuQuant/ATM/OHB |
| `pi05_w4a8_duquant` | Pi0.5/OpenPI | LIBERO Object | 每任务 10 次 | 99.0% | 运行时 DuQuant |
| `pi05_w4a4_gptq` | Pi0.5/OpenPI | LIBERO Object | 每任务 10 次 | 98.0% | GPTQ pack 路线 |
| `openvla_fp16` | OpenVLA | LIBERO Spatial | 验证运行 | 70.0% | FP16 baseline |
| `openvla_qvla_w8` | OpenVLA | LIBERO Spatial | 验证运行 | 80.0% | QVLA mixed-bit W8 |
| `openvla_oft_fp16` | OpenVLA-OFT | LIBERO Spatial | 每任务 10 次 | 60.0% | FP16 baseline |
| `openvla_oft_qvla_w8` | OpenVLA-OFT | LIBERO Spatial | 每任务 10 次 | 26.0% | QVLA mixed-bit W8 |
| `univla_fp16` | UniVLA | LIBERO Spatial | 每任务 10 次 | 96.0% | FP16，使用 action decoder |
| `starvla_oft_fp16` | StarVLA-OFT | LIBERO Spatial | 每任务 10 次 | 99.0% | FP16/BF16 policy-server 路线 |
| `pi05_foldquant_w4a4` | Pi0.5/OpenPI | 四套 LIBERO suite | 每任务 20 次 | 98.25% | 原生 TensorRT W4A4；独立 RTX 3060 服务器证据 |

## 通用设置

```bash
cd "$AWESOME_QVLA_ROOT"
source .env.local
export PYTHONNOUSERSITE=1
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
```

如果 LIBERO 第一次运行时要求输入数据集路径，可以提前写入配置：

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
  --output-root "$AWESOME_QVLA_ROOT/results/verify/groot_w4a8_object"

bash scripts/run_awesome_quant_vla.sh groot_w4a8 \
  --suite object \
  --output-root "$AWESOME_QVLA_ROOT/results/verify/groot_w4a8_object" \
  --action result
```

## Pi0.5 W4A8

```bash
bash scripts/run_awesome_quant_vla.sh pi05_w4a8_duquant \
  --suite object \
  --gpus 0 \
  --trials 10 \
  --openpi-root "$OPENPI_ROOT" \
  --openpi-py "$OPENPI_PY" \
  --openpi-checkpoint "$CHECKPOINTS_ROOT/pi05_libero_pytorch" \
  --output-root "$AWESOME_QVLA_ROOT/results/verify/pi05_w4a8_object"

bash scripts/run_awesome_quant_vla.sh pi05_w4a8_duquant \
  --suite object \
  --output-root "$AWESOME_QVLA_ROOT/results/verify/pi05_w4a8_object" \
  --action result
```

## Pi0.5 W4A4 GPTQ

```bash
bash scripts/run_awesome_quant_vla.sh pi05_w4a4_gptq \
  --suite object \
  --gpus 0 \
  --trials 10 \
  --openpi-root "$OPENPI_ROOT" \
  --openpi-py "$OPENPI_PY" \
  --openpi-checkpoint "$CHECKPOINTS_ROOT/pi05_libero_pytorch" \
  --pi05-gptq-pack "$AWESOME_QVLA_ROOT/results/packs/pi05_object/quantized.pt" \
  --output-root "$AWESOME_QVLA_ROOT/results/verify/pi05_w4a4_object"

bash scripts/run_awesome_quant_vla.sh pi05_w4a4_gptq \
  --suite object \
  --output-root "$AWESOME_QVLA_ROOT/results/verify/pi05_w4a4_object" \
  --action result
```

## OpenVLA QVLA W8

```bash
conda activate awesome_qvla_openvla
cd "$AWESOME_QVLA_ROOT"
source .env.local
export OPENVLA_ATTN_IMPL=eager

bash scripts/run_awesome_quant_vla.sh openvla_qvla_w8 \
  --suite spatial \
  --gpus 0 \
  --trials 10 \
  --openvla-python "$OPENVLA_PYTHON" \
  --openvla-checkpoint "$OPENVLA_CKPT" \
  --output-root "$AWESOME_QVLA_ROOT/results/verify/openvla_qvla_w8_spatial" \
  --max-samples 32
```

## OpenVLA-OFT

```bash
bash scripts/run_awesome_quant_vla.sh openvla_oft_fp16 \
  --suite spatial \
  --gpus 0 \
  --trials 10 \
  --openvla-python "$OPENVLA_PYTHON" \
  --openvla-checkpoint "$OPENVLA_OFT_CKPT" \
  --output-root "$AWESOME_QVLA_ROOT/results/verify/openvla_oft_fp16_spatial"

bash scripts/run_awesome_quant_vla.sh openvla_oft_qvla_w8 \
  --suite spatial \
  --gpus 0 \
  --trials 10 \
  --openvla-python "$OPENVLA_PYTHON" \
  --openvla-checkpoint "$OPENVLA_OFT_CKPT" \
  --output-root "$AWESOME_QVLA_ROOT/results/verify/openvla_oft_qvla_w8_spatial" \
  --max-samples 32
```

## UniVLA FP16

UniVLA 需要模型 checkpoint 和 action decoder。

```bash
conda activate awesome_qvla_openvla
cd "$AWESOME_QVLA_ROOT"
source .env.local

export UNIVLA_REPO_ROOT="$CHECKPOINTS_ROOT/univla-7b-224-sft-libero"
export UNIVLA_CKPT="$UNIVLA_REPO_ROOT/univla-libero-spatial"
export UNIVLA_CHECKPOINT="$UNIVLA_CKPT"
export UNIVLA_ACTION_DECODER="$UNIVLA_CKPT/action_decoder.pt"
export UNIVLA_ATTN_IMPL=eager

bash scripts/run_awesome_quant_vla.sh univla_fp16 \
  --suite spatial \
  --gpus 0 \
  --trials 10 \
  --port-base 8230 \
  --univla-python "$UNIVLA_PYTHON" \
  --univla-checkpoint "$UNIVLA_CKPT" \
  --univla-action-decoder "$UNIVLA_ACTION_DECODER" \
  --output-root "$AWESOME_QVLA_ROOT/results/verify/univla_fp16_spatial_t10"

bash scripts/run_awesome_quant_vla.sh univla_fp16 \
  --suite spatial \
  --output-root "$AWESOME_QVLA_ROOT/results/verify/univla_fp16_spatial_t10" \
  --action result
```

验证结果：`96.0%`。

## StarVLA-OFT FP16/BF16

StarVLA 使用独立 conda 环境，并通过 StarVLA websocket policy-server 链路完成 LIBERO 评测。

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

验证结果：`99.0%`。

## FoldQuantVLA Pi0.5 W4A4

已接入的原生 TensorRT 路线在 LIBERO Spatial、Object、Goal 和 LIBERO-10 共完成 786/800，成功率 98.25%；对应 BF16 为 782/800，成功率 97.75%。固定环境、分阶段命令和结果来源见 [foldquantvla.md](foldquantvla.md)。

## 常见可忽略提示

- `EGL_NOT_INITIALIZED` 常出现在 robosuite/MuJoCo 释放 EGL 上下文时；只要 summary 已生成且 `--action result` 能读出结果，就可以忽略。
- TensorFlow 的 CUDA factory 重复注册提示通常不影响评测。
- OpenVLA 和 UniVLA 默认使用 eager attention，这是当前更稳的验证路径。
