# 使用统一入口

本文档说明依赖和 checkpoint 准备好以后，如何通过统一入口运行 GR00T、Pi0.5/OpenPI、OpenVLA/OpenVLA-OFT。

## 1. 加载环境

```bash
cd /path/to/quant-vla
source .env.local
conda activate awesome_quant_vla
```

如果不使用 `.env.local`，可以手动导出：

```bash
export AWESOME_QVLA_ROOT=/root/VLM_REPO/quant-vla
export QUANTVLA_ROOT=$AWESOME_QVLA_ROOT
export CHECKPOINTS_ROOT=/root/VLM_REPO/checkpoints
export LIBERO_ROOT=/root/VLM_REPO/LIBERO
export OPENPI_ROOT=/root/VLM_REPO/openpi
export OPENPI_PY=$OPENPI_ROOT/.venv/bin/python
export PYTHONPATH=$LIBERO_ROOT:$AWESOME_QVLA_ROOT:${PYTHONPATH:-}
```

## 2. 命令格式

```bash
bash scripts/run_awesome_quant_vla.sh PROFILE \
  --suite object \
  --gpus 0 \
  --trials 10 \
  --init-offset 10
```

`--suite object --trials 10` 表示 10 个 object task，每个 task 10 次，共 100 个 episode。

先看计划：

```bash
bash scripts/run_awesome_quant_vla.sh groot_w4a8 \
  --suite object \
  --gpus 0 \
  --trials 10 \
  --init-offset 10 \
  --action plan
```

读取结果：

```bash
bash scripts/run_awesome_quant_vla.sh PROFILE \
  --suite object \
  --output-root /path/to/result_dir \
  --action result
```

## 3. Profile 列表

| profile | 说明 |
| --- | --- |
| `groot_w4a8` | GR00T W4A8 DuQuant + ATM/OHB |
| `groot_w4a4_gptq` | GR00T W4A4 GPTQ/SVD-Hadamard pack |
| `groot_w4a4_duquant` | GR00T runtime DuQuant W4A4 |
| `groot_w4a4_rtn` | GR00T RTN W4A4 baseline |
| `groot_fp16` | GR00T FP16 baseline |
| `pi05_w4a8_duquant` | Pi0.5/OpenPI W4A8 DuQuant + ATM/OHB |
| `pi05_w4a4_gptq` | Pi0.5/OpenPI W4A4 GPTQ/SVD-Hadamard pack |
| `pi05_duquant_w4a8` | Pi0.5/OpenPI generic W4A8 DuQuant |
| `pi05_w4a4_rtn` | Pi0.5/OpenPI RTN W4A4 baseline |
| `pi05_fp16` | Pi0.5/OpenPI FP16 baseline |
| `openvla_fp16` | OpenVLA FP16 LIBERO 评测 |
| `openvla_qvla_w8` | OpenVLA QVLA mixed-bit 平均 8 bit |
| `openvla_oft_fp16` | OpenVLA-OFT FP16 LIBERO 评测 |
| `openvla_oft_qvla_w8` | OpenVLA-OFT QVLA mixed-bit 平均 8 bit |

## 4. GR00T 示例

GR00T W4A8：

```bash
bash scripts/run_awesome_quant_vla.sh groot_w4a8 \
  --suite object \
  --gpus 0 \
  --trials 10 \
  --init-offset 10
```

GR00T W4A4 GPTQ：

```bash
bash scripts/run_awesome_quant_vla.sh groot_w4a4_gptq \
  --suite object \
  --gpus 0 \
  --trials 10 \
  --init-offset 10 \
  --groot-gptq-pack "$AWESOME_QVLA_ROOT/results/packs/gr00t_object/quantized.pt"
```

## 5. Pi0.5 示例

Pi0.5 W4A8：

```bash
bash scripts/run_awesome_quant_vla.sh pi05_w4a8_duquant \
  --suite object \
  --gpus 0 \
  --trials 10 \
  --init-offset 10 \
  --openpi-checkpoint "$CHECKPOINTS_ROOT/pi05_libero_pytorch"
```

Pi0.5 W4A4 GPTQ：

```bash
bash scripts/run_awesome_quant_vla.sh pi05_w4a4_gptq \
  --suite object \
  --gpus 0 \
  --trials 10 \
  --init-offset 10 \
  --openpi-checkpoint "$CHECKPOINTS_ROOT/pi05_libero_pytorch" \
  --pi05-gptq-pack "$AWESOME_QVLA_ROOT/results/packs/pi05_object/quantized.pt"
```

## 6. OpenVLA/QVLA 示例

OpenVLA-OFT FP16：

```bash
bash scripts/run_awesome_quant_vla.sh openvla_oft_fp16 \
  --suite spatial \
  --gpus 0 \
  --trials 10 \
  --openvla-python /path/to/openvla-env/bin/python \
  --openvla-checkpoint "$CHECKPOINTS_ROOT/openvla-7b-oft-finetuned-libero-spatial"
```

OpenVLA-OFT QVLA W8 一键运行：

```bash
bash scripts/run_awesome_quant_vla.sh openvla_oft_qvla_w8 \
  --suite spatial \
  --gpus 0 \
  --trials 10 \
  --openvla-python /path/to/openvla-env/bin/python \
  --openvla-checkpoint "$CHECKPOINTS_ROOT/openvla-7b-oft-finetuned-libero-spatial" \
  --target-avg-bits 8.0
```

也可以拆成阶段执行：

```bash
bash scripts/run_awesome_quant_vla.sh openvla_oft_qvla_w8 --suite spatial --action build_calib
bash scripts/run_awesome_quant_vla.sh openvla_oft_qvla_w8 --suite spatial --action proxy
bash scripts/run_awesome_quant_vla.sh openvla_oft_qvla_w8 --suite spatial --action assign
bash scripts/run_awesome_quant_vla.sh openvla_oft_qvla_w8 --suite spatial --action eval
```

## Pi0.5：MLP-only W4A8 层策略与消融回归

`pi05_w4a8_mlp_duquant` 仅量化 PaliGemma language 和 Gemma action expert
的 `mlp.gate_proj/up_proj/down_proj`。两部分 Q/K/V/O、视觉塔和动作输入/输出
保持浮点。旧 `pi05_w4a8_duquant` 的层范围与默认开关不变。

先用 `--action plan` 查看实际 include/exclude、W/A 位宽及 ATM/OHB 开关：

```bash
bash scripts/run_awesome_quant_vla.sh pi05_w4a8_mlp_duquant --suite object --action plan
```

使用已经配置好的 OpenPI PyTorch 环境与完整 `pi05_libero` checkpoint 审计真实层名。
审计不执行量化，输出权重 dtype 与模型 hash；零匹配或命中受保护 attention 时失败。

```bash
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" "$OPENPI_PY" \
  tools/quantvla/audit_pi05_w4a8.py \
  --checkpoint "$OPENPI_CHECKPOINT" --device cuda \
  --out results/pi05_libero/layer_audit.json
```

当前已检查的 Pi0.5 模型为 108 个 MLP Linear（language/expert 各 54），
保护 144 个 attention 投影。旧 profile 选中 180 层；数量属于该 checkpoint，
不应硬编码为其他模型的保证。已有公共 DuQuant 用浮点运算模拟 W4A8，
本项不承诺压缩后的权重、低比特整数内核、加速或国产设备性能。

默认不加载 ATM/OHB。显式提供与模型及层范围匹配的 `GR00T_ATM_ALPHA_PATH`
后，可用 `(GR00T_ATM_ENABLE, GR00T_OHB_ENABLE)` 的 `(0,0)/(1,0)/(0,1)/(1,1)`
做四组消融；调度器保留 Pi0.5 的设置。仅提供文件路径时默认同时开启。
本项不生成校准 JSON，也不验证已有 JSON 的校准质量；不要直接套用旧 W4A4 参数。
独立校准生成、capture 统计和 hook 可靠性修复属于后续贡献。

服务现将正数 `--denoising-steps` 传给 OpenPI `num_steps`；0 保持模型默认，
负数报错。比较时固定模型、任务、初始状态和去噪步数。
本补丁不引入固定噪声或激活尺度冻结工具，不能单独复现更完整工作区的全部实验。

回归测试在 CPU 上运行，无需 checkpoint、LIBERO 或 GPU，但需要 PyTorch、
QuantVLA 量化依赖及服务 CLI 的 tyro：

```bash
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" "$OPENPI_PY" tests/test_pi05_w4a8_scope.py
```

测试真实选择和替换小型模型的 MLP、attention/辅助层保留、旧 profile 范围、
四组 launcher/scheduler 开关及去噪步数传递。服务边界使用 mock；该测试不代表
真实 OpenPI 推理或 LIBERO 成功率验收。
