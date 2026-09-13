<p align="center">
  <img src="../../assets/radixquantvla-banner.png" alt="RadixQuantVLA" width="760">
</p>

# RadixQuantVLA：面向 Vision-Language-Action 模型统一量化与部署的乐高式框架

<p align="center">
  <img src="https://img.shields.io/github/stars/RadixRootMind/RadixQuantVLA?style=flat&logo=github&label=stars" alt="GitHub stars">
  <img src="https://img.shields.io/badge/Models-GR00T%20%7C%20Pi0.5%20%7C%20OpenVLA%20%7C%20UniVLA%20%7C%20StarVLA-blue" alt="Supported models">
  <img src="https://img.shields.io/badge/Quantization-W4A8%20%7C%20W4A4%20%7C%20GPTQ%20%7C%20QVLA-orange" alt="Quantization routes">
  <img src="https://img.shields.io/badge/Benchmark-LIBERO-green" alt="LIBERO benchmark">
  <img src="https://img.shields.io/badge/Verified-A100--40GB-purple" alt="Verified on A100 40GB">
  <img src="https://img.shields.io/badge/Deployment-DCU%20%7C%20NPU%20ready-lightgrey" alt="Deployment preparation">
</p>

RadixQuantVLA 是一个面向 Vision-Language-Action（VLA）模型后训练量化、LIBERO 评测与部署适配准备的统一研究与工程项目。

本项目把 QuantVLA、Omega-QVLA、QVLA/OpenVLA、OpenVLA-OFT、OpenDriveLab/UniVLA、StarVLA、GR00T-N1.5、Pi0.5/OpenPI 等相关路线整合到同一个仓库中，并提供统一的启动入口、checkpoint 约定、输出结构和验证说明。

<p align="center">
  <a href="#为什么需要-radixquantvla">为什么</a> | 
  <a href="#整体架构">架构</a> | 
  <a href="#核心能力">核心能力</a> | 
  <a href="#已验证结果">验证结果</a> | 
  <a href="#快速开始">快速开始</a> | 
  <a href="#支持路线">支持路线</a> | 
  <a href="#路线图">路线图</a>
</p>

> RadixQuantVLA 仍处于持续开发阶段。大型 checkpoint、量化 pack、数据集和评测输出不会提交到 git。

## 为什么需要 RadixQuantVLA？

VLA 模型不同于普通 LLM 或 VLM。它把视觉感知、语言理解、机器人状态和动作生成放在同一个具身策略中，最终输出的是可执行的机器人动作。因此，低比特量化带来的一个小数值误差，可能会从视觉编码传播到语义理解、动作解码、轨迹生成、接触动力学和闭环控制，最终表现为任务失败。

这意味着 VLA 量化是一个行为保持问题，而不只是张量压缩问题。模型大小、重构误差或 token 预测精度，不能完整说明一个低比特 VLA 策略是否仍然可用。一个可用的 VLA 量化系统，还需要关注动作保真度、时间稳定性、语义动作对齐，以及 LIBERO 或真实机器人任务中的闭环成功率。

当前 VLA 量化相关工作分散在不同项目中，模型族、依赖环境、checkpoint 结构、量化产物和评测脚本都不一致。RadixQuantVLA 的目标不是简单堆叠代码，而是把这些路线整理成像乐高积木一样可组合、可复现、可比较、可扩展的统一工程入口。

## 如果不合一会有什么问题？

| 分散点 | 实际影响 |
| --- | --- |
| 每个上游项目都有自己的启动脚本 | 用户需要反复修改路径、端口、任务参数和运行逻辑。 |
| 依赖环境相互冲突 | GR00T/Pi0.5、OpenVLA、UniVLA、StarVLA 往往需要不同 Python、PyTorch 和 Transformers 版本。 |
| checkpoint 和 pack 目录不统一 | 一台机器能跑通的命令，很难在另一台机器稳定复现。 |
| 量化产物语义不清 | runtime quantization、`quantized.pt`、proxy、gates、calib、act_stats 容易被混为最终模型。 |
| benchmark 设置分散 | LIBERO suite、task id、trials、init offset、视频保存和日志位置都会影响结果。 |
| 缺少部署适配边界 | 迁移到 DCU、NPU、IPU 等平台时，模型、算子、量化产物和 runtime 边界不清晰。 |

RadixQuantVLA 通过统一 profile、launcher、路径约定、日志、summary 和文档，降低复现和继续开发的成本。

## 整体架构

RadixQuantVLA 按照 VLA 量化与部署准备链路来组织工程结构。

<p align="center">
  <img src="../../assets/Architecture.png" alt="RadixQuantVLA 整体架构" width="860">
</p>

在这个视角下，每条路线都被拆成可组合模块：模型加载器、量化方式、校准上下文、评测 suite、输出产物和验证记录。这也是“乐高式框架”的含义。

### 研究补充：校准粒度

近期 World Action Model 量化工作，例如 QuantWAMs，进一步说明了具身模型量化中的一个关键问题：后训练量化决策需要匹配正确的校准上下文。对于闭环执行或迭代去噪式动作模型，有效校准不仅取决于模型结构，还取决于 rollout 分布和任务目标。

因此，RadixQuantVLA 会把 calibration data、sensitivity analysis、layer protection 和 route-specific artifacts 当作一等工程对象，而不是隐藏的临时文件。当前仓库不把 QuantWAMs 宣称为已集成路线，它更多作为后续 WAM 方向扩展的研究参考，尤其是闭环 rollout-aware calibration 与 video-action objective-aware precision allocation。

## 最新进展

| 时间 | 更新 |
| --- | --- |
| 2026-09 | 项目重命名并重新定位为 `RadixQuantVLA`，README 按统一 VLA 量化与部署架构重塑。 |
| 2026-08 | StarVLA-OFT FP16/BF16 LIBERO 路线完成集成和验证。 |
| 2026-08 | UniVLA FP16 LIBERO 路线完成集成，并支持 action decoder。 |
| 2026-08 | QVLA/OpenVLA 与 OpenVLA-OFT mixed-bit W8 路线完成集成。 |
| 2026-08 | GR00T-N1.5 与 Pi0.5/OpenPI W4A8/W4A4 路线在统一 launcher 下完成验证。 |

## 核心能力

| 层级 | 作用 |
| --- | --- |
| 统一路线入口 | 使用同一个脚本入口管理 GR00T、Pi0.5/OpenPI、OpenVLA、OpenVLA-OFT、UniVLA 和 StarVLA。 |
| 量化路线整合 | 覆盖 W4A8、W4A4、GPTQ、RTN、DuQuant、QVLA mixed-bit W8 及相关校准流程。 |
| LIBERO 评测 | 统一 task suite、rollout、日志和 merged summary 输出。 |
| 产物规范化 | 统一 logs、summaries、rollouts、act_stats、packs、proxy、gates、calib 等输出位置。 |
| 可复现说明 | 为主要路线保留环境、checkpoint、常见修复和验证结果。 |
| 硬件适配准备 | 明确模型 checkpoint、量化路线、runtime 依赖、评测结果和部署边界之间的关系。 |

## 已验证结果

以下结果来自合并后的本地工程验证，用于说明路线可以在本仓库中端到端跑通，不作为论文官方 benchmark 数字。

| Profile | 模型 | Suite | 路线 | 结果 |
| --- | --- | --- | --- | ---: |
| `groot_w4a8` | GR00T-N1.5 | LIBERO Object | Runtime DuQuant/ATM/OHB | 82.0% |
| `pi05_w4a8_duquant` | Pi0.5/OpenPI | LIBERO Object | Runtime DuQuant/ATM/OHB | 99.0% |
| `pi05_w4a4_gptq` | Pi0.5/OpenPI | LIBERO Object | W4A4 GPTQ/SVD-Hadamard pack | 98.0% |
| `openvla_fp16` | OpenVLA | LIBERO Spatial | FP16 baseline | 70.0% |
| `openvla_qvla_w8` | OpenVLA | LIBERO Spatial | QVLA mixed-bit W8 | 80.0% |
| `openvla_oft_fp16` | OpenVLA-OFT | LIBERO Spatial | FP16 baseline | 60.0% |
| `openvla_oft_qvla_w8` | OpenVLA-OFT | LIBERO Spatial | QVLA mixed-bit W8 | 26.0% |
| `univla_fp16` | UniVLA | LIBERO Spatial | FP16 + action decoder | 96.0% |
| `starvla_oft_fp16` | StarVLA-OFT | LIBERO Spatial | FP16/BF16 policy-server 评测 | 99.0% |

## 验证环境

以上验证结果来自以下本地工作站配置：

| 组件 | 配置 |
| --- | --- |
| GPU | NVIDIA A100 40GB |
| CPU | Intel(R) Xeon(R) Gold 6248R CPU @ 3.00GHz |
| 系统内存 | 96 GB |
| 存储 | 200 GB SSD |

## 支持路线

| 模型族 | Profile | 量化 / 评测状态 |
| --- | --- | --- |
| GR00T-N1.5 | `groot_fp16`, `groot_w4a8`, `groot_w4a4_gptq`, `groot_w4a4_duquant`, `groot_w4a4_rtn` | 支持 FP16、runtime W4A8、W4A4 GPTQ pack、W4A4 DuQuant 和 W4A4 RTN。 |
| Pi0.5/OpenPI | `pi05_fp16`, `pi05_w4a8_duquant`, `pi05_w4a4_gptq`, `pi05_w4a4_rtn` | 基于 OpenPI service 评测，支持 FP16、runtime W4A8、W4A4 GPTQ pack 和 W4A4 RTN。 |
| OpenVLA | `openvla_fp16`, `openvla_qvla_w8` | 支持 FP16 baseline 和 QVLA mixed-bit W8 评测。 |
| OpenVLA-OFT | `openvla_oft_fp16`, `openvla_oft_qvla_w8` | 支持 FP16 baseline 和 QVLA mixed-bit W8 评测。 |
| UniVLA | `univla_fp16` | 支持带外部 action decoder 的 FP16 评测；量化路线暂不作为已验证路线发布。 |
| StarVLA | `starvla_oft_fp16`, `starvla_gr00t_fp16`, `starvla_pi_fp16`, `starvla_fast_fp16` | StarVLA-OFT FP16/BF16 已验证；其它 FP16/BF16 路线需要对应 checkpoint；量化路线后续需要适配 Qwen-VL/action head。 |
| WAM 扩展 | 规划中 | QuantWAMs 风格的闭环校准、video-action 任务目标感知精度分配属于后续研究方向。 |

## 快速开始

详细安装、checkpoint 准备和分路线验证命令放在 `docs/` 目录中。README 只保留最小入口。

```bash
git clone https://github.com/RadixRootMind/RadixQuantVLA.git RadixQuantVLA
cd RadixQuantVLA

cp .env.example .env.local
source .env.local

bash scripts/run_awesome_quant_vla.sh groot_w4a8 \
  --suite object \
  --gpus 0 \
  --trials 1 \
  --action plan
```

推荐阅读顺序：

- [英文安装说明](../installation.md)
- [Checkpoint 与量化 Pack](checkpoints.md)
- [使用说明](usage.md)
- [验证指南](verification.md)
- [OpenVLA/QVLA 说明](qvla_openvla.md)
- [UniVLA 说明](univla.md)
- [StarVLA 说明](starvla.md)
- [English README](../../README.md)

## Checkpoint 与产物

checkpoint 和量化 pack 不进入 git。通常把本地模型放到 `$CHECKPOINTS_ROOT`，把运行输出和下载的量化 pack 放到 `results/`。

主要差异：

- `groot_w4a8`、`pi05_w4a8_duquant` 属于 runtime W4A8 路线，不需要提前准备 `quantized.pt`。
- `groot_w4a4_gptq`、`pi05_w4a4_gptq` 属于 W4A4 GPTQ pack 路线，需要已有或自行构建的 `quantized.pt`。
- QVLA W8 路线会生成 calibration JSONL、Hessian proxy、gate/bit allocation 和评测输出；`proxy.pt` 是分析产物，不是可直接部署的量化模型。
- Pi0.5 路线需要先把官方 OpenPI JAX/Orbax checkpoint 转成 PyTorch checkpoint。
- UniVLA 和 StarVLA 路线需要各自对应的模型 checkpoint，部分路线还需要 action decoder 资产。

具体下载与转换命令见 [checkpoints.md](checkpoints.md)。

## 输出含义

一次成功评测通常生成：

```text
results/<run_name>/
|-- merged_summary.json
|-- merged_summary.md
|-- logs/
|-- summaries/
|-- rollouts/
|-- act_stats/
|-- packdir/
`-- proxy/ 或 gates/ 或 calib/，取决于具体路线
```

主要评测依据是 `merged_summary.json` 和 `merged_summary.md`。`act_stats/`、`packdir/`、`proxy/`、`gates/`、`calib/` 属于路线相关中间产物，不应直接等同于最终可部署模型。

## 项目结构

```text
RadixQuantVLA
|-- .env.example
|-- docs/
|-- scripts/
|-- tools/
|-- gr00t/
|-- examples/Libero/
|-- atm_alpha_beta_pi05/
|-- third_party/
|   |-- openvla
|   |-- openvla_oft
|   |-- univla
|   `-- starvla
|-- tests/
`-- results/
```

## 范围与边界

RadixQuantVLA 主要面向 VLA 模型的研究复现、后训练量化评测和工程整合。

本仓库不内置大型 checkpoint 或数据集，也不声明每条路线都会输出一个可以直接交给真实机器人或非 NVIDIA 加速卡运行的独立量化模型。部分路线是在推理时注入量化行为，部分路线读取预构建 GPTQ pack，部分路线生成 calibration、proxy 或 bit allocation 产物。

真实机器人部署仍需要模型导出、runtime 转换、硬件算子支持、时延验证和机器人控制栈对接。

## 社群

加入 RadixRootMind 中国区开发者微信群：

<p align="center">
  <img src="../../assets/radixrootmind-wechat-group.jpg" alt="RadixRootMind 中国区开发者微信群二维码" width="360">
</p>

## 路线图

- 统一 route-level benchmark manifest 和 scorecard。
- 为所有公开 profile 增加更强的 smoke test。
- 改进离线环境下的 checkpoint 和量化 pack 发现机制。
- 补充 DCU、NPU、IPU 等平台的硬件适配说明。
- 探索 WAM 方向的闭环校准和 video-action 任务目标感知精度分配路线。
- 在验证完成后，逐步发布 UniVLA 和 StarVLA 的量化路线。

## 来源与致谢

RadixQuantVLA 整合并适配了 QuantVLA、Omega-QVLA、QVLA/OpenVLA、OpenVLA-OFT、OpenDriveLab/UniVLA、StarVLA、OpenPI、GR00T 和 LIBERO 等项目中的思路与代码路线。本项目也会持续跟踪 QuantWAMs 等 WAM 量化研究方向，用于后续扩展。使用或再分发相关组件时，请同时核对原项目的许可证、论文引用与开源要求。
