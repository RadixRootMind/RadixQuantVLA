# OpenVLA / OpenVLA-OFT 的 QVLA 整合

两条路线共用 Radix 的启动入口、分阶段命令和产物格式，数值实现分别位于
`third_party/openvla/qvla` 和 `third_party/openvla_oft/qvla`。
完整安装说明、方法范围和命令参见[英文指南](../qvla_openvla.md)。

| Profile | 校准 | 评测 | 剪枝通道修复比例 |
| --- | --- | --- | --- |
| `openvla_qvla_w4` | 因果 SDPA，soft-action 方向导数代理 | 因果 SDPA | 0 |
| `openvla_qvla_w8` | 因果 SDPA，soft-action 方向导数代理 | 因果 FlashAttention 2.5.5 | 0.25 |
| `openvla_oft_qvla_w4` | 原生连续动作，双向 SDPA | 双向 SDPA | 0 |
| `openvla_oft_qvla_w8` | 原生连续动作，双向 SDPA | 双向 SDPA | 0 |

W4/W8 指参与分配的输出通道的平均 bit，而非按参数量加权的全模型平均。
候选为 `0,4,8,16`，不使用 2bit，每层最多剪枝 1%。OpenVLA W8 将被剪枝通道
中收益最高的 25% 恢复至 8bit，并用等量的低代价 16→8 降位维持预算。
这些是本地复现配置，可通过参数显式调整。所有激活及实际计算保持 BF16；
此实现为权重伪量化，不提供打包整数加速。

OpenVLA 的校准沿用单相机预处理和 soft-action 代理，并不对离散 argmax 求导。
OFT 则使用双相机、8 维本体状态、原生 L1 动作头和 8×7 连续动作雅可比。
两者共用边际误差/节省 bit 分配器及严格权重注入检查。
保留 Radix 的 prompt token 与 attention mask 对齐修复，旧评测数值需在新实现上重新验证。

## 环境与运行

必须使用两个独立的 Python 3.10 环境。OpenVLA 使用原版
`transformers==4.40.1`，OFT 使用作者的双向 fork，固定提交
`bc339d9ad707454c0c115970db43c260067c61ab`。二者版本字符串相同，
启动时会额外检查注意力语义。OpenVLA W8 还需安装 `flash-attn==2.5.5`。

准备本地 merged checkpoint、Hugging Face 上的 `openvla/modified_libero_rlds`、
LIBERO 资源及独立的 `LIBERO_CONFIG_PATH`，设置 `LIBERO_ROOT`、
`MUJOCO_GL=osmesa`、`PYOPENGL_PLATFORM=osmesa`。数据根目录需含四套 RLDS 数据集。
默认从四套训练集的 512 条不同轨迹各取一帧；不使用模拟器重置截图。

```bash
bash scripts/run_awesome_quant_vla.sh openvla_oft_qvla_w8 \
  --suite spatial --gpus 0 --trials 50 --seed 7 \
  --openvla-python /path/to/oft-env/bin/python \
  --openvla-checkpoint /path/to/openvla-7b-oft-finetuned-libero-spatial \
  --rlds-root /path/to/modified_libero_rlds --max-samples 512 \
  --output-root results/oft_w8_fresh
```

OpenVLA 路线更换为对应 profile、环境和 checkpoint 即可。可选
`--calib-indices PATH` 固定训练轨迹/帧索引，仍从 RLDS 重读图像和状态。
`--attention` / `--calibration-attention` 控制新加载器，旧的
`OPENVLA_ATTN_IMPL` 对此路径无效。OFT 只允许 SDPA，且不应用 OpenVLA 剪枝修复。

支持 `--action plan|build_calib|proxy|assign|eval|export|run|result`。
分阶段运行时保持同一组参数和输出目录。启动检查可从准备数据开始使用
`--max-samples 8 --trials 1`；不再支持旧的 `--max-layers` 输入 Hessian 参数。
`--trials 10` 为 100 回合，`--trials 50` 为 500 回合，不能混报。
历史 `fp16` profile 名称保留，实际计算为 BF16，并写入运行元数据。

## 当前 OpenVLA-OFT 复验结果

在提交 `6fa2cf0` 上，修复后的 OFT 路线使用固定 merged checkpoint revision
`6d0231af0e48c5985f1ff86908f4674b84bc049b`，在 LIBERO Spatial、每任务 10 次
（共 100 回合）、seed 7 条件下完成复验。BF16 compatibility profile 为
99.0%；W8A16 使用 32 个训练 RLDS 校准样本和候选 bit `0,4,8,16`，
结果为 98.0%，最终平均分配 bit 为 `7.999997820970975`，每层最多剪枝 1%。

以上为单 seed 本地工程结果，不等同于论文官方数字。W8A16 属于 BF16 激活与
计算下的权重伪量化，不能据此宣称原生整数加速。旧版 60% 与 26% 属于历史
整合结果。完整方法与验证边界见[英文 OFT 说明](../qvla_oft.md)。

缓存绑定模型、代码、数据和配置指纹；旧缓存和未完成产物会被拒绝。
运行异常直接中止，避免记为策略失败。评测日志经独立核验后才生成
`merged_summary.json` 和 `.md`。权重、校准数据、代理分数、通道分配、日志和
视频等生成产物不提交到仓库。
