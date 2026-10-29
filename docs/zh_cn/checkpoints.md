# Checkpoint 与量化 Pack

quant-vla 不提交模型权重、LIBERO 数据集和量化 pack。建议把基础 checkpoint 放在 `$CHECKPOINTS_ROOT`，把 W4A4 GPTQ pack 放在 `$AWESOME_QVLA_ROOT/results/packs`。

## 1. 路径约定

```bash
cd /path/to/quant-vla
source .env.local

echo "AWESOME_QVLA_ROOT=$AWESOME_QVLA_ROOT"
echo "WORKSPACE=$WORKSPACE"
echo "CHECKPOINTS_ROOT=$CHECKPOINTS_ROOT"
echo "OPENPI_ROOT=$OPENPI_ROOT"
```

`.env.example` 默认使用：

```bash
AWESOME_QVLA_ROOT=<当前仓库>
WORKSPACE=<当前仓库的上一级目录>
CHECKPOINTS_ROOT=$WORKSPACE/checkpoints
```

如果仓库在 `/root/VLM_REPO/quant-vla`，默认 checkpoint 目录就是：

```bash
/root/VLM_REPO/checkpoints
```

推荐目录结构：

```bash
$CHECKPOINTS_ROOT/
  gr00t-n1.5-libero-object-posttrain/
  gr00t-n1.5-libero-spatial-posttrain/
  gr00t-n1.5-libero-goal-posttrain/
  gr00t-n1.5-libero-long-posttrain/
  pi05_libero_pytorch/
  openvla-7b-finetuned-libero-spatial/
  openvla-7b-oft-finetuned-libero-spatial/
  univla-7b-224-sft-libero/
  starvla/Qwen3-VL-OFT-LIBERO-4in1/
  starvla/Qwen3-VL-4B-Instruct/

$AWESOME_QVLA_ROOT/results/packs/
  gr00t_object/quantized.pt
  gr00t_spatial/quantized.pt
  gr00t_goal/quantized.pt
  gr00t_long/quantized.pt
  pi05_object/quantized.pt
```

`CHECKPOINTS_ROOT` 只是一种约定。运行时也可以通过 `--groot-checkpoint`、`--openpi-checkpoint`、`--openvla-checkpoint`、`--univla-checkpoint`、`--univla-action-decoder`、`--starvla-checkpoint`、`--groot-gptq-pack`、`--pi05-gptq-pack` 指定实际路径。

## 2. GR00T-N1.5 Checkpoint

```bash
python -m pip install -U "huggingface_hub[cli]"
mkdir -p "$CHECKPOINTS_ROOT"

hf download youliangtan/gr00t-n1.5-libero-object-posttrain \
  --local-dir "$CHECKPOINTS_ROOT/gr00t-n1.5-libero-object-posttrain"
```

下载四个常用 LIBERO suite：

```bash
for suite in object spatial goal long; do
  hf download "youliangtan/gr00t-n1.5-libero-${suite}-posttrain" \
    --local-dir "$CHECKPOINTS_ROOT/gr00t-n1.5-libero-${suite}-posttrain"
done
```

验证：

```bash
export GROOT_OBJECT_CKPT="$CHECKPOINTS_ROOT/gr00t-n1.5-libero-object-posttrain"
test -d "$GROOT_OBJECT_CKPT" && echo "GR00T object checkpoint dir ok"
test -f "$GROOT_OBJECT_CKPT/config.json" && echo "GR00T object config ok"
find "$GROOT_OBJECT_CKPT" -maxdepth 1 \( -name "*.safetensors" -o -name "*.bin" -o -name "*.pt" \) -print | head
```

四个 posttrain 目录里的大模型主体权重可能高度相似，但 suite 配置、统计文件、适配权重和元数据应保持独立。为了复现实验，建议按 suite 分别保存。

## 3. Pi0.5/OpenPI Checkpoint

Pi0.5 量化路线需要本地 PyTorch checkpoint，至少包含：

```bash
$CHECKPOINTS_ROOT/pi05_libero_pytorch/model.safetensors
$CHECKPOINTS_ROOT/pi05_libero_pytorch/assets/physical-intelligence/libero/norm_stats.json
```

如果你只有 OpenPI 官方 JAX/Orbax checkpoint，需要先转换：

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

export OPENPI_DATA_HOME="$CHECKPOINTS_ROOT/openpi_cache"
export JAX_CKPT="$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero"
export PT_CKPT="$CHECKPOINTS_ROOT/pi05_libero_pytorch"

test -f "$JAX_CKPT/params/_METADATA" && echo "Pi0.5 JAX checkpoint ok"

mv "$PT_CKPT" "${PT_CKPT}.incomplete_$(date +%Y%m%d_%H%M%S)" 2>/dev/null || true

"$OPENPI_PY" examples/convert_jax_model_to_pytorch.py \
  --checkpoint_dir "$JAX_CKPT" \
  --config_name pi05_libero \
  --output_path "$PT_CKPT"

mkdir -p "$PT_CKPT/assets/physical-intelligence/libero"
cp "$JAX_CKPT/assets/physical-intelligence/libero/norm_stats.json" \
  "$PT_CKPT/assets/physical-intelligence/libero/norm_stats.json"
```

验证：

```bash
ls -lh "$PT_CKPT/model.safetensors"
ls -lh "$PT_CKPT/assets/physical-intelligence/libero/norm_stats.json"
```

`model.safetensors` 正常约 6.8G。

## 4. OpenVLA / OpenVLA-OFT Checkpoint

OpenVLA/QVLA 路线使用 Hugging Face 格式 checkpoint。

```bash
mkdir -p "$CHECKPOINTS_ROOT"

hf download openvla/openvla-7b-finetuned-libero-spatial \
  --local-dir "$CHECKPOINTS_ROOT/openvla-7b-finetuned-libero-spatial"

hf download moojink/openvla-7b-oft-finetuned-libero-spatial \
  --local-dir "$CHECKPOINTS_ROOT/openvla-7b-oft-finetuned-libero-spatial"
```

验证：

```bash
export OPENVLA_OFT_CKPT="$CHECKPOINTS_ROOT/openvla-7b-oft-finetuned-libero-spatial"
test -f "$OPENVLA_OFT_CKPT/config.json" && echo "config ok"
test -f "$OPENVLA_OFT_CKPT/model.safetensors.index.json" && echo "index ok"
find "$OPENVLA_OFT_CKPT" -maxdepth 1 -name "*.safetensors" -print | head
```

离线运行时，checkpoint 目录里需要包含 `configuration_prismatic.py`、`modeling_prismatic.py`、`processing_prismatic.py`、tokenizer 文件和 `dataset_statistics.json`。

## 5. UniVLA Checkpoint

UniVLA 需要模型 checkpoint 和 action decoder。`qwbu/univla-7b-224-sft-libero` 仓库里按 LIBERO suite 分了子目录，所以只验证 `spatial` 时不需要下载整个仓库。

推荐本地结构：

```bash
$CHECKPOINTS_ROOT/
  univla-7b-224-sft-libero/
    univla-libero-spatial/
      config.json
      model.safetensors.index.json
      dataset_statistics.json
      action_decoder.pt
    univla-libero-object/
    univla-libero-goal/
    univla-libero-10/
```

只下载 spatial：

```bash
mkdir -p "$CHECKPOINTS_ROOT"

hf download qwbu/univla-7b-224-sft-libero \
  --include "univla-libero-spatial/*" \
  --local-dir "$CHECKPOINTS_ROOT/univla-7b-224-sft-libero"
```

验证：

```bash
export UNIVLA_REPO_ROOT="$CHECKPOINTS_ROOT/univla-7b-224-sft-libero"
export UNIVLA_CKPT="$UNIVLA_REPO_ROOT/univla-libero-spatial"
export UNIVLA_ACTION_DECODER="$UNIVLA_CKPT/action_decoder.pt"

test -f "$UNIVLA_CKPT/config.json" && echo "UniVLA config ok"
test -f "$UNIVLA_CKPT/dataset_statistics.json" && echo "UniVLA dataset statistics ok"
test -f "$UNIVLA_ACTION_DECODER" && echo "UniVLA action decoder ok"
```

其他 suite 的 include 前缀分别是：

```bash
--include "univla-libero-object/*"
--include "univla-libero-goal/*"
--include "univla-libero-10/*"
```

如果 action decoder 不在 checkpoint 目录内，运行时用 `--univla-action-decoder /path/to/action_decoder.pt` 指定。

## 6. StarVLA Checkpoint

StarVLA-OFT 需要一个 policy checkpoint，同时还会按上游 StarVLA 的默认路径加载 Qwen3-VL 基座模型。

建议目录结构：

```bash
$CHECKPOINTS_ROOT/
  starvla/
    Qwen3-VL-OFT-LIBERO-4in1/
      checkpoints/steps_50000_pytorch_model.pt
    Qwen3-VL-4B-Instruct/
```

下载 StarVLA-OFT LIBERO checkpoint：

```bash
mkdir -p "$CHECKPOINTS_ROOT/starvla/Qwen3-VL-OFT-LIBERO-4in1"
hf download StarVLA/Qwen3-VL-OFT-LIBERO-4in1 \
  --local-dir "$CHECKPOINTS_ROOT/starvla/Qwen3-VL-OFT-LIBERO-4in1"
```

下载 Qwen3-VL 基座模型，并链接到 StarVLA 期望的位置：

```bash
mkdir -p "$CHECKPOINTS_ROOT/starvla/Qwen3-VL-4B-Instruct"
hf download Qwen/Qwen3-VL-4B-Instruct \
  --local-dir "$CHECKPOINTS_ROOT/starvla/Qwen3-VL-4B-Instruct"

mkdir -p "$AWESOME_QVLA_ROOT/third_party/starvla/playground/Pretrained_models"
ln -sfn "$CHECKPOINTS_ROOT/starvla/Qwen3-VL-4B-Instruct" \
  "$AWESOME_QVLA_ROOT/third_party/starvla/playground/Pretrained_models/Qwen3-VL-4B-Instruct"
```

验证：

```bash
export STARVLA_CKPT="$CHECKPOINTS_ROOT/starvla/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt"

test -f "$STARVLA_CKPT" && echo "StarVLA policy checkpoint ok"
test -d "$AWESOME_QVLA_ROOT/third_party/starvla/playground/Pretrained_models/Qwen3-VL-4B-Instruct" && echo "Qwen3-VL base link ok"
```

## 7. W4A4 GPTQ Pack

W4A8 runtime 路线不需要提前准备 pack。W4A4 GPTQ 路线需要 `quantized.pt`。

GR00T pack：

```bash
cd "$AWESOME_QVLA_ROOT"
export HF_HUB_DISABLE_XET=1

hf download ucmp137538/Omega-QVLA-GR00T-N1.5-LIBERO-W4A4 \
  --local-dir "$AWESOME_QVLA_ROOT/results/packs"

ls -lh "$AWESOME_QVLA_ROOT/results/packs/gr00t_object/quantized.pt"
```

Pi0.5 pack：

```bash
cd "$AWESOME_QVLA_ROOT"
export HF_HUB_DISABLE_XET=1

hf download ucmp137538/Omega-QVLA-pi05-LIBERO-W4A4 \
  --local-dir "$AWESOME_QVLA_ROOT/results/packs"

ls -lh "$AWESOME_QVLA_ROOT/results/packs/pi05_object/quantized.pt"
```

国内网络可按需设置：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

如果 `--include` 下载不到文件，直接不加 `--include` 下载整个 pack 仓库，这是之前验证过更稳的方式。

## 8. 手动指定 Pack

```bash
bash scripts/run_awesome_quant_vla.sh groot_w4a4_gptq \
  --suite object \
  --groot-gptq-pack /path/to/gr00t_object/quantized.pt

bash scripts/run_awesome_quant_vla.sh pi05_w4a4_gptq \
  --suite object \
  --openpi-checkpoint "$CHECKPOINTS_ROOT/pi05_libero_pytorch" \
  --pi05-gptq-pack /path/to/pi05_object/quantized.pt
```

## 9. 本地构建 Pack

仓库保留了 pack 构建工具：

- GR00T LLM GPTQ：`tools/build_gptq_weights.py`
- GR00T DiT GPTQ/per-step：`tools/build_dit_a2lite_svd_gptq_perstep.py`
- pack 合并：`tools/merge_packs.py`
- Pi0.5 GPTQ/per-step：`tools/build_pi05_a2lite_gptq_perstep.py`
- Pi0.5 SVDQuant-style build：`tools/build_pi05_svdquant_weights.py`

本地构建比直接评测更慢，也更依赖显存和环境。建议先用小样本 calibration 验证 pack 能加载，再扩大样本数。

## 10. FoldQuantVLA Checkpoint 与部署产物

FoldQuant 复用现有 Pi0.5 PyTorch checkpoint 和按 suite 区分的 GR00T N1.5 checkpoint，不读取 `quantized.pt` pack。`quantize` 阶段会写出完整量化 checkpoint 目录，后续再生成 ONNX 与 TensorRT engine。

```text
$CHECKPOINTS_ROOT/pi05_libero_pytorch/
$CHECKPOINTS_ROOT/gr00t-n1.5-libero-<suite>-posttrain/

<output-root>/quantized/
<output-root>/onnx/
<output-root>/engines/
```

校准数据要求和完整命令见 [foldquantvla.md](foldquantvla.md)。
