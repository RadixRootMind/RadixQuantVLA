# Checkpoints

quant-vla does not commit model weights, LIBERO datasets, or quantized packs. Put them anywhere and point the scripts to them with environment variables or CLI options.

## 1. Path Variables

Load the same path configuration used by the launch scripts before running any download command:

```bash
cd /path/to/quant-vla
source .env.local

echo "AWESOME_QVLA_ROOT=$AWESOME_QVLA_ROOT"
echo "WORKSPACE=$WORKSPACE"
echo "CHECKPOINTS_ROOT=$CHECKPOINTS_ROOT"
echo "OPENPI_ROOT=$OPENPI_ROOT"
```

`CHECKPOINTS_ROOT` is the directory where base model checkpoints are stored. By default, `.env.example` defines:

```bash
AWESOME_QVLA_ROOT=<this repository>
WORKSPACE=<parent directory of this repository>
CHECKPOINTS_ROOT=$WORKSPACE/checkpoints
```

For example, if the repository is:

```bash
/opt/quant-vla
```

then the default checkpoint directory is:

```bash
/opt/checkpoints
```

If you prefer the shorter path used in several examples:

```bash
/opt/ckpts
```

set it explicitly in `.env.local`:

```bash
export CHECKPOINTS_ROOT=/opt/ckpts
```

After that, all download commands in this document will place files under `$CHECKPOINTS_ROOT`.

Suggested layout:

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
```

For quantized packs, the default directory is under the repository itself:

```bash
$AWESOME_QVLA_ROOT/results/packs/
  gr00t_object/quantized.pt
  gr00t_spatial/quantized.pt
  gr00t_goal/quantized.pt
  gr00t_long/quantized.pt
  pi05_object/quantized.pt
```

So there are two separate storage roots:

| Variable | Stores | Default example if repo is `/opt/quant-vla` |
| --- | --- | --- |
| `CHECKPOINTS_ROOT` | FP/base model checkpoints | `/opt/checkpoints` or your override such as `/opt/ckpts` |
| `AWESOME_QVLA_ROOT/results/packs` | W4A4 quantized packs | `/opt/quant-vla/results/packs` |

Do not run the download commands with an empty `CHECKPOINTS_ROOT`. Check it first:

```bash
test -n "$CHECKPOINTS_ROOT"
mkdir -p "$CHECKPOINTS_ROOT"
```

This is only a convention, not a requirement. You can override individual paths with:

- `--groot-checkpoint`
- `--openpi-checkpoint`
- `--groot-gptq-pack`
- `--pi05-gptq-pack`
- `--openvla-checkpoint`
- `--univla-checkpoint`
- `--univla-action-decoder`
- `--starvla-checkpoint`

## 2. GR00T-N1.5 Base Checkpoints

Install Hugging Face CLI if needed:

```bash
pip install -U "huggingface_hub[cli]"
```

Download one suite:

```bash
mkdir -p "$CHECKPOINTS_ROOT"

hf download youliangtan/gr00t-n1.5-libero-object-posttrain \
  --local-dir "$CHECKPOINTS_ROOT/gr00t-n1.5-libero-object-posttrain"
```

This writes files to:

```bash
$CHECKPOINTS_ROOT/gr00t-n1.5-libero-object-posttrain
```

For example, if `CHECKPOINTS_ROOT=/opt/ckpts`, the final path is:

```bash
/opt/ckpts/gr00t-n1.5-libero-object-posttrain
```

Download the four common LIBERO suites:

```bash
mkdir -p "$CHECKPOINTS_ROOT"

for suite in object spatial goal long; do
  hf download "youliangtan/gr00t-n1.5-libero-${suite}-posttrain" \
    --local-dir "$CHECKPOINTS_ROOT/gr00t-n1.5-libero-${suite}-posttrain"
done
```

Verify:

```bash
export GROOT_OBJECT_CKPT="$CHECKPOINTS_ROOT/gr00t-n1.5-libero-object-posttrain"

test -d "$GROOT_OBJECT_CKPT" && echo "GR00T object checkpoint dir ok"
test -f "$GROOT_OBJECT_CKPT/config.json" && echo "GR00T object config ok"
find "$GROOT_OBJECT_CKPT" -maxdepth 1 \( -name "*.safetensors" -o -name "*.bin" -o -name "*.pt" \) -print | head
```

## 3. Pi0.5/OpenPI Base Checkpoint

For FP16 OpenPI evaluation, the OpenPI runtime can use the public asset path directly:

```bash
export OPENPI_CHECKPOINT=gs://openpi-assets/checkpoints/pi05_libero
```

For quantized Pi0.5 runs, this repository expects a local PyTorch checkpoint directory containing `model.safetensors`, for example:

```bash
export OPENPI_CHECKPOINT="$CHECKPOINTS_ROOT/pi05_libero_pytorch"

test -f "$OPENPI_CHECKPOINT/model.safetensors" && echo "Pi0.5 PyTorch checkpoint ok"
test -f "$OPENPI_CHECKPOINT/assets/physical-intelligence/libero/norm_stats.json" && echo "Pi0.5 norm stats ok"
```

If the directory only contains `assets/` and does not contain `model.safetensors`, it is incomplete and quantized Pi0.5 routes will fail. Convert the official JAX/Orbax checkpoint to PyTorch first.

Prepare the OpenPI PyTorch conversion environment:

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

"$OPENPI_PY" - <<'PY'
import transformers
from openpi.models_pytorch import pi0_pytorch
print("transformers:", transformers.__version__)
print("pi0_pytorch import ok")
PY
```

Convert the cached JAX checkpoint:

```bash
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

ls -lh "$PT_CKPT/model.safetensors"
ls -lh "$PT_CKPT/assets/physical-intelligence/libero/norm_stats.json"
```

If the JAX checkpoint is not cached locally, point the converter at the public asset path if your network supports it:

```bash
"$OPENPI_PY" examples/convert_jax_model_to_pytorch.py \
  --checkpoint_dir gs://openpi-assets/checkpoints/pi05_libero \
  --config_name pi05_libero \
  --output_path "$CHECKPOINTS_ROOT/pi05_libero_pytorch"
```

If your OpenPI checkout downloads the official assets into its cache, you can keep the converted directory there and point `OPENPI_CHECKPOINT` to it. A symlink under `CHECKPOINTS_ROOT` is also fine:

```bash
ln -s /path/to/converted/pi05_libero "$CHECKPOINTS_ROOT/pi05_libero_pytorch"
```

The important part is the final path passed to `--openpi-checkpoint`, not where the files physically live.

## 4. Quantized Packs

W4A4 GPTQ/SVD-Hadamard profiles require quantized packs in addition to the FP checkpoint.

Suggested local layout:

```bash
$AWESOME_QVLA_ROOT/results/packs/
  gr00t_object/quantized.pt
  gr00t_spatial/quantized.pt
  gr00t_goal/quantized.pt
  gr00t_long/quantized.pt
  pi05_object/quantized.pt
```

Older locally built GR00T packs may also use `<suite>_MERGED/quantized.pt`, for example `object_MERGED/quantized.pt`. The unified launcher accepts either layout.

Known upstream pack repositories:

| Route | Hugging Face repository | Default local target |
| --- | --- | --- |
| GR00T W4A4 GPTQ | `ucmp137538/Omega-QVLA-GR00T-N1.5-LIBERO-W4A4` | `$AWESOME_QVLA_ROOT/results/packs/gr00t_<suite>/quantized.pt` |
| Pi0.5 W4A4 GPTQ | `ucmp137538/Omega-QVLA-pi05-LIBERO-W4A4` | `$AWESOME_QVLA_ROOT/results/packs/pi05_<suite>/quantized.pt` |

Download the object-suite GR00T pack:

```bash
cd "$AWESOME_QVLA_ROOT"
python -m pip install -U "huggingface_hub[cli]"

export HF_HUB_DISABLE_XET=1
export SUITE=object

hf download ucmp137538/Omega-QVLA-GR00T-N1.5-LIBERO-W4A4 \
  --include "gr00t_${SUITE}/quantized.pt" \
  --local-dir "$AWESOME_QVLA_ROOT/results/packs"

ls -lh "$AWESOME_QVLA_ROOT/results/packs/gr00t_${SUITE}/quantized.pt"
```

For `SUITE=object`, this writes the pack to:

```bash
$AWESOME_QVLA_ROOT/results/packs/gr00t_object/quantized.pt
```

If the include filter returns `Fetching 0 files`, download the repository without `--include` and then use the `gr00t_<suite>/quantized.pt` file:

```bash
hf download ucmp137538/Omega-QVLA-GR00T-N1.5-LIBERO-W4A4 \
  --local-dir "$AWESOME_QVLA_ROOT/results/packs"

find "$AWESOME_QVLA_ROOT/results/packs" -path "*gr00t_${SUITE}/quantized.pt" -print
```

Download the object-suite Pi0.5 pack:

```bash
cd "$AWESOME_QVLA_ROOT"
python -m pip install -U "huggingface_hub[cli]"

export HF_HUB_DISABLE_XET=1
export SUITE=object

hf download ucmp137538/Omega-QVLA-pi05-LIBERO-W4A4 \
  --include "pi05_${SUITE}/quantized.pt" \
  --local-dir "$AWESOME_QVLA_ROOT/results/packs"

ls -lh "$AWESOME_QVLA_ROOT/results/packs/pi05_${SUITE}/quantized.pt"
```

If your network cannot reach Hugging Face directly, set your mirror or proxy before running `hf download`, for example:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

You can override pack paths directly:

```bash
bash scripts/run_awesome_quant_vla.sh groot_w4a4_gptq \
  --suite object \
  --groot-gptq-pack /path/to/gr00t_object_quantized.pt

bash scripts/run_awesome_quant_vla.sh pi05_w4a4_gptq \
  --suite object \
  --openpi-checkpoint "$OPENPI_CHECKPOINT" \
  --pi05-gptq-pack /path/to/pi05_object_quantized.pt
```

If you publish prebuilt packs for this project, document their Hugging Face repository here and download them into the layout above, or pass their exact path through the CLI.

If `groot_w4a4_gptq` exits with a message saying the GPTQ pack was not found, first check the resolved path:

```bash
ls -lh "$AWESOME_QVLA_ROOT/results/packs/gr00t_object/quantized.pt"
ls -lh "$AWESOME_QVLA_ROOT/results/packs/object_MERGED/quantized.pt"
find "$AWESOME_QVLA_ROOT/results/packs" -name quantized.pt -print
```

Then either place the pack at the default layout or pass the actual file:

```bash
bash scripts/run_awesome_quant_vla.sh groot_w4a4_gptq \
  --suite object \
  --groot-gptq-pack /path/to/gr00t_object/quantized.pt
```


## 5. OpenVLA and OpenVLA-OFT Checkpoints

OpenVLA/QVLA profiles use Hugging Face-format OpenVLA checkpoints. Suggested local layout:

```bash
$CHECKPOINTS_ROOT/
  openvla-7b-finetuned-libero-spatial/
  openvla-7b-oft-finetuned-libero-spatial/
```

Download examples:

```bash
mkdir -p "$CHECKPOINTS_ROOT"

hf download openvla/openvla-7b-finetuned-libero-spatial \
  --local-dir "$CHECKPOINTS_ROOT/openvla-7b-finetuned-libero-spatial"

hf download moojink/openvla-7b-oft-finetuned-libero-spatial \
  --local-dir "$CHECKPOINTS_ROOT/openvla-7b-oft-finetuned-libero-spatial"
```

If a model card or upstream repository uses a different namespace, keep the local directory name stable or pass the exact checkpoint path:

```bash
bash scripts/run_awesome_quant_vla.sh openvla_oft_qvla_w8 \
  --suite spatial \
  --openvla-checkpoint /path/to/openvla-7b-oft-finetuned-libero-spatial
```

Verify:

```bash
export OPENVLA_OFT_CKPT="$CHECKPOINTS_ROOT/openvla-7b-oft-finetuned-libero-spatial"

test -f "$OPENVLA_OFT_CKPT/config.json" && echo "config ok"
test -f "$OPENVLA_OFT_CKPT/model.safetensors.index.json" && echo "index ok"
find "$OPENVLA_OFT_CKPT" -maxdepth 1 -name "*.safetensors" -print | head
```

For offline use, make sure `configuration_prismatic.py`, `modeling_prismatic.py`, `processing_prismatic.py`, tokenizer files, and `dataset_statistics.json` are present in the checkpoint directory.

## 6. UniVLA Checkpoint

UniVLA requires both the Hugging Face model checkpoint and an action decoder. The Hugging Face repository contains separate LIBERO suite subdirectories, so you can download only the suite you want to evaluate.

Suggested local layout:

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

Download only the spatial checkpoint:

```bash
mkdir -p "$CHECKPOINTS_ROOT"

hf download qwbu/univla-7b-224-sft-libero \
  --include "univla-libero-spatial/*" \
  --local-dir "$CHECKPOINTS_ROOT/univla-7b-224-sft-libero"
```

Download other LIBERO suites by changing the include prefix:

```bash
# object
hf download qwbu/univla-7b-224-sft-libero \
  --include "univla-libero-object/*" \
  --local-dir "$CHECKPOINTS_ROOT/univla-7b-224-sft-libero"

# goal
hf download qwbu/univla-7b-224-sft-libero \
  --include "univla-libero-goal/*" \
  --local-dir "$CHECKPOINTS_ROOT/univla-7b-224-sft-libero"

# long / LIBERO-10
hf download qwbu/univla-7b-224-sft-libero \
  --include "univla-libero-10/*" \
  --local-dir "$CHECKPOINTS_ROOT/univla-7b-224-sft-libero"
```

Verify:

```bash
export UNIVLA_REPO_ROOT="$CHECKPOINTS_ROOT/univla-7b-224-sft-libero"
export UNIVLA_CKPT="$UNIVLA_REPO_ROOT/univla-libero-spatial"
export UNIVLA_ACTION_DECODER="$UNIVLA_CKPT/action_decoder.pt"

test -f "$UNIVLA_CKPT/config.json" && echo "UniVLA config ok"
test -f "$UNIVLA_CKPT/dataset_statistics.json" && echo "UniVLA dataset statistics ok"
test -f "$UNIVLA_ACTION_DECODER" && echo "UniVLA action decoder ok"
```

If your checkpoint stores the decoder elsewhere, pass it explicitly:

```bash
bash scripts/run_awesome_quant_vla.sh univla_fp16 \
  --suite spatial \
  --univla-checkpoint /path/to/univla-7b-224-sft-libero/univla-libero-spatial \
  --univla-action-decoder /path/to/univla-7b-224-sft-libero/univla-libero-spatial/action_decoder.pt
```

## 7. StarVLA Checkpoint

StarVLA-OFT uses a policy checkpoint plus a Qwen3-VL base model expected by the upstream StarVLA path.

Suggested layout:

```bash
$CHECKPOINTS_ROOT/
  starvla/
    Qwen3-VL-OFT-LIBERO-4in1/
      checkpoints/steps_50000_pytorch_model.pt
    Qwen3-VL-4B-Instruct/
```

Download the StarVLA-OFT LIBERO checkpoint:

```bash
mkdir -p "$CHECKPOINTS_ROOT/starvla/Qwen3-VL-OFT-LIBERO-4in1"
hf download StarVLA/Qwen3-VL-OFT-LIBERO-4in1 \
  --local-dir "$CHECKPOINTS_ROOT/starvla/Qwen3-VL-OFT-LIBERO-4in1"
```

Download the Qwen3-VL base model and link it to StarVLA's expected path:

```bash
mkdir -p "$CHECKPOINTS_ROOT/starvla/Qwen3-VL-4B-Instruct"
hf download Qwen/Qwen3-VL-4B-Instruct \
  --local-dir "$CHECKPOINTS_ROOT/starvla/Qwen3-VL-4B-Instruct"

mkdir -p "$AWESOME_QVLA_ROOT/third_party/starvla/playground/Pretrained_models"
ln -sfn "$CHECKPOINTS_ROOT/starvla/Qwen3-VL-4B-Instruct" \
  "$AWESOME_QVLA_ROOT/third_party/starvla/playground/Pretrained_models/Qwen3-VL-4B-Instruct"
```

Verify:

```bash
export STARVLA_CKPT="$CHECKPOINTS_ROOT/starvla/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt"

test -f "$STARVLA_CKPT" && echo "StarVLA policy checkpoint ok"
test -d "$AWESOME_QVLA_ROOT/third_party/starvla/playground/Pretrained_models/Qwen3-VL-4B-Instruct" && echo "Qwen3-VL base link ok"
```

## 8. Build Packs Locally

The repository includes pack-building tools:

- GR00T LLM GPTQ: `tools/build_gptq_weights.py`
- GR00T DiT GPTQ/per-step: `tools/build_dit_a2lite_svd_gptq_perstep.py`
- Pack merge: `tools/merge_packs.py`
- Pi0.5 GPTQ/per-step: `tools/build_pi05_a2lite_gptq_perstep.py`
- Pi0.5 SVDQuant-style build: `tools/build_pi05_svdquant_weights.py`

Pack building is slower and more hardware-sensitive than evaluation. Start from a small calibration run, verify that the pack loads, then scale up calibration samples.

## 9. FoldQuantVLA Checkpoints and Artifacts

FoldQuant uses the existing Pi0.5 PyTorch and suite-specific GR00T N1.5 checkpoints. It does not use a `quantized.pt` pack. The `quantize` stage writes a complete quantized checkpoint directory; later stages write ONNX graphs and TensorRT engines beneath the selected output root.

```text
$CHECKPOINTS_ROOT/pi05_libero_pytorch/
$CHECKPOINTS_ROOT/gr00t-n1.5-libero-<suite>-posttrain/

<output-root>/quantized/
<output-root>/onnx/
<output-root>/engines/
```

See [foldquantvla.md](foldquantvla.md) for calibration dataset requirements and complete commands.
