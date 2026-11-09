# QVLA weight-only quantization on OpenVLA-OFT

This workflow calibrates channel importance from the **native continuous action
Jacobian**, allocates mixed weight bits, and evaluates with the bundled official
OFT LIBERO evaluator. It supports the merged two-camera, proprioceptive,
L1-regression LIBERO checkpoint with eight actions per policy call.

## Method and scope

- Calibration uses the Appendix G variance-times-Jacobian proxy:
  `E[||J^T r||_channel^2 * mean(delta_channel^2)]`, where
  `r = Rademacher / sqrt(56)`. Here `J` differentiates the native continuous
  8×7 action chunk with respect to the channel output. It is not an input-Gram
  Hessian or a gradient of token classification loss.
- Both camera calls of a shared vision module are accumulated before forming a
  score. A full-shape forward identity check guards against BF16 kernel artifacts.
  The first eight calibration examples must exactly match official inference.
- The global allocator uses marginal cost
  `(S_low - S_high) / (high - low)` for adjacent `16 -> 8 -> 4 -> 0` demotions.
  The default bit set excludes 2bit; the default per-layer pruning cap is 1%.
  This cap and the suite-balanced calibration sampler are explicit local choices.
- W4/W8 denotes **mean output-channel bits in the allocated modules**. It is not
  a parameter-weighted model size budget. The multimodal projector, continuous
  action head, proprio projector and language-model output head stay BF16.
- Activations remain BF16. Weights are fake quantized and evaluated with BF16
  operators. These scripts do not implement packed integer inference or claim
  deployment speedups or memory savings.
- This is a proxy-driven core implementation. Author-specific calibration
  indices, instruction-only subsets, most-sensitive-channel forward refinement
  and short-rollout cross-validation are not supplied or reproduced here.

No experiment results, calibration images, channel assignments, model weights,
logs or videos are included in this code release.

## Installation

Use an isolated **Python 3.10** environment for OFT. Run the following commands
from `third_party/openvla_oft/`, not from the repository root:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Choose a CUDA wheel compatible with your system. The example uses CUDA 11.8.
python -m pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
  --index-url https://download.pytorch.org/whl/cu118
python -m pip install -e .
python -m pip install -r qvla/requirements.txt

git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO
git -C third_party/LIBERO checkout 8f1084e3132a39270c3a13ebe37270a43ece2a01
python -m pip install -e third_party/LIBERO
export LIBERO_ROOT="$PWD/third_party/LIBERO"
```

The package pins the author's bidirectional Transformers fork to
`bc339d9ad707454c0c115970db43c260067c61ab`. A functional startup check rejects
stock Transformers even if its version string matches. FlashAttention is not
required for this SDPA inference path.

For headless simulation, install the system OSMesa library and ffmpeg, then set:

```bash
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export LIBERO_CONFIG_PATH="$PWD/out/libero_config"
```

Create a local LIBERO config before the first LIBERO import to avoid its
interactive initialization prompt:

```bash
python - <<'PY'
import os
from pathlib import Path
import yaml
root = Path(os.environ["LIBERO_ROOT"]).resolve() / "libero/libero"
config = Path(os.environ["LIBERO_CONFIG_PATH"])
config.mkdir(parents=True, exist_ok=True)
datasets = root.parent / "datasets"
datasets.mkdir(exist_ok=True)
paths = {"benchmark_root": root, "bddl_files": root / "bddl_files",
         "init_states": root / "init_files", "assets": root / "assets",
         "datasets": datasets}
(config / "config.yaml").write_text(yaml.safe_dump({k: str(v) for k, v in paths.items()}))
PY
```

Ensure LIBERO's assets and default initial states are present; see the bundled
[LIBERO guide](../LIBERO.md) and the [upstream LIBERO repository](https://github.com/Lifelong-Robot-Learning/LIBERO).
Keep this config private to your run instead of overwriting another project's
LIBERO config.

## Checkpoint and training data

Download the **merged** Spatial checkpoint, including its continuous action
head and proprio projector. An adapter-only checkpoint is insufficient.

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="moojink/openvla-7b-oft-finetuned-libero-spatial",
    revision="6d0231af0e48c5985f1ff86908f4674b84bc049b",
    local_dir="models/libero_spatial",
    ignore_patterns=["adapter/*"],
)
snapshot_download(
    repo_id="openvla/modified_libero_rlds", repo_type="dataset",
    local_dir="data/modified_libero_rlds",
)
PY
```

The RLDS root must contain `libero_{spatial,object,goal,10}_no_noops/1.0.0/`.
Calibration uses training trajectories only; simulator evaluation uses LIBERO's
default initial states. The runtime creates its own checkpoint view: binary
weights are symlinked and metadata is copied, so upstream's config/code sync
does not modify your downloaded checkpoint.

## Prepare calibration records

```bash
python -m qvla.prepare_calibration \
  --data_root data/modified_libero_rlds \
  --out_dir out/calibration \
  --trajectories 512 --suites spatial,object,goal,10 --seed 7
```

This samples one random frame from each of 512 distinct training trajectories,
balanced across the four suites. Records contain instruction, 8D state, both
camera paths and image SHA256 values. Paths are relative to `records.jsonl`.
Raw RLDS images are kept without additional rotation, resizing or center crop;
the official OFT image helper applies inference preprocessing exactly once.

To reuse your own suite/episode/frame list, pass `--indices_jsonl path/to/indices.jsonl`.
Its required fields are `suite` (for example `libero_spatial`), `episode_index`
and `step_index`; an optional instruction is checked against the RLDS record.
This preserves the input ordering. Indices and datasets are not bundled.

## Calibrate and allocate

Choose one GPU explicitly. The default calibration allocator cap is 55% of that
GPU; leave additional free memory for other users of a shared machine.

```bash
CUDA_VISIBLE_DEVICES=0 python -m qvla.calibrate_actions \
  --pretrained_checkpoint models/libero_spatial \
  --calib_jsonl out/calibration/records.jsonl \
  --out_dir out/action_sensitivity --seed 7

python -m qvla.assign_gates_from_sensitivity \
  --proxy_pt out/action_sensitivity/proxy.pt \
  --bits 0,4,8,16 --target_avg_bits 4 --max_prune_ratio 0.01 \
  --out_json out/gates_w4.json

python -m qvla.assign_gates_from_sensitivity \
  --proxy_pt out/action_sensitivity/proxy.pt \
  --bits 0,4,8,16 --target_avg_bits 8 --max_prune_ratio 0.01 \
  --out_json out/gates_w8.json
```

Calibration writes `status.json`, the primary `proxy.pt`, and a diagnostic
`directional.pt`. Only `proxy.pt` is used by the allocator. Intermediate files
are saved for inspection; consume them only after status becomes `complete`.
For a startup check use a fresh output directory with `--max_samples 8`.

The deprecated `sensitivity_hessian_proxy.py` filename redirects to
`calibrate_actions` and accepts its arguments. The previous input-Gram estimator
is no longer the OFT calibration implementation.

## Evaluate Spatial

Run W4A16 and W8A16 in sequence, or choose distinct GPUs if running in parallel.
Each default evaluation runs 10 tasks × 50 trials with seed 7, two cameras,
proprio, center crop and eight open-loop actions. The inference memory fraction
defaults to 42% and can be configured with `--memory_fraction`.
These fractions reflect shared-GPU defaults; raise them explicitly when using
a smaller dedicated GPU if the model does not fit within the selected fraction.

```bash
CUDA_VISIBLE_DEVICES=0 python -m qvla.run_eval \
  --pretrained_checkpoint models/libero_spatial \
  --gates_path out/gates_w4.json --task_suite_name libero_spatial \
  --num_trials_per_task 50 --seed 7 --local_log_dir out/w4a16

CUDA_VISIBLE_DEVICES=0 python -m qvla.run_eval \
  --pretrained_checkpoint models/libero_spatial \
  --gates_path out/gates_w8.json --task_suite_name libero_spatial \
  --num_trials_per_task 50 --seed 7 --local_log_dir out/w8a16

python -m qvla.verify_eval --run_dir out/w4a16
python -m qvla.verify_eval --run_dir out/w8a16
```

Use `--num_trials_per_task 1` with a fresh output directory for a 10-episode
smoke check. Every command refuses to overwrite an existing run directory.

Evaluation uses the official OFT initialization and continuous action head.
Module names, gate lengths and bit choices are validated before modifying any
weights; invalid inputs fail instead of falling back to a median bit width.
Zero-bit channels zero their weights and retain their original biases.
Runtime exceptions abort the run instead of being counted as policy failures.

Each run keeps `run.json`, a canonical text log under `logs/`,
`verified_episodes.json` and videos under its own `rollouts/` directory.
Completion requires matching all trial/cumulative/task counters and the official
return value. Source and gate hashes, protected component hashes and the full
official configuration are recorded. `verify_eval --stdout_path ...` can also
check a separately captured Rich-formatted console log, including wrapped lines.

## Optional checkpoint export

Evaluation can inject weights directly; export is optional and writes another
full BF16 checkpoint, including the original action head, proprio projector and
statistics. It is not a packed low-bit model.

```bash
CUDA_VISIBLE_DEVICES=0 python -m qvla.inject_fake_w \
  --pretrained_checkpoint models/libero_spatial \
  --gates_path out/gates_w4.json --out_dir out/export_w4a16
```

## Checks

```bash
python -m pip install pytest
CUDA_VISIBLE_DEVICES='' python -m pytest qvla/tests -q
```

CPU tests cover the explicit action Jacobian, shared-camera score reduction,
action unnormalization dtype order, marginal allocation, pruning budgets,
strict gate injection, checkpoint isolation and wrapped-log auditing. They use
synthetic data and include no experiment outputs.
