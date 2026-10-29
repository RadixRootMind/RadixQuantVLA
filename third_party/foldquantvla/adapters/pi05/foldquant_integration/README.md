# FoldQuant on Pi0 / Pi0.5 (openpi)

This adapter integrates FoldQuant with openpi, using upstream
policy transforms, websocket serving, and LIBERO evaluation. The shared
[`foldquant`](../../../foldquant) package provides quantization, ONNX export,
and TensorRT plugins.

Two modules of the policy are replaced by FoldQuant plugin graphs:

| module | attribute | graph | schemes |
|---|---|---|---|
| LLM (PaliGemma Gemma-2B decoder, prefix pass) | `policy._model.paligemma_with_expert.paligemma.language_model` | `llm_bf16.onnx` | `w4a4_srg` (default), `w4a4_sg`, `w4a8_srg`, `w8a8_sr`, `w8a8_s`, `w8a8` |
| action expert (Gemma-300M, one denoise step) | `policy._model.paligemma_with_expert.gemma_expert` + the projections around it | `expert_bf16.onnx` | `w4a4_shg` (default), `w4a4_sh`, `w4a4_sr`, `w8a8_sh`, `w8a8` |

Pi's inference splits into a prefix pass and a suffix loop, and the graphs
follow that split rather than the module tree:

- **Prefix.** SigLIP on the three cameras and the prompt token embedding
  (`embed_prefix`, PyTorch) produce `prefix_embs`; the LLM graph takes
  those with the 4-D additive attention mask and `position_ids`, and
  returns the stacked post-RoPE KV cache `[layers, 2, 1, kv_heads, S, 256]`
  in bf16. The prefix length `S` is a constant of the checkpoint (the
  processor pads the prompt to `max_token_len` and the camera set is fixed:
  3 × 256 image tokens + 200 text tokens = 968 for `pi05_libero`), so the
  export pins it from a captured call and the engine is fully static.
- **Suffix.** The expert graph is one `denoise_step`: `x_t`, `timestep`,
  `prefix_pad_masks` and the KV stack in, `velocity` out (Pi0 adds
  `state`). Everything a step does is inside it (the action / time
  projections, the 18 Gemma-300M layers cross-attending the cached prefix
  with the adaRMS modulation (Pi0.5) or the state token (Pi0), the output
  projection); the 10-step Euler loop stays in PyTorch.

For comparison, either module can instead take an NVIDIA ModelOpt INT8
SmoothQuant Q/DQ graph (`modelopt_w8a8_smoothquant`) under the same file name
and I/O contract; see [ModelOpt INT8 SmoothQuant baseline](#modelopt-int8-smoothquant-baseline).

The KV stack is the contract between the two seams, so each engine can also
be installed alone: the runtime stacks the PyTorch `DynamicCache` when the
LLM stays float, and rebuilds one from the engine's stack when the expert
stays float.

Use `none` to keep a module in PyTorch. The all-float baseline is `export` on
the unquantized checkpoint (`--checkpoint-dir <checkpoint> --dataset-path
<dataset>`): both graphs traced with no plugin nodes. `verify` compares all
engines against the BF16 PyTorch policy.

Engines are served by [`runtime.py`](runtime.py), which rebinds
`paligemma_with_expert.forward` (prefix branch only) and
`PI0Pytorch.denoise_step` on the loaded instance; nothing else in the policy
changes, so the same `Policy` runs float, partly or fully quantized, behind
the same `WebsocketPolicyServer`.

The rebinding needs the eager model. Upstream builds `PI0Pytorch` with
`pytorch_compile_mode = "max-autotune"` (`sample_actions` is traced and
replayed as CUDA graphs). A traced call keeps running the prefix pass it
captured, so a seam bound on the instance afterwards is skipped
(`denoise_step` then receives a `DynamicCache` instead of a KV stack), and
tensors a calibration hook keeps across replays are overwritten.
`calibration.load_policy(..., compile=False)` clears that mode on the train
config; `install_engines` and the calibration capture refuse a compiled
model. The reference serving arm keeps upstream's compiled configuration,
and `benchmark` times it as its own arm.

## Environment

On a Jetson Orin, use `scripts/deployment/orin/install_deps.sh` instead of the
commands below: it routes torch to the Jetson wheel index (PyPI's aarch64 torch
does not run on sm_87), takes JetPack's TensorRT from the system rather than
`tensorrt-cu12`, and applies the same transformers patches without touching the
uv cache. Python 3.10 is what JetPack 6.2 ships and the only version that index
publishes wheels for.

On x86, the upstream pins apply (Python 3.11, torch 2.7.1, transformers 4.53.2 with
openpi's patched Gemma copied over it). From this directory:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync --python 3.11
source .venv/bin/activate
cp -r --remove-destination src/openpi/models_pytorch/transformers_replace/* \
    .venv/lib/python3.11/site-packages/transformers/
uv pip install "tensorrt-cu12==10.15.1.29"
uv pip install -e ../.. "onnx<1.18"    # the foldquant package into it
python -m foldquant.kernels build      # compile the plugin libraries for this GPU / TensorRT
```

Two deviations from a plain `uv pip install`: `--remove-destination`, because
uv hardlinks site-packages into its wheel cache and copying *over* them would
rewrite the cached `transformers` wheel for every environment on the machine;
and `onnx<1.18`, because upstream pins `ml-dtypes==0.4.1` (an override in
`pyproject.toml`) and `onnx` 1.18+ imports `ml_dtypes.float4_e2m1fn`, which
that version lacks.

`foldquant.kernels build` needs `nvcc` and the TensorRT headers. Binaries are
cached per `(SM, arch, TensorRT major.minor)` and matched exactly, so another
GPU or TensorRT version compiles its own copy. The checkpoint must be the **PyTorch** form
(`model.safetensors`); convert a JAX checkpoint with upstream's
`examples/convert_jax_model_to_pytorch.py` first. The LIBERO client runs in
its own environment (`examples/libero/README.md`, "Without Docker"). On a
Jetson Orin that client's pins (Python 3.8, a CUDA 11.3 torch) cannot run; a
Python 3.10 environment at `examples/libero/.venv` with `numpy==1.26.4`,
`robosuite==1.4.0`, `mujoco==2.3.7`, `bddl easydict hydra-core einops termcolor
thop gym tqdm tyro pyyaml imageio[ffmpeg] opencv-python-headless matplotlib`,
the Jetson torch (only for LIBERO's stored initial states) and
`-e packages/openpi-client` does. The Orin environment above installs neither
openpi nor openpi-client: put `src` and `packages/openpi-client/src` on
`PYTHONPATH` beside the repository root. A calibration dataset that is a
subset of a larger one (sparse episode indices) is loaded as it is, without
Hub access.

## Workflow

Every step takes `--config` (the upstream training config; default
`pi05_libero`) and `--checkpoint-dir`. The calibration dataset is any
LeRobot-format LIBERO dataset (`observation.images.image`,
`observation.images.wrist_image`, `observation.state`, tasks); the
integration builds the client's observation dictionaries from it, so the
policy sees exactly what the websocket server hands it.

1. **Calibrate and emit the FoldQuant graphs.** Observations are drawn from
   a seeded, episode-balanced plan and run through `Policy.infer` with
   explicit, seeded flow-matching noise, so every capture sees the
   inference-time tensors and the same inputs emit byte-identical graphs.

   ```bash
   python -m foldquant_integration.quantize \
       --checkpoint-dir <pi05_libero PyTorch checkpoint> \
       --dataset-path <LeRobot LIBERO dataset> --num-calib 128 \
       --llm-scheme w4a4_srg --expert-scheme w4a4_shg \
       --output-dir exports/pi05_w4a4/quantized
   ```

   `quantize` writes the quantized model, `exports/pi05_w4a4/quantized`: the
   base checkpoint with each quantized projection's weight replaced by its
   integer codes (`qweight`) and per-row scale (`weight_scale`), the same
   bytes the engines carry, plus the SmoothQuant scales and fold settings.
   Everything unquantized, tokenizer and processor files included, is copied
   from the base as it is. It runs in PyTorch with fake-quant layers in place
   of the quantized projections, and the plugin graphs are exported from it,
   with no dataset:

   ```bash
   python -m foldquant_integration.export --checkpoint-dir exports/pi05_w4a4/quantized \
       --output-dir exports/pi05_w4a4/onnx
   ```

   That is the paper's W4A4 arm. Its W4A4 o/d8 arm keeps `o_proj` and
   `down_proj` at INT8: add `--llm-params '{"site_bits": {"o": 8, "down": 8}}'`.

   `--cascade` calibrates the expert while the LLM runs under FoldQuant's
   fake-quant emulation of its own fold. `--llm-scheme none` /
   `--expert-scheme none` leaves a module in PyTorch. Keep `--num-calib`
   at 128 or more for a GPTQ (`_g`) arm.

2. **Build the engine directory.**

   ```bash
   python -m foldquant_integration.build_engines \
       --onnx-dir exports/pi05_w4a4/onnx --engine-dir exports/pi05_w4a4/engines
   ```

   Loads the plugin libraries, then compiles each FoldQuant graph strongly
   typed. Both graphs are static at the captured prefix length;
   `foldquant_engines.json` records what was built from where, and
   `foldquant_export.json` travels with the engines so the runtime tools
   know which plugin libraries to load.

3. **Verify, serve or evaluate, benchmark.**

   ```bash
   python -m foldquant_integration.verify --checkpoint-dir ... --dataset-path ... \
       --engine-dir exports/pi05_w4a4/engines

   # upstream's client/server protocol, server side quantized:
   python -m foldquant_integration.serve --checkpoint-dir ... --engine-dir exports/pi05_w4a4/engines
   MUJOCO_GL=egl <libero venv>/bin/python examples/libero/main.py --args.task-suite-name libero_spatial

   # the same pairing over every suite, resumable:
   MUJOCO_GL=egl python -m foldquant_integration.eval_libero --protocol p3 --checkpoint-dir ... \
       --engine-dir exports/pi05_w4a4/engines \
       --client-python examples/libero/.venv/bin/python --output exports/pi05_w4a4/libero

   python -m foldquant_integration.benchmark --checkpoint-dir ... --dataset-path ... \
       --arms w4a4=exports/pi05_w4a4/engines
   ```

   `verify` scores the prefix KV stack and the action chunk the client
   receives under seeded flow-matching noise on held-out observations from
   episodes the calibration never saw, and records every observation's
   drift in `verify.json`; `--split-from <quantized engine dir>` scores
   another directory on that arm's held-out set. `serve` is upstream's
   `WebsocketPolicyServer` around a policy with the engines installed, so
   upstream's client runs unchanged. Serving the bf16 arm compiles on the
   first inference (about 50 s on an Orin) while upstream's server answers no
   websocket pings, which openpi-client treats as a dead connection; `serve`
   therefore runs one warm-up inference on upstream's example observation
   before it opens the port (`--no-warmup` skips it). `eval_libero` starts that server and
   runs `libero_client.py` (upstream's client loop with a selectable step
   budget) per suite in the client environment; `--protocol p3` is the
   paper's campaign (20 trials per task from LIBERO's stored initial states,
   520 steps on every suite, seed 7, replan every 5 steps), the default keeps
   upstream's 50 trials and per-suite budgets. `summary.json` records every
   episode and resumes only into the same run. `benchmark` times the input transforms, prefix
   embedding, prefix LLM pass, the denoise loop and the whole
   `Policy.infer` for the eager PyTorch arm and each `--arms` engine
   directory in one process, plus upstream's `torch.compile(max-autotune)`
   serving configuration end to end only (the component stopwatches would
   break its graphs).

Plugin graphs are emitted at batch 1; the upstream client sends one
observation per request.

### Quantized models

`quantize` writes the arm as a quantized checkpoint, `<output-dir>`:
the base checkpoint with every quantized projection's weight replaced by its
integer codes and per-row scale, the rest copied from the base. `eval_libero`,
`serve` and `verify` run it in PyTorch with the engines' arithmetic when
`--checkpoint-dir` names it; the bf16 baseline runs from the base checkpoint, since
the quantized projections' bf16 weights are not in it. `export` turns it into
the plugin ONNX graphs and `build_engines` into engines, without calibration
data, and `python -m foldquant.quantized push` uploads it to the Hugging Face
Hub (private by default). The full description, with measured agreement against
the engines, is in the
[GR00T N1.7 README](../../groot_n1_7/foldquant_integration/README.md#fake-quant-checkpoints-pytorch-the-hub-then-onnx-and-engines).

## ModelOpt INT8 SmoothQuant baseline

`modelopt_w4a16_awq` (INT4 weights, group 128, bf16 activations) works the
same way as the SmoothQuant arm below; its weights go through the
`Int4GroupwiseGemmPlugin`, so the engine really runs INT4.

`modelopt_w8a8_smoothquant` is not a FoldQuant fold. It applies NVIDIA ModelOpt's
INT8 SmoothQuant recipe to the same two modules, so a FoldQuant arm and the
ModelOpt baseline can be built, verified and served by the same tools and
compared on a robot. It needs `nvidia-modelopt==0.45.0` (and `ninja`, for its
CUDA extension) in the environment.

```bash
python -m foldquant_integration.quantize --checkpoint-dir ... --dataset-path ... \
    --num-calib 64 --seed 0 \
    --llm-scheme modelopt_w8a8_smoothquant --expert-scheme modelopt_w8a8_smoothquant \
    --output-dir exports/pi05_modelopt_w8a8_sq/quantized
python -m foldquant_integration.export --checkpoint-dir exports/pi05_modelopt_w8a8_sq/quantized \
    --output-dir exports/pi05_modelopt_w8a8_sq/onnx
python -m foldquant_integration.build_engines \
    --onnx-dir exports/pi05_modelopt_w8a8_sq/onnx --engine-dir exports/pi05_modelopt_w8a8_sq/engines
python -m foldquant_integration.verify --checkpoint-dir ... --dataset-path ... \
    --engine-dir exports/pi05_modelopt_w8a8_sq/engines --split-from exports/pi05_w4a4/engines
```

What the arm does (`foldquant/modelopt_int8.py`, `modelopt_export.py`):

| step | this arm |
|---|---|
| calibration data | `--num-calib` observations (64 in the paper), seeded noise per observation; one bf16 policy replay records every prefix pass and every denoise step before anything is quantized |
| quantized scopes | LLM: `paligemma.language_model`; expert: `Pi05ExpertView` over the live expert (`expert_model.model.layers.*`, `action_in_proj`, `action_out_proj`, `time_mlp_in`, `time_mlp_out`) |
| calibration replay | LLM: the captured `prefix_embs` / 4-D mask / `position_ids` through `paligemma_with_expert.forward`; expert: the captured `x_t` / `timestep` / `prefix_pad_masks` / KV stack through `denoise_step`; the expert sees float-LLM caches (no cascade) |
| config | `mtq.INT8_SMOOTHQUANT_CFG`: per-channel INT8 weights, per-tensor static INT8 activations, SmoothQuant pre-quant scales |
| excluded leaves | Linear / Conv whose name matches `*norm*`, `*layernorm*`, `*final_action*`, `*action_proj*`: on Pi0.5 that is the adaRMS `dense` modulation of every expert norm; `action_in_proj` / `action_out_proj` and the time MLP are quantized |
| checkpoint | `quantize` saves the quantized model with the smoothed weights, every enabled quantizer's `amax` / `pre_quant_scale` (`foldquant.<module>.modelopt.*` tensors), the ModelOpt mode state (`foldquant_modelopt_<module>.pt`) and one captured call as the trace; `export` restores the quantizers on the live policy and traces from it |
| export | the KV-stack contract trace wrappers over the quantized live modules (same bindings and dtypes), legacy TorchScript exporter at opset 20 (`--modelopt-opset`), dtype repairs, graph outputs cast back to the runtime dtype (ModelOpt's Q/DQ dequantizes to float32), one external-data sidecar, default ScatterND `reduction` stripped for TensorRT 10.3, export refused when no Q/DQ node survived |
| engine | strongly typed, by the unchanged `build_engines` (no plugin library) |

`--cascade` and `--llm-params` / `--expert-params` are refused for this
scheme. `foldquant_export.json` records, per module, the excluded leaves,
the inserted / enabled quantizer counts, the Q/DQ node counts and the repairs
made; the quantized model holds every enabled quantizer's `amax` and
`pre_quant_scale` under ModelOpt's names relative to the quantized scope, for a
key-by-key comparison with the same module quantized elsewhere.

The reference policy keeps upstream openpi's mixed precision (the norms, and
the projections around the expert, in float32); the graphs use the FoldQuant
bindings and the engine directory is served by `runtime.install_engines`, so
the baseline and the FoldQuant arms differ only in the quantization recipe.

Measured on the SO101 multitask checkpoint (`pi05_so101`, 64 calibration
observations, 32 held-out observations shared with the `w8a8_sr` + `w8a8_sh`
arm through `--split-from`, Jetson AGX Orin, TensorRT 10.3):

| arm | kv_stack cos mean | position cos min | action cos mean / median / min | action max abs mean / worst |
|---|---|---|---|---|
| `modelopt_w8a8_smoothquant` both modules | 0.97719 | 0.595 | 0.99931 / 0.99955 / 0.99624 | 4.15 / 12.55 |
| `w8a8_sr` LLM + `w8a8_sh` expert | 0.99660 | 0.882 | 0.99997 / 0.99998 / 0.99979 | 0.90 / 1.97 |

The gap to `w8a8` is the recipe (per-tensor static activations), not the
port: the same ModelOpt recipe built elsewhere drifts from its bf16 policy by a
comparable action cosine (mean 0.9989).

## Using a custom checkpoint

Every tool takes `--config`, which upstream resolves from its own list of
`TrainConfig` entries. A checkpoint fine-tuned elsewhere carries a config not
in that list, and the tree under `src/openpi` is used unchanged, so upstream's
file is not edited.

`FOLDQUANT_PI05_PLUGIN` names a Python file imported before any config is
resolved. Upstream keeps its configs in a module-level dict, so the plugin
registers the entry itself:

```python
# my_plugin.py: add its directory to sys.path so sibling imports resolve.
from openpi.training import config as _config
from openpi.training.config import DataConfig, TrainConfig
import my_policy                      # the checkpoint's own transforms

_config._CONFIGS_DICT.setdefault("pi05_mine", TrainConfig(name="pi05_mine", ...))
```

```bash
export FOLDQUANT_PI05_PLUGIN=/path/to/my_plugin.py
python -m foldquant_integration.quantize \
    --checkpoint-dir <ckpt> --config pi05_mine --dataset-path <lerobot dataset> \
    --output-dir exports/mine/quantized
```

An import error in the plugin is raised, not swallowed; otherwise it would
surface as a misleading "config not found".

### Dataset columns

Calibration reads two camera streams and a state vector out of the LeRobot
dataset. The defaults are LIBERO's column names
(`observation.images.image`, `observation.images.wrist_image`,
`observation.state`); a dataset that has them is read with them, and only a
column the dataset does **not** carry is looked up in the train config's repack
transform.

The order matters: `pi05_libero` repacks from `image` / `wrist_image` /
`state` (the `physical-intelligence/libero` columns), while the LeRobot
conversion calibrated on here stores `observation.images.image`. The dataset is
the authority on its own columns and the config is the fallback. If neither
answers, the error lists the columns that are present.

### Video timestamps

`load_dataset` takes a `video_backend`. LeRobot fetches frames **by timestamp**
and checks the result to 1e-4 s, so a dataset whose `.mp4` files carry a
non-zero container `start_time` either raises or, worse, returns a neighbouring
frame (an index-based loader would not notice). Fix the dataset, losslessly:

```bash
ffmpeg -nostdin -fflags +genpts -i in.mp4 -c copy -reset_timestamps 1 out.mp4
```

Raising the tolerance instead would silently accept the wrong frame.

Some recordings start the video a couple of frames after the episode (the
SO101 set starts at 0.066 s). Those first steps have no frame at all, so the
sampler skips them and draws from the first step every camera can serve.

## Smoke check

`w8a8_sr` LLM + `w4a4_sh` expert, 16 calibration observations, 8 held-out
observations from episodes the calibration never saw, float64 cosines
against the bf16 eager PyTorch policy under the same seeded flow-matching
noise (`verify`), `pi05_libero` checkpoint at its 10 denoising steps, one
RTX 4070 Ti SUPER (sm89), TensorRT 10.15:

| seam | cos mean | cos min | note |
|---|---|---|---|
| `kv_stack` (18 layers × K,V × 968 × 256) | 0.99922 | 0.99910 | per-position min 0.988 |
| action chunk (10 × 7, normalised) | 0.99987 | 0.99984 | max abs 0.026 |

`benchmark` on the same engines (10 iterations, median): PyTorch eager
165.3 ms (prefix LLM 58.4, denoise loop 79.1); upstream's serving default,
`torch.compile(max-autotune)`, 100.1 ms end to end; FoldQuant 85.7 ms
(prefix LLM 26.8, denoise loop 30.8): 1.93× over eager, 1.17× over the
compiled baseline, with SigLIP, the prefix embedding and the Euler loop
still in eager PyTorch in the FoldQuant arm. Paper numbers use 128
calibration observations and the server's harness; this is the
installation check.

## Files

| file | role |
|---|---|
| `calibration.py` | upstream policy / dataset loading, seeded sample plan, client-format observations, forward loop |
| `quantize.py` | scheme validation, shape capture, `export_llm` / `export_expert`, the quantized model |
| `export.py` | quantized model -> the plugin graphs and manifests, from the recorded codes |
| `modelopt_export.py` | ModelOpt INT8 SmoothQuant baseline: seam capture, live-module quantization, Q/DQ export under the FoldQuant contract |
| `build_engines.py` | plugin load + `foldquant.runtime.builder.build_engine` per component |
| `runtime.py` | engine installer (`install_engines`), the two rebinds, KV-stack helpers, `PrefixCapture` |
| `verify.py` | held-out PyTorch-vs-engine drift report |
| `serve.py` | upstream `WebsocketPolicyServer` with engines installed |
| `eval_libero.py` | LIBERO sweep through upstream's client, per-suite `summary.json` |
| `benchmark.py` | component and end-to-end timing over the PyTorch arm and engine directories |
| `_upstream.py` | paths, component table |

## Device memory per arm

`memory.py` measures one arm in one fresh process and reports two numbers to
read together:

* **as served** (`--keep-replaced-weights`): what `serve` holds, with the
  replaced PyTorch weights still resident on the GPU;
* **floor** (default for an engine arm): the engines plus the PyTorch
  components the runtime still executes (SigLIP and the PaliGemma embedding table).

The value is the median `cudaMemGetInfo` plateau (total minus free) over 60
timed calls after 10 warm-ups; `memory --help` gives the details. Run the eager
arm first so the engine arms can report their decoded-action cosine against it:

```bash
python -m foldquant_integration.memory --checkpoint-dir <ckpt> --dataset-path <LeRobot LIBERO> --reference-actions ref.npz
python -m foldquant_integration.memory --checkpoint-dir <ckpt> --dataset-path <LeRobot LIBERO> --engine-dir exports/<arm>/engines --reference-actions ref.npz
python -m foldquant_integration.memory --checkpoint-dir <ckpt> --dataset-path <LeRobot LIBERO> --engine-dir exports/<arm>/engines --keep-replaced-weights
```
