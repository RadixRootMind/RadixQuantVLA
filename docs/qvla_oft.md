# Repaired QVLA on OpenVLA-OFT

The OFT route uses `third_party/openvla_oft/qvla`. The unified Radix launcher
and legacy `tools/qvla` CLI entrypoints dispatch to this package when OFT is
selected. The causal OpenVLA route uses its own QVLA package under the same
stage adapter; see the [profile and environment guide](qvla_openvla.md).

Calibration uses native8x7 continuous actions with both cameras and8D proprio,
through the same preprocessing, L1 head, action statistics and bidirectional
SDPA implementation as evaluation. Channel scores use the AppendixG
variance-times-Jacobian proxy. Scoring and injection share one BF16 quantizer.
Allocation uses incremental error per saved bit with a per-layer pruning cap.

This is a proxy-driven QVLA core implementation. Unpublished author calibration
indices, sensitive-channel forward refinement and rollout cross-validation are
not reproduced. Default candidates are0,4,8,16 (no2bit), with at most1% pruning
per layer; these are explicit local choices. Activations and operators remain
BF16. This path does not provide packed integer speedups or storage savings.

## Setup

Use Python3.10 in a dedicated OFT environment. Install the pinned author's
Transformers fork; stock Transformers is rejected by a bidirectional-attention
functional check. Set up LIBERO and OSMesa using the
[backend guide](../third_party/openvla_oft/qvla/README.md).

```bash
python -m pip install -e third_party/openvla_oft
python -m pip install -r third_party/openvla_oft/qvla/requirements.txt

export OPENVLA_PYTHON=/path/to/oft-env/bin/python
export OPENVLA_OFT_CKPT=/path/to/openvla-7b-oft-finetuned-libero-spatial
export QVLA_RLDS_ROOT=/path/to/modified_libero_rlds
export LIBERO_ROOT=/path/to/LIBERO
export LIBERO_CONFIG_PATH=/path/to/private/libero_config
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
```

The local merged checkpoint must include its action head and proprio projector.
The RLDS root contains `libero_{spatial,object,goal,10}_no_noops/1.0.0/` training
datasets from `openvla/modified_libero_rlds`. Default sampling selects512
distinct training trajectories, one frame each, balanced across four suites.
Both raw camera images and proprio are saved; simulator reset screenshots are
not used, and sample counts are checked.

## Unified launcher

Use a **fresh output directory**:

```bash
bash scripts/run_awesome_quant_vla.sh openvla_oft_qvla_w8 \
  --suite spatial --gpus 0 --trials 50 --seed 7 \
  --openvla-python "$OPENVLA_PYTHON" \
  --openvla-checkpoint "$OPENVLA_OFT_CKPT" \
  --rlds-root "$QVLA_RLDS_ROOT" --max-samples 512 \
  --bits 0,4,8,16 --proxy-bits 0,4,8,16 --max-prune-ratio 0.01 \
  --output-root results/oft_qvla_spatial_fresh
```

`--trials 10` runs100 episodes, matching the historical Radix protocol;
`--trials 50` runs the full500. Report these protocols separately.
Optional `--calib-indices /path/to/indices.jsonl` fixes suite/episode/frame
selection; images and proprio are reread from training RLDS and all scores and
gates are recomputed. Indices and generated artifacts are not bundled.

Individual `--action build_calib|proxy|assign|eval|export` stages are supported.
To evaluate the same gates separately, pass `--action eval`, an explicit
`--gates-json`, and a new output directory. The `openvla_oft_fp16` compatibility
profile actually computes in BF16 and now uses the same guarded OFT loader,
seed and wait settings. Its run metadata records the true dtype.

## Checks and outputs

- An offline all16bit injector check must leave all target weights and protected
  components byte-identical; this does not run baseline robot rollouts.
- The first8 gradient-enabled actions must exactly match official inference.
  Observed modules check full-shape forward identity while computing scores.
- Bad gate names, lengths, bits and pruning budgets fail before weight mutation.
  OFT has no median-bit fallback.
- Runtime exceptions abort evaluation instead of reducing policy SR.
- Completion requires task/trial/cumulative counters and final SR to agree.
  Source and protected-component hashes are recorded; videos are kept per run.
- Stage manifests bind caches to checkpoint hashes, code/environment, training
  records and configuration. Unverified or stale artifacts are rejected.

`merged_summary.json`/`.md` is written only after evaluator and independent
verification succeed. Detailed metadata/logs/videos live under `logs/eval/`.
No new experiment results, weights, data, proxies, gates or videos are included
in the code patch. Published60%/26% rows refer to the historical integration.

```bash
cd third_party/openvla_oft
CUDA_VISIBLE_DEVICES='' python -m pytest qvla/tests -q
```
