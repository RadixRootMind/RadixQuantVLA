# OpenVLA / OpenVLA-OFT QVLA routes

Both routes use Radix's unified launcher, stage commands and output layout.
The numerical pipelines live in `third_party/{openvla,openvla_oft}/qvla`.
The legacy `tools/qvla` entrypoints dispatch to the selected package through
`tools/qvla/oft_cli.py`; its historical filename and `.oft_*` manifests are
retained for compatibility.

## Profiles and precision

| Profile | Calibration | Evaluation | Pruned-channel repair |
| --- | --- | --- | --- |
| `openvla_qvla_w4` | Causal SDPA, soft-action surrogate | Causal SDPA | 0 |
| `openvla_qvla_w8` | Causal SDPA, soft-action surrogate | Causal FlashAttention 2.5.5 | 0.25 |
| `openvla_oft_qvla_w4` | Native continuous actions, bidirectional SDPA | Bidirectional SDPA | 0 |
| `openvla_oft_qvla_w8` | Native continuous actions, bidirectional SDPA | Bidirectional SDPA | 0 |

W4/W8 is the mean output-channel bit budget across allocated modules, rather
than a parameter-weighted whole-model budget. Weights use symmetric per-output-
channel RTN with BF16 scales, then BF16 computation. Activations stay BF16
(A16). There is no packed integer storage or acceleration in these routes.
The `openvla_fp16` and `openvla_oft_fp16` compatibility names compute in BF16;
metadata records the actual dtype.

Default candidates are `0,4,8,16`, with at most 1% zero-bit channels per layer.
The allocator uses marginal error per saved bit. OpenVLA W8 additionally restores
25% of pruned channels with the largest score benefit, compensating with the
same number of low-cost 16-to-8 demotions to conserve the budget. This is a local
recipe, configurable with `--restore-pruned-fraction`; OFT requires zero.
The projector, language-model output head, and OFT action/proprio heads remain
BF16. Zero-bit weights retain the original bias.

## Separate environments

Use Python 3.10 and PyTorch 2.2.0. OpenVLA requires stock causal
`transformers==4.40.1`; OFT requires the author's bidirectional fork pinned to
`bc339d9ad707454c0c115970db43c260067c61ab`. Their version strings are identical,
so use separate virtual environments. Functional checks verify the attention
semantics and record the actual loaded layer classes.

Follow the [OpenVLA setup](../third_party/openvla/qvla/README.md) or
[OFT setup](../third_party/openvla_oft/qvla/README.md). The OpenVLA W8 profile
also needs `flash-attn==2.5.5` installed for the selected PyTorch/CUDA build.
An existing installation overlay can be selected with `--flash-attn-path PATH`
or `OPENVLA_FLASH_ATTN_PATH`; no machine-specific path is embedded in the code.
`--attention` and `--calibration-attention` explicitly control the new loader;
legacy `OPENVLA_ATTN_IMPL` does not control these packages. OFT requires SDPA.

## Calibration data and method scope

Download `openvla/modified_libero_rlds` from Hugging Face. The RLDS root contains
`libero_{spatial,object,goal,10}_no_noops/1.0.0/`. Default calibration selects
512 distinct training trajectories, one frame each, balanced over four suites.
An optional `--calib-indices PATH` preserves a supplied suite/episode/frame list
while rereading images and state from training data. Reset screenshots are not
used. These local sample choices are not unpublished paper calibration indices.

OpenVLA uses its earlier single-camera recipe: Lanczos resize to 224 and one
90%-area center crop. Its score is the squared directional derivative of a
softmax action-bin expectation, conditioned on generated action prefixes.
This is a differentiable surrogate for the token policy; it is not the derivative
of discrete argmax actions. Cached generation and teacher-forced forwards can
differ under BF16 kernels; token agreement is recorded rather than assumed.
The port retains Radix's prompt-token/attention-mask alignment fix, so old
rollout scores are not automatically evidence for the new code.

OFT uses raw front/wrist images, 8D proprio, official preprocessing, L1 head,
action unnormalization and the native 8x7 continuous-action Jacobian. It scores
the Appendix G variance-times-Jacobian proxy and checks native forward identity.
See [OFT method details](qvla_oft.md). Neither route includes the unpublished
sensitive-channel forward refinement or rollout cross-validation stages.

## Unified commands

Set `LIBERO_ROOT`, a private `LIBERO_CONFIG_PATH`, `MUJOCO_GL=osmesa`, and
`PYOPENGL_PLATFORM=osmesa` as described in the setup guides. Select one visible
GPU. Use a fresh output directory for each run and set explicit checkpoint and
Python paths for the selected family.

```bash
bash scripts/run_awesome_quant_vla.sh openvla_qvla_w8 \
  --suite spatial --gpus 0 --trials 50 --seed 7 \
  --openvla-python /path/to/causal-env/bin/python \
  --openvla-checkpoint /path/to/openvla-7b-finetuned-libero-spatial \
  --rlds-root /path/to/modified_libero_rlds --max-samples 512 \
  --output-root results/openvla_w8_fresh

bash scripts/run_awesome_quant_vla.sh openvla_oft_qvla_w8 \
  --suite spatial --gpus 0 --trials 50 --seed 7 \
  --openvla-python /path/to/oft-env/bin/python \
  --openvla-checkpoint /path/to/openvla-7b-oft-finetuned-libero-spatial \
  --rlds-root /path/to/modified_libero_rlds --max-samples 512 \
  --output-root results/oft_w8_fresh
```

Select the corresponding `_w4` profile for W4A16. `--trials 1` is a 10-episode
smoke check, `--trials 10` is 100 episodes, and `--trials 50` is 500 episodes.
Report each protocol separately. No numerical outcome is bundled with this update.

`--action plan|build_calib|proxy|assign|eval|export|run|result` is supported.
For staged runs, repeat the same arguments/output directory and change only
`--action`. For a startup check use `--max-samples 8 --trials 1` from preparation
onward. The old input-Hessian `--max-layers` option is unsupported. For an
independent evaluation use `--action eval --gates-json PATH` and a fresh output.

Stage manifests bind reuse to checkpoint hashes, code/environment, training
manifest and configuration. Stale, partial and untracked artifacts are rejected.
Runtime faults abort evaluation. Gates are fully validated before mutation.
`merged_summary.json` and `.md` are produced only after task/trial/cumulative
counters and an independent canonical-log verification agree. Per-run logs,
metadata and videos live under `logs/eval/`. Optional `export` writes a full
BF16 checkpoint with injected fake weights.
