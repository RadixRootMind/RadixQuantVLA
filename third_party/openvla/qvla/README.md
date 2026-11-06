# Causal OpenVLA QVLA backend

This package ports the locally validated OpenVLA weight-only recipe to the
Radix stage interface. See the [route guide](../../../docs/qvla_openvla.md) for
profiles, commands and method limitations. No experiment results or artifacts
are bundled.

## Install

Use a dedicated Python 3.10 environment, separate from OpenVLA-OFT. From the
repository root:

```bash
python3.10 -m venv .venv-openvla
source .venv-openvla/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
  --index-url https://download.pytorch.org/whl/cu118
python -m pip install -e third_party/openvla
python -m pip install -r third_party/openvla/qvla/requirements.txt

# Needed for the default OpenVLA W8 evaluator, after installing torch:
python -m pip install flash-attn==2.5.5 --no-build-isolation
```

Install LIBERO at commit `8f1084e3132a39270c3a13ebe37270a43ece2a01`,
robosuite 1.4.1, MuJoCo 2.3.7, ffmpeg and the system OSMesa library. Follow the
[LIBERO/private config instructions](../../openvla_oft/qvla/README.md#installation)
using the causal environment. Stock Transformers 4.40.1 is required here;
OFT's fork changes SDPA semantics and is rejected by a functional causal check.

Download the merged `openvla/openvla-7b-finetuned-libero-spatial` checkpoint and
`openvla/modified_libero_rlds` training dataset from Hugging Face. The checkpoint
must include its processor/tokenizer and action normalization statistics.
Pass local asset paths through the unified launcher rather than editing sources.

## Semantics and checks

Calibration preserves the original single-camera, prepared-image recipe and
soft-action directional sensitivity. Prepared manifests carry image checksums,
prompts and `preprocessed=true`; this prevents applying center crop twice.
Scores use BF16 fake-weight perturbations and FP32 accumulation. The projector
and language-model output head remain protected. W8 pruning repair is applied
by the Radix profile, or explicitly through the allocator's
`--restore_pruned_fraction 0.25`; direct module defaults do not enable it.

The shared quantizer validates all channel assignments before any mutation.
The evaluator wraps the bundled LIBERO evaluator, fails on runtime errors,
isolates videos and checks source/protected-parameter identities. It uses
Radix's prompt-token/attention-mask alignment fix. Earlier rollout numbers must
be revalidated with this implementation and the recorded attention backend.

```bash
cd third_party/openvla
CUDA_VISIBLE_DEVICES='' python -m pytest qvla/tests -q
```

Tests use synthetic inputs to check directional scores, allocation budgets,
quantization identity, causal attention and canonical-log consistency. CI tests
this package with stock Transformers and OFT separately with its pinned fork.
