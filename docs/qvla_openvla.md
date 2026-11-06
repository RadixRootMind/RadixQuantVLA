# OpenVLA/QVLA Route

For the repaired OpenVLA-OFT action-Jacobian workflow, use
[qvla_oft.md](qvla_oft.md). The input-Hessian description and60%/26% rows below
describe the previous integration, not the repaired OFT path.

quant-vla includes the QVLA/OpenVLA route from AutoLab-SAI-SJTU/QVLA as an integrated third project family.

## Included Backends

| Profile | Backend | Quantization | Launcher |
| --- | --- | --- | --- |
| `openvla_fp16` | `third_party/openvla` | FP16 | `scripts/run_openvla_qvla.sh` |
| `openvla_qvla_w8` | `third_party/openvla` | QVLA mixed-bit W8 | `scripts/run_openvla_qvla.sh` |
| `openvla_oft_fp16` | `third_party/openvla_oft` | FP16 | `scripts/run_openvla_qvla.sh` |
| `openvla_oft_qvla_w8` | `third_party/openvla_oft` | QVLA mixed-bit W8 | `scripts/run_openvla_qvla.sh` |

## Implementation Notes

- `tools/qvla/build_calib_jsonl.py` prepares a small LIBERO calibration JSONL file.
- `tools/qvla/sensitivity_hessian_proxy.py` computes the QVLA sensitivity proxy.
- `tools/qvla/greedy_bit_allocator.py` selects layer bit widths for the target average bit budget.
- `tools/qvla/run_eval.py` loads the original OpenVLA/OpenVLA-OFT model path, injects QVLA fake-weight settings, then runs LIBERO evaluation.

The OpenVLA path intentionally reuses the original project loader where possible. This keeps behavior close to the upstream project while allowing the merged launcher to control quantization and output layout.

## Validated Settings

| Route | Suite | Result |
| --- | --- | ---: |
| `openvla_fp16` | `spatial` | 70.0% |
| `openvla_qvla_w8` | `spatial` | 80.0% |
| `openvla_oft_fp16` | `spatial` | 60.0% |
| `openvla_oft_qvla_w8` | `spatial` | 26.0% |

These numbers are local validation outputs. They verify integration correctness, not final benchmark claims.

## Key Compatibility Fixes

- OpenVLA is run with eager attention by default through `OPENVLA_ATTN_IMPL=eager`.
- OpenVLA action prediction appends the expected prompt delimiter token and matching attention mask when needed.
- The QVLA proxy path provides a label fallback for OpenVLA-OFT forward passes.
- Evaluation can be run without saving videos, or with `imageio[ffmpeg]` installed when videos are desired.

## Minimal Commands

```bash
conda activate awesome_qvla_openvla
cd "$AWESOME_QVLA_ROOT"
source .env.local
export OPENVLA_ATTN_IMPL=eager

bash scripts/run_awesome_quant_vla.sh openvla_qvla_w8 \
  --suite spatial \
  --gpus 0 \
  --trials 10 \
  --openvla-python "$OPENVLA_PYTHON" \
  --openvla-checkpoint "$OPENVLA_CKPT" \
  --output-root "$AWESOME_QVLA_ROOT/results/verify/openvla_qvla_w8_spatial" \
  --max-samples 32
```

Read the result:

```bash
bash scripts/run_awesome_quant_vla.sh openvla_qvla_w8 \
  --suite spatial \
  --output-root "$AWESOME_QVLA_ROOT/results/verify/openvla_qvla_w8_spatial" \
  --action result
```

