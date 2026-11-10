# Merge Notes

This document records the current code-level integration state for `quant-vla` after the QuantVLA, Omega-QVLA, QVLA/OpenVLA, UniVLA, and StarVLA merge work.

## Merged Project Families

| Source family | Current status | Main files |
| --- | --- | --- |
| QuantVLA | Integrated | `gr00t/quantization/`, `scripts/run_groot_benchmark.sh`, `scripts/run_awesome_quant_vla.sh` |
| Omega-QVLA | Integrated | GPTQ builders, Pi0.5 runtime/eval scripts, W4A4 pack route |
| QVLA/OpenVLA | Integrated and validated | `third_party/openvla`, `third_party/openvla_oft`, `tools/qvla`, `scripts/run_openvla_qvla.sh` |
| OpenDriveLab/UniVLA | Integrated and validated | `third_party/univla`, `scripts/run_univla_libero.sh` |
| StarVLA | Integrated and validated for StarVLA-OFT FP16/BF16 | `third_party/starvla`, `scripts/run_starvla_libero.sh`, `tools/starvla/` |

## Current Validation Baseline

These are local engineering validation results, not official paper benchmark numbers.

| Profile | Model | Suite | Result |
| --- | --- | --- | ---: |
| `groot_w4a8` | GR00T-N1.5 | LIBERO Object | 82.0% |
| `pi05_w4a8_duquant` | Pi0.5/OpenPI | LIBERO Object | 99.0% |
| `pi05_w4a4_gptq` | Pi0.5/OpenPI | LIBERO Object | 98.0% |
| `openvla_fp16` | OpenVLA | LIBERO Spatial | 70.0% |
| `openvla_qvla_w8` | OpenVLA | LIBERO Spatial | 80.0% |
| `openvla_oft_fp16` | OpenVLA-OFT | LIBERO Spatial | 99.0% |
| `openvla_oft_qvla_w8` | OpenVLA-OFT | LIBERO Spatial | 98.0% |
| `univla_fp16` | UniVLA | LIBERO Spatial | 96.0% |
| `starvla_oft_fp16` | StarVLA-OFT | LIBERO Spatial | 99.0% |

## Notes

- GR00T and Pi0.5 W4A8 routes use runtime quantization and do not require a prebuilt `quantized.pt` pack.
- GR00T and Pi0.5 W4A4 GPTQ routes require prebuilt or locally generated `quantized.pt` packs.
- OpenVLA/QVLA routes generate calibration, proxy, gate, and bit-allocation artifacts; `proxy.pt` is an analysis artifact, not a standalone deployable quantized model.
- The repaired OpenVLA-OFT rows use 100 Spatial episodes, seed 7, and the same merged checkpoint. W8A16 uses 32 RLDS calibration samples and fake-weight quantization with BF16 activations/compute. Earlier `60.0%`/`26.0%` rows are legacy records rather than the current baseline.
- UniVLA and StarVLA are currently exposed as validated FP16/BF16 LIBERO evaluation routes. Quantized variants need additional action-decoder or Qwen-VL/action-head-aware adapters before being marked as validated.
