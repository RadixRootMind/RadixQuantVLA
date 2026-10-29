#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_paths.sh"

die() {
    echo "[foldquantvla] ERROR: $*" >&2
    exit 2
}

require_dir() {
    [[ -d "$1" ]] || die "$2 not found: $1"
}

require_file() {
    [[ -f "$1" ]] || die "$2 not found: $1"
}

require_dataset() {
    [[ -n "${FOLDQUANT_DATASET:-}" ]] || die "set --foldquant-dataset or FOLDQUANT_DATASET"
    require_dir "${FOLDQUANT_DATASET}" "calibration dataset"
}

print_result() {
    local summary="${OUTPUT_ROOT}/libero/summary.json"
    require_file "$summary" "FoldQuant LIBERO summary"
    "${FOLDQUANT_PYTHON}" - "$summary" "${OUTPUT_ROOT}" "${PROFILE}" "${FOLDQUANT_FAMILY}" \
        "${TASK_SUITE}" "${WBITS}" "${ABITS}" <<'PY'
import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
output_root = Path(sys.argv[2])
profile, family, suite = sys.argv[3:6]
wbits, abits = (int(x) for x in sys.argv[6:8])
with source.open("r", encoding="utf-8") as f:
    data = json.load(f)
if isinstance(data.get("overall"), dict):
    totals = data["overall"]
    episodes = totals.get("episodes")
elif isinstance(data.get("totals"), dict):
    totals = data["totals"]
    episodes = totals.get("num_episodes")
else:
    totals = data
    episodes = totals.get("total_episodes")
rate = totals.get("success_rate", totals.get("total_success_rate"))
if rate is None:
    raise SystemExit(f"success rate not found in {source}")
successes = totals.get("successes", totals.get("total_successes"))
merged = {
    "profile": profile,
    "backend": "foldquantvla",
    "model_family": family,
    "suite": suite,
    "wbits": wbits,
    "abits": abits,
    "total_successes": successes,
    "total_episodes": episodes,
    "total_success_rate": float(rate),
    "source_summary": str(source.relative_to(output_root)),
}
(output_root / "merged_summary.json").write_text(
    json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
rows = [
    "# FoldQuantVLA result",
    "",
    f"- Profile: `{profile}`",
    f"- Model family: `{family}`",
    f"- Suite: `{suite}`",
    f"- Precision: W{wbits}A{abits}",
    f"- Successes: {successes if successes is not None else 'n/a'} / {episodes if episodes is not None else 'n/a'}",
    f"- Success rate: {100 * float(rate):.1f}%",
    f"- Native summary: `{source.relative_to(output_root)}`",
    "",
]
(output_root / "merged_summary.md").write_text("\n".join(rows), encoding="utf-8")
print(f"{100 * float(rate):.1f} %")
PY
}

run_module() {
    "${FOLDQUANT_PYTHON}" -m "foldquant_integration.$1" "${@:2}"
}

quantize_model() {
    require_dataset
    require_dir "${FOLDQUANT_CHECKPOINT}" "base checkpoint"
    mkdir -p "$(dirname "${FOLDQUANT_QUANTIZED_DIR}")"
    if [[ "${FOLDQUANT_FAMILY}" == "pi05" ]]; then
        run_module quantize \
            --checkpoint-dir "${FOLDQUANT_CHECKPOINT}" \
            --dataset-path "${FOLDQUANT_DATASET}" \
            --output-dir "${FOLDQUANT_QUANTIZED_DIR}" \
            --config "${OPENPI_CONFIG}" \
            --llm-scheme "${FOLDQUANT_LLM_SCHEME}" \
            --expert-scheme "${FOLDQUANT_ACTION_SCHEME}" \
            --num-calib "${FOLDQUANT_NUM_CALIB}" \
            --seed "${FOLDQUANT_SEED}" \
            --device "${FOLDQUANT_DEVICE}"
    else
        run_module quantize \
            --model-path "${FOLDQUANT_CHECKPOINT}" \
            --dataset-path "${FOLDQUANT_DATASET}" \
            --output-dir "${FOLDQUANT_QUANTIZED_DIR}" \
            --embodiment-tag "${FOLDQUANT_EMBODIMENT_TAG}" \
            --data-config "${FOLDQUANT_DATA_CONFIG}" \
            --denoising-steps "${DENOISING_STEPS}" \
            --llm-scheme "${FOLDQUANT_LLM_SCHEME}" \
            --dit-scheme "${FOLDQUANT_ACTION_SCHEME}" \
            --num-calib "${FOLDQUANT_NUM_CALIB}" \
            --seed "${FOLDQUANT_SEED}" \
            --device "${FOLDQUANT_DEVICE}"
    fi
}

export_onnx() {
    require_dir "${FOLDQUANT_QUANTIZED_DIR}" "quantized checkpoint"
    mkdir -p "${FOLDQUANT_ONNX_DIR}"
    if [[ "${FOLDQUANT_FAMILY}" == "pi05" ]]; then
        run_module export \
            --checkpoint-dir "${FOLDQUANT_QUANTIZED_DIR}" \
            --output-dir "${FOLDQUANT_ONNX_DIR}" \
            --device "${FOLDQUANT_DEVICE}"
    else
        run_module export \
            --model-path "${FOLDQUANT_QUANTIZED_DIR}" \
            --output-dir "${FOLDQUANT_ONNX_DIR}" \
            --embodiment-tag "${FOLDQUANT_EMBODIMENT_TAG}" \
            --device "${FOLDQUANT_DEVICE}"
    fi
}

build_engines() {
    require_file "${FOLDQUANT_ONNX_DIR}/foldquant_export.json" "FoldQuant ONNX manifest"
    mkdir -p "${FOLDQUANT_ENGINE_DIR}"
    run_module build_engines \
        --onnx-dir "${FOLDQUANT_ONNX_DIR}" \
        --engine-dir "${FOLDQUANT_ENGINE_DIR}" \
        --workspace-mb "${FOLDQUANT_WORKSPACE_MB}"
}

verify_engines() {
    require_dataset
    require_file "${FOLDQUANT_ENGINE_DIR}/foldquant_export.json" "FoldQuant engine manifest"
    if [[ "${FOLDQUANT_FAMILY}" == "pi05" ]]; then
        run_module verify \
            --checkpoint-dir "${FOLDQUANT_CHECKPOINT}" \
            --dataset-path "${FOLDQUANT_DATASET}" \
            --engine-dir "${FOLDQUANT_ENGINE_DIR}" \
            --output "${OUTPUT_ROOT}/verify.json" \
            --config "${OPENPI_CONFIG}" \
            --num-samples "${FOLDQUANT_VERIFY_SAMPLES}" \
            --seed "${FOLDQUANT_VERIFY_SEED}" \
            --device "${FOLDQUANT_DEVICE}"
    else
        run_module verify \
            --model-path "${FOLDQUANT_CHECKPOINT}" \
            --dataset-path "${FOLDQUANT_DATASET}" \
            --engine-dir "${FOLDQUANT_ENGINE_DIR}" \
            --output "${OUTPUT_ROOT}/verify.json" \
            --embodiment-tag "${FOLDQUANT_EMBODIMENT_TAG}" \
            --data-config "${FOLDQUANT_DATA_CONFIG}" \
            --denoising-steps "${DENOISING_STEPS}" \
            --num-samples "${FOLDQUANT_VERIFY_SAMPLES}" \
            --seed "${FOLDQUANT_VERIFY_SEED}"
    fi
}

benchmark_engines() {
    require_dataset
    require_file "${FOLDQUANT_ENGINE_DIR}/foldquant_export.json" "FoldQuant engine manifest"
    if [[ "${FOLDQUANT_FAMILY}" == "pi05" ]]; then
        run_module benchmark \
            --checkpoint-dir "${FOLDQUANT_CHECKPOINT}" \
            --dataset-path "${FOLDQUANT_DATASET}" \
            --arms "${FOLDQUANT_LABEL}=${FOLDQUANT_ENGINE_DIR}" \
            --config "${OPENPI_CONFIG}" \
            --num-iterations "${FOLDQUANT_BENCH_ITERATIONS}" \
            --warmup "${FOLDQUANT_BENCH_WARMUP}" \
            --seed "${FOLDQUANT_VERIFY_SEED}" \
            --output "${OUTPUT_ROOT}/benchmark.json" \
            --device "${FOLDQUANT_DEVICE}"
    else
        run_module benchmark \
            --model-path "${FOLDQUANT_CHECKPOINT}" \
            --dataset-path "${FOLDQUANT_DATASET}" \
            --arms "${FOLDQUANT_LABEL}=${FOLDQUANT_ENGINE_DIR}" \
            --embodiment-tag "${FOLDQUANT_EMBODIMENT_TAG}" \
            --data-config "${FOLDQUANT_DATA_CONFIG}" \
            --denoising-steps "${DENOISING_STEPS}" \
            --num-iterations "${FOLDQUANT_BENCH_ITERATIONS}" \
            --warmup "${FOLDQUANT_BENCH_WARMUP}" \
            --seed "${FOLDQUANT_VERIFY_SEED}" \
            --output "${OUTPUT_ROOT}/benchmark.json"
    fi
}

eval_libero() {
    require_file "${FOLDQUANT_ENGINE_DIR}/foldquant_export.json" "FoldQuant engine manifest"
    mkdir -p "${OUTPUT_ROOT}/libero"
    export MUJOCO_GL="${MUJOCO_GL:-egl}"
    export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
    if [[ "${FOLDQUANT_FAMILY}" == "pi05" ]]; then
        require_file "${FOLDQUANT_CLIENT_PYTHON}" "Pi0.5 LIBERO client Python"
        run_module eval_libero \
            --checkpoint-dir "${FOLDQUANT_CHECKPOINT}" \
            --client-python "${FOLDQUANT_CLIENT_PYTHON}" \
            --engine-dir "${FOLDQUANT_ENGINE_DIR}" \
            --output "${OUTPUT_ROOT}/libero" \
            --config "${OPENPI_CONFIG}" \
            --suites "${TASK_SUITE}" \
            --protocol "${FOLDQUANT_PROTOCOL}" \
            --num-trials-per-task "${NUM_TRIALS_PER_TASK}" \
            --num-steps-wait "${GR00T_EVAL_INIT_OFFSET}" \
            --replan-steps "${REPLAN_STEPS}" \
            --seed "${FOLDQUANT_EVAL_SEED}" \
            --port "${PORT_BASE}"
    else
        run_module eval_libero \
            --model-path "${FOLDQUANT_CHECKPOINT}" \
            --engine-dir "${FOLDQUANT_ENGINE_DIR}" \
            --output "${OUTPUT_ROOT}/libero" \
            --embodiment-tag "${FOLDQUANT_EMBODIMENT_TAG}" \
            --data-config "${FOLDQUANT_DATA_CONFIG}" \
            --denoising-steps "${DENOISING_STEPS}" \
            --suites "${TASK_SUITE}" \
            --protocol "${FOLDQUANT_PROTOCOL}" \
            --n-episodes "${NUM_TRIALS_PER_TASK}" \
            --num-steps-wait "${GR00T_EVAL_INIT_OFFSET}" \
            --seed "${FOLDQUANT_EVAL_SEED}"
    fi
    print_result
}

build_kernels() {
    "${FOLDQUANT_PYTHON}" -m foldquant.kernels build
    "${FOLDQUANT_PYTHON}" -m foldquant.kernels status
}

AWESOME_QVLA_ROOT="${AWESOME_QVLA_ROOT:-${QUANTVLA_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}}"
QUANTVLA_ROOT="${QUANTVLA_ROOT:-${AWESOME_QVLA_ROOT}}"
FOLDQUANT_ROOT="${FOLDQUANT_ROOT:-${QUANTVLA_ROOT}/third_party/foldquantvla}"
FOLDQUANT_FAMILY="${FOLDQUANT_FAMILY:?FOLDQUANT_FAMILY is required}"
FOLDQUANT_PYTHON="${FOLDQUANT_PYTHON:-python}"
FOLDQUANT_DEVICE="${FOLDQUANT_DEVICE:-cuda}"
FOLDQUANT_NUM_CALIB="${FOLDQUANT_NUM_CALIB:-128}"
FOLDQUANT_SEED="${FOLDQUANT_SEED:-0}"
FOLDQUANT_VERIFY_SEED="${FOLDQUANT_VERIFY_SEED:-42}"
FOLDQUANT_VERIFY_SAMPLES="${FOLDQUANT_VERIFY_SAMPLES:-16}"
FOLDQUANT_EVAL_SEED="${FOLDQUANT_EVAL_SEED:-7}"
FOLDQUANT_PROTOCOL="${FOLDQUANT_PROTOCOL:-p3}"
FOLDQUANT_WORKSPACE_MB="${FOLDQUANT_WORKSPACE_MB:-8192}"
FOLDQUANT_BENCH_ITERATIONS="${FOLDQUANT_BENCH_ITERATIONS:-20}"
FOLDQUANT_BENCH_WARMUP="${FOLDQUANT_BENCH_WARMUP:-5}"
FOLDQUANT_MAX_SHARD_GIB="${FOLDQUANT_MAX_SHARD_GIB:-100}"
FOLDQUANT_EMBODIMENT_TAG="${FOLDQUANT_EMBODIMENT_TAG:-new_embodiment}"
FOLDQUANT_DATA_CONFIG="${FOLDQUANT_DATA_CONFIG:-examples.Libero.custom_data_config:LiberoDataConfig}"
FOLDQUANT_QUANTIZED_DIR="${FOLDQUANT_QUANTIZED_DIR:-${OUTPUT_ROOT}/quantized}"
FOLDQUANT_ONNX_DIR="${FOLDQUANT_ONNX_DIR:-${OUTPUT_ROOT}/onnx}"
FOLDQUANT_ENGINE_DIR="${FOLDQUANT_ENGINE_DIR:-${OUTPUT_ROOT}/engines}"

require_dir "${FOLDQUANT_ROOT}/foldquant" "FoldQuant core"
if [[ "${FOLDQUANT_PYTHON}" != */* ]]; then
    FOLDQUANT_PYTHON="$(command -v "${FOLDQUANT_PYTHON}" || true)"
fi
[[ -n "${FOLDQUANT_PYTHON}" ]] || die "FoldQuant Python is not available"
require_file "${FOLDQUANT_PYTHON}" "FoldQuant Python"

if [[ "${FOLDQUANT_FAMILY}" == "pi05" ]]; then
    ADAPTER_ROOT="${FOLDQUANT_ROOT}/adapters/pi05"
    export FOLDQUANT_OPENPI_ROOT="${FOLDQUANT_OPENPI_ROOT:-${OPENPI_ROOT}}"
    FOLDQUANT_CLIENT_PYTHON="${FOLDQUANT_CLIENT_PYTHON:-${OPENPI_ROOT}/examples/libero/.venv/bin/python}"
    EXTRA_PYTHONPATH="${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src"
elif [[ "${FOLDQUANT_FAMILY}" == "groot_n1_5" ]]; then
    ADAPTER_ROOT="${FOLDQUANT_ROOT}/adapters/groot_n1_5"
    export FOLDQUANT_GROOT_ROOT="${FOLDQUANT_GROOT_ROOT:-${QUANTVLA_ROOT}}"
    EXTRA_PYTHONPATH="${QUANTVLA_ROOT}"
else
    die "unknown family ${FOLDQUANT_FAMILY}"
fi

require_dir "${ADAPTER_ROOT}/foldquant_integration" "FoldQuant adapter"
export FOLDQUANT_LIBERO_DIR="${FOLDQUANT_LIBERO_DIR:-${LIBERO_ROOT:-$(quantvla_default_libero_root)}}"
export FOLDQUANT_MAX_SHARD_GIB
export PYTHONNOUSERSITE=1
export PYTHONPATH="${ADAPTER_ROOT}:${FOLDQUANT_ROOT}:${EXTRA_PYTHONPATH}:${PYTHONPATH:-}"

# RadixQuantVLA's other quantizers patch modules during import. FoldQuant must
# own the LLM and action-module replacements for these profiles.
unset GR00T_GPTQ_PATH_OVERRIDE GR00T_GPTQ_PATH GR00T_GPTQ_MISSING
unset GR00T_DUQUANT_ENABLE GR00T_DUQUANT_PATH GR00T_RTN_ENABLE
unset GR00T_ATM_ENABLE GR00T_ATM_ALPHA_PATH GR00T_ATM_SCOPE
unset GR00T_OHB_ENABLE GR00T_OHB_SCOPE GR00T_DIT_CORRECTION_PATH
unset OPENPI_GPTQ_PATH OPENPI_GPTQ_INCLUDE OPENPI_GPTQ_EXCLUDE
unset OPENPI_DUQUANT_INCLUDE OPENPI_DUQUANT_EXCLUDE

mkdir -p "${OUTPUT_ROOT}"

case "${ACTION}" in
    run|eval) eval_libero ;;
    quantize) quantize_model ;;
    export) export_onnx ;;
    build) build_engines ;;
    kernels) build_kernels ;;
    verify) verify_engines ;;
    benchmark) benchmark_engines ;;
    result) print_result ;;
    pipeline)
        quantize_model
        export_onnx
        build_engines
        verify_engines
        benchmark_engines
        eval_libero
        ;;
    *) die "unsupported action ${ACTION}" ;;
esac
