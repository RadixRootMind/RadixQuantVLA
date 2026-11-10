#!/bin/bash
# OpenVLA/OpenVLA-OFT QVLA runner for Awesome-quant-vla.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_paths.sh"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_openvla_qvla.sh [options]

Actions:
  --action plan|build_calib|proxy|assign|eval|export|run|result

Options:
  --profile NAME
  --backend openvla|openvla_oft
  --suite object|spatial|goal|long
  --checkpoint PATH
  --python PATH
  --gpus 0[,1]
  --trials N
  --port-base PORT        Reserved for interface symmetry.
  --output-root PATH
  --calib-jsonl PATH
  --proxy-pt PATH
  --gates-json PATH
  --target-avg-bits FLOAT
  --bits 0,4,8,16
  --proxy-bits 0,4,8,16
  --max-samples N
  --libero-root PATH
  --center-crop True|False
  --method qvla|fp16
  --rlds-root PATH       Modified LIBERO RLDS training root (required for calibration).
  --calib-indices PATH   Optional fixed training trajectory/frame index JSONL.
  --max-prune-ratio N    Per-layer pruning cap (default 0.01).
  --seed N              Default 7.
  --attention BACKEND    Evaluation: sdpa, eager, flash_attention_2.
  --calibration-attention BACKEND
                        Calibration defaults to sdpa for both families.
  --restore-pruned-fraction N
                        OpenVLA W8 defaults to 0.25; other profiles use 0.
  --flash-attn-path PATH Optional installed FA2 overlay for OpenVLA W8.
EOF
}

die() {
    echo "[openvla-qvla] ERROR: $*" >&2
    exit 2
}

suite_to_task_suite() {
    case "$1" in
        long) echo "libero_10" ;;
        spatial) echo "libero_spatial" ;;
        goal) echo "libero_goal" ;;
        object) echo "libero_object" ;;
        libero_10|libero_spatial|libero_goal|libero_object|libero_90) echo "$1" ;;
        *) die "unknown suite '$1'" ;;
    esac
}

suite_to_short() {
    case "$1" in
        libero_10) echo "long" ;;
        libero_spatial) echo "spatial" ;;
        libero_goal) echo "goal" ;;
        libero_object) echo "object" ;;
        libero_90) echo "90" ;;
        *) echo "$1" ;;
    esac
}

suite_to_openvla_suffix() {
    case "$1" in
        long|libero_10) echo "10" ;;
        spatial|libero_spatial) echo "spatial" ;;
        goal|libero_goal) echo "goal" ;;
        object|libero_object) echo "object" ;;
        *) die "OpenVLA checkpoint suffix is not defined for suite '$1'" ;;
    esac
}

print_result() {
    local summary="$1/merged_summary.json"
    [[ -f "$summary" ]] || die "summary not found: $summary"
    "$PYTHON_BIN" - "$summary" <<'PY'
import json, sys
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    data = json.load(f)
print(f"{100 * data['total_success_rate']:.1f} %")
PY
}

PROFILE="${PROFILE:-}"
ACTION="${ACTION:-run}"
BACKEND="${OPENVLA_BACKEND:-openvla_oft}"
METHOD="${METHOD:-qvla}"
SUITE="${SUITE:-spatial}"
GPU_LIST="${GPU_LIST:-0}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-1}"
GR00T_EVAL_INIT_OFFSET="${GR00T_EVAL_INIT_OFFSET:-10}"
CHECKPOINTS_ROOT="${CHECKPOINTS_ROOT:-${HOME}/ckpts}"
AWESOME_QVLA_ROOT="${AWESOME_QVLA_ROOT:-${QUANTVLA_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}}"
QUANTVLA_ROOT="${QUANTVLA_ROOT:-${AWESOME_QVLA_ROOT}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
OPENVLA_CHECKPOINT="${OPENVLA_CHECKPOINT:-}"
OPENVLA_PYTHON="${OPENVLA_PYTHON:-python}"
CALIB_JSONL="${CALIB_JSONL:-}"
PROXY_PT="${PROXY_PT:-}"
GATES_JSON="${GATES_JSON:-}"
TARGET_AVG_BITS="${TARGET_AVG_BITS:-8.0}"
BITS="${BITS:-}"
PROXY_BITS="${PROXY_BITS:-}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
QVLA_RLDS_ROOT="${QVLA_RLDS_ROOT:-}"
QVLA_CALIB_INDICES="${QVLA_CALIB_INDICES:-}"
QVLA_MAX_PRUNE_RATIO="${QVLA_MAX_PRUNE_RATIO:-0.01}"
MAX_LAYERS="${MAX_LAYERS:-}"
LIBERO_ROOT_ARG="${LIBERO_ROOT:-}"
CENTER_CROP="${CENTER_CROP:-True}"
SEED="${SEED:-7}"
QVLA_ATTENTION="${QVLA_ATTENTION:-}"
QVLA_CALIBRATION_ATTENTION="${QVLA_CALIBRATION_ATTENTION:-}"
QVLA_RESTORE_PRUNED_FRACTION="${QVLA_RESTORE_PRUNED_FRACTION:-}"
OPENVLA_FLASH_ATTN_PATH="${OPENVLA_FLASH_ATTN_PATH:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --action|--mode) ACTION="${2:?missing --action value}"; shift 2 ;;
        --profile) PROFILE="${2:?missing --profile value}"; shift 2 ;;
        --backend) BACKEND="${2:?missing --backend value}"; shift 2 ;;
        --method) METHOD="${2:?missing --method value}"; shift 2 ;;
        --suite) SUITE="${2:?missing --suite value}"; shift 2 ;;
        --checkpoint|--openvla-checkpoint) OPENVLA_CHECKPOINT="${2:?missing checkpoint value}"; shift 2 ;;
        --python|--openvla-python) OPENVLA_PYTHON="${2:?missing python value}"; shift 2 ;;
        --gpus|--gpu-list) GPU_LIST="${2:?missing --gpus value}"; shift 2 ;;
        --trials) NUM_TRIALS_PER_TASK="${2:?missing --trials value}"; shift 2 ;;
        --init-offset) GR00T_EVAL_INIT_OFFSET="${2:?missing --init-offset value}"; shift 2 ;;
        --port-base) PORT_BASE="${2:?missing --port-base value}"; shift 2 ;;
        --output-root) OUTPUT_ROOT="${2:?missing --output-root value}"; shift 2 ;;
        --checkpoints-root) CHECKPOINTS_ROOT="${2:?missing --checkpoints-root value}"; shift 2 ;;
        --calib-jsonl) CALIB_JSONL="${2:?missing --calib-jsonl value}"; shift 2 ;;
        --proxy-pt) PROXY_PT="${2:?missing --proxy-pt value}"; shift 2 ;;
        --gates-json) GATES_JSON="${2:?missing --gates-json value}"; shift 2 ;;
        --target-avg-bits) TARGET_AVG_BITS="${2:?missing --target-avg-bits value}"; shift 2 ;;
        --bits) BITS="${2:?missing --bits value}"; shift 2 ;;
        --proxy-bits) PROXY_BITS="${2:?missing --proxy-bits value}"; shift 2 ;;
        --max-samples) MAX_SAMPLES="${2:?missing --max-samples value}"; shift 2 ;;
        --max-layers) MAX_LAYERS="${2:?missing --max-layers value}"; shift 2 ;;
        --rlds-root) QVLA_RLDS_ROOT="${2:?missing --rlds-root value}"; shift 2 ;;
        --calib-indices) QVLA_CALIB_INDICES="${2:?missing --calib-indices value}"; shift 2 ;;
        --max-prune-ratio) QVLA_MAX_PRUNE_RATIO="${2:?missing --max-prune-ratio value}"; shift 2 ;;
        --libero-root) LIBERO_ROOT_ARG="${2:?missing --libero-root value}"; shift 2 ;;
        --center-crop) CENTER_CROP="${2:?missing --center-crop value}"; shift 2 ;;
        --seed) SEED="${2:?missing --seed value}"; shift 2 ;;
        --attention) QVLA_ATTENTION="${2:?missing --attention value}"; shift 2 ;;
        --calibration-attention) QVLA_CALIBRATION_ATTENTION="${2:?missing --calibration-attention value}"; shift 2 ;;
        --restore-pruned-fraction) QVLA_RESTORE_PRUNED_FRACTION="${2:?missing --restore-pruned-fraction value}"; shift 2 ;;
        --flash-attn-path) OPENVLA_FLASH_ATTN_PATH="${2:?missing --flash-attn-path value}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option '$1'" ;;
    esac
done

case "$ACTION" in
    plan|build_calib|proxy|assign|eval|export|run|result) ;;
    *) die "unknown action '$ACTION'" ;;
esac
case "$BACKEND" in
    openvla) BACKEND_ROOT="${QUANTVLA_ROOT}/third_party/openvla" ;;
    openvla_oft|openvla-oft) BACKEND="openvla_oft"; BACKEND_ROOT="${QUANTVLA_ROOT}/third_party/openvla_oft" ;;
    *) die "unknown backend '$BACKEND'" ;;
esac
[[ -d "$BACKEND_ROOT" ]] || die "backend root not found: $BACKEND_ROOT"
if [[ "$BACKEND" == "openvla_oft" ]]; then
    BITS="${BITS:-0,4,8,16}"
    PROXY_BITS="${PROXY_BITS:-0,4,8,16}"
    MAX_SAMPLES="${MAX_SAMPLES:-512}"
else
    BITS="${BITS:-0,4,8,16}"
    PROXY_BITS="${PROXY_BITS:-0,4,8,16}"
    MAX_SAMPLES="${MAX_SAMPLES:-512}"
fi

if [[ -z "$PROFILE" ]]; then
    if [[ "$METHOD" == "fp16" ]]; then
        if [[ "$BACKEND" == "openvla_oft" ]]; then
            PROFILE="openvla_oft_fp16"
        else
            PROFILE="openvla_fp16"
        fi
    else
        if [[ "$BACKEND" == "openvla_oft" ]]; then
            PROFILE="openvla_oft_qvla_w8"
        else
            PROFILE="openvla_qvla_w8"
        fi
    fi
fi

TASK_SUITE="$(suite_to_task_suite "$SUITE")"
SUITE_SHORT="$(suite_to_short "$TASK_SUITE")"
SUITE_SUFFIX="$(suite_to_openvla_suffix "$TASK_SUITE")"
SUITE="$SUITE_SHORT"
OUTPUT_ROOT="${OUTPUT_ROOT:-${QUANTVLA_ROOT}/results/awesome_quant_vla/${PROFILE}_${SUITE}}"
if [[ "$BACKEND" == "openvla_oft" ]]; then
    CALIB_JSONL="${CALIB_JSONL:-${OUTPUT_ROOT}/calib/records.jsonl}"
else
    CALIB_JSONL="${CALIB_JSONL:-${OUTPUT_ROOT}/calib/calib.jsonl}"
fi
PROXY_PT="${PROXY_PT:-${OUTPUT_ROOT}/proxy/proxy.pt}"
GATES_JSON="${GATES_JSON:-${OUTPUT_ROOT}/gates/greedy_bits.json}"
LOG_DIR="${OUTPUT_ROOT}/logs/eval"
EXPORT_DIR="${OUTPUT_ROOT}/export/fakew"
PYTHON_BIN="$OPENVLA_PYTHON"

if [[ -z "$OPENVLA_CHECKPOINT" ]]; then
    if [[ "$BACKEND" == "openvla_oft" ]]; then
        OPENVLA_CHECKPOINT="${CHECKPOINTS_ROOT}/openvla-7b-oft-finetuned-libero-${SUITE_SUFFIX}"
    else
        OPENVLA_CHECKPOINT="${CHECKPOINTS_ROOT}/openvla-7b-finetuned-libero-${SUITE_SUFFIX}"
    fi
fi

FIRST_GPU="${GPU_LIST%%,*}"
export CUDA_VISIBLE_DEVICES="$FIRST_GPU"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-$FIRST_GPU}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export QVLA_BACKEND_ROOT="$BACKEND_ROOT"
export QVLA_ATTENTION QVLA_CALIBRATION_ATTENTION QVLA_RESTORE_PRUNED_FRACTION OPENVLA_FLASH_ATTN_PATH
if [[ -n "$LIBERO_ROOT_ARG" ]]; then
    export LIBERO_ROOT="$LIBERO_ROOT_ARG"
fi
export PYTHONPATH="${BACKEND_ROOT}:${QUANTVLA_ROOT}/tools:${QUANTVLA_ROOT}${LIBERO_ROOT:+:${LIBERO_ROOT}}:${PYTHONPATH:-}"

print_plan() {
    cat <<EOF
================================================================
[openvla-qvla] plan
  profile        : ${PROFILE}
  action         : ${ACTION}
  backend        : ${BACKEND}
  method         : ${METHOD}
  suite          : ${SUITE} (${TASK_SUITE})
  project root   : ${QUANTVLA_ROOT}
  backend root   : ${BACKEND_ROOT}
  python         : ${PYTHON_BIN}
  checkpoint     : ${OPENVLA_CHECKPOINT}
  output         : ${OUTPUT_ROOT}
  gpus           : ${GPU_LIST}
  trials/task    : ${NUM_TRIALS_PER_TASK}
  calib jsonl    : ${CALIB_JSONL}
  proxy pt       : ${PROXY_PT}
  gates json     : ${GATES_JSON}
  bits           : ${BITS}
  proxy bits     : ${PROXY_BITS}
  target avg bits: ${TARGET_AVG_BITS}
================================================================
EOF
}

run_build_calib() {
    mkdir -p "$(dirname "$CALIB_JSONL")"
    "$PYTHON_BIN" "${QUANTVLA_ROOT}/tools/qvla/build_calib_jsonl.py" \
        --task-suite-name "$TASK_SUITE" \
        --out-jsonl "$CALIB_JSONL" \
        --num-samples "$MAX_SAMPLES" \
        --seed "$SEED"
}

run_proxy() {
    mkdir -p "$(dirname "$PROXY_PT")"
    local extra=()
    if [[ -n "$MAX_LAYERS" ]]; then
        extra+=(--max_layers "$MAX_LAYERS")
    fi
    "$PYTHON_BIN" "${QUANTVLA_ROOT}/tools/qvla/sensitivity_hessian_proxy.py" \
        --backend-root "$BACKEND_ROOT" \
        --pretrained_checkpoint "$OPENVLA_CHECKPOINT" \
        --calib_jsonl "$CALIB_JSONL" \
        --out_path "$PROXY_PT" \
        --bits "$PROXY_BITS" \
        --max_samples "$MAX_SAMPLES" \
        --device cuda:0 \
        "${extra[@]}"
}

run_assign() {
    mkdir -p "$(dirname "$GATES_JSON")"
    "$PYTHON_BIN" "${QUANTVLA_ROOT}/tools/qvla/assign_gates_from_sensitivity.py" \
        --proxy_pt "$PROXY_PT" \
        --bits "$BITS" \
        --target_avg_bits "$TARGET_AVG_BITS" \
        --out_json "$GATES_JSON"
}

run_eval_qvla() {
    mkdir -p "$LOG_DIR"
    local args=(
        --backend-root "$BACKEND_ROOT"
        --pretrained_checkpoint "$OPENVLA_CHECKPOINT"
        --gates_path "$GATES_JSON"
        --task_suite_name "$TASK_SUITE"
        --num_trials_per_task "$NUM_TRIALS_PER_TASK"
        --num-steps-wait "$GR00T_EVAL_INIT_OFFSET"
        --local_log_dir "$LOG_DIR"
        --center-crop "$CENTER_CROP"
        --run-id-note "${METHOD}_${BACKEND}_${SUITE}_w${TARGET_AVG_BITS}"
        --seed "$SEED"
    )
    if [[ -n "${LIBERO_ROOT:-}" ]]; then
        args+=(--libero_root "$LIBERO_ROOT")
    fi
    "$PYTHON_BIN" "${QUANTVLA_ROOT}/tools/qvla/run_eval.py" "${args[@]}"
    run_parse
}

run_eval_fp16() {
    mkdir -p "$LOG_DIR"
    "$PYTHON_BIN" "${BACKEND_ROOT}/experiments/robot/libero/run_libero_eval.py" \
        --model_family openvla \
        --pretrained_checkpoint "$OPENVLA_CHECKPOINT" \
        --task_suite_name "$TASK_SUITE" \
        --num_trials_per_task "$NUM_TRIALS_PER_TASK" \
        --num_steps_wait "$GR00T_EVAL_INIT_OFFSET" \
        --local_log_dir "$LOG_DIR" \
        --center_crop "$CENTER_CROP" \
        --run_id_note "bf16_${BACKEND}_${SUITE}" \
        --seed "$SEED" \
        --use_wandb False
    run_parse
}

run_parse() {
    "$PYTHON_BIN" "${QUANTVLA_ROOT}/tools/qvla/parse_openvla_eval.py" \
        --log-dir "$LOG_DIR" \
        --output-root "$OUTPUT_ROOT" \
        --task-suite-name "$TASK_SUITE" \
        --benchmark "${PROFILE}_${SUITE}" \
        --quantization "$([[ "$METHOD" == "fp16" ]] && echo BF16 || echo QVLA-W${TARGET_AVG_BITS}A16)" \
        --gpus "$GPU_LIST"
}

run_export() {
    mkdir -p "$EXPORT_DIR"
    "$PYTHON_BIN" "${QUANTVLA_ROOT}/tools/qvla/inject_fake_w.py" \
        --backend-root "$BACKEND_ROOT" \
        --pretrained_checkpoint "$OPENVLA_CHECKPOINT" \
        --gates_path "$GATES_JSON" \
        --out_dir "$EXPORT_DIR"
}

if [[ "$ACTION" != "result" ]]; then
    print_plan
fi

if [[ -f "${BACKEND_ROOT}/qvla/calibrate_actions.py" ]]; then
    if [[ "$METHOD" == "fp16" ]]; then
        run_id_note="bf16_${BACKEND}_${SUITE}"
    else
        run_id_note="qvla_${BACKEND}_${SUITE}_w${TARGET_AVG_BITS}a16"
    fi
    oft_args=(
        --stage "$ACTION" --backend-root "$BACKEND_ROOT" --profile "$PROFILE" --method "$METHOD"
        --pretrained_checkpoint "$OPENVLA_CHECKPOINT" --output-root "$OUTPUT_ROOT"
        --calib_jsonl "$CALIB_JSONL" --proxy_pt "$PROXY_PT" --gates_path "$GATES_JSON"
        --local_log_dir "$LOG_DIR" --out_dir "$EXPORT_DIR" --task_suite_name "$TASK_SUITE"
        --trials "$NUM_TRIALS_PER_TASK" --max-samples "$MAX_SAMPLES"
        --bits "$BITS" --proxy-bits "$PROXY_BITS" --target-avg-bits "$TARGET_AVG_BITS"
        --max-prune-ratio "$QVLA_MAX_PRUNE_RATIO" --seed "$SEED"
        --center-crop "$CENTER_CROP" --num-steps-wait "$GR00T_EVAL_INIT_OFFSET"
        --run-id-note "$run_id_note"
    )
    [[ -z "$QVLA_RLDS_ROOT" ]] || oft_args+=(--rlds-root "$QVLA_RLDS_ROOT")
    [[ -z "$QVLA_CALIB_INDICES" ]] || oft_args+=(--calib-indices "$QVLA_CALIB_INDICES")
    [[ -z "$LIBERO_ROOT_ARG" ]] || oft_args+=(--libero-root "$LIBERO_ROOT_ARG")
    [[ -z "$MAX_LAYERS" ]] || oft_args+=(--max_layers "$MAX_LAYERS")
    exec "$PYTHON_BIN" "${QUANTVLA_ROOT}/tools/qvla/oft_cli.py" "${oft_args[@]}"
fi

case "$ACTION" in
    plan) exit 0 ;;
    result) print_result "$OUTPUT_ROOT" ;;
    build_calib) run_build_calib ;;
    proxy) run_proxy ;;
    assign) run_assign ;;
    eval)
        if [[ "$METHOD" == "fp16" ]]; then run_eval_fp16; else run_eval_qvla; fi
        ;;
    export) run_export ;;
    run)
        if [[ "$METHOD" == "fp16" ]]; then
            run_eval_fp16
        else
            [[ -f "$CALIB_JSONL" ]] || run_build_calib
            [[ -f "$PROXY_PT" ]] || run_proxy
            [[ -f "$GATES_JSON" ]] || run_assign
            run_eval_qvla
        fi
        ;;
esac
