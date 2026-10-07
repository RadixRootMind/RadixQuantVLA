#!/bin/bash
#
# Awesome-quant-vla unified benchmark entrypoint.
#
# This script is intentionally a dispatcher inside one codebase. It does not
# call external project directories. Profiles are mapped onto the merged runtime
# scripts in this repository.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_paths.sh"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_awesome_quant_vla.sh --profile PROFILE [options]
  bash scripts/run_awesome_quant_vla.sh PROFILE [options]

Actions:
  --action run       Run benchmark. Default.
  --action plan      Print resolved configuration only.
  --action result    Print total_success_rate from merged_summary.json.
  --action build_calib|proxy|assign|eval|export
                     OpenVLA/QVLA staged actions.

Common options:
  --suite object|spatial|goal|long
  --gpus 0[,1,2,3]
  --trials N
  --init-offset N
  --port-base PORT
  --output-root PATH
  --checkpoints-root PATH
  --conda-root PATH
  --conda-env NAME

GR00T profiles:
  groot_w4a8             GR00T W4A8: DuQuant + ATM/OHB
  groot_w4a4_gptq        GR00T W4A4: merged GPTQ/SVD-Hadamard pack
  groot_w4a4_duquant     GR00T runtime DuQuant W4A4
  groot_w4a4_rtn         GR00T RTN W4A4 baseline
  groot_fp16             GR00T FP16 baseline

Pi0.5/OpenPI profiles:
  pi05_w4a4_gptq         Pi0.5 W4A4: GPTQ/SVD-Hadamard pack
  pi05_w4a8_duquant      Pi0.5 W4A8: DuQuant + Pi0.5 ATM/OHB
  pi05_w4a8_mlp_duquant  Pi0.5 W4A8: MLP only, attention kept floating point
  pi05_duquant_w4a8      Pi0.5 DuQuant W4A8 generic include regex
  pi05_w4a4_rtn          Pi0.5 RTN W4A4 baseline
  pi05_fp16              Pi0.5 FP16 baseline

OpenVLA/QVLA profiles:
  openvla_fp16           OpenVLA FP16 LIBERO eval
  openvla_qvla_w8        OpenVLA QVLA mixed-bit target avg 8
  openvla_oft_fp16       OpenVLA-OFT FP16 LIBERO eval
  openvla_oft_qvla_w8    OpenVLA-OFT QVLA mixed-bit target avg 8

UniVLA profiles:
  univla_fp16            UniVLA LIBERO FP16 eval with action decoder

StarVLA profiles:
  starvla_oft_fp16       StarVLA-OFT LIBERO FP16/BF16 eval
  starvla_gr00t_fp16     StarVLA-GR00T LIBERO FP16/BF16 eval; pass matching checkpoint
  starvla_pi_fp16        StarVLA-PI LIBERO FP16/BF16 eval; pass matching checkpoint
  starvla_fast_fp16      StarVLA-FAST LIBERO FP16/BF16 eval; pass matching checkpoint

Pack/checkpoint options:
  --groot-gptq-pack PATH  Pack for groot_w4a4_gptq.
  --pi05-gptq-pack PATH   Pack for pi05_w4a4_gptq.
  --groot-checkpoint PATH Override GR00T checkpoint.
  --openpi-root PATH      Default: ~/openpi.
  --openpi-py PATH        Default: $OPENPI_ROOT/.venv/bin/python.
  --openpi-checkpoint PATH
  --openvla-backend openvla|openvla_oft
  --openvla-checkpoint PATH
  --openvla-python PATH
  --univla-checkpoint PATH
  --univla-action-decoder PATH
  --univla-python PATH
  --starvla-checkpoint PATH
  --starvla-python PATH
  --starvla-unnorm-key KEY
  --starvla-use-bf16 0|1
  --save-video True|False
  --max-tasks N
  --calib-jsonl PATH
  --proxy-pt PATH
  --gates-json PATH
  --target-avg-bits FLOAT
  --max-samples N
  --max-layers N

Examples:
  bash scripts/run_awesome_quant_vla.sh groot_w4a8 --suite object --gpus 0 --trials 10
  bash scripts/run_awesome_quant_vla.sh groot_w4a4_gptq --suite object --gpus 0 --trials 10
  bash scripts/run_awesome_quant_vla.sh pi05_w4a4_gptq --suite object --gpus 0 --trials 10
  bash scripts/run_awesome_quant_vla.sh pi05_w4a8_duquant --suite object --gpus 0 --trials 10
  bash scripts/run_awesome_quant_vla.sh pi05_w4a4_gptq --suite object --action result
  bash scripts/run_awesome_quant_vla.sh univla_fp16 --suite spatial --gpus 0 --trials 10
  bash scripts/run_awesome_quant_vla.sh starvla_oft_fp16 --suite spatial --gpus 0 --trials 10
EOF
}

die() {
    echo "[awesome-quant-vla] ERROR: $*" >&2
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

suite_to_groot_ckpt_suffix() {
    case "$1" in
        long|libero_10) echo "libero-long-posttrain" ;;
        spatial|libero_spatial) echo "libero-spatial-posttrain" ;;
        goal|libero_goal) echo "libero-goal-posttrain" ;;
        object|libero_object) echo "libero-object-posttrain" ;;
        *) die "GR00T checkpoint suffix is not defined for suite '$1'" ;;
    esac
}

suite_to_pi05_atm_json() {
    case "$1" in
        long|libero_10) echo "10_w4a4_n16.json" ;;
        spatial|libero_spatial) echo "spatial_w4a4_n16.json" ;;
        goal|libero_goal) echo "goal_w4a4_n16.json" ;;
        object|libero_object) echo "object_w4a4_n16.json" ;;
        *) die "Pi0.5 ATM JSON is not defined for suite '$1'" ;;
    esac
}

print_result() {
    local summary="$1/merged_summary.json"
    local py
    if [[ ! -f "$summary" ]]; then
        die "summary not found: $summary"
    fi
    if command -v python3 >/dev/null 2>&1; then
        py=python3
    else
        py=python
    fi
    "$py" - "$summary" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
print(f"{100 * data['total_success_rate']:.1f} %")
PY
}

print_plan() {
    cat <<EOF
================================================================
[awesome-quant-vla] plan
  profile          : ${PROFILE}
  action           : ${ACTION}
  suite            : ${SUITE} (${TASK_SUITE})
  project root     : ${QUANTVLA_ROOT}
  conda env        : ${QUANTVLA_CONDA_ENV}
  checkpoints root : ${CHECKPOINTS_ROOT}
  output           : ${OUTPUT_ROOT}
  gpus / port      : ${GPU_LIST} / ${PORT_BASE}
  trials per task  : ${NUM_TRIALS_PER_TASK}
  ${PLAN_WAIT_LABEL:-init offset}      : ${GR00T_EVAL_INIT_OFFSET}
  wbits / abits    : ${WBITS} / ${ABITS}
  launcher         : ${LAUNCHER}
EOF
    if [[ "$MODEL_KIND" == "groot" ]]; then
        cat <<EOF
  groot checkpoint : ${CHECKPOINT}
  preset           : ${PRESET:-manual}
  llm quant/rot    : ${LLM_QUANT:-default} / ${LLM_ROT:-default}
  dit quant/rot    : ${DIT_QUANT:-default} / ${DIT_ROT:-default}
  dit attention    : ${DIT_ATTN:-default}
  groot gptq pack  : ${GR00T_GPTQ_PATH_OVERRIDE:-}
EOF
    elif [[ "$MODEL_KIND" == "pi05" ]]; then
        cat <<EOF
  openpi root      : ${OPENPI_ROOT}
  openpi python    : ${OPENPI_PY}
  openpi checkpoint: ${OPENPI_CHECKPOINT}
  openpi config    : ${OPENPI_CONFIG}
  method           : ${METHOD}
  pi05 gptq pack   : ${OPENPI_GPTQ_PATH:-}
  pi05 atm json    : ${GR00T_ATM_ALPHA_PATH:-}
  pi05 atm / ohb   : ${GR00T_ATM_ENABLE:-0} / ${GR00T_OHB_ENABLE:-0}
  duquant include : ${OPENPI_DUQUANT_INCLUDE:-}
  duquant exclude : ${OPENPI_DUQUANT_EXCLUDE:-}
EOF
    elif [[ "$MODEL_KIND" == "openvla" ]]; then
        cat <<EOF
  openvla backend  : ${OPENVLA_BACKEND}
  openvla python   : ${OPENVLA_PYTHON}
  openvla checkpoint: ${OPENVLA_CHECKPOINT}
  method           : ${METHOD}
  calib jsonl      : ${CALIB_JSONL:-}
  proxy pt         : ${PROXY_PT:-}
  gates json       : ${GATES_JSON:-}
  target avg bits  : ${TARGET_AVG_BITS:-}
EOF
    elif [[ "$MODEL_KIND" == "starvla" ]]; then
        cat <<EOF
  starvla python   : ${STARVLA_PYTHON}
  starvla checkpoint: ${STARVLA_CHECKPOINT}
  method           : ${METHOD}
  use bf16         : ${STARVLA_USE_BF16}
  save video       : ${SAVE_VIDEO}
  max tasks        : ${MAX_TASKS}
  unnorm key       : ${STARVLA_UNNORM_KEY:-auto}
EOF
    else
        cat <<EOF
  univla python    : ${UNIVLA_PYTHON}
  univla checkpoint: ${UNIVLA_CHECKPOINT}
  action decoder   : ${UNIVLA_ACTION_DECODER}
  method           : ${METHOD}
EOF
    fi
    echo "================================================================"
}

PROFILE="${PROFILE:-}"
ACTION="${ACTION:-run}"
SUITE="${SUITE:-object}"
GPU_LIST="${GPU_LIST:-0}"
PORT_BASE="${PORT_BASE:-8000}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-10}"
GR00T_EVAL_INIT_OFFSET="${GR00T_EVAL_INIT_OFFSET:-10}"
CHECKPOINTS_ROOT="${CHECKPOINTS_ROOT:-${HOME}/ckpts}"
QUANTVLA_CONDA_ENV="${QUANTVLA_CONDA_ENV:-$(quantvla_default_conda_env)}"
OUTPUT_ROOT_ARG="${OUTPUT_ROOT:-}"
GROOT_GPTQ_PACK="${GROOT_GPTQ_PACK:-}"
PI05_GPTQ_PACK="${PI05_GPTQ_PACK:-}"
GROOT_CHECKPOINT="${GROOT_CHECKPOINT:-}"
OPENPI_ROOT_ARG="${OPENPI_ROOT:-}"
OPENPI_PY_ARG="${OPENPI_PY:-}"
OPENPI_CHECKPOINT_ARG="${OPENPI_CHECKPOINT:-}"
OPENPI_CONFIG="${OPENPI_CONFIG:-pi05_libero}"
OPENPI_DEVICE="${OPENPI_DEVICE:-cuda}"
OPENVLA_BACKEND_ARG="${OPENVLA_BACKEND:-}"
OPENVLA_CHECKPOINT_ARG="${OPENVLA_CHECKPOINT:-}"
OPENVLA_PYTHON_ARG="${OPENVLA_PYTHON:-}"
UNIVLA_CHECKPOINT_ARG="${UNIVLA_CHECKPOINT:-${UNIVLA_CKPT:-}}"
UNIVLA_ACTION_DECODER_ARG="${UNIVLA_ACTION_DECODER:-}"
UNIVLA_PYTHON_ARG="${UNIVLA_PYTHON:-}"
STARVLA_CHECKPOINT_ARG="${STARVLA_CHECKPOINT:-}"
STARVLA_PYTHON_ARG="${STARVLA_PYTHON:-}"
STARVLA_UNNORM_KEY="${STARVLA_UNNORM_KEY:-}"
STARVLA_USE_BF16="${STARVLA_USE_BF16:-1}"
SAVE_VIDEO="${SAVE_VIDEO:-False}"
MAX_TASKS="${MAX_TASKS:--1}"
CALIB_JSONL_ARG="${CALIB_JSONL:-}"
PROXY_PT_ARG="${PROXY_PT:-}"
GATES_JSON_ARG="${GATES_JSON:-}"
TARGET_AVG_BITS="${TARGET_AVG_BITS:-8.0}"
BITS="${BITS:-0,2,4,8,16}"
PROXY_BITS="${PROXY_BITS:-0,2,4,8}"
MAX_SAMPLES="${MAX_SAMPLES:-32}"
MAX_LAYERS="${MAX_LAYERS:-}"
DENOISING_STEPS="${DENOISING_STEPS:-8}"
REPLAN_STEPS="${REPLAN_STEPS:-5}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile) PROFILE="${2:?missing --profile value}"; shift 2 ;;
        --action|--mode) ACTION="${2:?missing --action value}"; shift 2 ;;
        --suite) SUITE="${2:?missing --suite value}"; shift 2 ;;
        --gpus|--gpu-list) GPU_LIST="${2:?missing --gpus value}"; shift 2 ;;
        --trials) NUM_TRIALS_PER_TASK="${2:?missing --trials value}"; shift 2 ;;
        --init-offset) GR00T_EVAL_INIT_OFFSET="${2:?missing --init-offset value}"; shift 2 ;;
        --port-base) PORT_BASE="${2:?missing --port-base value}"; shift 2 ;;
        --output-root) OUTPUT_ROOT_ARG="${2:?missing --output-root value}"; shift 2 ;;
        --checkpoints-root) CHECKPOINTS_ROOT="${2:?missing --checkpoints-root value}"; shift 2 ;;
        --conda-root) CONDA_ROOT="${2:?missing --conda-root value}"; export CONDA_ROOT; shift 2 ;;
        --conda-env) QUANTVLA_CONDA_ENV="${2:?missing --conda-env value}"; shift 2 ;;
        --groot-gptq-pack) GROOT_GPTQ_PACK="${2:?missing --groot-gptq-pack value}"; shift 2 ;;
        --pi05-gptq-pack) PI05_GPTQ_PACK="${2:?missing --pi05-gptq-pack value}"; shift 2 ;;
        --groot-checkpoint|--checkpoint) GROOT_CHECKPOINT="${2:?missing checkpoint value}"; shift 2 ;;
        --openpi-root) OPENPI_ROOT_ARG="${2:?missing --openpi-root value}"; shift 2 ;;
        --openpi-py) OPENPI_PY_ARG="${2:?missing --openpi-py value}"; shift 2 ;;
        --openpi-checkpoint) OPENPI_CHECKPOINT_ARG="${2:?missing --openpi-checkpoint value}"; shift 2 ;;
        --openpi-config) OPENPI_CONFIG="${2:?missing --openpi-config value}"; shift 2 ;;
        --openpi-device) OPENPI_DEVICE="${2:?missing --openpi-device value}"; shift 2 ;;
        --openvla-backend) OPENVLA_BACKEND_ARG="${2:?missing --openvla-backend value}"; shift 2 ;;
        --openvla-checkpoint) OPENVLA_CHECKPOINT_ARG="${2:?missing --openvla-checkpoint value}"; shift 2 ;;
        --openvla-python) OPENVLA_PYTHON_ARG="${2:?missing --openvla-python value}"; shift 2 ;;
        --univla-checkpoint) UNIVLA_CHECKPOINT_ARG="${2:?missing --univla-checkpoint value}"; shift 2 ;;
        --univla-action-decoder|--action-decoder) UNIVLA_ACTION_DECODER_ARG="${2:?missing --univla-action-decoder value}"; shift 2 ;;
        --univla-python) UNIVLA_PYTHON_ARG="${2:?missing --univla-python value}"; shift 2 ;;
        --starvla-checkpoint) STARVLA_CHECKPOINT_ARG="${2:?missing --starvla-checkpoint value}"; shift 2 ;;
        --starvla-python) STARVLA_PYTHON_ARG="${2:?missing --starvla-python value}"; shift 2 ;;
        --starvla-unnorm-key) STARVLA_UNNORM_KEY="${2:?missing --starvla-unnorm-key value}"; shift 2 ;;
        --starvla-use-bf16) STARVLA_USE_BF16="${2:?missing --starvla-use-bf16 value}"; shift 2 ;;
        --save-video) SAVE_VIDEO="${2:?missing --save-video value}"; shift 2 ;;
        --max-tasks) MAX_TASKS="${2:?missing --max-tasks value}"; shift 2 ;;
        --calib-jsonl) CALIB_JSONL_ARG="${2:?missing --calib-jsonl value}"; shift 2 ;;
        --proxy-pt) PROXY_PT_ARG="${2:?missing --proxy-pt value}"; shift 2 ;;
        --gates-json) GATES_JSON_ARG="${2:?missing --gates-json value}"; shift 2 ;;
        --target-avg-bits) TARGET_AVG_BITS="${2:?missing --target-avg-bits value}"; shift 2 ;;
        --bits) BITS="${2:?missing --bits value}"; shift 2 ;;
        --proxy-bits) PROXY_BITS="${2:?missing --proxy-bits value}"; shift 2 ;;
        --max-samples) MAX_SAMPLES="${2:?missing --max-samples value}"; shift 2 ;;
        --max-layers) MAX_LAYERS="${2:?missing --max-layers value}"; shift 2 ;;
        --denoising-steps) DENOISING_STEPS="${2:?missing --denoising-steps value}"; shift 2 ;;
        --replan-steps) REPLAN_STEPS="${2:?missing --replan-steps value}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        --) shift; break ;;
        -*)
            die "unknown option '$1'"
            ;;
        *)
            if [[ -z "$PROFILE" ]]; then
                PROFILE="$1"
                shift
            else
                die "unexpected positional argument '$1'"
            fi
            ;;
    esac
done

[[ -n "$PROFILE" ]] || { usage; exit 2; }

case "$ACTION" in
    run|plan|result|build_calib|proxy|assign|eval|export) ;;
    *) die "unknown action '$ACTION'" ;;
esac

TASK_SUITE="$(suite_to_task_suite "$SUITE")"
SUITE_SHORT="$(suite_to_short "$TASK_SUITE")"
SUITE="$SUITE_SHORT"
AWESOME_QVLA_ROOT="${AWESOME_QVLA_ROOT:-${QUANTVLA_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}}"
QUANTVLA_ROOT="${QUANTVLA_ROOT:-${AWESOME_QVLA_ROOT}}"
OUTPUT_ROOT="${OUTPUT_ROOT_ARG:-${QUANTVLA_ROOT}/results/awesome_quant_vla/${PROFILE}_${SUITE}}"
OPENPI_ROOT="${OPENPI_ROOT_ARG:-${HOME}/openpi}"
OPENPI_PY="${OPENPI_PY_ARG:-${OPENPI_ROOT}/.venv/bin/python}"
OPENPI_CHECKPOINT="${OPENPI_CHECKPOINT_ARG:-${CHECKPOINTS_ROOT}/pi05_libero_pytorch}"

export AWESOME_QVLA_ROOT QUANTVLA_ROOT CHECKPOINTS_ROOT QUANTVLA_CONDA_ENV
export PROFILE ACTION SUITE GPU_LIST PORT_BASE NUM_TRIALS_PER_TASK OUTPUT_ROOT
export GR00T_EVAL_INIT_OFFSET
export DENOISING_STEPS

MODEL_KIND=""
LAUNCHER=""
PRESET=""
WBITS="${WBITS:-4}"
ABITS="${ABITS:-8}"

case "$PROFILE" in
    groot_w4a8|groot_quantvla_w4a8|quantvla_groot_w4a8)
        MODEL_KIND="groot"
        LAUNCHER="${SCRIPT_DIR}/run_groot_benchmark.sh"
        PRESET="quantvla_w4a8"
        WBITS=4
        ABITS=8
        ;;
    groot_w4a4_gptq|groot_w4a4)
        MODEL_KIND="groot"
        LAUNCHER="${SCRIPT_DIR}/run_groot_benchmark.sh"
        PRESET=""
        LLM_QUANT="gptq"
        DIT_QUANT="gptq"
        LLM_ROT="svd_hadamard"
        DIT_ROT="svd_hadamard"
        DIT_ATTN="${DIT_ATTN:-1}"
        DIT_PERSTEP="${DIT_PERSTEP:-1}"
        WBITS=4
        ABITS=4
        if [[ -n "${GROOT_GPTQ_PACK}" ]]; then
            GR00T_GPTQ_PATH_OVERRIDE="${GROOT_GPTQ_PACK}"
        elif [[ -f "${QUANTVLA_ROOT}/results/packs/gr00t_${SUITE}/quantized.pt" ]]; then
            GR00T_GPTQ_PATH_OVERRIDE="${QUANTVLA_ROOT}/results/packs/gr00t_${SUITE}/quantized.pt"
        else
            GR00T_GPTQ_PATH_OVERRIDE="${QUANTVLA_ROOT}/results/packs/${SUITE}_MERGED/quantized.pt"
        fi
        GR00T_GPTQ_MISSING="${GR00T_GPTQ_MISSING:-fallback}"
        ;;
    groot_w4a4_duquant|groot_duquant_w4a4)
        MODEL_KIND="groot"
        LAUNCHER="${SCRIPT_DIR}/run_groot_benchmark.sh"
        PRESET=""
        LLM_QUANT="duquant"
        DIT_QUANT="duquant"
        LLM_ROT="${LLM_ROT:-svd_hadamard}"
        DIT_ROT="${DIT_ROT:-svd_hadamard}"
        DIT_ATTN="${DIT_ATTN:-1}"
        WBITS=4
        ABITS=4
        ;;
    groot_w4a4_rtn|groot_rtn_w4a4)
        MODEL_KIND="groot"
        LAUNCHER="${SCRIPT_DIR}/run_groot_benchmark.sh"
        PRESET=""
        LLM_QUANT="rtn"
        DIT_QUANT="rtn"
        DIT_ATTN="${DIT_ATTN:-1}"
        WBITS=4
        ABITS=4
        ;;
    groot_fp16|groot_all16|groot_none)
        MODEL_KIND="groot"
        LAUNCHER="${SCRIPT_DIR}/run_groot_benchmark.sh"
        PRESET="fp16"
        WBITS=16
        ABITS=16
        ;;
    pi05_w4a4_gptq|pi05_w4a4)
        MODEL_KIND="pi05"
        LAUNCHER="${SCRIPT_DIR}/run_pi05_libero_benchmark.sh"
        METHOD="gptq"
        WBITS=4
        ABITS=4
        OPENPI_GPTQ_PATH="${PI05_GPTQ_PACK:-${QUANTVLA_ROOT}/results/packs/pi05_${SUITE}/quantized.pt}"
        PI05_GPTQ_INCLUDE_DEFAULT='.*paligemma_with_expert\.(paligemma\.model\.language_model|gemma_expert\.model)\.layers\.[0-9]+\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj).*'
        PI05_GPTQ_EXCLUDE_DEFAULT='(?:^|\.)(vision|vision_tower|vision_model|embeddings|embed_tokens|norm|ln|layernorm|lm_head|pos_embed|timestep|state_encoder|action_encoder|action_decoder|state_proj|action_in_proj|action_out_proj|time_mlp|time_mlp_in|time_mlp_out)(?:\.|$)'
        OPENPI_GPTQ_INCLUDE="${OPENPI_GPTQ_INCLUDE:-$PI05_GPTQ_INCLUDE_DEFAULT}"
        OPENPI_GPTQ_EXCLUDE="${OPENPI_GPTQ_EXCLUDE:-$PI05_GPTQ_EXCLUDE_DEFAULT}"
        GR00T_GPTQ_MISSING="${GR00T_GPTQ_MISSING:-error}"
        ;;
    pi05_w4a8_duquant|pi05_quantvla_w4a8|quantvla_pi05_w4a8)
        MODEL_KIND="pi05"
        LAUNCHER="${SCRIPT_DIR}/run_pi05_libero_benchmark.sh"
        METHOD="duquant"
        WBITS=4
        ABITS=8
        PI05_QVLA_INCLUDE_DEFAULT='(.*paligemma_with_expert\.paligemma\.model\.language_model\.layers\.[0-9]+\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)|.*paligemma_with_expert\.gemma_expert\.model\.layers\.[0-9]+\.mlp\.(gate_proj|up_proj|down_proj))$'
        PI05_QVLA_EXCLUDE_DEFAULT='(?:^|\.)(vision|norm|ln|layernorm|embed|lm_head|pos_embed|timestep|state_encoder|action_encoder|action_decoder|state_proj|action_in_proj|action_out_proj|time_mlp)(?:\.|$)'
        OPENPI_DUQUANT_INCLUDE="${OPENPI_DUQUANT_INCLUDE:-$PI05_QVLA_INCLUDE_DEFAULT}"
        OPENPI_DUQUANT_EXCLUDE="${OPENPI_DUQUANT_EXCLUDE:-$PI05_QVLA_EXCLUDE_DEFAULT}"
        GR00T_ATM_SCOPE="pi05"
        GR00T_OHB_SCOPE="pi05"
        GR00T_ATM_ENABLE="${GR00T_ATM_ENABLE:-1}"
        GR00T_OHB_ENABLE="${GR00T_OHB_ENABLE:-1}"
        GR00T_OHB_FALLBACK="${GR00T_OHB_FALLBACK:-1.0}"
        GR00T_ATM_ALPHA_PATH="${GR00T_ATM_ALPHA_PATH:-${QUANTVLA_ROOT}/atm_alpha_beta_pi05/$(suite_to_pi05_atm_json "$SUITE")}"
        ;;
    pi05_w4a8_mlp_duquant|pi05_w4a8_mlp)
        MODEL_KIND="pi05"
        LAUNCHER="${SCRIPT_DIR}/run_pi05_libero_benchmark.sh"
        METHOD="duquant"
        WBITS=4
        ABITS=8
        # Conservative first-pass scope: MLP projections only. Attention
        # Q/K/V/O remain floating point until matching calibration is made.
        PI05_MLP_W4A8_INCLUDE='.*paligemma_with_expert\.(paligemma\.model\.language_model|gemma_expert\.model)\.layers\.[0-9]+\.mlp\.(gate_proj|up_proj|down_proj)$'
        OPENPI_DUQUANT_INCLUDE="${OPENPI_DUQUANT_INCLUDE:-$PI05_MLP_W4A8_INCLUDE}"
        PI05_MLP_W4A8_EXCLUDE='(?:^|\.)(vision|norm|ln|layernorm|embed|lm_head|pos_embed|timestep|state_encoder|action_encoder|action_decoder|state_proj|action_in_proj|action_out_proj|time_mlp)(?:\.|$)'
        OPENPI_DUQUANT_EXCLUDE="${OPENPI_DUQUANT_EXCLUDE:-$PI05_MLP_W4A8_EXCLUDE}"
        # Keep the profile neutral by default. Supplying a Pi0.5 calibration
        # JSON explicitly enables the same profile for ATM/OHB ablations.
        if [[ -n "${GR00T_ATM_ALPHA_PATH:-}" ]]; then
            GR00T_ATM_SCOPE="pi05"
            GR00T_OHB_SCOPE="pi05"
            GR00T_ATM_ENABLE="${GR00T_ATM_ENABLE:-1}"
            GR00T_OHB_ENABLE="${GR00T_OHB_ENABLE:-1}"
            GR00T_OHB_FALLBACK="${GR00T_OHB_FALLBACK:-1.0}"
        else
            GR00T_ATM_SCOPE=""
            GR00T_OHB_SCOPE=""
            GR00T_ATM_ENABLE=0
            GR00T_OHB_ENABLE=0
        fi
        export GR00T_DUQUANT_PERM_SCORE=weight
        ;;
    pi05_duquant_w4a8)
        MODEL_KIND="pi05"
        LAUNCHER="${SCRIPT_DIR}/run_pi05_libero_benchmark.sh"
        METHOD="duquant"
        WBITS=4
        ABITS=8
        ;;
    pi05_w4a4_rtn|pi05_rtn_w4a4)
        MODEL_KIND="pi05"
        LAUNCHER="${SCRIPT_DIR}/run_pi05_libero_benchmark.sh"
        METHOD="rtn"
        WBITS=4
        ABITS=4
        ;;
    pi05_fp16|pi05_all16|pi05_none)
        MODEL_KIND="pi05"
        LAUNCHER="${SCRIPT_DIR}/run_pi05_libero_benchmark.sh"
        METHOD="fp16"
        WBITS=16
        ABITS=16
        ;;
    openvla_fp16)
        MODEL_KIND="openvla"
        LAUNCHER="${SCRIPT_DIR}/run_openvla_qvla.sh"
        METHOD="fp16"
        OPENVLA_BACKEND="${OPENVLA_BACKEND_ARG:-openvla}"
        WBITS=16
        ABITS=16
        ;;
    openvla_qvla_w8|openvla_qvla)
        MODEL_KIND="openvla"
        LAUNCHER="${SCRIPT_DIR}/run_openvla_qvla.sh"
        METHOD="qvla"
        OPENVLA_BACKEND="${OPENVLA_BACKEND_ARG:-openvla}"
        TARGET_AVG_BITS="${TARGET_AVG_BITS:-8.0}"
        WBITS=8
        ABITS=16
        ;;
    openvla_oft_fp16|openvla-oft_fp16)
        MODEL_KIND="openvla"
        LAUNCHER="${SCRIPT_DIR}/run_openvla_qvla.sh"
        METHOD="fp16"
        OPENVLA_BACKEND="${OPENVLA_BACKEND_ARG:-openvla_oft}"
        WBITS=16
        ABITS=16
        ;;
    openvla_oft_qvla_w8|openvla-oft_qvla_w8|openvla_oft_qvla)
        MODEL_KIND="openvla"
        LAUNCHER="${SCRIPT_DIR}/run_openvla_qvla.sh"
        METHOD="qvla"
        OPENVLA_BACKEND="${OPENVLA_BACKEND_ARG:-openvla_oft}"
        TARGET_AVG_BITS="${TARGET_AVG_BITS:-8.0}"
        WBITS=8
        ABITS=16
        ;;
    starvla_oft_fp16|starvla_fp16)
        MODEL_KIND="starvla"
        LAUNCHER="${SCRIPT_DIR}/run_starvla_libero.sh"
        METHOD="fp16"
        WBITS=16
        ABITS=16
        ;;
    starvla_gr00t_fp16|starvla_pi_fp16|starvla_fast_fp16)
        MODEL_KIND="starvla"
        LAUNCHER="${SCRIPT_DIR}/run_starvla_libero.sh"
        METHOD="fp16"
        WBITS=16
        ABITS=16
        ;;
    univla_fp16|univla_libero_fp16)
        MODEL_KIND="univla"
        LAUNCHER="${SCRIPT_DIR}/run_univla_libero.sh"
        METHOD="fp16"
        WBITS=16
        ABITS=16
        ;;
    *)
        die "unknown profile '$PROFILE'"
        ;;
esac

if [[ "$MODEL_KIND" == "groot" ]]; then
    CKPT_SUFFIX="$(suite_to_groot_ckpt_suffix "$SUITE")"
    CHECKPOINT="${GROOT_CHECKPOINT:-${CHECKPOINTS_ROOT}/gr00t-n1.5-${CKPT_SUFFIX}}"
    export CHECKPOINT PRESET WBITS ABITS
    export LLM_QUANT="${LLM_QUANT:-}" DIT_QUANT="${DIT_QUANT:-}"
    export LLM_ROT="${LLM_ROT:-}" DIT_ROT="${DIT_ROT:-}"
    export DIT_ATTN="${DIT_ATTN:-}" DIT_PERSTEP="${DIT_PERSTEP:-}"
    export GR00T_GPTQ_PATH_OVERRIDE="${GR00T_GPTQ_PATH_OVERRIDE:-}"
    export GR00T_GPTQ_MISSING="${GR00T_GPTQ_MISSING:-fallback}"
elif [[ "$MODEL_KIND" == "pi05" ]]; then
    export METHOD WBITS ABITS
    export OPENPI_ROOT OPENPI_PY OPENPI_CHECKPOINT OPENPI_CONFIG OPENPI_DEVICE REPLAN_STEPS
    export OPENPI_GPTQ_PATH="${OPENPI_GPTQ_PATH:-}"
    export OPENPI_GPTQ_INCLUDE="${OPENPI_GPTQ_INCLUDE:-}"
    export OPENPI_GPTQ_EXCLUDE="${OPENPI_GPTQ_EXCLUDE:-}"
    export OPENPI_DUQUANT_INCLUDE="${OPENPI_DUQUANT_INCLUDE:-}"
    export OPENPI_DUQUANT_EXCLUDE="${OPENPI_DUQUANT_EXCLUDE:-}"
    export GR00T_ATM_SCOPE="${GR00T_ATM_SCOPE:-}" GR00T_OHB_SCOPE="${GR00T_OHB_SCOPE:-}"
    export GR00T_ATM_ENABLE="${GR00T_ATM_ENABLE:-}" GR00T_OHB_ENABLE="${GR00T_OHB_ENABLE:-}"
    export GR00T_OHB_FALLBACK="${GR00T_OHB_FALLBACK:-}"
    export GR00T_ATM_ALPHA_PATH="${GR00T_ATM_ALPHA_PATH:-}"
elif [[ "$MODEL_KIND" == "openvla" ]]; then
    OPENVLA_PYTHON="${OPENVLA_PYTHON_ARG:-python}"
    OPENVLA_SUITE_SUFFIX="$SUITE"
    if [[ "$SUITE" == "long" ]]; then
        OPENVLA_SUITE_SUFFIX="10"
    fi
    if [[ -n "$OPENVLA_CHECKPOINT_ARG" ]]; then
        OPENVLA_CHECKPOINT="$OPENVLA_CHECKPOINT_ARG"
    elif [[ "$OPENVLA_BACKEND" == "openvla_oft" ]]; then
        OPENVLA_CHECKPOINT="${CHECKPOINTS_ROOT}/openvla-7b-oft-finetuned-libero-${OPENVLA_SUITE_SUFFIX}"
    else
        OPENVLA_CHECKPOINT="${CHECKPOINTS_ROOT}/openvla-7b-finetuned-libero-${OPENVLA_SUITE_SUFFIX}"
    fi
    CALIB_JSONL="${CALIB_JSONL_ARG:-${OUTPUT_ROOT}/calib/calib.jsonl}"
    PROXY_PT="${PROXY_PT_ARG:-${OUTPUT_ROOT}/proxy/proxy.pt}"
    GATES_JSON="${GATES_JSON_ARG:-${OUTPUT_ROOT}/gates/greedy_bits.json}"
    export METHOD WBITS ABITS OPENVLA_BACKEND OPENVLA_PYTHON OPENVLA_CHECKPOINT
    export CALIB_JSONL PROXY_PT GATES_JSON TARGET_AVG_BITS BITS PROXY_BITS MAX_SAMPLES MAX_LAYERS
elif [[ "$MODEL_KIND" == "starvla" ]]; then
    STARVLA_PYTHON="${STARVLA_PYTHON_ARG:-python}"
    STARVLA_CHECKPOINT="${STARVLA_CHECKPOINT_ARG:-${CHECKPOINTS_ROOT}/starvla/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt}"
    export METHOD WBITS ABITS STARVLA_PYTHON STARVLA_CHECKPOINT STARVLA_UNNORM_KEY STARVLA_USE_BF16 SAVE_VIDEO MAX_TASKS
else
    UNIVLA_PYTHON="${UNIVLA_PYTHON_ARG:-${OPENVLA_PYTHON_ARG:-python}}"
    UNIVLA_SUITE_SUFFIX="$SUITE"
    if [[ "$SUITE" == "long" ]]; then
        UNIVLA_SUITE_SUFFIX="10"
    fi
    UNIVLA_REPO_ROOT="${UNIVLA_REPO_ROOT:-${CHECKPOINTS_ROOT}/univla-7b-224-sft-libero}"
    UNIVLA_CHECKPOINT="${UNIVLA_CHECKPOINT_ARG:-${UNIVLA_REPO_ROOT}/univla-libero-${UNIVLA_SUITE_SUFFIX}}"
    UNIVLA_ACTION_DECODER="${UNIVLA_ACTION_DECODER_ARG:-${UNIVLA_CHECKPOINT}/action_decoder.pt}"
    export METHOD WBITS ABITS UNIVLA_PYTHON UNIVLA_CHECKPOINT UNIVLA_ACTION_DECODER
    export OPENVLA_ATTN_IMPL="${OPENVLA_ATTN_IMPL:-eager}"
    export UNIVLA_ATTN_IMPL="${UNIVLA_ATTN_IMPL:-${OPENVLA_ATTN_IMPL}}"
fi

if [[ "$MODEL_KIND" == "openvla" || "$MODEL_KIND" == "univla" || "$MODEL_KIND" == "starvla" ]]; then
    PLAN_WAIT_LABEL="steps wait"
else
    PLAN_WAIT_LABEL="init offset"
fi

if [[ "$ACTION" != "result" ]]; then
    print_plan
fi

case "$ACTION" in
    plan)
        exit 0
        ;;
    result)
        print_result "$OUTPUT_ROOT"
        exit 0
        ;;
    run)
        bash "$LAUNCHER"
        ;;
    build_calib|proxy|assign|eval|export)
        [[ "$MODEL_KIND" == "openvla" ]] || die "action '$ACTION' is only supported for OpenVLA/QVLA profiles"
        bash "$LAUNCHER"
        ;;
esac
