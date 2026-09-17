#!/usr/bin/env bash
set -euo pipefail

EVAL_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TRAINING_ROOT=${CORL_TRAINING_ROOT:-$(cd "${EVAL_ROOT}/.." && pwd)}
PYTHON_BIN=${EVAL_PYTHON:-python3}
mode=${1:-all}
if (( $# )); then shift; fi

export CORL_EVAL_ROOT=${EVAL_ROOT}
export CORL_TRAINING_ROOT=${TRAINING_ROOT}
export PYTHONPATH="${EVAL_ROOT}:${TRAINING_ROOT}:${TRAINING_ROOT}/AgentDyn/src${PYTHONPATH:+:${PYTHONPATH}}"

export BASE_MODEL_URL=${BASE_MODEL_URL:-http://127.0.0.1:8100/v1}
export PPO_DEFENDER_URL=${PPO_DEFENDER_URL:-http://127.0.0.1:8101/v1}
export NOPOP_DEFENDER_URL=${NOPOP_DEFENDER_URL:-http://127.0.0.1:8102/v1}
export COPPO_DEFENDER_URL=${COPPO_DEFENDER_URL:-http://127.0.0.1:8103/v1}
export CORL_DEFENDER_URL=${CORL_DEFENDER_URL:-http://127.0.0.1:8104/v1}
export COPPO_ATTACKER_URL=${COPPO_ATTACKER_URL:-http://127.0.0.1:8110/v1}

export BASE_MODEL_PATH=${BASE_MODEL_PATH:-${TRAINING_ROOT}/models/base-model}
export PPO_DEFENDER_PATH=${PPO_DEFENDER_PATH:-${TRAINING_ROOT}/models/ppo-defender}
export NOPOP_DEFENDER_PATH=${NOPOP_DEFENDER_PATH:-${TRAINING_ROOT}/models/nopop-defender}
export COPPO_DEFENDER_PATH=${COPPO_DEFENDER_PATH:-${TRAINING_ROOT}/checkpoints/corl/defender/global_step_430/actor/huggingface}
export COPPO_ATTACKER_PATH=${COPPO_ATTACKER_PATH:-${TRAINING_ROOT}/models/selected-coppo-attacker}
export CORL_DEFENDER_PATH=${CORL_DEFENDER_PATH:-${TRAINING_ROOT}/sft_output/defender_sft/checkpoint-360}

run_config() {
    local config=$1
    shift
    "${PYTHON_BIN}" -m evaluation.agentdyn.defender_eval \
        --config "${EVAL_ROOT}/evaluation/agentdyn/configs/${config}" "$@"
}

cd "${EVAL_ROOT}"
case "${mode}" in
    adaptive) run_config main_adaptive.yaml "$@" ;;
    clean-fixed) run_config main_clean_fixed.yaml "$@" ;;
    all)
        run_config main_clean_fixed.yaml "$@"
        run_config main_adaptive.yaml "$@"
        ;;
    *) echo "Usage: scripts/run_main.sh [all|adaptive|clean-fixed] [evaluator options]" >&2; exit 2 ;;
esac
