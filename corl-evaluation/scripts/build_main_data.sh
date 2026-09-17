#!/usr/bin/env bash
set -euo pipefail

EVAL_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TRAINING_ROOT=${CORL_TRAINING_ROOT:-$(cd "${EVAL_ROOT}/.." && pwd)}
PYTHON_BIN=${EVAL_PYTHON:-python3}
export CORL_TRAINING_ROOT=${TRAINING_ROOT}
export PYTHONPATH="${EVAL_ROOT}:${TRAINING_ROOT}:${TRAINING_ROOT}/AgentDyn/src${PYTHONPATH:+:${PYTHONPATH}}"

args=(
    --parquet "${TRAINING_ROOT}/training_data/agentdyn_official_overlap_eval.parquet"
    --panel-dir "${EVAL_ROOT}/evaluation/agentdyn/panels"
)
if [[ -s "${TRAINING_ROOT}/training_data/agentdyn_train.parquet" ]]; then
    args+=(--train-parquet "${TRAINING_ROOT}/training_data/agentdyn_train.parquet")
else
    args+=(--train-parquet "")
fi

cd "${EVAL_ROOT}"
exec "${PYTHON_BIN}" -m evaluation.agentdyn.build_official_overlap "${args[@]}" "$@"
