#!/usr/bin/env bash
set -euo pipefail

EVAL_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TRAINING_ROOT=${CORL_TRAINING_ROOT:-$(cd "${EVAL_ROOT}/.." && pwd)}
PYTHON_BIN=${EVAL_PYTHON:-python3}
export CORL_TRAINING_ROOT=${TRAINING_ROOT}
export PYTHONPATH="${EVAL_ROOT}:${TRAINING_ROOT}:${TRAINING_ROOT}/AgentDyn/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${EVAL_ROOT}"
exec "${PYTHON_BIN}" scripts/run_ood_injecagent.py "$@"
