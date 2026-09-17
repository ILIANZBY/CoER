#!/usr/bin/env bash
# Canonical entry point for the paper's three-stage CoRL pipeline.
# Attacker SFT -> bilateral Co-PPO (with populations) -> Defender SFT.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
stage=${1:-}
if [[ -z "${stage}" ]]; then
    echo "Usage: training/run.sh <attacker-sft|coppo|defender-sft>" >&2
    exit 2
fi
shift

case "${stage}" in
    attacker-sft)
        exec bash "${SCRIPT_DIR}/attacker_sft/run.sh" "$@"
        ;;
    coppo|corl)
        exec bash "${SCRIPT_DIR}/corl/run.sh" "$@"
        ;;
    defender-sft)
        exec bash "${SCRIPT_DIR}/defender_sft/run.sh" "$@"
        ;;
    *)
        echo "Unknown stage: ${stage}" >&2
        exit 2
        ;;
esac
