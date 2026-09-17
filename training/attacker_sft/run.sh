#!/usr/bin/env bash
# Stage 1: successful-trajectory, all-attacker-turn SFT.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec bash "${SCRIPT_DIR}/../common/submit_sft.sh" attacker "$@"
