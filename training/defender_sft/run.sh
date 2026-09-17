#!/usr/bin/env bash
# Stage 3: teacher repair SFT initialized from the selected Co-PPO defender.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec bash "${SCRIPT_DIR}/../common/submit_sft.sh" defender "$@"
