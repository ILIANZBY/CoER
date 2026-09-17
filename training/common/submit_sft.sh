#!/usr/bin/env bash
# Shared submission path for the two supervised stages.
set -euo pipefail
ROLE=${1:?usage: submit_sft.sh attacker|defender}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR=${CORL_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}
RAY_ADDRESS=${RAY_ADDRESS:-http://127.0.0.1:8265}
RAY_BIN=${RAY_BIN:-ray}
RAY_NUM_NODES=${RAY_NUM_NODES:-2}
RAY_GPUS_PER_NODE=${RAY_GPUS_PER_NODE:-8}
RAY_CPUS_PER_NODE=${RAY_CPUS_PER_NODE:-8}

case "$ROLE" in
    attacker)
        MODEL_PATH=${ATTACKER_MODEL_PATH:-${PROJECT_DIR}/models/base-model}
        DATA_PATH=${ATTACKER_DATA_PATH:-${PROJECT_DIR}/sft_data/attacker_sft.jsonl}
        OUTPUT_DIR=${ATTACKER_OUTPUT_DIR:-${PROJECT_DIR}/sft_output/attacker_sft}
        MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-57344}
        ;;
    defender)
        MODEL_PATH=${DEFENDER_MODEL_PATH:-${PROJECT_DIR}/checkpoints/corl/defender/global_step_430/actor/huggingface}
        DATA_PATH=${DEFENDER_DATA_PATH:-${PROJECT_DIR}/sft_data/defender_sft.jsonl}
        OUTPUT_DIR=${DEFENDER_OUTPUT_DIR:-${PROJECT_DIR}/sft_output/defender_sft}
        MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-16384}
        if (( RAY_NUM_NODES * RAY_GPUS_PER_NODE != 16 )); then
            echo "ERROR: paper Defender SFT uses 16 GPUs, batch 16 and 720 updates" >&2
            exit 2
        fi
        ;;
    *) echo "Unknown SFT role: $ROLE" >&2; exit 2 ;;
esac
if [[ "${SFT_DRY_RUN:-0}" == "1" ]]; then
    printf '%s\n' "SFT preflight: $ROLE, all assistant turns, no truncation, fresh start" \
        "Resources: ${RAY_NUM_NODES} x ${RAY_GPUS_PER_NODE} GPUs; max sequence ${MAX_SEQ_LENGTH}" \
        "Model: $MODEL_PATH" "Data: $DATA_PATH" "Output: $OUTPUT_DIR"
    exit 0
fi
test -r "$MODEL_PATH/config.json"
test -r "$DATA_PATH"
if [[ -d "$OUTPUT_DIR" && -n "$(ls -A "$OUTPUT_DIR")" ]]; then
    echo "ERROR: SFT output must be empty: $OUTPUT_DIR" >&2
    exit 2
fi
command -v "$RAY_BIN" >/dev/null
RAY_BUNDLE_DIR=$(mktemp -d "${TMPDIR:-/tmp}/corl-sft.XXXXXX")
trap 'rm -rf -- "${RAY_BUNDLE_DIR}"' EXIT
for source in sft_train.py sft_data.py sft_entrypoint.sh ray_multinode_torchrun.py ds_config_zero3.json; do
    cp "${SCRIPT_DIR}/${source}" "${RAY_BUNDLE_DIR}/"
done
export NO_PROXY=${RAY_NO_PROXY:-*}
export no_proxy=$NO_PROXY
RAY_AUTH_ARGS=()
if [[ -n "${RAY_HEADERS_JSON:-}" ]]; then RAY_AUTH_ARGS=(--headers "$RAY_HEADERS_JSON"); fi
"$RAY_BIN" job submit --no-wait --address "$RAY_ADDRESS" \
    --submission-id "${RAY_SUBMISSION_ID:-corl-${ROLE}-sft-$(date +%Y%m%d-%H%M%S)}" \
    --working-dir "$RAY_BUNDLE_DIR" --entrypoint-num-cpus 1 --entrypoint-num-gpus 0 \
    "${RAY_AUTH_ARGS[@]}" -- bash sft_entrypoint.sh \
    "$ROLE" "$MAX_SEQ_LENGTH" "$RAY_NUM_NODES" "$RAY_GPUS_PER_NODE" "$RAY_CPUS_PER_NODE" \
    "$MODEL_PATH" "$DATA_PATH" "$OUTPUT_DIR"
