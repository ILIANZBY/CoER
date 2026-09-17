#!/usr/bin/env bash
set -euo pipefail
ROLE=${1:?role required}
MAX_SEQ_LENGTH=${2:?sequence length required}
NUM_NODES=${3:?node count required}
GPUS_PER_NODE=${4:?GPU count required}
CPUS_PER_NODE=${5:?CPU count required}
MODEL_PATH=${6:?model required}
DATA_PATH=${7:?dataset required}
OUTPUT_DIR=${8:?output required}
BUNDLE_DIR="$(cd "$(dirname "$0")" && pwd)"
case "$ROLE" in
    attacker)
        ROLE_ARGS=(--learning_rate 1e-5 --gradient_accumulation_steps 2 --warmup_ratio 0.05
            --expected_examples 3995 --save_strategy epoch)
        ;;
    defender)
        if (( NUM_NODES * GPUS_PER_NODE != 16 )); then
            echo "ERROR: Defender SFT requires global batch 16" >&2; exit 2
        fi
        # Do not replace this with one epoch: update 360 uses the 720-step schedule.
        ROLE_ARGS=(--learning_rate 5e-6 --gradient_accumulation_steps 1 --warmup_ratio 0.03
            --expected_examples 5760 --save_strategy steps --save_steps 360)
        ;;
    *) echo "Unknown SFT role: $ROLE" >&2; exit 2 ;;
esac
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec python "${BUNDLE_DIR}/ray_multinode_torchrun.py" \
    --num-nodes "$NUM_NODES" --gpus-per-node "$GPUS_PER_NODE" --cpus-per-node "$CPUS_PER_NODE" \
    --train-script "${BUNDLE_DIR}/sft_train.py" -- \
    --model_name_or_path "$MODEL_PATH" --data_path "$DATA_PATH" --output_dir "$OUTPUT_DIR" \
    --max_seq_length "$MAX_SEQ_LENGTH" --per_device_train_batch_size 1 \
    --num_train_epochs 2 --weight_decay 0.01 --seed 42 --data_seed 42 \
    --bf16 True --gradient_checkpointing True --deepspeed "${BUNDLE_DIR}/ds_config_zero3.json" \
    --logging_steps 5 --save_total_limit 2 --lr_scheduler_type cosine \
    --report_to none --run_name "${ROLE}_sft" "${ROLE_ARGS[@]}"
