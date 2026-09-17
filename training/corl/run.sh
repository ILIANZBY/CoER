#!/usr/bin/env bash
# Adaptive adversarial co-evolution: attacker PPO + defender PPO.
#
# Default layout uses 7 nodes x 8 GPUs (56 GPUs total):
#   node 1: attacker actor + critic + ref (TP=2, CP=1, DP=4, offloaded)
#   node 2: defender actor + critic + ref (TP=2, CP=4, offloaded)
#   node 3: current attacker vLLM (4 replicas at TP=2)
#   node 4: old attacker population (4 replicas at TP=2)
#   nodes 5-6: current defender vLLM (16 replicas at TP=1)
#   node 7: old defender population (4 replicas at TP=2)
#
# One environment rollout contains all adaptive injection turns. Both the outer
# co-training repeat and vLLM rollout.n are fixed to 1; do not increase either
# for PPO. Increase AGENT_LOOP_NUM_WORKERS or rollout replicas for throughput.

set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_USE_V1=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export no_proxy="*"
export NO_PROXY="*"

# Keep injected payloads intact in long tool results.  The custom loop uses
# COTRAIN_MAX_TOOL_RESPONSE_LENGTH; the generic multi-turn path is configured
# separately below so it cannot fall back to its 256-character default.
# 0 disables tool-response truncation.  This keeps the sampled attacker
# payload and the defender PPO context identical, even for long results.
COTRAIN_MAX_TOOL_RESPONSE_LENGTH=${COTRAIN_MAX_TOOL_RESPONSE_LENGTH:-0}
MULTI_TURN_MAX_TOOL_RESPONSE_LENGTH=${MULTI_TURN_MAX_TOOL_RESPONSE_LENGTH:-0}
COTRAIN_WANDB_VERBOSE_TASK_METRICS=${COTRAIN_WANDB_VERBOSE_TASK_METRICS:-0}
export COTRAIN_MAX_TOOL_RESPONSE_LENGTH

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="${CORL_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${PROJECT_DIR}"

############################ Models and output ############################

ATTACKER_MODEL_PATH=${ATTACKER_MODEL_PATH:-"${PROJECT_DIR}/sft_output/attacker_sft"}
DEFENDER_MODEL_PATH=${DEFENDER_MODEL_PATH:-"${PROJECT_DIR}/models/base-model"}
PROJECT_NAME=${PROJECT_NAME:-"corl"}
EXP_NAME=${EXP_NAME:-"corl"}
if [[ "${COTRAIN_TRAINING_MODE:-dual}" != "dual" ]]; then
    echo "ERROR: only bilateral Co-PPO is supported" >&2
    exit 1
fi
CKPT_BASE_DIR=${CKPT_BASE_DIR:-"${PROJECT_DIR}/checkpoints/corl"}
ATK_CKPT_DIR=${ATK_CKPT_DIR:-"${CKPT_BASE_DIR}/attacker"}
DEF_CKPT_DIR=${DEF_CKPT_DIR:-"${CKPT_BASE_DIR}/defender"}
export COTRAIN_ROLLOUT_SAVE_DIR=${COTRAIN_ROLLOUT_SAVE_DIR:-"${CKPT_BASE_DIR}/rollout_logs"}
POPULATION_STATE_DIR=${POPULATION_STATE_DIR:-"${CKPT_BASE_DIR}/population"}
# Canonical runs start fresh. Never silently overwrite an earlier experiment.
for checkpoint_dir in "${ATK_CKPT_DIR}" "${DEF_CKPT_DIR}" "${POPULATION_STATE_DIR}"; do
    if [[ -d "${checkpoint_dir}" ]] && [[ -n "$(ls -A "${checkpoint_dir}")" ]]; then
        echo "ERROR: choose an empty checkpoint directory: ${checkpoint_dir}" >&2
        exit 1
    fi
done

validate_hf_model() {
    local model_dir=$1
    local role=$2
    if [[ ! -f "${model_dir}/config.json" ]]; then
        echo "ERROR: ${role} model is incomplete: missing ${model_dir}/config.json" >&2
        return 1
    fi
    if [[ ! -f "${model_dir}/pytorch_model.bin" ]] && \
       ! compgen -G "${model_dir}/*.safetensors" >/dev/null; then
        echo "ERROR: ${role} model has no pytorch_model.bin or *.safetensors: ${model_dir}" >&2
        return 1
    fi
}

if [[ "${ADV_EVO_SKIP_MODEL_CHECK:-0}" != "1" ]]; then
    validate_hf_model "${ATTACKER_MODEL_PATH}" attacker
    validate_hf_model "${DEFENDER_MODEL_PATH}" defender
fi

############################ Cluster resources ############################

NNODES_ATK_TRAIN=${NNODES_ATK_TRAIN:-1}
NNODES_DEF_TRAIN=${NNODES_DEF_TRAIN:-1}
NNODES_ROLLOUT=${NNODES_ROLLOUT:-5}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

CURRENT_ATK_ROLLOUT_NNODES=${CURRENT_ATK_ROLLOUT_NNODES:-1}
CURRENT_ATK_ROLLOUT_GPUS_PER_NODE=${CURRENT_ATK_ROLLOUT_GPUS_PER_NODE:-8}
OLD_ATK_ROLLOUT_NNODES=${OLD_ATK_ROLLOUT_NNODES:-1}
OLD_ATK_ROLLOUT_GPUS_PER_NODE=${OLD_ATK_ROLLOUT_GPUS_PER_NODE:-8}
CURRENT_DEF_ROLLOUT_NNODES=${CURRENT_DEF_ROLLOUT_NNODES:-2}
CURRENT_DEF_ROLLOUT_GPUS_PER_NODE=${CURRENT_DEF_ROLLOUT_GPUS_PER_NODE:-8}
OLD_DEF_ROLLOUT_NNODES=${OLD_DEF_ROLLOUT_NNODES:-1}
OLD_DEF_ROLLOUT_GPUS_PER_NODE=${OLD_DEF_ROLLOUT_GPUS_PER_NODE:-8}

ROLE_ROLLOUT_NODES=$((
    CURRENT_ATK_ROLLOUT_NNODES + OLD_ATK_ROLLOUT_NNODES
    + CURRENT_DEF_ROLLOUT_NNODES + OLD_DEF_ROLLOUT_NNODES
))
if (( NNODES_ROLLOUT != ROLE_ROLLOUT_NODES )); then
    echo "ERROR: NNODES_ROLLOUT=${NNODES_ROLLOUT}, but role allocations require ${ROLE_ROLLOUT_NODES} nodes" >&2
    exit 1
fi

ATK_TRAIN_GPUS=$((NNODES_ATK_TRAIN * NGPUS_PER_NODE))
DEF_TRAIN_GPUS=$((NNODES_DEF_TRAIN * NGPUS_PER_NODE))
CURRENT_ATK_ROLLOUT_GPUS=$((CURRENT_ATK_ROLLOUT_NNODES * CURRENT_ATK_ROLLOUT_GPUS_PER_NODE))
OLD_ATK_ROLLOUT_GPUS=$((OLD_ATK_ROLLOUT_NNODES * OLD_ATK_ROLLOUT_GPUS_PER_NODE))
CURRENT_DEF_ROLLOUT_GPUS=$((CURRENT_DEF_ROLLOUT_NNODES * CURRENT_DEF_ROLLOUT_GPUS_PER_NODE))
OLD_DEF_ROLLOUT_GPUS=$((OLD_DEF_ROLLOUT_NNODES * OLD_DEF_ROLLOUT_GPUS_PER_NODE))
TOTAL_GPUS=$((
    ATK_TRAIN_GPUS + DEF_TRAIN_GPUS
    + CURRENT_ATK_ROLLOUT_GPUS + OLD_ATK_ROLLOUT_GPUS
    + CURRENT_DEF_ROLLOUT_GPUS + OLD_DEF_ROLLOUT_GPUS
))
BASE_TRAIN_NNODES=${NNODES_ATK_TRAIN}
if (( BASE_TRAIN_NNODES <= 0 )); then
    echo "ERROR: active trainer node count must be positive" >&2
    exit 1
fi

ATK_TP=${ATK_TP:-2}
ATK_PP=${ATK_PP:-1}
# The text-only Qwen3.5 SFT checkpoint uses the GatedDeltaNet path. Its
# current Megatron/MBridge import has an inconsistent CP slice for ``dt_bias``
# (CP=2 produces [4] vs [8] heads on the first PPO batch). Keep CP disabled
# for this model and use DP=4 to occupy the same eight attacker GPUs. The
# reference co-training checkpoint is multimodal ``qwen3_5`` and can still
# override this to CP=2 explicitly.
ATK_CP=${ATK_CP:-1}
DEF_TP=${DEF_TP:-2}
DEF_PP=${DEF_PP:-1}
DEF_CP=${DEF_CP:-4}
EP=${EP:-1}
ETP=${ETP:-1}

CURRENT_ATK_TP=${CURRENT_ATK_TP:-2}
# Current defender serving has always used TP=1.  Keep it explicit so resource
# profiles can reduce only the number of identical serving replicas without
# silently changing per-request model parallelism.
CURRENT_DEF_TP=${CURRENT_DEF_TP:-1}
OLD_ATK_TP=${OLD_ATK_TP:-2}
OLD_DEF_TP=${OLD_DEF_TP:-2}

PARAM_OFFLOAD=${PARAM_OFFLOAD:-True}
OPTIMIZER_OFFLOAD=${OPTIMIZER_OFFLOAD:-True}
GRAD_OFFLOAD=${GRAD_OFFLOAD:-True}

############################ Co-evolution policy ############################

N_ROLLOUTS_PER_PROMPT=1
ROLLOUT_N=1

# This is the only mechanism switch used by the no-population ablation.  The
# default remains the production co-evolution setup.
POPULATION_ENABLED=${POPULATION_ENABLED:-True}
case "${POPULATION_ENABLED}" in
    True|False) ;;
    *)
        echo "ERROR: POPULATION_ENABLED must be True or False" >&2
        exit 1
        ;;
esac

PROB_CURR_CURR=${PROB_CURR_CURR:-0.40}
PROB_OLD_ATK_CURR_DEF=${PROB_OLD_ATK_CURR_DEF:-0.25}
PROB_CURR_ATK_OLD_DEF=${PROB_CURR_ATK_OLD_DEF:-0.25}
PROB_FIXED_TEMPLATE=${PROB_FIXED_TEMPLATE:-0.10}
# Native clean rows (about 25% of agentdyn_train.parquet) are routed by data
# metadata. Do not add a second random clean stream on injected rows.
PROB_CLEAN=${PROB_CLEAN:-0.0}

POPULATION_SIZE=${POPULATION_SIZE:-4}
POPULATION_UPDATE_INTERVAL=${POPULATION_UPDATE_INTERVAL:-20}
# Historical serving stays one complete refresh behind the current policy.
# At defender step 600 the newest probation candidate is therefore step 580,
# rather than a duplicate of the current step-600 endpoint.
POPULATION_CANDIDATE_LAG_STEPS=${POPULATION_CANDIDATE_LAG_STEPS:-20}
POPULATION_MIN_EVAL_SAMPLES_PER_BUCKET=${POPULATION_MIN_EVAL_SAMPLES_PER_BUCKET:-32}
POPULATION_MIN_EVAL_BUCKETS=${POPULATION_MIN_EVAL_BUCKETS:-7}
POPULATION_REQUIRED_BUCKETS=${POPULATION_REQUIRED_BUCKETS:-"slack,shopping,workspace,dailylife,banking,github,travel"}
POPULATION_PROBATION_SAMPLING_PROB=${POPULATION_PROBATION_SAMPLING_PROB:-0.75}
# Allow in-flight trajectories to drain before replacing historical weights.
POPULATION_DRAIN_TIMEOUT_S=${POPULATION_DRAIN_TIMEOUT_S:-3600}
POPULATION_RANDOM_SEED=${POPULATION_RANDOM_SEED:-0}

if ! awk \
    -v p1="${PROB_CURR_CURR}" \
    -v p2="${PROB_OLD_ATK_CURR_DEF}" \
    -v p3="${PROB_CURR_ATK_OLD_DEF}" \
    -v p4="${PROB_FIXED_TEMPLATE}" \
    -v p5="${PROB_CLEAN}" \
    'BEGIN {
        values[1]=p1; values[2]=p2; values[3]=p3; values[4]=p4; values[5]=p5;
        sum=0;
        for (i=1; i<=5; i++) {
            if (values[i] !~ /^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$/) exit 1;
            if (values[i] < 0 || values[i] > 1) exit 1;
            sum += values[i];
        }
        if (sum < 0.99999999 || sum > 1.00000001) exit 1;
    }'; then
    echo "ERROR: injected-row model-pair probabilities must each be in [0,1] and sum to 1.0" >&2
    exit 1
fi

if (( CURRENT_ATK_ROLLOUT_GPUS % CURRENT_ATK_TP != 0 || \
      CURRENT_DEF_ROLLOUT_GPUS % CURRENT_DEF_TP != 0 )); then
    echo "ERROR: current rollout GPU counts must be divisible by their tensor-parallel sizes" >&2
    exit 1
fi
CURRENT_ATK_REPLICAS=$((CURRENT_ATK_ROLLOUT_GPUS / CURRENT_ATK_TP))
CURRENT_DEF_REPLICAS=$((CURRENT_DEF_ROLLOUT_GPUS / CURRENT_DEF_TP))
if (( CURRENT_DEF_REPLICAS <= 0 )); then
    echo "ERROR: current defender rollout pool must have at least one replica" >&2
    exit 1
fi
if (( CURRENT_ATK_REPLICAS <= 0 )); then
    echo "ERROR: dual training requires at least one current attacker replica" >&2
    exit 1
fi

if [[ "${POPULATION_ENABLED}" == "True" ]]; then
if (( OLD_ATK_ROLLOUT_GPUS % OLD_ATK_TP != 0 || OLD_DEF_ROLLOUT_GPUS % OLD_DEF_TP != 0 )); then
    echo "ERROR: historical rollout GPU counts must be divisible by their tensor-parallel sizes" >&2
    exit 1
fi
OLD_ATK_REPLICAS=$((OLD_ATK_ROLLOUT_GPUS / OLD_ATK_TP))
OLD_DEF_REPLICAS=$((OLD_DEF_ROLLOUT_GPUS / OLD_DEF_TP))
if (( POPULATION_SIZE != OLD_ATK_REPLICAS || POPULATION_SIZE != OLD_DEF_REPLICAS )); then
    echo "ERROR: POPULATION_SIZE=${POPULATION_SIZE} must equal old attacker/defender replicas (${OLD_ATK_REPLICAS}/${OLD_DEF_REPLICAS})" >&2
    exit 1
fi
if (( POPULATION_UPDATE_INTERVAL <= 0 || POPULATION_MIN_EVAL_SAMPLES_PER_BUCKET <= 0 || POPULATION_MIN_EVAL_BUCKETS <= 0 )); then
    echo "ERROR: population update interval and evaluation quotas must be positive" >&2
    exit 1
fi
if ! [[ "${POPULATION_CANDIDATE_LAG_STEPS}" =~ ^[0-9]+$ ]] || \
   (( POPULATION_CANDIDATE_LAG_STEPS % POPULATION_UPDATE_INTERVAL != 0 )); then
    echo "ERROR: POPULATION_CANDIDATE_LAG_STEPS must be a non-negative multiple of POPULATION_UPDATE_INTERVAL" >&2
    exit 1
fi
IFS=',' read -r -a POPULATION_REQUIRED_BUCKET_ARRAY <<< "${POPULATION_REQUIRED_BUCKETS}"
if (( ${#POPULATION_REQUIRED_BUCKET_ARRAY[@]} == 0 )); then
    echo "ERROR: POPULATION_REQUIRED_BUCKETS must not be empty" >&2
    exit 1
fi
POPULATION_REQUIRED_BUCKET_SEEN=","
for bucket in "${POPULATION_REQUIRED_BUCKET_ARRAY[@]}"; do
    if [[ -z "${bucket}" || "${bucket}" =~ [[:space:]] ]]; then
        echo "ERROR: population bucket names must be non-empty and contain no whitespace: ${POPULATION_REQUIRED_BUCKETS}" >&2
        exit 1
    fi
    case "${POPULATION_REQUIRED_BUCKET_SEEN}" in
        *",${bucket},"*)
            echo "ERROR: duplicate population bucket: ${bucket}" >&2
            exit 1
            ;;
    esac
    POPULATION_REQUIRED_BUCKET_SEEN+="${bucket},"
done
if (( POPULATION_MIN_EVAL_BUCKETS > ${#POPULATION_REQUIRED_BUCKET_ARRAY[@]} )); then
    echo "ERROR: POPULATION_MIN_EVAL_BUCKETS=${POPULATION_MIN_EVAL_BUCKETS} exceeds required bucket count ${#POPULATION_REQUIRED_BUCKET_ARRAY[@]}" >&2
    exit 1
fi
if ! awk -v value="${POPULATION_PROBATION_SAMPLING_PROB}" \
    'BEGIN {
        if (value !~ /^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$/) exit 1;
        if (value < 0 || value > 1) exit 1;
    }'; then
    echo "ERROR: POPULATION_PROBATION_SAMPLING_PROB must be in [0,1]" >&2
    exit 1
fi
if ! awk -v value="${POPULATION_DRAIN_TIMEOUT_S}" \
    'BEGIN {
        if (value !~ /^[+]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$/) exit 1;
        if (value <= 0) exit 1;
    }'; then
    echo "ERROR: POPULATION_DRAIN_TIMEOUT_S must be finite and positive" >&2
    exit 1
fi
if ! awk -v old_atk="${PROB_OLD_ATK_CURR_DEF}" -v old_def="${PROB_CURR_ATK_OLD_DEF}" \
    'BEGIN { if (old_atk <= 0 || old_def <= 0) exit 1 }'; then
    echo "ERROR: both historical model-pair probabilities must be positive for probation evaluation" >&2
    exit 1
fi
else
    if (( POPULATION_SIZE != 0 )); then
        echo "ERROR: POPULATION_SIZE must be 0 when population is disabled" >&2
        exit 1
    fi
    if (( OLD_ATK_ROLLOUT_NNODES != 0 || OLD_ATK_ROLLOUT_GPUS_PER_NODE != 0 || \
          OLD_DEF_ROLLOUT_NNODES != 0 || OLD_DEF_ROLLOUT_GPUS_PER_NODE != 0 )); then
        echo "ERROR: historical rollout resources must all be 0 when population is disabled" >&2
        exit 1
    fi
    if ! awk -v old_atk="${PROB_OLD_ATK_CURR_DEF}" -v old_def="${PROB_CURR_ATK_OLD_DEF}" \
        'BEGIN { if (old_atk != 0 || old_def != 0) exit 1 }'; then
        echo "ERROR: historical model-pair probabilities must both be 0 when population is disabled" >&2
        exit 1
    fi
fi

############################ Sequence and PPO settings ############################

ATK_LR=${ATK_LR:-5e-7}
ATK_CRITIC_LR=${ATK_CRITIC_LR:-1e-5}
# Paper Table 12: initial critic-only calibration before actor updates.
ATK_CRITIC_WARMUP=${ATK_CRITIC_WARMUP:-40}
ATK_CRITIC_LR_WARMUP=${ATK_CRITIC_LR_WARMUP:-20}
ATK_KL_LOSS_COEF=${ATK_KL_LOSS_COEF:-0.001}
ATK_CLIP_LOW=${ATK_CLIP_LOW:-0.20}
ATK_CLIP_HIGH=${ATK_CLIP_HIGH:-0.28}
ATK_MAX_PROMPT_LENGTH=${ATK_MAX_PROMPT_LENGTH:-8192}
ATK_MAX_RESPONSE_LENGTH=${ATK_MAX_RESPONSE_LENGTH:-49152}
ATK_MAX_MODEL_LEN=${ATK_MAX_MODEL_LEN:-57344}
# Auditing the exact checkpoint tokenizer gives a 2,328-token maximum across
# all 11,655 supervised turns.  3,072 leaves 32% headroom for exploration but
# stops residual runaway generations earlier than the old 4,096-token cap.
ATK_TURN_MAX_TOKENS=${ATK_TURN_MAX_TOKENS:-3072}
ATK_TEMPERATURE=${ATK_TEMPERATURE:-1.0}
# Per-GPU dynamic micro-batch token budget.
ATK_TOKEN_LEN_PER_GPU=${ATK_TOKEN_LEN_PER_GPU:-32768}
# Target-tool/parameter progress remains telemetry only. PPO reward is the
# authoritative AgentDojo ASR plus the 0.05 strict-format/non-empty bonus.

DEF_LR=${DEF_LR:-5e-7}
DEF_CRITIC_WARMUP=${DEF_CRITIC_WARMUP:-100}


DEF_CRITIC_LR_WARMUP=${DEF_CRITIC_LR_WARMUP:-40}
DEF_CLIP_LOW=${DEF_CLIP_LOW:-0.20}
DEF_CLIP_HIGH=${DEF_CLIP_HIGH:-0.24}
DEF_MAX_PROMPT_LENGTH=${DEF_MAX_PROMPT_LENGTH:-15360}
DEF_MAX_RESPONSE_LENGTH=${DEF_MAX_RESPONSE_LENGTH:-16384}
DEF_TURN_MAX_TOKENS=${DEF_TURN_MAX_TOKENS:-8192}
DEF_MAX_MODEL_LEN=${DEF_MAX_MODEL_LEN:-40960}
DEF_TOKEN_LEN_PER_GPU=${DEF_TOKEN_LEN_PER_GPU:-32768}
# Starting all vLLM replicas at once can exhaust the per-node shared-memory
# broadcast pool before the engine cores finish warmup.  Limit initialization
# concurrency; this affects startup only, not steady-state rollout throughput.
VLLM_STANDALONE_INIT_CONCURRENCY=${VLLM_STANDALONE_INIT_CONCURRENCY:-8}

if ! awk -v value="${ATK_TEMPERATURE}" \
    'BEGIN {
        if (value !~ /^[+]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$/) exit 1;
        if (value < 0 || value > 2) exit 1;
    }'; then
    echo "ERROR: ATK_TEMPERATURE must be a finite number in [0,2], got ${ATK_TEMPERATURE}" >&2
    exit 1
fi
if ! [[ "${ATK_TURN_MAX_TOKENS}" =~ ^[0-9]+$ ]] || (( ATK_TURN_MAX_TOKENS < 2328 )); then
    echo "ERROR: ATK_TURN_MAX_TOKENS must be an integer >= 2328, got ${ATK_TURN_MAX_TOKENS}" >&2
    exit 1
fi
if ! [[ "${ATK_TOKEN_LEN_PER_GPU}" =~ ^[0-9]+$ && "${DEF_TOKEN_LEN_PER_GPU}" =~ ^[0-9]+$ ]] \
    || (( ATK_TOKEN_LEN_PER_GPU <= 0 || DEF_TOKEN_LEN_PER_GPU <= 0 )); then
    echo "ERROR: attacker/defender token limits must be positive integers" >&2
    exit 1
fi
if ! [[ "${ATK_CRITIC_WARMUP}" =~ ^[0-9]+$ && "${DEF_CRITIC_WARMUP}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: critic warmup steps must be non-negative integers" >&2
    exit 1
fi
if (( DEF_CRITIC_WARMUP < ATK_CRITIC_WARMUP )); then
    echo "ERROR: defender warmup must be >= attacker warmup to preserve attacker bootstrap" >&2
    exit 1
fi

PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-128}
# VAPO's token-level sequence normalization.  This is distinct from the
# older token-mean aggregation, which overweights long responses.
PPO_LOSS_AGG_MODE=${PPO_LOSS_AGG_MODE:-seq-mean-token-sum-norm}
LOG_PROB_TOKEN_LEN_PER_GPU=${LOG_PROB_TOKEN_LEN_PER_GPU:-65536}
SAVE_FREQ=${SAVE_FREQ:-10}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-50}
if ! [[ "${TOTAL_EPOCHS}" =~ ^[0-9]+$ ]] || (( TOTAL_EPOCHS <= 0 )); then
    echo "ERROR: TOTAL_EPOCHS must be a positive integer, got ${TOTAL_EPOCHS}" >&2
    exit 1
fi

TRAIN_PROMPT_BSZ=0
GEN_PROMPT_BSZ=${GEN_PROMPT_BSZ:-8}
TOTAL_ROLLOUT_STEPS=${TOTAL_ROLLOUT_STEPS:-null}
STALENESS_THRESHOLD=${STALENESS_THRESHOLD:-0.5}
MAX_POLICY_LAG=${MAX_POLICY_LAG:-1}
TRIGGER_PARAMETER_SYNC_STEP=${TRIGGER_PARAMETER_SYNC_STEP:-2}
TRIGGER_PARAMETER_SYNC_STEP=${TRIGGER_PARAMETER_SYNC_STEP:-2}
REQUIRE_BATCHES=${REQUIRE_BATCHES:-1}
AGENT_LOOP_NUM_WORKERS=${AGENT_LOOP_NUM_WORKERS:-128}

AGENT_LOOP_CONFIG="agent_loop.yaml"
AGENT_LOOP_CONFIG_SOURCE="${SCRIPT_DIR}/${AGENT_LOOP_CONFIG}"
TRAIN_PATH=${TRAIN_PATH:-"${PROJECT_DIR}/training_data/agentdyn_train.parquet"}
TEST_PATH=${TEST_PATH:-"${PROJECT_DIR}/training_data/agentdyn_val.parquet"}
RAY_ADDRESS=${RAY_ADDRESS:-"http://127.0.0.1:8265"}

if [[ "${ADV_EVO_DRY_RUN:-0}" == "1" ]]; then
    ALGORITHM_SUMMARY="attacker PPO + defender PPO"
    printf '%s\n' \
        "adv-evo preflight OK" \
        "  training_mode=dual" \
        "  attacker=${ATTACKER_MODEL_PATH}" \
        "  defender=${DEFENDER_MODEL_PATH}" \
        "  algorithm=${ALGORITHM_SUMMARY}" \
        "  rollout_n=${ROLLOUT_N}, n_rollouts_per_prompt=${N_ROLLOUTS_PER_PROMPT}" \
        "  model_pair_probs=curr_curr:${PROB_CURR_CURR},old_atk_curr_def:${PROB_OLD_ATK_CURR_DEF},curr_atk_old_def:${PROB_CURR_ATK_OLD_DEF},fixed_template:${PROB_FIXED_TEMPLATE},clean:${PROB_CLEAN}" \
        "  GPUs=${TOTAL_GPUS}: train(atk=${ATK_TRAIN_GPUS},def=${DEF_TRAIN_GPUS}), rollout(curr_atk=${CURRENT_ATK_ROLLOUT_GPUS},old_atk=${OLD_ATK_ROLLOUT_GPUS},curr_def=${CURRENT_DEF_ROLLOUT_GPUS},old_def=${OLD_DEF_ROLLOUT_GPUS})" \
        "  current_replicas=attacker:${CURRENT_ATK_REPLICAS}(TP=${CURRENT_ATK_TP}),defender:${CURRENT_DEF_REPLICAS}(TP=${CURRENT_DEF_TP})" \
        "  ppo_mini_batch=${PPO_MINI_BATCH_SIZE}, loss_agg=${PPO_LOSS_AGG_MODE}" \
        "  save_freq=${SAVE_FREQ}, total_epochs=${TOTAL_EPOCHS}" \
        "  tool_response_max_chars=${COTRAIN_MAX_TOOL_RESPONSE_LENGTH}, generic_multi_turn_max_chars=${MULTI_TURN_MAX_TOOL_RESPONSE_LENGTH}" \
        "  wandb_verbose_task_metrics=${COTRAIN_WANDB_VERBOSE_TASK_METRICS}" \
        "  attacker_clip=${ATK_CLIP_LOW}/${ATK_CLIP_HIGH}, defender_clip=${DEF_CLIP_LOW}/${DEF_CLIP_HIGH}" \
        "  attacker_kl_loss_coef=${ATK_KL_LOSS_COEF}" \
        "  critic-only warmup steps: attacker=${ATK_CRITIC_WARMUP}, defender=${DEF_CRITIC_WARMUP}" \
        "  attacker context=${ATK_MAX_PROMPT_LENGTH}+${ATK_MAX_RESPONSE_LENGTH}, train_token_cap=${ATK_TOKEN_LEN_PER_GPU}, CP=${ATK_CP}, turn_max_tokens=${ATK_TURN_MAX_TOKENS}" \
        "  attacker_temperature=${ATK_TEMPERATURE}" \
        "  attacker_target_dense_shaping=off (AgentDojo ASR is authoritative)" \
        "  population_enabled=${POPULATION_ENABLED}, population_size=${POPULATION_SIZE}, update_interval=${POPULATION_UPDATE_INTERVAL}, candidate_lag=${POPULATION_CANDIDATE_LAG_STEPS}, eval_quota=${POPULATION_MIN_EVAL_BUCKETS}x${POPULATION_MIN_EVAL_SAMPLES_PER_BUCKET}, required_buckets=${POPULATION_REQUIRED_BUCKETS}" \
        "  population_state=${POPULATION_STATE_DIR}, probation_sampling_prob=${POPULATION_PROBATION_SAMPLING_PROB}, drain_timeout_s=${POPULATION_DRAIN_TIMEOUT_S}" \
        "  defender context=${DEF_MAX_PROMPT_LENGTH}+${DEF_MAX_RESPONSE_LENGTH}, train_token_cap=${DEF_TOKEN_LEN_PER_GPU}, turn_max_tokens=${DEF_TURN_MAX_TOKENS}" \
        "  vllm_standalone_init_concurrency=${VLLM_STANDALONE_INIT_CONCURRENCY}" \
        "  ray=${RAY_ADDRESS}"
    exit 0
fi

############################ Ray working bundle ############################

# Assemble the transient runtime bundle on the node-local filesystem. Shared
# filesystem metadata operations can otherwise make bundle creation slower
# than a Ray-job submission timeout and leave partial bundles in the project.
RAY_BUNDLE_DIR=$(mktemp -d "${TMPDIR:-/tmp}/ray_bundle_adv_evo.XXXXXX")
cleanup_bundle() {
    rm -rf -- "${RAY_BUNDLE_DIR}"
}
trap cleanup_bundle EXIT

mkdir -p "${RAY_BUNDLE_DIR}/AgentDyn" "${RAY_BUNDLE_DIR}/config"
cp -a "${PROJECT_DIR}/cotrain" "${RAY_BUNDLE_DIR}/"
# The runtime needs only this entry config; copying the whole directory can
# follow transient hidden links created by local tooling and abort submission.
cp -a "${PROJECT_DIR}/config/cotrain_config.yaml" "${RAY_BUNDLE_DIR}/config/"
cp -a "${PROJECT_DIR}/verl" "${RAY_BUNDLE_DIR}/"
cp -a "${PROJECT_DIR}/model_configs" "${RAY_BUNDLE_DIR}/"
if [[ -d "${PROJECT_DIR}/recipe" ]]; then
    cp -a "${PROJECT_DIR}/recipe" "${RAY_BUNDLE_DIR}/"
fi
cp -a "${AGENT_LOOP_CONFIG_SOURCE}" "${RAY_BUNDLE_DIR}/"
cp -a "${PROJECT_DIR}/pyproject.toml" "${RAY_BUNDLE_DIR}/"
cp -a "${PROJECT_DIR}/setup.py" "${RAY_BUNDLE_DIR}/"
cp -a "${PROJECT_DIR}/AgentDyn/src" "${RAY_BUNDLE_DIR}/AgentDyn/"

# Megatron's GatedDeltaNet context-parallel helper is decorated with
# ``torch.compile`` upstream. Dynamo keeps its full ``dt_bias`` while alpha
# is CP-local, which makes the first packed training step fail to broadcast.
# Eager Megatron still uses the same CUDA and NCCL kernels across all GPUs.
RUNTIME_ENV_JSON="{\"working_dir\":\".\",\"excludes\":[\"/.git/\",\"*.log\",\"logs/\",\"rollout_logs/\",\"__pycache__/\",\"training_data/\",\"*.jsonl\",\"*.parquet\",\"*.safetensors\",\"*.bin\",\"*.pt\",\"*.gguf\"],\"env_vars\":{\"VLLM_USE_V1\":\"1\",\"VLLM_ALLREDUCE_USE_SYMM_MEM\":\"0\",\"VERL_QWEN35_TEXT_VLLM_REGISTRY\":\"1\",\"TORCH_COMPILE_DISABLE\":\"1\",\"PYTHONPATH\":\".:./AgentDyn/src\",\"HYDRA_FULL_ERROR\":\"1\",\"ATTACKER_MODEL_PATH\":\"${ATTACKER_MODEL_PATH}\",\"DEFENDER_MODEL_PATH\":\"${DEFENDER_MODEL_PATH}\",\"COTRAIN_ROLLOUT_SAVE_DIR\":\"${COTRAIN_ROLLOUT_SAVE_DIR}\",\"COTRAIN_MAX_TOOL_RESPONSE_LENGTH\":\"${COTRAIN_MAX_TOOL_RESPONSE_LENGTH}\",\"MULTI_TURN_MAX_TOOL_RESPONSE_LENGTH\":\"${MULTI_TURN_MAX_TOOL_RESPONSE_LENGTH}\",\"COTRAIN_WANDB_VERBOSE_TASK_METRICS\":\"${COTRAIN_WANDB_VERBOSE_TASK_METRICS}\"},\"pip\":[\"numpy>=1.26,<2\",\"scipy>=1.11,<1.13\",\"scikit-learn>=1.4,<1.5\",\"transformers>=4.55.0\",\"pydantic>=2.0\",\"pydantic-settings>=2.0.0\",\"jinja2>=3.0.0\",\"deepdiff\",\"pyyaml\",\"openai>=1.0.0\",\"tenacity\",\"click>=8.0\",\"cohere\",\"anthropic\",\"google-genai\",\"cupy-cuda12x\",\"email-validator\",\"docstring-parser\"]}"

ray job submit --address "${RAY_ADDRESS}" --working-dir "${RAY_BUNDLE_DIR}" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    --no-wait \
    -- \
    python3 cotrain/orchestrator.py \
    --config-path=../config \
    --config-name=cotrain_config.yaml \
    +cotrain.attacker_train_nnodes=${NNODES_ATK_TRAIN} \
    +cotrain.defender_train_nnodes=${NNODES_DEF_TRAIN} \
    cotrain.attacker_algorithm=ppo \
    cotrain.attacker_model_path="${ATTACKER_MODEL_PATH}" \
    cotrain.defender_model_path="${DEFENDER_MODEL_PATH}" \
    cotrain.attacker_ckpt_dir="${ATK_CKPT_DIR}" \
    cotrain.defender_ckpt_dir="${DEF_CKPT_DIR}" \
    cotrain.population_enabled=${POPULATION_ENABLED} \
    cotrain.n_rollouts_per_prompt=${N_ROLLOUTS_PER_PROMPT} \
    cotrain.prob_curr_curr=${PROB_CURR_CURR} \
    cotrain.prob_old_atk_curr_def=${PROB_OLD_ATK_CURR_DEF} \
    cotrain.prob_curr_atk_old_def=${PROB_CURR_ATK_OLD_DEF} \
    cotrain.prob_fixed_template=${PROB_FIXED_TEMPLATE} \
    cotrain.prob_clean=${PROB_CLEAN} \
    cotrain.population_size=${POPULATION_SIZE} \
    cotrain.population_update_interval=${POPULATION_UPDATE_INTERVAL} \
    +cotrain.population_candidate_lag_steps=${POPULATION_CANDIDATE_LAG_STEPS} \
    cotrain.population_min_eval_samples_per_bucket=${POPULATION_MIN_EVAL_SAMPLES_PER_BUCKET} \
    cotrain.population_min_eval_buckets=${POPULATION_MIN_EVAL_BUCKETS} \
    cotrain.population_required_buckets="[${POPULATION_REQUIRED_BUCKETS}]" \
    cotrain.population_probation_sampling_prob=${POPULATION_PROBATION_SAMPLING_PROB} \
    cotrain.population_state_dir="${POPULATION_STATE_DIR}" \
    cotrain.population_drain_timeout_s=${POPULATION_DRAIN_TIMEOUT_S} \
    cotrain.population_random_seed=${POPULATION_RANDOM_SEED} \
    cotrain.attacker_lr=${ATK_LR} \
    +cotrain.attacker_critic_lr=${ATK_CRITIC_LR} \
    +cotrain.attacker_critic_warmup=${ATK_CRITIC_WARMUP} \

    +cotrain.attacker_critic_lr_warmup=${ATK_CRITIC_LR_WARMUP} \
    +cotrain.attacker_kl_loss_coef=${ATK_KL_LOSS_COEF} \
    +cotrain.attacker_clip_ratio_low=${ATK_CLIP_LOW} \
    +cotrain.attacker_clip_ratio_high=${ATK_CLIP_HIGH} \
    cotrain.defender_lr=${DEF_LR} \
    +cotrain.defender_critic_warmup=${DEF_CRITIC_WARMUP} \

    +cotrain.defender_critic_lr_warmup=${DEF_CRITIC_LR_WARMUP} \
    +cotrain.defender_clip_ratio_low=${DEF_CLIP_LOW} \
    +cotrain.defender_clip_ratio_high=${DEF_CLIP_HIGH} \
    cotrain.attacker_max_prompt_length=${ATK_MAX_PROMPT_LENGTH} \
    cotrain.attacker_max_response_length=${ATK_MAX_RESPONSE_LENGTH} \
    cotrain.defender_max_prompt_length=${DEF_MAX_PROMPT_LENGTH} \
    cotrain.defender_max_response_length=${DEF_MAX_RESPONSE_LENGTH} \
    +cotrain.attacker_turn_max_tokens=${ATK_TURN_MAX_TOKENS} \
    +cotrain.attacker_rollout_temperature=${ATK_TEMPERATURE} \
    +cotrain.defender_turn_max_tokens=${DEF_TURN_MAX_TOKENS} \
    +cotrain.attacker_token_len_per_gpu=${ATK_TOKEN_LEN_PER_GPU} \
    +cotrain.attacker_critic_token_len_per_gpu=${ATK_TOKEN_LEN_PER_GPU} \
    +cotrain.defender_token_len_per_gpu=${DEF_TOKEN_LEN_PER_GPU} \
    +cotrain.defender_critic_token_len_per_gpu=${DEF_TOKEN_LEN_PER_GPU} \
    cotrain.attacker_cp=${ATK_CP} \
    cotrain.defender_cp=${DEF_CP} \
    +cotrain.current_attacker_tp=${CURRENT_ATK_TP} \
    +cotrain.current_defender_tp=${CURRENT_DEF_TP} \
    cotrain.old_attacker_tp=${OLD_ATK_TP} \
    cotrain.old_defender_tp=${OLD_DEF_TP} \
    cotrain.old_model_gpu_mem_util=0.85 \
    cotrain.old_attacker_max_model_len=${ATK_MAX_MODEL_LEN} \
    cotrain.old_defender_max_model_len=${DEF_MAX_MODEL_LEN} \
    +cotrain.current_attacker_max_model_len=${ATK_MAX_MODEL_LEN} \
    +cotrain.current_defender_max_model_len=${DEF_MAX_MODEL_LEN} \
    cotrain.current_attacker_rollout_nnodes=${CURRENT_ATK_ROLLOUT_NNODES} \
    cotrain.current_attacker_rollout_gpus_per_node=${CURRENT_ATK_ROLLOUT_GPUS_PER_NODE} \
    cotrain.current_defender_rollout_nnodes=${CURRENT_DEF_ROLLOUT_NNODES} \
    cotrain.current_defender_rollout_gpus_per_node=${CURRENT_DEF_ROLLOUT_GPUS_PER_NODE} \
    cotrain.old_attacker_rollout_nnodes=${OLD_ATK_ROLLOUT_NNODES} \
    cotrain.old_attacker_rollout_gpus_per_node=${OLD_ATK_ROLLOUT_GPUS_PER_NODE} \
    cotrain.old_defender_rollout_nnodes=${OLD_DEF_ROLLOUT_NNODES} \
    cotrain.old_defender_rollout_gpus_per_node=${OLD_DEF_ROLLOUT_GPUS_PER_NODE} \
    cotrain.attacker_exp_name="${EXP_NAME}_attacker" \
    cotrain.defender_exp_name="${EXP_NAME}_defender" \
    algorithm.adv_estimator=gae \
    algorithm.gamma=1.0 \
    algorithm.lam=0.95 \
    algorithm.norm_adv_by_std_in_grpo=False \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    algorithm.rollout_correction.bypass_mode=True \
    +algorithm.filter_groups.enable=False \
    data.train_files="${TRAIN_PATH}" \
    data.val_files="${TEST_PATH}" \
    data.train_batch_size=${TRAIN_PROMPT_BSZ} \
    data.gen_batch_size=${GEN_PROMPT_BSZ} \
    data.max_prompt_length=${ATK_MAX_PROMPT_LENGTH} \
    data.max_response_length=${ATK_MAX_RESPONSE_LENGTH} \
    data.truncation=error \
    data.filter_overlong_prompts=False \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="${ATTACKER_MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.actor.optim.lr=${ATK_LR} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=20 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.optim.use_checkpoint_opt_param_scheduler=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ATK_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_rollout_log_probs=True \
    actor_rollout_ref.actor.loss_agg_mode=${PPO_LOSS_AGG_MODE} \
    actor_rollout_ref.actor.calculate_entropy=True \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    ++actor_rollout_ref.actor.policy_loss.loss_mode=vanilla \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.actor.megatron.vanilla_mbridge=True \
    actor_rollout_ref.actor.megatron.use_remove_padding=True \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=False \
    actor_rollout_ref.actor.checkpoint.load_contents="[model]" \
    actor_rollout_ref.actor.checkpoint.save_contents="[model,optimizer,extra]" \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${ATK_TP} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${ATK_PP} \
    actor_rollout_ref.actor.megatron.context_parallel_size=${ATK_CP} \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${EP} \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${ETP} \
    actor_rollout_ref.actor.megatron.param_offload=${PARAM_OFFLOAD} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${OPTIMIZER_OFFLOAD} \
    actor_rollout_ref.actor.megatron.grad_offload=${GRAD_OFFLOAD} \
    actor_rollout_ref.actor.megatron.dtype=bfloat16 \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=flash \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${CURRENT_ATK_TP} \
    actor_rollout_ref.rollout.standalone_init_concurrency=${VLLM_STANDALONE_INIT_CONCURRENCY} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=${MULTI_TURN_MAX_TOOL_RESPONSE_LENGTH} \
    actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=middle \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.max_num_batched_tokens=65536 \
    actor_rollout_ref.rollout.max_model_len=${ATK_MAX_MODEL_LEN} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${LOG_PROB_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.agent.default_agent_loop=adv_evo_adversarial \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=${AGENT_LOOP_CONFIG} \
    actor_rollout_ref.rollout.agent.num_workers=${AGENT_LOOP_NUM_WORKERS} \
    '+actor_rollout_ref.rollout.agent.custom_import_modules=["cotrain.agent_loop","cotrain.adv_evo_agent_loop"]' \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${LOG_PROB_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=False \
    actor_rollout_ref.ref.megatron.param_offload=${PARAM_OFFLOAD} \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${ATK_TP} \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${ATK_PP} \
    actor_rollout_ref.ref.megatron.context_parallel_size=${ATK_CP} \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${EP} \
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${ETP} \
    ++actor_rollout_ref.ref.megatron.override_transformer_config.attention_backend=flash \
    critic.enable=True \
    critic.strategy=megatron \
    critic.model.path="${ATTACKER_MODEL_PATH}" \
    critic.optim.use_checkpoint_opt_param_scheduler=False \
    critic.checkpoint.load_contents="[model]" \
    critic.checkpoint.save_contents="[model,optimizer,extra]" \
    reward.reward_manager.name=naive \
    trainer.logger='["console"]' \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    trainer.nnodes=${BASE_TRAIN_NNODES} \
    trainer.device=cuda \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.default_local_dir="${CKPT_BASE_DIR}" \
    trainer.val_before_train=False \
    trainer.resume_mode=disable \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.test_freq=-1 \
    rollout.nnodes=${NNODES_ROLLOUT} \
    rollout.n_gpus_per_node=${NGPUS_PER_NODE} \
    rollout.n=${ROLLOUT_N} \
    rollout.total_rollout_steps=${TOTAL_ROLLOUT_STEPS} \
    async_training.staleness_threshold=${STALENESS_THRESHOLD} \
    +cotrain.max_policy_lag=${MAX_POLICY_LAG} \
    async_training.trigger_parameter_sync_step=${TRIGGER_PARAMETER_SYNC_STEP} \
    async_training.require_batches=${REQUIRE_BATCHES} \
    async_training.use_trainer_do_validate=False
