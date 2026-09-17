# Three-stage CoER training

The updated paper defines **Attacker SFT → Co-PPO → Defender SFT**. Population refresh belongs inside Co-PPO. There is no subsequent defender-only RL stage.

Aligned with the manuscript revision supplied on 2026-09-17. Legacy `corl` paths and command aliases are retained for compatibility; CoER is the paper's method name.

## Prerequisites

Use Python 3.12 and `uv`, a Ray cluster and shared model/data storage. Co-PPO defaults to 56 GPUs: 8 attacker-training, 8 defender-training, and 8/8/16/8 current-attacker/historical-attacker/current-defender/historical-defender serving GPUs. SFT uses 16 GPUs.

The GPU stack must support Qwen3.5-9B, exact rollout token IDs/log-probabilities and the local model patches: PyTorch/CUDA, Megatron/MBridge, vLLM, FlashAttention, Transformers and (for SFT) DeepSpeed. The SFT trainer uses Transformers' `processing_class` and Qwen3.5 selective logits. Generic dependency bounds in `setup.py` are not a verified reproduction lockfile. Record actual package/CUDA versions and model/tokenizer hashes with each run. Distributed training has not been validated on the CPU-only development machine.

Examples use shared-storage placeholders. Do not commit actual endpoints, credentials or mount names. External experiment tracking is off by default. Models and generated training corpora are not bundled.

## 1. Attacker SFT

```bash
ATTACKER_MODEL_PATH=/path/to/base-model \
ATTACKER_DATA_PATH=/path/to/attacker_sft.jsonl \
ATTACKER_OUTPUT_DIR=/path/to/new-attacker-sft-output \
bash training/run.sh attacker-sft
```

Retain 3,995 verified-success conversations, deduplicated as described in the paper. **All 11,655 attacker turns receive supervision**, including earlier unsuccessful attempts in successful conversations. Legacy per-turn `loss_mask` values are ignored. Do not supervise only the final successful payload.

Both SFT stages share `common/sft_data.py` and `common/sft_train.py`. JSONL records have a `conversations` list of `{ "role": ..., "content": ... }` messages. Supported roles are system, user, assistant and tool. Only assistant content and its terminating ChatML token receive labels; prompts, user/tool messages, padding and headers are masked. Literal control-token text inside content cannot create a new loss span. Overlength examples fail rather than silently losing context.

Keep system instructions, tool schemas, reasoning and serialized assistant tool calls in `content` as recorded. Structured `tool_calls` must be converted to the model's training protocol upstream; the loader rejects them rather than silently dropping calls. Tool messages render as user-side `<tool_response>` blocks. Group multiple tool responses into one user block upstream when the template requires it. This is a pre-serialized conversation interface, not a general OpenAI-message converter.

Attacker launcher defaults: two epochs, LR `1e-5`, cosine schedule, warmup `0.05`, weight decay `0.01`, micro-batch 1, accumulation 2, maximum sequence 57,344. The expected corpus size is checked. These are launcher defaults, distinct from the verified Defender-SFT settings below.

## 2. Co-PPO

```bash
ATTACKER_MODEL_PATH=/path/to/attacker-sft-output \
DEFENDER_MODEL_PATH=/path/to/base-model \
TRAIN_PATH=/path/to/agentdyn_train.parquet \
TEST_PATH=/path/to/agentdyn_val.parquet \
CKPT_BASE_DIR=/path/to/new-coppo-checkpoints \
bash training/run.sh coppo
```

`corl/run.sh` launches adaptive-attack Co-PPO; `corl` remains a command alias. Training uses 12,705 rows (9,529 injected, 3,176 native clean), with 3,186 internal-validation rows. Official evaluation configurations must not enter training or checkpoint selection.

One sampled configuration yields one interleaved execution with at most 20 defender turns. The attacker acts only at reached sites and emits strictly parsed `<attack>` payloads. Terminal trajectory reward applies to all generated attacker responses, not only the last successful turn. Attacker reward: `I_atk + 0.05 × (1 − I_atk) × I_fmt`. Defender reward: `+1` safe completion, `−0.2` safe failure, `−1` compromise.

Separate critics, masked GAE (`γ=1`, `λ=0.95`), asymmetric clipped PPO and KL penalties update current-role tokens only. Historical/opponent/tool tokens never receive the current trainer's gradient. Paper Table 3: actor LR `5e-7` for both roles, critic LR `1e-5`, minibatch 128, KL `0.001`, clip low `0.20`, clip high `0.28/0.24`, and initial critic warmup `40/100` updates for attacker/defender.

| Injected route | Probability | Updated roles |
|---|---:|---|
| Current attacker / current defender | 0.40 | Both |
| Historical attacker / current defender | 0.25 | Defender |
| Current attacker / historical defender | 0.25 | Attacker |
| Template / current defender | 0.10 | Defender |

Native clean rows train the current defender separately. Each historical population has four serving slots including one probation slot. Eligibility requires 32 reward-bearing samples in each of seven suites. Fitness uses a seven-suite macro mean; probation probability is 0.75. Save every 10 versions, consider refresh every 20, and enforce a maximum producer-policy lag of one version outside initial frozen-actor critic warmup.

NoPop remains available: set `POPULATION_ENABLED=False`, `POPULATION_SIZE=0`, both historical probabilities and both historical GPU allocations to zero; explicitly adjust the remaining mixture and rollout node count. This remains bilateral training, unlike the deleted frozen-population defender-only mode.

## 3. Defender SFT

```bash
DEFENDER_MODEL_PATH=/path/to/coppo-d430/actor/huggingface \
DEFENDER_DATA_PATH=/path/to/defender_sft.jsonl \
DEFENDER_OUTPUT_DIR=/path/to/new-defender-sft-output \
bash training/run.sh defender-sft
```

Initialize from Co-PPO d430, selected by the highest defender reward during training (paper Section 5.1). Retained attackers challenge teacher defenders executing from initial task states. Keep well-formed, normally terminated, verifier-approved safe completions after deterministic quality screening, per-configuration selection and deduplication. Expected corpus: 5,760 demonstrations, comprising 4,907 attacked and 853 untriggered replays. Replays share the cross-entropy objective; no preference or explicit KL loss is used.

Paper Table 4: full-parameter BF16 ZeRO-3, all assistant turns supervised, LR `5e-6`, cosine schedule, warmup `0.03`, weight decay `0.01`, seed 42, sequence cap 16,384 without truncation. Global batch 16 uses 16 GPUs × micro-batch 1 × accumulation 1.

**Configure two epochs / 720 updates, and evaluate `checkpoint-360`.** This checkpoint inherits the full 720-update learning-rate schedule (22 warmup updates). Do not schedule a separate one-epoch run or substitute the final 720-step model. Dataset generation and the original selected checkpoint are required inputs, not recreated automatically by the launcher.

## Validation and restart boundaries

```bash
SFT_DRY_RUN=1 bash training/run.sh attacker-sft
ADV_EVO_DRY_RUN=1 ADV_EVO_SKIP_MODEL_CHECK=1 bash training/run.sh coppo
SFT_DRY_RUN=1 bash training/run.sh defender-sft
python -m unittest discover -s tests/release -v
```

Public launchers start fresh and refuse non-empty output directories. Custom actor-graft, recovery warmup, topology migration and defender-only frozen-population code are removed. Generic upstream checkpoint saving/loading and asynchronous pause/resume for weight synchronization remain infrastructure features, not additional stages. Long evaluation jobs retain validated resume support.
