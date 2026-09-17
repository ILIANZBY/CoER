# CoER evaluation

Evaluation is isolated from the training entry points. It imports shared environments through `CORL_TRAINING_ROOT` and writes runtime artifacts below `results/` and `logs/`. It can be copied as an evaluation-only project alongside the training source; it is not a separate Git remote in this checkout.

Aligned with the manuscript revision supplied on 2026-09-17. Existing `corl` labels and environment variables remain compatible aliases for the paper's CoER method.

## Scope and availability

| Paper experiment | Available locally |
|---|---|
| Q1: main AgentDyn comparison | Runner, fixed panel builders, report scripts |
| Q2: external InjecAgent | Runner and validated report scripts |
| Q2: external AgentLAB | Adapter/raw results not yet imported |
| Q3: 2×2 initialization × repair data | Generic evaluator accepts explicitly configured checkpoints; paper-specific manifests not bundled |
| Q4: historical-attacker success union (four checkpoints × two attempts) | Paper-specific ordered attempt manifest and result bundle not bundled |
| Q5: fixed-opponent attacker learning | Generic attacker evaluator; snapshot manifests not bundled |

README tables are paper transcriptions, not raw result bundles. No remote results are included unless their provenance and contents have actually been checked. Neither internal validation nor the external benchmark results establishes strict domain-, injection- or payload-OOD generalization. Q4 reuses retained attackers; it is not a newly optimized best response. The subsequent RL reference in the paper's cross-play matrix is not part of the final three-stage training pipeline or the canonical five-defender main comparison.

A remote AgentLAB snapshot and its orchestration scripts have been retrieved into ignored local artifacts, outside this public subproject. The snapshot contains 2,847 unique completed model/task pairs (949 each for Base, Co-PPO d430 and SFT360), but mixes GPT-5.4 with Qwen fallback. It is not the corrected GPT-5.4-only result set in the updated paper's Appendix Table 12. That table reports CoER attack/task/safe counts of 134/786/732 over 949 selected trajectories. The remote scripts still depend on private gateways and recovery infrastructure and are not a release-ready AgentLAB runner.

## Environment

Use Python 3.12 and `uv`. From the training root:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e AgentDyn -e 'corl-evaluation[test]'
```

Main evaluation calls already-running compatible model endpoints. InjecAgent execution additionally needs Ray GPU workers and vLLM. Set `EVAL_PYTHON` to choose the interpreter. Report scripts use only the Python standard library.

## Main experiment

From this directory:

```bash
bash scripts/build_main_data.sh
bash scripts/run_main.sh all --plan-only
bash scripts/run_main.sh all
python scripts/report_main.py --require-complete
```

The canonical defender labels are `base`, `ppo`, `nopop`, `coppo`, and `corl`. Configure `BASE_MODEL_URL`/`BASE_MODEL_PATH`, and `{PPO,NOPOP,COPPO,CORL}_DEFENDER_URL`/`{PPO,NOPOP,COPPO,CORL}_DEFENDER_PATH`. Use `COPPO_ATTACKER_URL`/`COPPO_ATTACKER_PATH` for the same selected common attacker across defenders. Endpoint served-model names must match the YAML `model` fields.

`coppo` denotes the selected pre-refinement defender (paper d430); `corl` denotes CoER's Defender SFT `checkpoint-360`, not a second RL stage. Baseline and attacker checkpoint paths are placeholders and must be supplied. The paper selects d430 by highest defender reward during training and SFT360 after one data epoch within a two-epoch job. The report never selects a checkpoint by test-set performance.

The panel has 1,514 raw configurations: 157 clean, 168 fixed (42 cases × four templates), 1,189 adaptive. The predeclared metric exclusions remove two inconsistent adaptive cases, leaving 1,512 eligible configurations. Coverage, exclusions and model failures are separate fields; infrastructure/evaluator errors must not count as successful defense.

## InjecAgent external benchmark

```bash
export BASE_MODEL_PATH=/path/to/base
export CORL_DEFENDER_PATH=/path/to/defender-sft/checkpoint-360
bash scripts/run_ood.sh
python scripts/report_ood_injecagent.py --require-complete
```

The retained launcher compares Base and CoER under base/enhanced settings and DH/DS attack families. Its `run_ood.sh` name is a legacy routing name, not a claim of strict OOD generalization. The launcher shards deterministically and the reporter validates produced records against canonical benchmark inputs. `third_party/InjecAgent/` keeps its own license. DS-S1 denotes the first data-stealing step; do not confuse it with subsequent exfiltration completion.

The updated paper's Appendix Table 13 reports aggregate ASR for Base, Co-PPO and CoER, separately from the earlier DH/DS subtype summaries in Table 14d. CoER's supplied counts are 0/1,043 (base) and 20/1,016 (enhanced); the baseline denominators differ, and Co-PPO counts are not supplied. The scope of excluded cases remains unresolved in the paper. Do not invent records, infer missing denominators, or silently reinterpret subtype reports as those aggregate results. The packaged default launcher is not a complete reproduction of the new three-defender aggregate table.

## Metrics and result integrity

Utility is task success even if an attack also succeeds. ASR is attack success; Safe-U is joint task success without compromise. Overall U uses clean + fixed + adaptive eligible cases; overall ASR uses only eligible attacked cases. Learned-attacker evaluation additionally reports effective ASR, reach and strict-format validity.

Do not average away seeds, discard hard failures after inspection, infer full coverage from a running summary, or treat AgentLAB's completed-task count as a metric-valid denominator. Evaluation resume remains supported only when manifests/fingerprints match.

Before publishing results, prepare a separately reviewed bundle containing anonymized checkpoint aliases, benchmark/panel hashes, model/tokenizer versions, decoding settings, seeds, exclusions, coverage and aggregate metrics. Raw execution logs can expose endpoints, local paths, task contents and credentials; they are excluded from the source export. Never relabel paper-transcribed values as independently reproduced results.
