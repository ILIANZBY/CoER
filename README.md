<div align="center">

# CoER

### Defending against Adaptive Indirect Prompt Injection via Adversarial Co-Evolution and Refinement

Anonymous research code accompanying the ICLR 2027 submission.

[Method](#method) · [Results](#results) · [Models and datasets](#models-and-datasets) · [Getting started](#getting-started) · [Training](training/README.md) · [Evaluation](corl-evaluation/README.md)

</div>

**CoER (Co-Evolution and Refinement)** trains tool-using agents to resist adaptive indirect prompt injection while completing legitimate tasks. Its three stages are **Attacker SFT → Co-PPO → Defender SFT**: initialize an adaptive attacker, co-evolve both roles against current and historical opponents, then refine the selected defender using verified teacher demonstrations under retained attacks.

On the main seven-suite evaluation, overall ASR decreases from **38.45% to 0.22%**, and task utility increases from **63.23% to 76.32%**. Against the common adaptive attacker, CoER achieves **0.25% ASR** and **75.40% Safe-U**. These are paper-reported point estimates, not a claim of universal robustness.

![The paper's three-stage CoER framework](docs/assets/paper/figure2-corl-framework.png)

*Figure 2 from the paper. Historical opponent sampling and population refresh belong inside Co-PPO, not a separate stage. Online RL ends before Defender SFT.*

## Method

Adaptive IPI is modeled as a finite-horizon, partially observable, general-sum Markov game. The attacker replaces lower-trust tool-return content only when the defender reaches an eligible injection site. Later payloads adapt to a bounded public trace and prior attempts within the same execution. Private defender reasoning, hidden prompts, parameters and checkpoint identity are not attacker observations. Task completion and compromise are checked independently: an execution can satisfy both.

![Adaptive indirect prompt injection within one execution](docs/assets/paper/figure1-adaptive-ipi.png)

*Figure 1 from the paper. A failed attempt can inform a later injection within the same stateful execution.*

| Stage | Learning signal | Output |
|---|---|---|
| 1. Attacker SFT | All attacker turns in 3,995 verified-success conversations (11,655 turns) | Initial adaptive attacker |
| 2. Co-PPO | Separate PPO updates on current-role tokens, task/security verifiers, dynamic historical populations | Co-evolved defender and retained attackers |
| 3. Defender SFT | All assistant turns in 5,760 verified teacher demonstrations | Final CoER defender |

Co-PPO uses an injected-trajectory mixture of `0.40 / 0.25 / 0.25 / 0.10`: current/current, historical-attacker/current-defender, current-attacker/historical-defender and template/current-defender. Native clean examples are routed separately. Each role maintains four historical slots with seven-suite eligibility checks and periodic refresh.

Defender SFT executes teachers from **task initial states**, not resumed failure prefixes. Its corpus contains 4,907 attacked trajectories and 853 untriggered replays. These replays are attack-configured runs whose injection site was not reached—not native-clean examples. Co-PPO d430 is selected by the highest defender reward during training; the final model is Defender-SFT update 360, after one pass through the data within a configured two-epoch/720-update job. See the [training protocol](training/README.md) for data contracts, masks and checkpoint selection.

## Results

Aligned with the manuscript revision supplied on **2026-09-17**. These numbers are transcribed from the paper, **not recomputed from a bundled raw-results release**. All values are percentages. Utility (U) measures task success; ASR measures attack success; Safe-U requires task success without compromise. Overall U/Safe-U pool all 1,512 eligible conditions; overall ASR uses only the 1,355 attacked executions.

### Main experiment

The common-attacker panel has 1,514 raw configurations and 1,512 metric-eligible configurations: 157 clean, 168 fixed and 1,187 adaptive. Two inconsistent adaptive configurations are excluded by the predeclared rule. Checkpoints and exclusions must be fixed before evaluation.

| Defender | Clean U ↑ | Fixed U ↑ | Fixed ASR ↓ | Fixed Safe-U ↑ | Adaptive U ↑ | Adaptive ASR ↓ | Adaptive Safe-U ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base | 79.62 | 66.67 | 12.50 | 61.90 | 60.57 | 42.12 | 38.75 |
| PPO | 74.52 | 67.26 | 5.95 | 66.67 | 60.57 | 30.75 | 46.08 |
| NoPop | 77.71 | 69.64 | 4.76 | 67.86 | 67.14 | 28.48 | 51.73 |
| Co-PPO | 78.34 | 72.62 | 2.98 | 71.43 | 68.41 | 26.37 | 54.09 |
| **CoER** | **79.62** | **79.76** | **0.00** | **79.76** | **75.40** | **0.25** | **75.40** |

*Paper Table 1a. NoPop is bilateral Co-PPO without historical populations; Co-PPO is the intermediate defender before supervised refinement. CoER has 3 compromised adaptive executions out of 1,187; adaptive Safe-U is 895/1,187.*

### Official fixed-attack evaluation

| Defender | AgentDojo attacked U ↑ | AgentDojo ASR ↓ | AgentDyn attacked U ↑ | AgentDyn ASR ↓ |
|---|---:|---:|---:|---:|
| Base | 84.62 | 13.28 | 54.29 | 31.79 |
| Co-PPO | 86.09 | 5.16 | 57.32 | 17.14 |
| CoER | 86.30 | 2.21 | 66.79 | 2.50 |

*Paper Table 2 and Appendix Tables 11/14. The official `important_instructions` protocol covers 1,509 attacked pairs across seven suites; it is separate from the 168-case fixed-template panel above. Values are case-weighted within each benchmark, not seven-suite macro averages. Aggregate gains do not rule out per-suite regressions.*

### External benchmarks

| AgentLAB defender | ASR ↓ | Task success ↑ | Safe-U ↑ |
|---|---:|---:|---:|
| Base | 37.41 | 59.43 | 37.83 |
| Co-PPO | 15.81 | 78.61 | 70.39 |
| CoER | 14.12 | 82.82 | 77.13 |

*Paper Appendix Tables 12/14: GPT-5.4 Task-Injection attacks, 949 selected trajectories per defender. CoER counts are 134 attack successes, 786 task successes and 732 safe completions. Safe-U is taken from joint outcomes, not inferred from marginal rates. An older, mixed-attacker local snapshot is not the source of this table.*

| InjecAgent defender | Base payload ASR ↓ | Successes / denominator | Enhanced payload ASR ↓ | Successes / denominator |
|---|---:|---:|---:|---:|
| Base | 7.49 | 77/1,028 | 22.69 | 221/974 |
| Co-PPO | 4.40 | Not supplied | 17.30 | Not supplied |
| CoER | 0.00 | 0/1,043 | 1.97 | 20/1,016 |

*Paper Appendix Table 13 (Table 2 rounds to one decimal). Supplied denominators differ from the nominal 1,054 cases; the scope of excluded cases remains unverified. Do not combine these updated aggregates with the earlier DH/DS subtype summaries, infer missing counts, or treat InjecAgent ASR as task utility.*

These are separate cross-protocol and cross-benchmark results, not evidence of strict domain-, injection- or payload-OOD generalization. CoER's historical-union ASR rises to **4.97%** with four retained attackers and two attempts each; these are not freshly optimized best responses. The paper reports one training run, and comparisons do not establish training-seed significance.

## Models and datasets

Hugging Face destinations are pending account confirmation, privacy/safety review and release authorization. No public download is claimed yet; model weights and generated training corpora are not bundled in the source repository.

| Planned artifact | Paper scope | Availability |
|---|---|---|
| Attacker model | Retained Co-PPO attacker (local a200 candidate) | Local checkpoint; upload pending |
| Defender model | Co-PPO d430 → Defender-SFT update 360 | Local checkpoint; upload pending |
| Attacker SFT dataset | 3,995 conversations; 11,655 supervised attacker turns | Original corpus retrieval and review pending |
| Defender SFT dataset | 5,760 trajectories: 4,907 attacked + 853 untriggered replay | Original corpus retrieval and review pending |
| RL dataset | 12,705 training rows + 3,186 disjoint internal-validation rows | Original corpus retrieval and review pending |

The RL dataset keeps training and validation as separate splits. These are executable task configurations, not the official evaluation panel or a dump of rollout logs. Expected paper counts do not substitute for inspecting the original files. See the [publication plan](docs/publishing.md) before distributing any artifact.

## Getting started

Use Python 3.12 and `uv`. On a compatible CUDA system:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e '.[agentdyn,mcore]' -e AgentDyn
```

Training additionally requires compatible PyTorch/CUDA, Megatron/MBridge, vLLM, FlashAttention and DeepSpeed installations. This checkout is not a locked, one-command reproduction environment; see [prerequisites](training/README.md#prerequisites). Weights and generated training corpora are not included.

Inspect launch configurations without starting training or model services:

```bash
SFT_DRY_RUN=1 bash training/run.sh attacker-sft
ADV_EVO_DRY_RUN=1 ADV_EVO_SKIP_MODEL_CHECK=1 bash training/run.sh coppo
SFT_DRY_RUN=1 bash training/run.sh defender-sft
```

Full commands and shared-storage inputs are in [training/README.md](training/README.md). Public launchers start fresh and reject non-empty output directories. No defender-only frozen-population RL stage or actor-graft recovery workflow is included.

## Evaluation

[`corl-evaluation/`](corl-evaluation/README.md) is an isolated evaluation subproject with separate main and external-benchmark entry points. Existing `corl` directory names, command aliases, checkpoint labels and environment variables are retained for compatibility; the paper and final method are named **CoER**.

| Track | Code in this checkout | Raw paper results |
|---|---|---|
| AgentDyn main | Runner, panel builders, report scripts | Not bundled |
| InjecAgent external benchmark | Runner and record-validating report | Not bundled |
| AgentLAB external benchmark | Not yet imported | Not bundled |

Internal AgentDyn validation is not domain-OOD or injection-OOD. The legacy `run_ood.sh` filename does not establish a strict OOD split. Missing remote assets are not replaced by invented runs or reconstructed episode records.

```bash
cd corl-evaluation
bash scripts/build_main_data.sh
bash scripts/run_main.sh all --plan-only
```

## Repository layout

```text
training/           Three-stage launchers and shared supervised training
cotrain/            Adaptive rollout, bilateral PPO, dynamic populations
config/             Co-PPO configuration
AgentDyn/           Stateful environments and deterministic verifiers
verl/               Distributed RL infrastructure
model_configs/      Model-specific distributed configuration
corl-evaluation/    Main and external-benchmark evaluation (legacy path)
docs/assets/paper/  Original paper figures, independent of the source PDF
scripts/           Utilities, including anonymous source export
```

## Anonymous release and responsible use

See the [anonymous release checklist](docs/anonymous_release.md) and [GitHub / Hugging Face publication plan](docs/publishing.md). The source exporter omits Git history, local tooling configuration, paper source files, credentials, logs, checkpoints and generated trajectories. Required upstream licenses and notices remain intact; upstream attribution does not identify this submission's authors.

This is dual-use security research for controlled evaluation and defense development. Run attacks only in authorized, isolated environments; review datasets and logs before release. Citation metadata and author/project links are deferred until after anonymous review.

## License

The infrastructure builds on [verl](https://github.com/verl-project/verl); the environments build on AgentDojo. See [LICENSE](LICENSE), [Notice.txt](Notice.txt), [AgentDyn/LICENSE](AgentDyn/LICENSE), and the evaluation subproject's third-party licenses. Changes to upstream components are identified in [THIRD_PARTY.md](THIRD_PARTY.md).
