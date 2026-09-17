# AgentDyn environment dependency

This directory contains the stateful tool environments and deterministic verifiers used by CoRL, adapted from the public AgentDyn benchmark and its AgentDojo foundation.

The canonical CoRL protocol and commands are in the [main README](../README.md), [training guide](../training/README.md), and [evaluation guide](../corl-evaluation/README.md). Upstream standalone benchmark defaults do not define the paper's training stages or evaluation panel.

## Installation

From the repository root, in the active Python 3.12 environment:

```bash
uv pip install -e AgentDyn
```

The seven suites are banking, daily life, GitHub, shopping, Slack, travel and workspace. Synthetic names, addresses, organizations and file contents are part of executable benchmark fixtures, not author metadata. Do not remove them independently of the associated verifiers.

## Upstream attribution

AgentDyn extends [AgentDojo](https://github.com/ethz-spylab/agentdojo); the upstream [AgentDyn paper](https://arxiv.org/abs/2602.03117) describes its original benchmark. Local CoRL integrations and third-party provenance are documented in [THIRD_PARTY.md](../THIRD_PARTY.md). Preserve [LICENSE](LICENSE) when redistributing.
