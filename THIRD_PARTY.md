# Third-party code and notices

This anonymous research snapshot includes modified third-party components. The names in their licenses and copyright headers identify upstream rightsholders, not the authors or affiliations of this submission.

| Component | Upstream / license | Local modifications |
|---|---|---|
| `verl/` and distributed-training utilities | [verl](https://github.com/verl-project/verl), Apache-2.0; see `LICENSE` and `Notice.txt` | Adaptive co-training integration, role-attributed asynchronous sampling, model/serving support, checkpoint and synchronization changes |
| `AgentDyn/` | [AgentDojo](https://github.com/ethz-spylab/agentdojo), MIT; see `AgentDyn/LICENSE` | Extended task suites, environments, verifiers and serialization used by CoRL |
| `corl-evaluation/third_party/InjecAgent/` | [InjecAgent](https://github.com/uiuc-kang-lab/InjecAgent); see its `LICENCE` | Local evaluation integration; benchmark input data retained for validation |

Preserve applicable upstream notices when redistributing. Public benchmark fixtures may contain fictional names, organizations and addresses required by their tasks/verifiers. They are not submission-author metadata and should not be mechanically deleted as if they were author PII.
