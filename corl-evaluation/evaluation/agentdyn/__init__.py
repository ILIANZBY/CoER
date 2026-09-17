"""Held-out AgentDyn evaluation for attackers and defenders.

The package intentionally does not import VERL's training stack.  Evaluation
only needs AgentDojo, pandas and an OpenAI-compatible inference endpoint, which
keeps it usable in a small, separate Ray job while training continues.
"""

from .dataset import EvalCase, load_eval_cases
from .types import AttackSpec, EndpointConfig, EvalSettings

__all__ = [
    "AttackSpec",
    "EndpointConfig",
    "EvalCase",
    "EvalSettings",
    "load_eval_cases",
]
