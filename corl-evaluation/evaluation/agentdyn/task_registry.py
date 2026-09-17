from __future__ import annotations

import copy
import importlib
import os
import re
import sys
from pathlib import Path
from typing import Literal

_REPO_ROOT = Path(
    os.environ.get("CORL_TRAINING_ROOT", Path(__file__).resolve().parents[3])
).resolve()
_AGENTDYN_SRC = _REPO_ROOT / "AgentDyn" / "src"
if str(_AGENTDYN_SRC) not in sys.path:
    sys.path.insert(0, str(_AGENTDYN_SRC))

from agentdojo.default_suites import import_default_suites  # noqa: E402

TaskCatalog = Literal["v2_training", "canonical_v1"]

_SUITE_NAMES = (
    "banking",
    "travel",
    "workspace",
    "shopping",
    "slack",
    "github",
    "dailylife",
)


def register_v2_training_tasks() -> None:
    """Register the corrected training tasks on AgentDojo's v1 suite objects."""

    importlib.import_module("agentdojo.task_suite.load_suites")
    for suite_name in _SUITE_NAMES:
        importlib.import_module(
            f"agentdojo.default_suites.v2_training.{suite_name}.user_tasks"
        )
        importlib.import_module(
            f"agentdojo.default_suites.v2_training.{suite_name}.injection_tasks"
        )


def _numeric_injection_task_modules(suites: dict) -> list[tuple[str, str, str]]:
    modules: list[tuple[str, str, str]] = []
    for suite_name, suite in suites.items():
        for task_id, task in suite.injection_tasks.items():
            if re.fullmatch(r"injection_task_\d+", task_id):
                modules.append((suite_name, task_id, type(task).__module__))
    return modules


def load_task_suites(benchmark_version: str, task_catalog: TaskCatalog) -> dict:
    """Load and freeze one internally consistent task registry.

    ``v2_training`` keeps the original v1 case-ID matrix while installing the
    corrected contracts and validators used by current CORL training.
    ``canonical_v1`` is available for strict reproduction of the old registry,
    but fails closed if this process has already installed the v2 overrides.
    """

    if task_catalog not in ("v2_training", "canonical_v1"):
        raise ValueError(f"unsupported task catalog: {task_catalog}")
    if task_catalog == "canonical_v1" and benchmark_version != "v1":
        raise ValueError("canonical_v1 task catalog requires benchmark_version=v1")

    if task_catalog == "v2_training":
        register_v2_training_tasks()

    suites = import_default_suites(benchmark_version)
    numeric_modules = _numeric_injection_task_modules(suites)
    v2_modules = [item for item in numeric_modules if ".v2_training." in item[2]]
    if task_catalog == "v2_training" and len(v2_modules) != len(numeric_modules):
        missing = sorted(set(numeric_modules) - set(v2_modules))[:5]
        raise RuntimeError(
            "v2_training task registration is incomplete; refusing mixed registry: "
            f"{missing}"
        )
    if task_catalog == "canonical_v1" and v2_modules:
        raise RuntimeError(
            "canonical_v1 was requested after v2_training modified the process-wide "
            "AgentDojo registry; start this evaluation in a fresh process"
        )

    # AgentDojo registries are mutable process-wide singletons.  A private copy
    # prevents a later runner in the same process from changing this run's tasks.
    return copy.deepcopy(suites)
