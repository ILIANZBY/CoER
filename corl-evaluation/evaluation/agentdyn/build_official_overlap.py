from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import tempfile
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pandas as pd

from cotrain.attack_contract import (
    find_user_injection_verifier_conflicts,
    has_ground_truth_action_conflict,
    resolve_attack_contract,
)
from cotrain.ppo_correctness import build_vector_placeholders
from agentdojo.agent_pipeline.ground_truth_pipeline import GroundTruthPipeline
from agentdojo.functions_runtime import FunctionsRuntime
from agentdojo.types import get_text_content_as_str

from .build_panels import _diverse_attack_core
from .dataset import EvalCase, load_eval_cases, write_panel
from .task_registry import TaskCatalog, load_task_suites


EXPECTED_SUITES = {
    "banking": {"users": 16, "injection_tasks": 9, "pairs": 144},
    "dailylife": {"users": 20, "injection_tasks": 10, "pairs": 200},
    "github": {"users": 20, "injection_tasks": 9, "pairs": 180},
    "shopping": {"users": 20, "injection_tasks": 9, "pairs": 180},
    "slack": {"users": 21, "injection_tasks": 5, "pairs": 105},
    "travel": {"users": 20, "injection_tasks": 7, "pairs": 140},
    "workspace": {"users": 40, "injection_tasks": 6, "pairs": 240},
}
EXPECTED_CLEAN_CASES = 157
EXPECTED_ATTACKED_CASES = 1189
FIXED_CORE_PER_SUITE = 6
FIXED_TEMPLATES = 4
EXPECTED_V2_TRAINING_METRIC_EXCLUSIONS = {
    "banking:user_task_15:injection_task_4",
    "travel:user_task_3:injection_task_5",
}
INCOHERENT_PAIR_REASON = (
    "user_ground_truth_triggers_injection_security_goal"
)


def _numeric_task_id(task_id: str, prefix: str) -> int:
    match = re.fullmatch(rf"{re.escape(prefix)}(\d+)", task_id)
    if match is None:
        raise ValueError(f"not an original numeric task ID: {task_id}")
    return int(match.group(1))


def _original_task_ids(task_ids: list[str] | set[str]) -> list[str]:
    return sorted(
        (task_id for task_id in task_ids if re.fullmatch(r"user_task_\d+", task_id)),
        key=lambda task_id: _numeric_task_id(task_id, "user_task_"),
    )


def _injection_task_ids(task_ids: list[str] | set[str]) -> list[str]:
    return sorted(
        (
            task_id
            for task_id in task_ids
            if re.fullmatch(r"injection_task_\d+", task_id)
        ),
        key=lambda task_id: _numeric_task_id(task_id, "injection_task_"),
    )


def _injection_candidates(suite, user_task) -> tuple[str, ...]:
    defaults = suite.get_injection_vector_defaults()
    canaries = {vector_id: f"<CANARY---{vector_id}--->" for vector_id in defaults}
    environment = suite.load_and_inject_default_environment(canaries)
    runtime = FunctionsRuntime(suite.tools)
    _, _, _, messages, _ = GroundTruthPipeline(user_task).query(
        user_task.PROMPT,
        runtime,
        environment,
    )
    candidates = tuple(
        vector_id
        for vector_id, canary in canaries.items()
        if any(
            message["content"] is not None
            and canary in get_text_content_as_str(message["content"])
            for message in messages
        )
    )
    if not candidates:
        raise ValueError(f"{suite.name}:{user_task.ID} has no reachable injection vector")
    return candidates


def _row(
    *,
    suite_name: str,
    user_task_id: str,
    user_prompt: str,
    injection_task_id: str | None,
    injections: dict[str, str],
    task_catalog: TaskCatalog,
) -> dict[str, str]:
    return {
        "data_source": "agentdyn_official_overlap_eval",
        "prompt": json.dumps(
            [{"role": "user", "content": user_prompt}], ensure_ascii=False
        ),
        "extra_info": json.dumps(
            {
                "suite_name": suite_name,
                "user_task_id": user_task_id,
                "injection_task_id": injection_task_id,
                "injections": injections,
                "user_task_prompt": user_prompt,
                "case_matrix": "AgentDyn plus AgentDojo canonical v1 numeric IDs",
                "task_catalog": task_catalog,
            },
            ensure_ascii=False,
        ),
    }


def _build_rows(
    benchmark_version: str,
    task_catalog: TaskCatalog,
) -> tuple[list[dict[str, str]], dict, dict[str, str]]:
    suites = load_task_suites(benchmark_version, task_catalog)
    rows: list[dict[str, str]] = []
    suite_stats: dict[str, dict[str, int]] = {}
    metric_exclusions: dict[str, str] = {}

    for suite_name, expected in EXPECTED_SUITES.items():
        print(f"building official overlap suite: {suite_name}", flush=True)
        suite = suites[suite_name]
        user_task_ids = _original_task_ids(set(suite.user_tasks))
        injection_task_ids = _injection_task_ids(set(suite.injection_tasks))
        actual = {
            "users": len(user_task_ids),
            "injection_tasks": len(injection_task_ids),
            "pairs": len(user_task_ids) * len(injection_task_ids),
        }
        if actual != expected:
            raise ValueError(
                f"official task counts changed for {suite_name}: "
                f"expected={expected} actual={actual}"
            )
        validated_contracts = 0
        excluded_pairs = 0

        for user_task_id in user_task_ids:
            user_task = suite.user_tasks[user_task_id]
            vector_ids = _injection_candidates(suite, user_task)
            base_environment = suite.load_and_inject_default_environment(
                build_vector_placeholders({vector_id: "" for vector_id in vector_ids})
            )
            environment = base_environment.model_copy(deep=True)
            if hasattr(user_task, "init_environment"):
                environment = user_task.init_environment(environment)
            runtime = FunctionsRuntime(suite.tools)
            verifier_conflicts = find_user_injection_verifier_conflicts(
                suite,
                user_task,
                {
                    injection_task_id: suite.injection_tasks[injection_task_id]
                    for injection_task_id in injection_task_ids
                },
                base_environment,
            )
            rows.append(
                _row(
                    suite_name=suite_name,
                    user_task_id=user_task_id,
                    user_prompt=user_task.PROMPT,
                    injection_task_id=None,
                    injections={},
                    task_catalog=task_catalog,
                )
            )
            for injection_task_id in injection_task_ids:
                injection_task = suite.injection_tasks[injection_task_id]
                resolve_attack_contract(
                    injection_task,
                    environment,
                    available_tool_names=runtime.functions.keys(),
                )
                validated_contracts += 1
                case_id = f"{suite_name}:{user_task_id}:{injection_task_id}"
                if (
                    injection_task_id in verifier_conflicts
                    or has_ground_truth_action_conflict(
                        user_task.ground_truth(environment),
                        injection_task.ground_truth(environment),
                    )
                ):
                    metric_exclusions[case_id] = INCOHERENT_PAIR_REASON
                    excluded_pairs += 1
                payload = f"TODO: {injection_task.GOAL}"
                rows.append(
                    _row(
                        suite_name=suite_name,
                        user_task_id=user_task_id,
                        user_prompt=user_task.PROMPT,
                        injection_task_id=injection_task_id,
                        injections={vector_id: payload for vector_id in vector_ids},
                        task_catalog=task_catalog,
                    )
                )
        suite_stats[suite_name] = {
            **actual,
            "validated_contracts": validated_contracts,
            "metric_exclusions": excluded_pairs,
        }

    return rows, suite_stats, metric_exclusions


def _user_keys(cases: list[EvalCase]) -> set[tuple[str, str]]:
    return {(case.suite_name, case.user_task_id) for case in cases}


def _normalized_prompt(prompt: str) -> str:
    return " ".join(prompt.split())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the official AgentDyn + AgentDojo overlap evaluation set"
    )
    parser.add_argument("--benchmark-version", default="v1")
    parser.add_argument(
        "--task-catalog",
        choices=("v2_training", "canonical_v1"),
        default="v2_training",
        help=(
            "Use v2_training for the corrected contracts/validators used by CORL; "
            "canonical_v1 reproduces the old registry, including its broken contracts."
        ),
    )
    parser.add_argument(
        "--parquet",
        default="training_data/agentdyn_official_overlap_eval.parquet",
    )
    parser.add_argument("--panel-dir", default="evaluation/agentdyn/panels")
    parser.add_argument("--train-parquet", default="training_data/agentdyn_train.parquet")
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args(argv)

    rows, suite_stats, metric_exclusions = _build_rows(
        args.benchmark_version,
        args.task_catalog,
    )
    if (
        args.task_catalog == "v2_training"
        and set(metric_exclusions) != EXPECTED_V2_TRAINING_METRIC_EXCLUSIONS
    ):
        raise ValueError(
            "v2_training incoherent-pair set changed: "
            f"expected={sorted(EXPECTED_V2_TRAINING_METRIC_EXCLUSIONS)} "
            f"actual={sorted(metric_exclusions)}"
        )
    parquet_path = Path(args.parquet)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        parquet_bytes = io.BytesIO()
        pd.DataFrame(rows).to_parquet(parquet_bytes, index=False)
        with tempfile.NamedTemporaryFile(
            prefix=f".{parquet_path.stem}.",
            suffix=".parquet.tmp",
            dir=parquet_path.parent,
            delete=False,
        ) as temporary:
            temporary.write(parquet_bytes.getvalue())
            temporary_path = Path(temporary.name)
        temporary_path.replace(parquet_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    cases = [
        replace(
            case,
            metric_exclusion_reason=metric_exclusions.get(case.case_id),
        )
        for case in load_eval_cases(parquet_path)
    ]
    clean = [case for case in cases if case.is_clean]
    attacked = [case for case in cases if not case.is_clean]
    if len(clean) != EXPECTED_CLEAN_CASES or len(attacked) != EXPECTED_ATTACKED_CASES:
        raise ValueError(
            "official overlap size changed: "
            f"expected={EXPECTED_CLEAN_CASES}+{EXPECTED_ATTACKED_CASES} "
            f"actual={len(clean)}+{len(attacked)}"
        )

    fixed_core = _diverse_attack_core(attacked, FIXED_CORE_PER_SUITE, args.seed)
    expected_core_by_suite = {
        suite_name: FIXED_CORE_PER_SUITE for suite_name in EXPECTED_SUITES
    }
    core_by_suite = Counter(case.suite_name for case in fixed_core)
    if core_by_suite != expected_core_by_suite:
        raise ValueError(
            f"fixed core suite counts changed: {dict(core_by_suite)}"
        )

    panel_dir = Path(args.panel_dir)
    adaptive_path = panel_dir / "official1514_adaptive.jsonl"
    clean_fixed_path = panel_dir / "official1514_clean_fixed.jsonl"
    adaptive_rows = write_panel(adaptive_path, attacked)
    clean_fixed_rows = write_panel(clean_fixed_path, clean + fixed_core)

    train_cases = load_eval_cases(args.train_parquet) if args.train_parquet else []
    train_clean = [case for case in train_cases if case.is_clean]
    train_strip_prompts = {case.user_prompt.strip() for case in train_clean}
    train_normalized_prompts = {
        _normalized_prompt(case.user_prompt) for case in train_clean
    }
    strip_overlap_keys = {
        (case.suite_name, case.user_task_id)
        for case in clean
        if case.user_prompt.strip() in train_strip_prompts
    }
    normalized_overlap_keys = {
        (case.suite_name, case.user_task_id)
        for case in clean
        if _normalized_prompt(case.user_prompt) in train_normalized_prompts
    }
    normalized_adaptive_overlap = [
        case
        for case in attacked
        if (case.suite_name, case.user_task_id) in normalized_overlap_keys
    ]
    normalized_fixed_overlap = [
        case
        for case in fixed_core
        if (case.suite_name, case.user_task_id) in normalized_overlap_keys
    ]
    metadata = {
        "name": "official1514",
        "case_matrix_benchmark_version": args.benchmark_version,
        "task_catalog": args.task_catalog,
        "source": str(parquet_path.resolve()),
        "source_sha256": hashlib.sha256(parquet_path.read_bytes()).hexdigest(),
        "selection_seed": args.seed,
        "definition": (
            "All canonical-v1 numeric user-task and injection-task ID pairs from "
            "AgentDyn's three suites and AgentDojo's four suites, evaluated with "
            f"the {args.task_catalog} task catalog"
        ),
        "suite_counts": suite_stats,
        "unique_user_tasks": len(_user_keys(clean)),
        "unique_injection_test_cases": len(attacked),
        "metric_exclusions": {
            "count": len(metric_exclusions),
            "reason": INCOHERENT_PAIR_REASON,
            "case_ids": sorted(metric_exclusions),
            "adaptive_primary_metric_denominator": (
                len(attacked) - len(metric_exclusions)
            ),
        },
        "panels": {
            "adaptive": {
                "path": str(adaptive_path.resolve()),
                "rows": adaptive_rows,
                "attacked_by_suite": Counter(case.suite_name for case in attacked),
            },
            "clean_fixed": {
                "path": str(clean_fixed_path.resolve()),
                "rows": clean_fixed_rows,
                "clean_cases": len(clean),
                "fixed_core_cases": len(fixed_core),
                "fixed_core_by_suite": core_by_suite,
                "fixed_core_is_adaptive_subset": {
                    case.case_id for case in fixed_core
                }
                <= {case.case_id for case in attacked},
            },
        },
        "train_overlap": {
            "checked": bool(args.train_parquet),
            "case_ids": len(
                {case.case_id for case in cases}
                & {case.case_id for case in train_cases}
            ),
            "user_task_keys": len(_user_keys(cases) & _user_keys(train_cases)),
            "strip_exact_user_task_keys": len(strip_overlap_keys),
            "whitespace_normalized_user_task_keys": len(normalized_overlap_keys),
            "whitespace_normalized_keys": [
                f"{suite_name}:{user_task_id}"
                for suite_name, user_task_id in sorted(normalized_overlap_keys)
            ],
            "affected_clean_cases": len(normalized_overlap_keys),
            "affected_adaptive_cases": len(normalized_adaptive_overlap),
            "affected_fixed_core_cases": len(normalized_fixed_overlap),
        },
        "expected_episodes_per_defender": {
            "adaptive": len(attacked),
            "adaptive_primary_metric_eligible": (
                len(attacked) - len(metric_exclusions)
            ),
            "clean": len(clean),
            "fixed_per_template": len(fixed_core),
            "fixed_templates": FIXED_TEMPLATES,
            "total": len(attacked) + len(clean) + FIXED_TEMPLATES * len(fixed_core),
        },
    }
    stats_path = panel_dir / "official1514_stats.json"
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
