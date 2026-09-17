from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    row_index: int
    suite_name: str
    user_task_id: str
    injection_task_id: str | None
    injection_vector_ids: tuple[str, ...]
    dataset_injections: dict[str, str]
    user_prompt: str
    metric_exclusion_reason: str | None = None

    @property
    def is_clean(self) -> bool:
        return self.injection_task_id is None

    def panel_record(self) -> dict:
        record = asdict(self)
        record.pop("dataset_injections")
        record.pop("user_prompt")
        if self.metric_exclusion_reason is None:
            record.pop("metric_exclusion_reason")
        return record


def _parse_json_mapping(value, field: str) -> dict:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError(f"{field} must decode to an object")
    return value


def _user_prompt(row, extra_info: dict) -> str:
    prompt = row.get("prompt", "[]")
    if isinstance(prompt, str):
        prompt = json.loads(prompt)
    if isinstance(prompt, list):
        for message in prompt:
            if isinstance(message, dict) and message.get("role") == "user":
                return str(message.get("content", ""))
    return str(extra_info.get("user_task_prompt", ""))


def load_eval_cases(
    parquet_path: str | Path,
    *,
    panel_path: str | Path | None = None,
    suites: Iterable[str] | None = None,
    limit_per_suite: int | None = None,
    seed: int = 0,
) -> list[EvalCase]:
    """Load and deduplicate the held-out AgentDyn split.

    The source parquet repeats clean rows across attack families.  Clean cases
    are deduplicated by ``(suite, user_task)`` and attacked cases by
    ``(suite, user_task, injection_task)`` so normal utility is not reweighted
    by an unrelated number of attack templates.
    """

    frame = pd.read_parquet(parquet_path)
    cases: list[EvalCase] = []
    seen: set[tuple[str, str, str]] = set()
    for row_index, row in frame.iterrows():
        extra = _parse_json_mapping(row["extra_info"], "extra_info")
        suite_name = str(extra.get("suite_name", ""))
        user_task_id = str(extra.get("user_task_id", ""))
        injection_task_id = extra.get("injection_task_id") or None
        if not suite_name or not user_task_id:
            raise ValueError(f"row {row_index} is missing suite_name or user_task_id")
        dedup_key = (suite_name, user_task_id, str(injection_task_id or ""))
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        injections = extra.get("injections") or {}
        if not isinstance(injections, dict):
            raise ValueError(f"row {row_index} injections must be an object")
        case_id = f"{suite_name}:{user_task_id}:{injection_task_id or 'clean'}"
        cases.append(
            EvalCase(
                case_id=case_id,
                row_index=int(row_index),
                suite_name=suite_name,
                user_task_id=user_task_id,
                injection_task_id=str(injection_task_id) if injection_task_id else None,
                injection_vector_ids=tuple(str(key) for key in injections),
                dataset_injections={str(key): str(value) for key, value in injections.items()},
                user_prompt=_user_prompt(row, extra),
            )
        )

    if panel_path is not None:
        panel_records: dict[str, dict] = {}
        with Path(panel_path).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    record = json.loads(line)
                    case_id = str(record["case_id"])
                    if case_id in panel_records:
                        raise ValueError(
                            f"duplicate case_id in panel at line {line_number}: {case_id}"
                        )
                    reason = record.get("metric_exclusion_reason")
                    if reason is not None and (
                        not isinstance(reason, str) or not reason.strip()
                    ):
                        raise ValueError(
                            "metric_exclusion_reason must be a non-empty string "
                            f"for panel case {case_id}"
                        )
                    panel_records[case_id] = record
        allowed = set(panel_records)
        unknown = allowed - {case.case_id for case in cases}
        if unknown:
            preview = ", ".join(sorted(unknown)[:5])
            raise ValueError(f"panel contains {len(unknown)} cases absent from parquet: {preview}")
        cases = [
            replace(
                case,
                metric_exclusion_reason=panel_records[case.case_id].get(
                    "metric_exclusion_reason"
                ),
            )
            for case in cases
            if case.case_id in allowed
        ]

    suite_filter = set(suites or ())
    if suite_filter:
        cases = [case for case in cases if case.suite_name in suite_filter]

    if limit_per_suite is not None:
        if limit_per_suite <= 0:
            raise ValueError("limit_per_suite must be positive")
        rng = random.Random(seed)
        by_suite: dict[str, list[EvalCase]] = {}
        for case in cases:
            by_suite.setdefault(case.suite_name, []).append(case)
        selected: list[EvalCase] = []
        for suite_name in sorted(by_suite):
            group = sorted(by_suite[suite_name], key=lambda item: item.case_id)
            rng.shuffle(group)
            selected.extend(group[:limit_per_suite])
        cases = selected

    return sorted(cases, key=lambda item: item.case_id)


def write_panel(path: str | Path, cases: Iterable[EvalCase]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for case in sorted(cases, key=lambda item: item.case_id):
            handle.write(json.dumps(case.panel_record(), ensure_ascii=False) + "\n")
            count += 1
    return count
