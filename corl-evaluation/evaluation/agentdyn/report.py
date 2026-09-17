from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> dict[str, float | int | None]:
    if total <= 0:
        return {"successes": successes, "total": total, "rate": None, "ci95_low": None, "ci95_high": None}
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total)) / denominator
    return {
        "successes": successes,
        "total": total,
        "rate": rate,
        "ci95_low": max(0.0, center - margin),
        "ci95_high": min(1.0, center + margin),
    }


def _rate(rows: list[dict[str, Any]], field: str) -> dict[str, float | int | None]:
    values = [bool(row[field]) for row in rows if row.get(field) is not None]
    return _wilson(sum(values), len(values))


def _summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    excluded = [row for row in rows if row.get("metric_exclusion_reason")]
    eligible = [
        row
        for row in rows
        if row.get("metric_eligible") and not row.get("metric_exclusion_reason")
    ]
    attacked = [row for row in eligible if row.get("attack_kind") != "clean"]
    reached = [row for row in attacked if row.get("injection_reached")]
    adaptive = [row for row in attacked if row.get("attack_kind") == "adaptive"]
    attacker_attempts = sum(int(row.get("attacker_attempts") or 0) for row in adaptive)
    attacker_valid_turns = sum(int(row.get("attacker_valid_turns") or 0) for row in adaptive)
    elapsed = [float(row.get("elapsed_seconds") or 0.0) for row in rows]
    return {
        "episodes": len(rows),
        "case_exclusions": len(excluded),
        "metric_eligible": len(eligible),
        "infrastructure_errors": sum(bool(row.get("infra_error")) for row in rows),
        "evaluator_errors": sum(bool(row.get("evaluator_error")) for row in rows),
        "model_failures": sum(bool(row.get("model_failure")) for row in rows),
        "clean_tsr": _rate([row for row in eligible if row.get("attack_kind") == "clean"], "utility_success"),
        "overall_asr": _rate(attacked, "asr_success"),
        "injection_reach_rate": _rate(attacked, "injection_reached"),
        "effective_asr": _rate(reached, "asr_success"),
        "attacked_tsr": _rate(attacked, "utility_success"),
        "safe_utility": _rate(attacked, "safe_utility"),
        "attacker_format_validity": _wilson(attacker_valid_turns, attacker_attempts),
        "defender_tool_format_violation_episodes": sum(
            int(row.get("defender_tool_format_violations") or 0) > 0 for row in eligible
        ),
        "defender_length_finish_episodes": sum(
            int(row.get("defender_length_finish_count") or 0) > 0 for row in eligible
        ),
        "attacker_length_finish_episodes": sum(
            int(row.get("attacker_length_finish_count") or 0) > 0 for row in eligible
        ),
        "tool_response_overflow_episodes": sum(int(row.get("tool_response_overflows") or 0) > 0 for row in eligible),
        "elapsed_seconds": sum(elapsed),
    }


def summarize(rows: Iterable[dict[str, Any]], benchmark_kind: str) -> dict[str, Any]:
    rows = list(rows)
    if benchmark_kind == "defender":
        group_fields = ("defender_label", "attack_label")
    elif benchmark_kind == "attacker":
        group_fields = ("attacker_label", "defender_label")
    else:
        raise ValueError(f"unknown benchmark kind: {benchmark_kind}")

    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    suites: dict[tuple[str, ...], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = tuple(str(row.get(field) or "") for field in group_fields)
        groups[key].append(row)
        suites[key][str(row.get("suite_name") or "unknown")].append(row)

    output_groups = []
    metric_names = (
        "clean_tsr",
        "overall_asr",
        "injection_reach_rate",
        "effective_asr",
        "attacked_tsr",
        "safe_utility",
        "attacker_format_validity",
    )
    for key in sorted(groups):
        entry = {field: value for field, value in zip(group_fields, key, strict=True)}
        entry.update(_summarize_group(groups[key]))
        entry["by_suite"] = {suite: _summarize_group(suite_rows) for suite, suite_rows in sorted(suites[key].items())}
        entry["suite_macro_rates"] = {
            metric: (
                sum(values) / len(values)
                if (
                    values := [
                        suite_summary[metric]["rate"]
                        for suite_summary in entry["by_suite"].values()
                        if suite_summary[metric]["rate"] is not None
                    ]
                )
                else None
            )
            for metric in metric_names
        }
        output_groups.append(entry)
    return {
        "benchmark_kind": benchmark_kind,
        "total_episode_records": len(rows),
        "group_fields": list(group_fields),
        "groups": output_groups,
    }


def _flatten_summary(summary: dict[str, Any]) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []
    group_fields = summary["group_fields"]
    metric_names = (
        "clean_tsr",
        "overall_asr",
        "injection_reach_rate",
        "effective_asr",
        "attacked_tsr",
        "safe_utility",
        "attacker_format_validity",
    )
    for group in summary["groups"]:
        common = {field: group[field] for field in group_fields}
        for suite_name, values in [("ALL", group), *sorted(group["by_suite"].items())]:
            for metric in metric_names:
                stats = values[metric]
                flat.append(
                    {
                        **common,
                        "suite_name": suite_name,
                        "metric": metric,
                        **stats,
                        "episodes": values["episodes"],
                        "case_exclusions": values["case_exclusions"],
                        "metric_eligible": values["metric_eligible"],
                        "infrastructure_errors": values["infrastructure_errors"],
                        "evaluator_errors": values["evaluator_errors"],
                        "model_failures": values["model_failures"],
                    }
                )
    return flat


def write_summary(output_dir: str | Path, rows: Iterable[dict[str, Any]], benchmark_kind: str) -> dict[str, Any]:
    output_dir = Path(output_dir)
    summary = summarize(rows, benchmark_kind)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    flat = _flatten_summary(summary)
    if flat:
        with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)
    return summary


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    path = Path(path)
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows
