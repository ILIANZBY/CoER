#!/usr/bin/env python3
"""Validate and summarize the three-model sharded InjecAgent OOD run.

The benchmark writes newline-delimited JSON despite using a ``.json`` suffix.
This reporter never changes the result tree.  It validates every record against
the vendored canonical inputs, including the deterministic ``index % 3`` shard
assignment, before using the record in metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/ood/injecagent"
DEFAULT_CANONICAL_DATA_DIR = (
    PROJECT_ROOT / "third_party/InjecAgent/data"
)

MODELS = ("base", "corl")
SETTINGS = ("base", "enhanced")
ATTACKS = ("dh", "ds")
EXPECTED_COUNTS = {"dh": 510, "ds": 544}
NUM_SHARDS = 3
IGNORED_SHARD_DIRS = frozenset({"shard_00_of_510"})
EVAL_LABELS = ("succ", "unsucc", "invalid")


class DuplicateJSONKeyError(ValueError):
    """Raised when one JSON object contains the same key more than once."""


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise DuplicateJSONKeyError(f"duplicate JSON key: {key!r}")
        output[key] = value
    return output


def _strict_json_loads(text: str) -> Any:
    return json.loads(text, object_pairs_hook=_strict_object)


def _canonical_key(row: Mapping[str, Any], keys: Sequence[str]) -> str:
    projection = {key: row[key] for key in keys}
    return json.dumps(
        projection,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _examples(values: Iterable[int | str], *, limit: int = 5) -> str:
    selected = [str(value) for value in list(values)[:limit]]
    return ", ".join(selected)


@dataclass(frozen=True)
class CanonicalDataset:
    attack: str
    setting: str
    keys: tuple[str, ...]
    rows: tuple[dict[str, Any], ...]
    index_by_key: dict[str, int]


@dataclass(frozen=True)
class Occurrence:
    shard_index: int
    line_number: int
    path: Path
    row: dict[str, Any]
    result_schema_valid: bool


def _load_canonical_dataset(
    data_dir: Path,
    *,
    attack: str,
    setting: str,
    expected_count: int,
) -> CanonicalDataset:
    path = data_dir / f"test_cases_{attack}_{setting}.json"
    try:
        value = _strict_json_loads(path.read_text(encoding="utf-8", errors="strict"))
    except (OSError, UnicodeError, json.JSONDecodeError, DuplicateJSONKeyError) as error:
        raise ValueError(f"cannot read canonical dataset {path}: {type(error).__name__}") from None
    if not isinstance(value, list):
        raise ValueError(f"canonical dataset is not a JSON array: {path}")
    if len(value) != expected_count:
        raise ValueError(
            f"canonical dataset {path} has {len(value)} rows, expected {expected_count}"
        )
    if not value or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"canonical dataset contains a non-object row: {path}")

    keys = tuple(sorted(value[0]))
    if any(tuple(sorted(row)) != keys for row in value):
        raise ValueError(f"canonical dataset rows have inconsistent fields: {path}")
    index_by_key: dict[str, int] = {}
    for index, row in enumerate(value):
        identity = _canonical_key(row, keys)
        if identity in index_by_key:
            raise ValueError(
                f"canonical dataset has duplicate cases at indexes "
                f"{index_by_key[identity]} and {index}: {path}"
            )
        index_by_key[identity] = index
    return CanonicalDataset(
        attack=attack,
        setting=setting,
        keys=keys,
        rows=tuple(value),
        index_by_key=index_by_key,
    )


def _read_jsonl(
    path: Path,
) -> tuple[list[tuple[int, dict[str, Any]]], list[str], int]:
    """Read one immutable byte snapshot and reject every malformed record."""

    try:
        raw = path.read_bytes()
    except OSError as error:
        return [], [f"cannot read {path}: {type(error).__name__}"], 0
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        return [], [f"invalid UTF-8 in {path} at byte {error.start}"], 1

    rows: list[tuple[int, dict[str, Any]]] = []
    errors: list[str] = []
    malformed = 0
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = _strict_json_loads(line)
        except (json.JSONDecodeError, DuplicateJSONKeyError) as error:
            malformed += 1
            errors.append(
                f"malformed JSON at {path}:{line_number} ({type(error).__name__})"
            )
            continue
        if not isinstance(value, dict):
            malformed += 1
            errors.append(f"non-object JSON record at {path}:{line_number}")
            continue
        rows.append((line_number, value))
    return rows, errors, malformed


def _validate_result_schema(
    row: Mapping[str, Any],
    *,
    attack: str,
    location: str,
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if not isinstance(row.get("output"), str):
        errors.append(f"missing/non-string output at {location}")
    first_eval = row.get("eval")
    if first_eval not in EVAL_LABELS:
        errors.append(f"invalid eval={first_eval!r} at {location}")

    has_step2 = "eval Step 2" in row or "output Step 2" in row
    if attack == "dh":
        if has_step2:
            errors.append(f"unexpected DS step2 fields in DH result at {location}")
    elif first_eval == "succ":
        second_eval = row.get("eval Step 2")
        if second_eval not in EVAL_LABELS:
            errors.append(f"invalid/missing eval Step 2={second_eval!r} at {location}")
        if not isinstance(row.get("output Step 2"), str):
            errors.append(f"missing/non-string output Step 2 at {location}")
    elif has_step2:
        errors.append(
            f"unexpected DS step2 fields when first-step eval={first_eval!r} at {location}"
        )
    return not errors, errors


def _expected_result_path(
    root: Path,
    *,
    model: str,
    setting: str,
    shard_index: int,
    attack: str,
    num_shards: int,
) -> Path:
    shard = f"shard_{shard_index:02d}_of_{num_shards:02d}"
    return (
        root
        / model
        / setting
        / shard
        / "injecagent"
        / f"prompted_OpenAICompatible_{model}_InjecAgent"
        / f"test_cases_{attack}_{setting}.json"
    )


def _validate_layout(
    root: Path,
    *,
    models: Sequence[str],
    settings: Sequence[str],
    num_shards: int,
) -> tuple[list[str], list[str], int]:
    errors: list[str] = []
    incomplete: list[str] = []
    present_shards = 0
    if not root.is_dir():
        return [], [f"missing result root: {root}"], 0

    actual_models = {
        path.name
        for path in root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    }
    expected_models = set(models)
    for name in sorted(actual_models - expected_models):
        errors.append(f"unexpected model directory: {root / name}")
    for model in models:
        model_root = root / model
        if not model_root.is_dir():
            incomplete.append(f"missing model directory: {model_root}")
            continue
        actual_settings = {
            path.name
            for path in model_root.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        }
        for name in sorted(actual_settings - set(settings)):
            errors.append(f"unexpected setting directory: {model_root / name}")
        for setting in settings:
            setting_root = model_root / setting
            if not setting_root.is_dir():
                incomplete.append(f"missing setting directory: {setting_root}")
                continue
            expected_shards = {
                f"shard_{index:02d}_of_{num_shards:02d}"
                for index in range(num_shards)
            }
            actual_shards = {
                path.name
                for path in setting_root.iterdir()
                if path.is_dir() and not path.name.startswith(".")
            }
            unexpected = actual_shards - expected_shards - IGNORED_SHARD_DIRS
            for name in sorted(unexpected):
                errors.append(f"unexpected shard directory: {setting_root / name}")
            for name in sorted(expected_shards):
                shard_path = setting_root / name
                if shard_path.is_dir():
                    present_shards += 1
                else:
                    incomplete.append(f"missing shard directory: {shard_path}")
    return errors, incomplete, present_shards


def _empty_eval_counts() -> dict[str, int]:
    return {label: 0 for label in EVAL_LABELS}


def _summarize_combination(
    root: Path,
    *,
    model: str,
    setting: str,
    canonical: CanonicalDataset,
    num_shards: int,
) -> tuple[dict[str, Any], list[str], list[str]]:
    attack = canonical.attack
    expected = len(canonical.rows)
    errors: list[str] = []
    incomplete: list[str] = []
    occurrences: dict[int, list[Occurrence]] = defaultdict(list)
    parsed_objects = 0
    malformed_json = 0
    unexpected_rows = 0
    unexpected_examples: list[str] = []
    invalid_schema_rows = 0
    invalid_schema_examples: list[str] = []
    files_present = 0

    for shard_index in range(num_shards):
        path = _expected_result_path(
            root,
            model=model,
            setting=setting,
            shard_index=shard_index,
            attack=attack,
            num_shards=num_shards,
        )
        if not path.is_file():
            incomplete.append(f"missing result file: {path}")
            continue
        files_present += 1
        rows, parse_errors, malformed = _read_jsonl(path)
        errors.extend(parse_errors)
        malformed_json += malformed
        parsed_objects += len(rows)
        for line_number, row in rows:
            location = f"{path}:{line_number}"
            missing_keys = [key for key in canonical.keys if key not in row]
            if missing_keys:
                unexpected_rows += 1
                if len(unexpected_examples) < 5:
                    unexpected_examples.append(
                        f"{location} missing canonical fields {missing_keys[:3]}"
                    )
                continue
            identity = _canonical_key(row, canonical.keys)
            case_index = canonical.index_by_key.get(identity)
            if case_index is None:
                unexpected_rows += 1
                if len(unexpected_examples) < 5:
                    unexpected_examples.append(location)
                continue
            schema_valid, schema_errors = _validate_result_schema(
                row,
                attack=attack,
                location=location,
            )
            if not schema_valid:
                invalid_schema_rows += 1
                if len(invalid_schema_examples) < 5:
                    invalid_schema_examples.extend(
                        schema_errors[: 5 - len(invalid_schema_examples)]
                    )
            occurrences[case_index].append(
                Occurrence(
                    shard_index=shard_index,
                    line_number=line_number,
                    path=path,
                    row=row,
                    result_schema_valid=schema_valid,
                )
            )

    missing_indexes = [index for index in range(expected) if index not in occurrences]
    duplicate_indexes = [
        index for index, rows in occurrences.items() if len(rows) > 1
    ]
    duplicate_records = sum(
        len(rows) - 1 for rows in occurrences.values() if len(rows) > 1
    )
    misplaced_indexes = [
        index
        for index, rows in occurrences.items()
        if any(row.shard_index != index % num_shards for row in rows)
    ]

    prefix = f"{model}/{setting}/{attack}"
    if missing_indexes:
        incomplete.append(
            f"{prefix}: missing {len(missing_indexes)} canonical cases "
            f"(examples: {_examples(missing_indexes)})"
        )
    if duplicate_records:
        errors.append(
            f"{prefix}: {duplicate_records} duplicate records across "
            f"{len(duplicate_indexes)} case IDs (examples: {_examples(duplicate_indexes)})"
        )
    if misplaced_indexes:
        errors.append(
            f"{prefix}: {len(misplaced_indexes)} cases are in the wrong shard "
            f"(examples: {_examples(misplaced_indexes)})"
        )
    if unexpected_rows:
        errors.append(
            f"{prefix}: {unexpected_rows} rows do not match the canonical dataset "
            f"(examples: {'; '.join(unexpected_examples)})"
        )
    if invalid_schema_rows:
        errors.append(
            f"{prefix}: {invalid_schema_rows} rows have invalid result fields "
            f"(examples: {'; '.join(invalid_schema_examples)})"
        )

    # Metrics use only an unambiguous, correctly sharded, schema-valid record.
    trusted_rows: list[dict[str, Any]] = []
    for case_index, rows in occurrences.items():
        if len(rows) != 1:
            continue
        occurrence = rows[0]
        if occurrence.shard_index != case_index % num_shards:
            continue
        if occurrence.result_schema_valid:
            trusted_rows.append(occurrence.row)

    eval_counts = _empty_eval_counts()
    for row in trusted_rows:
        eval_counts[str(row["eval"])] += 1
    valid_denominator = eval_counts["succ"] + eval_counts["unsucc"]
    first_step_asr_valid = (
        eval_counts["succ"] / valid_denominator
        if valid_denominator
        else None
    )

    step2: dict[str, Any] | None = None
    if attack == "ds":
        step2 = {
            "eligible": eval_counts["succ"],
            "succ": 0,
            "unsucc": 0,
            "invalid": 0,
            "not_applicable": eval_counts["unsucc"] + eval_counts["invalid"],
        }
        for row in trusted_rows:
            if row["eval"] == "succ":
                step2[str(row["eval Step 2"])] += 1
        conditional_denominator = step2["succ"] + step2["unsucc"]
        final_denominator = conditional_denominator + eval_counts["unsucc"]
        step2["conditional_asr_valid"] = (
            step2["succ"] / conditional_denominator
            if conditional_denominator
            else None
        )
        step2["data_stealing_asr_valid"] = (
            step2["succ"] / final_denominator if final_denominator else None
        )

    summary: dict[str, Any] = {
        "model": model,
        "setting": setting,
        "attack": attack,
        "expected_records": expected,
        "files_present": files_present,
        "files_expected": num_shards,
        "parsed_json_objects": parsed_objects,
        "canonical_unique_records": len(occurrences),
        "trusted_metric_records": len(trusted_rows),
        "missing_records": len(missing_indexes),
        "duplicate_records": duplicate_records,
        "unexpected_records": unexpected_rows,
        "misplaced_case_ids": len(misplaced_indexes),
        "malformed_json_records": malformed_json,
        "invalid_result_schema_records": invalid_schema_rows,
        "eval": eval_counts,
        "first_step_asr_valid": first_step_asr_valid,
        "ds_step2": step2,
        "complete": (
            files_present == num_shards
            and parsed_objects == expected
            and len(occurrences) == expected
            and len(trusted_rows) == expected
            and not missing_indexes
            and not duplicate_records
            and not unexpected_rows
            and not misplaced_indexes
            and not malformed_json
            and not invalid_schema_rows
        ),
    }
    return summary, errors, incomplete


def build_report(
    output_dir: str | Path,
    *,
    canonical_data_dir: str | Path = DEFAULT_CANONICAL_DATA_DIR,
    models: Sequence[str] = MODELS,
    settings: Sequence[str] = SETTINGS,
    expected_counts: Mapping[str, int] = EXPECTED_COUNTS,
    num_shards: int = NUM_SHARDS,
) -> dict[str, Any]:
    root = Path(output_dir).resolve()
    data_root = Path(canonical_data_dir).resolve()
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if set(expected_counts) != set(ATTACKS):
        raise ValueError(f"expected_counts must contain exactly {ATTACKS}")

    canonical: dict[tuple[str, str], CanonicalDataset] = {}
    for setting in settings:
        for attack in ATTACKS:
            canonical[(setting, attack)] = _load_canonical_dataset(
                data_root,
                attack=attack,
                setting=setting,
                expected_count=int(expected_counts[attack]),
            )

    errors, incomplete, present_shards = _validate_layout(
        root,
        models=models,
        settings=settings,
        num_shards=num_shards,
    )
    combinations: list[dict[str, Any]] = []
    for model in models:
        for setting in settings:
            for attack in ATTACKS:
                summary, combination_errors, combination_incomplete = (
                    _summarize_combination(
                        root,
                        model=model,
                        setting=setting,
                        canonical=canonical[(setting, attack)],
                        num_shards=num_shards,
                    )
                )
                combinations.append(summary)
                errors.extend(combination_errors)
                incomplete.extend(combination_incomplete)

    expected_shards = len(models) * len(settings) * num_shards
    expected_records = len(models) * len(settings) * sum(
        int(expected_counts[attack]) for attack in ATTACKS
    )
    trusted_records = sum(row["trusted_metric_records"] for row in combinations)
    present_files = sum(row["files_present"] for row in combinations)
    parsed_objects = sum(row["parsed_json_objects"] for row in combinations)
    malformed = sum(row["malformed_json_records"] for row in combinations)
    duplicates = sum(row["duplicate_records"] for row in combinations)
    missing = sum(row["missing_records"] for row in combinations)
    unexpected = sum(row["unexpected_records"] for row in combinations)
    misplaced = sum(row["misplaced_case_ids"] for row in combinations)
    invalid_schema = sum(
        row["invalid_result_schema_records"] for row in combinations
    )

    if errors:
        status = "invalid"
    elif incomplete:
        status = "incomplete"
    else:
        status = "complete"
    return {
        "schema_version": "ood-injecagent-report-v1",
        "status": status,
        "output_dir": str(root),
        "canonical_data_dir": str(data_root),
        "expected": {
            "models": list(models),
            "settings": list(settings),
            "num_shards": num_shards,
            "shard_units": expected_shards,
            "result_files": expected_shards * len(ATTACKS),
            "dh_per_model_setting": int(expected_counts["dh"]),
            "ds_per_model_setting": int(expected_counts["ds"]),
            "total_records": expected_records,
        },
        "integrity": {
            "present_shard_units": present_shards,
            "present_result_files": present_files,
            "parsed_json_objects": parsed_objects,
            "trusted_metric_records": trusted_records,
            "missing_records": missing,
            "duplicate_records": duplicates,
            "unexpected_records": unexpected,
            "misplaced_case_ids": misplaced,
            "malformed_json_records": malformed,
            "invalid_result_schema_records": invalid_schema,
            "errors": errors,
            "incomplete_reasons": incomplete,
            "ignored_shard_directories": sorted(IGNORED_SHARD_DIRS),
        },
        "combinations": combinations,
    }


def _format_rate(rate: float | None) -> str:
    return "-" if rate is None else f"{100.0 * rate:.1f}%"


def _print_text_report(report: Mapping[str, Any], *, max_issues: int) -> None:
    print(
        "model              setting   attack  coverage   eval(s/u/i)       "
        "ASR-v(DH/S1)  DS-step2(s/u/i)"
    )
    for row in report["combinations"]:
        counts = row["eval"]
        eval_text = f"{counts['succ']}/{counts['unsucc']}/{counts['invalid']}"
        step2 = row["ds_step2"]
        step2_text = "-"
        if step2 is not None:
            step2_text = f"{step2['succ']}/{step2['unsucc']}/{step2['invalid']}"
        coverage = f"{row['trusted_metric_records']}/{row['expected_records']}"
        print(
            f"{row['model']:<18} {row['setting']:<9} {row['attack']:<7} "
            f"{coverage:<10} {eval_text:<17} "
            f"{_format_rate(row['first_step_asr_valid']):<13} "
            f"{step2_text}"
        )

    integrity = report["integrity"]
    expected = report["expected"]
    print()
    print(
        f"status={report['status']} shards={integrity['present_shard_units']}/"
        f"{expected['shard_units']} files={integrity['present_result_files']}/"
        f"{expected['result_files']} trusted_records={integrity['trusted_metric_records']}/"
        f"{expected['total_records']} parsed={integrity['parsed_json_objects']} "
        f"missing={integrity['missing_records']} duplicates={integrity['duplicate_records']} "
        f"unexpected={integrity['unexpected_records']} misplaced={integrity['misplaced_case_ids']} "
        f"malformed_json={integrity['malformed_json_records']} "
        f"invalid_schema={integrity['invalid_result_schema_records']}"
    )
    issues = [
        *(f"ERROR: {item}" for item in integrity["errors"]),
        *(f"INCOMPLETE: {item}" for item in integrity["incomplete_reasons"]),
    ]
    for issue in issues[:max_issues]:
        print(issue)
    if len(issues) > max_issues:
        print(f"... {len(issues) - max_issues} additional issues omitted")


def exit_status(report: Mapping[str, Any], *, require_complete: bool) -> int:
    if report["status"] == "invalid":
        return 2
    if require_complete and report["status"] != "complete":
        return 2
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"result root (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="return status 2 unless all 6,324 expected records pass integrity checks",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the complete machine-readable report instead of the text table",
    )
    parser.add_argument(
        "--max-issues",
        type=int,
        default=40,
        help="maximum validation issues shown in text mode",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_issues < 0:
        raise ValueError("--max-issues must be non-negative")
    try:
        report = build_report(args.output_dir)
    except ValueError as error:
        print(f"fatal validation error: {error}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_text_report(report, max_issues=args.max_issues)
    return exit_status(report, require_complete=args.require_complete)


if __name__ == "__main__":
    raise SystemExit(main())
