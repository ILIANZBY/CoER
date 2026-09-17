from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from .dataset import EvalCase, load_eval_cases, write_panel


def _stratified(cases: list[EvalCase], per_suite: int, seed: int) -> list[EvalCase]:
    rng = random.Random(seed)
    groups: dict[str, list[EvalCase]] = defaultdict(list)
    for case in cases:
        groups[case.suite_name].append(case)
    selected: list[EvalCase] = []
    for suite_name in sorted(groups):
        group = sorted(groups[suite_name], key=lambda item: item.case_id)
        rng.shuffle(group)
        selected.extend(group[:per_suite])
    return selected


def _stratified_attacks(cases: list[EvalCase], per_suite: int, seed: int) -> list[EvalCase]:
    """Round-robin injection tasks so a pilot is not dominated by one goal."""

    rng = random.Random(seed)
    by_suite_task: dict[str, dict[str, list[EvalCase]]] = defaultdict(lambda: defaultdict(list))
    for case in cases:
        by_suite_task[case.suite_name][str(case.injection_task_id)].append(case)
    selected: list[EvalCase] = []
    for suite_name in sorted(by_suite_task):
        buckets = by_suite_task[suite_name]
        for bucket in buckets.values():
            bucket.sort(key=lambda item: item.case_id)
            rng.shuffle(bucket)
        suite_selected: list[EvalCase] = []
        task_names = sorted(buckets)
        while len(suite_selected) < per_suite and task_names:
            remaining: list[str] = []
            for task_name in task_names:
                if buckets[task_name] and len(suite_selected) < per_suite:
                    suite_selected.append(buckets[task_name].pop())
                if buckets[task_name]:
                    remaining.append(task_name)
            task_names = remaining
        selected.extend(suite_selected)
    return selected


def _diverse_attack_core(cases: list[EvalCase], per_suite: int, seed: int) -> list[EvalCase]:
    """Select a small attacked core while spreading both users and goals."""

    if per_suite <= 0:
        raise ValueError("per_suite must be positive")
    if any(case.is_clean for case in cases):
        raise ValueError("_diverse_attack_core accepts attacked cases only")

    rng = random.Random(seed)
    groups: dict[str, list[EvalCase]] = defaultdict(list)
    for case in cases:
        groups[case.suite_name].append(case)

    selected: list[EvalCase] = []
    for suite_name in sorted(groups):
        remaining = sorted(groups[suite_name], key=lambda item: item.case_id)
        if len(remaining) < per_suite:
            raise ValueError(
                f"suite {suite_name} needs {per_suite} attacked cases but only has {len(remaining)}"
            )
        rng.shuffle(remaining)
        random_rank = {case.case_id: rank for rank, case in enumerate(remaining)}
        user_counts: Counter[str] = Counter()
        task_counts: Counter[str] = Counter()
        for _ in range(per_suite):
            chosen = min(
                remaining,
                key=lambda item: (
                    user_counts[item.user_task_id],
                    task_counts[str(item.injection_task_id)],
                    random_rank[item.case_id],
                    item.case_id,
                ),
            )
            remaining.remove(chosen)
            selected.append(chosen)
            user_counts[chosen.user_task_id] += 1
            task_counts[str(chosen.injection_task_id)] += 1
    return selected


def _stratified_total(cases: list[EvalCase], total: int, seed: int) -> list[EvalCase]:
    """Select exactly ``total`` cases while round-robining across suites.

    Unlike ``_stratified``, this supports totals that are not divisible by the
    number of suites.  The deterministic suite rotation prevents the extra
    cases from always going to alphabetically early suites.
    """

    rng = random.Random(seed)
    groups: dict[str, list[EvalCase]] = defaultdict(list)
    for case in cases:
        groups[case.suite_name].append(case)
    for group in groups.values():
        group.sort(key=lambda item: item.case_id)
        rng.shuffle(group)
    suites = sorted(groups)
    rng.shuffle(suites)
    selected: list[EvalCase] = []
    while len(selected) < total:
        made_progress = False
        for suite_name in suites:
            if groups[suite_name] and len(selected) < total:
                selected.append(groups[suite_name].pop())
                made_progress = True
        if not made_progress:
            raise ValueError(f"requested {total} cases but only found {len(selected)}")
    return selected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build deterministic AgentDyn evaluation panels")
    parser.add_argument("--parquet", default="training_data/agentdyn_val.parquet")
    parser.add_argument("--train-parquet", default="training_data/agentdyn_train.parquet")
    parser.add_argument("--output-dir", default="evaluation/agentdyn/panels")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument(
        "--extension-output",
        default=None,
        help="Write a new 30-clean/30-attacked panel here without rebuilding the standard panels.",
    )
    parser.add_argument(
        "--exclude-panel",
        default=None,
        help="Exclude case IDs already present in this panel when building an extension panel.",
    )
    args = parser.parse_args(argv)

    cases = load_eval_cases(args.parquet)
    train_cases = load_eval_cases(args.train_parquet) if args.train_parquet else []
    clean = [case for case in cases if case.is_clean]
    attacked = [case for case in cases if not case.is_clean]
    if args.extension_output:
        excluded_ids: set[str] = set()
        if args.exclude_panel:
            excluded_ids = {
                case.case_id
                for case in load_eval_cases(args.parquet, panel_path=args.exclude_panel)
            }
        extension_clean = _stratified_total(
            [case for case in clean if case.case_id not in excluded_ids], 30, args.seed + 40
        )
        extension_attacked = _stratified_total(
            [case for case in attacked if case.case_id not in excluded_ids], 30, args.seed + 41
        )
        count = write_panel(Path(args.extension_output), extension_clean + extension_attacked)
        print(
            json.dumps(
                {
                    "output": str(Path(args.extension_output).resolve()),
                    "seed": args.seed,
                    "excluded_cases": len(excluded_ids),
                    "clean_cases": len(extension_clean),
                    "attacked_cases": len(extension_attacked),
                    "total": count,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    output_dir = Path(args.output_dir)
    latest30_clean = _stratified_total(clean, 30, args.seed + 30)
    latest30_attacked = _stratified_total(attacked, 30, args.seed + 31)
    latest30_static = _stratified_total(latest30_attacked, 10, args.seed + 32)
    latest30_fixed = _stratified_total(latest30_attacked, 10, args.seed + 33)
    counts = {
        "full": write_panel(output_dir / "full.jsonl", clean + attacked),
        "clean_full": write_panel(output_dir / "clean_full.jsonl", clean),
        "attacked_full": write_panel(output_dir / "attacked_full.jsonl", attacked),
        "smoke": write_panel(
            output_dir / "smoke.jsonl",
            _stratified(clean, 2, args.seed) + _stratified(attacked, 2, args.seed),
        ),
        "pilot": write_panel(
            output_dir / "pilot.jsonl",
            _stratified(clean, 16, args.seed) + _stratified_attacks(attacked, 64, args.seed),
        ),
        "latest30_core": write_panel(
            output_dir / "latest30_core.jsonl", latest30_clean + latest30_attacked
        ),
        "latest30_attacked": write_panel(
            output_dir / "latest30_attacked.jsonl", latest30_attacked
        ),
        "latest30_static10": write_panel(
            output_dir / "latest30_static10.jsonl", latest30_static
        ),
        "latest30_fixed10": write_panel(
            output_dir / "latest30_fixed10.jsonl", latest30_fixed
        ),
    }
    stats = {
        "source": str(Path(args.parquet).resolve()),
        "source_sha256": hashlib.sha256(Path(args.parquet).read_bytes()).hexdigest(),
        "seed": args.seed,
        "deduplicated_cases": len(cases),
        "clean_cases": len(clean),
        "attacked_cases": len(attacked),
        "clean_by_suite": Counter(case.suite_name for case in clean),
        "attacked_by_suite": Counter(case.suite_name for case in attacked),
        "train_case_overlap": len({case.case_id for case in cases} & {case.case_id for case in train_cases}),
        "panels": counts,
    }
    with (output_dir / "stats.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
