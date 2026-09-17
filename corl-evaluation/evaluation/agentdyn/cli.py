from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import subprocess
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from cotrain.fixed_templates import TEMPLATE_NAMES

from .dataset import EvalCase, load_eval_cases
from .report import load_jsonl, write_summary
from .runner import AgentDynEpisodeRunner, episode_id
from .types import AttackSpec, EndpointConfig, EvalSettings, expand_env

logger = logging.getLogger(__name__)

EVALUATOR_PROTOCOL_VERSION = "official1514-v2-checkpoint-aligned"


def _training_root() -> Path:
    return Path(
        os.environ.get("CORL_TRAINING_ROOT", Path(__file__).resolve().parents[3])
    ).resolve()


def _read_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("evaluation config must contain a YAML object")
    return expand_env(raw)


def _endpoints(raw: dict[str, Any]) -> tuple[dict[str, EndpointConfig], set[str], set[str]]:
    defenders = {
        label: EndpointConfig.from_dict(label, values) for label, values in (raw.get("defenders") or {}).items()
    }
    attackers = {
        label: EndpointConfig.from_dict(label, values) for label, values in (raw.get("attackers") or {}).items()
    }
    overlap = set(defenders) & set(attackers)
    if overlap:
        raise ValueError(f"endpoint labels must be globally unique: {sorted(overlap)}")
    return {**defenders, **attackers}, set(defenders), set(attackers)


def _defender_specs(
    raw: dict[str, Any],
    defender_labels: set[str],
    attacker_labels: set[str],
) -> tuple[list[str], list[AttackSpec]]:
    section = raw.get("defender_benchmark") or {}
    selected_defenders = list(section.get("defenders") or sorted(defender_labels))
    unknown = set(selected_defenders) - defender_labels
    if unknown:
        raise ValueError(f"unknown defender endpoints: {sorted(unknown)}")
    attacks: list[AttackSpec] = []
    if section.get("clean", True):
        attacks.append(AttackSpec(kind="clean", label="clean"))
    if section.get("dataset_static", False):
        attacks.append(AttackSpec(kind="dataset", label="dataset_static"))
    templates = section.get("fixed_templates") or []
    if templates == "all":
        templates = TEMPLATE_NAMES
    for template in templates:
        if template not in TEMPLATE_NAMES:
            raise ValueError(f"unknown fixed template: {template}")
        attacks.append(
            AttackSpec(
                kind="fixed",
                label=f"fixed:{template}",
                template_name=str(template),
            )
        )
    for attacker in section.get("adaptive_attackers") or []:
        if attacker not in attacker_labels:
            raise ValueError(f"unknown adaptive attacker endpoint: {attacker}")
        attacks.append(
            AttackSpec(
                kind="adaptive",
                label=f"adaptive:{attacker}",
                attacker_label=str(attacker),
            )
        )
    if not attacks:
        raise ValueError("defender_benchmark selected no attacks")
    return selected_defenders, attacks


def _attacker_specs(
    raw: dict[str, Any],
    defender_labels: set[str],
    attacker_labels: set[str],
) -> tuple[list[str], list[AttackSpec]]:
    section = raw.get("attacker_benchmark") or {}
    selected_defenders = list(section.get("fixed_defenders") or [])
    selected_attackers = list(section.get("attackers") or [])
    unknown_defenders = set(selected_defenders) - defender_labels
    unknown_attackers = set(selected_attackers) - attacker_labels
    if unknown_defenders:
        raise ValueError(f"unknown fixed defender endpoints: {sorted(unknown_defenders)}")
    if unknown_attackers:
        raise ValueError(f"unknown attacker endpoints: {sorted(unknown_attackers)}")
    if not selected_defenders or not selected_attackers:
        raise ValueError("attacker_benchmark requires fixed_defenders and attackers")
    return selected_defenders, [
        AttackSpec(kind="adaptive", label=f"adaptive:{label}", attacker_label=label) for label in selected_attackers
    ]


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_training_root(), text=True
        ).strip()
    except Exception:
        return None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evaluator_code_sha256() -> str:
    """Hash every repository source file that can change metric semantics."""

    digest = hashlib.sha256()
    repository_root = _training_root()
    package_dir = Path(__file__).resolve().parent
    semantic_sources = list(package_dir.glob("*.py"))
    semantic_sources.extend(
        repository_root / "cotrain" / name
        for name in (
            "adaptive_attack_context.py",
            "attack_contract.py",
            "attacker_sft_contract.py",
            "fixed_templates.py",
            "ppo_correctness.py",
        )
    )
    semantic_sources.extend(
        (repository_root / "AgentDyn" / "src" / "agentdojo").rglob("*.py")
    )
    for path in sorted(set(semantic_sources)):
        digest.update(path.relative_to(repository_root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _validate_checkpoint_metadata(endpoints: dict[str, EndpointConfig]) -> None:
    for label, endpoint in endpoints.items():
        if not endpoint.checkpoint:
            continue
        if "global_step_N" in endpoint.checkpoint:
            raise ValueError(
                f"endpoint {label} still contains the moving checkpoint placeholder "
                "global_step_N; replace it with one frozen step"
            )
        checkpoint = Path(endpoint.checkpoint)
        if checkpoint.is_absolute() and not checkpoint.exists():
            raise FileNotFoundError(f"checkpoint metadata for {label} does not exist: {checkpoint}")
        if checkpoint.is_dir() and not (checkpoint / "config.json").exists():
            raise FileNotFoundError(f"checkpoint metadata for {label} has no config.json: {checkpoint}")


def _planned_episode_ids(
    cases: list[EvalCase],
    defenders: list[str],
    attacks: list[AttackSpec],
    generation_seeds: tuple[int, ...] | list[int],
) -> list[str]:
    identifiers = [
        episode_id(case, defender, attack, generation_seed)
        for case in cases
        for generation_seed in generation_seeds
        for attack in attacks
        if case.is_clean == (attack.kind == "clean")
        for defender in defenders
    ]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("evaluation plan contains duplicate episode IDs")
    return sorted(identifiers)


def _identifier_set_sha256(identifiers: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(identifiers)).encode()).hexdigest()


def _rewrite_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Atomically compact a resumable JSONL after dropping retryable rows."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            for row in rows:
                temporary.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        temporary_path.replace(path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _prepare_resume_rows(output_path: Path) -> list[dict[str, Any]]:
    """Keep successful records and remove infra failures so resume can retry."""

    existing = load_jsonl(output_path)
    episode_ids = [str(row["episode_id"]) for row in existing]
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("resume output contains duplicate episode IDs")
    retry_ids = {
        str(row["episode_id"])
        for row in existing
        if row.get("infra_error")
    }
    if not retry_ids:
        return existing

    retained = [
        row for row in existing if str(row["episode_id"]) not in retry_ids
    ]
    _rewrite_jsonl(output_path, retained)
    trace_path = output_path.with_name("traces.jsonl")
    if trace_path.exists():
        traces = [
            row
            for row in load_jsonl(trace_path)
            if str(row.get("episode_id") or "") not in retry_ids
        ]
        _rewrite_jsonl(trace_path, traces)
    logger.info(
        "resume compacted %d retryable infrastructure-error rows",
        len(retry_ids),
    )
    return retained


async def _run_jobs(
    *,
    benchmark_kind: str,
    runner: AgentDynEpisodeRunner,
    cases: list[EvalCase],
    defenders: list[str],
    attacks: list[AttackSpec],
    output_path: Path,
    resume: bool,
) -> tuple[list[dict[str, Any]], int]:
    existing = _prepare_resume_rows(output_path) if resume else []
    completed = {str(row["episode_id"]) for row in existing}
    jobs: list[tuple[EvalCase, str, AttackSpec, int]] = []
    # Interleave endpoint combinations instead of exhausting one defender at
    # a time.  The previous defender -> attack -> case ordering left most
    # served models idle for long runs (especially sampled attacker panels).
    # This ordering preserves the exact episode set and IDs while allowing the
    # first concurrency window to fan out across defenders and attackers.
    for case in cases:
        for generation_seed in runner.settings.generation_seeds:
            for attack in attacks:
                if case.is_clean != (attack.kind == "clean"):
                    continue
                for defender in defenders:
                    if episode_id(case, defender, attack, generation_seed) not in completed:
                        jobs.append((case, defender, attack, generation_seed))

    logger.info("planned=%d already_complete=%d", len(jobs), len(existing))
    semaphore = asyncio.Semaphore(runner.settings.concurrency)

    async def one(job):
        case, defender, attack, generation_seed = job
        async with semaphore:
            return await runner.run(
                case,
                defender_label=defender,
                attack=attack,
                generation_seed=generation_seed,
                benchmark_kind=benchmark_kind,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if resume else "w"
    newly_written = 0
    trace_path = output_path.with_name("traces.jsonl")
    trace_mode = "a" if resume else "w"
    with (
        output_path.open(mode, encoding="utf-8", buffering=1) as handle,
        trace_path.open(trace_mode, encoding="utf-8", buffering=1) as trace_handle,
    ):
        pending = [asyncio.create_task(one(job)) for job in jobs]
        for future in asyncio.as_completed(pending):
            row = await future
            trace = {
                "episode_id": row["episode_id"],
                "defender_trajectory": row.pop("defender_trajectory", []),
                "attacker_messages": row.pop("attacker_messages", []),
                "injection_records": row.pop("injection_records", []),
                "function_trace": row.pop("function_trace", []),
            }
            save_trace = runner.settings.trace_policy == "all" or (
                runner.settings.trace_policy == "failures"
                and (
                    bool(row.get("asr_success"))
                    or row.get("utility_success") is False
                    or bool(row.get("infra_error"))
                    or bool(row.get("evaluator_error"))
                    or bool(row.get("model_failure"))
                )
            )
            row["trace_saved"] = save_trace
            if save_trace:
                trace_handle.write(json.dumps(trace, ensure_ascii=False, default=str) + "\n")
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            existing.append(row)
            newly_written += 1
            if newly_written % 25 == 0:
                logger.info("completed=%d/%d", newly_written, len(jobs))
    return existing, newly_written


def _parser(benchmark_kind: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Held-out AgentDyn {benchmark_kind} benchmark")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--panel", default=None)
    parser.add_argument("--suite", action="append", dest="suites")
    parser.add_argument("--limit-per-suite", type=int, default=None)
    parser.add_argument("--selection-seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def run_cli(benchmark_kind: str, argv: list[str] | None = None) -> int:
    args = _parser(benchmark_kind).parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    raw = _read_config(args.config)
    settings = EvalSettings.from_dict(raw.get("settings"))
    endpoints, defender_labels, attacker_labels = _endpoints(raw)
    if benchmark_kind == "defender":
        defenders, attacks = _defender_specs(raw, defender_labels, attacker_labels)
    else:
        defenders, attacks = _attacker_specs(raw, defender_labels, attacker_labels)

    dataset_path = Path(raw.get("dataset", {}).get("parquet", "training_data/agentdyn_val.parquet"))
    panel_path = args.panel or raw.get("dataset", {}).get("panel")
    cases = load_eval_cases(
        dataset_path,
        panel_path=panel_path,
        suites=args.suites,
        limit_per_suite=args.limit_per_suite,
        seed=args.selection_seed,
    )
    output_dir = Path(
        args.output_dir or raw.get("output_dir") or f"eval_results/{benchmark_kind}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    output_path = output_dir / "episodes.jsonl"
    if output_path.exists() and not args.resume and not args.plan_only:
        raise FileExistsError(f"{output_path} exists; choose a new output directory or use --resume")

    clean_count = sum(case.is_clean for case in cases)
    attacked_count = len(cases) - clean_count
    planned_episode_ids = _planned_episode_ids(
        cases,
        defenders,
        attacks,
        settings.generation_seeds,
    )
    planned = len(planned_episode_ids)
    logger.info(
        "cases=%d clean=%d attacked=%d defenders=%d attacks=%d planned_episodes=%d",
        len(cases),
        clean_count,
        attacked_count,
        len(defenders),
        len(attacks),
        planned,
    )
    if args.plan_only:
        print(
            json.dumps(
                {
                    "benchmark_kind": benchmark_kind,
                    "cases": len(cases),
                    "clean_cases": clean_count,
                    "attacked_cases": attacked_count,
                    "defenders": defenders,
                    "attacks": [asdict(attack) for attack in attacks],
                    "planned_episodes": planned,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    _validate_checkpoint_metadata(endpoints)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "benchmark_kind": benchmark_kind,
        "evaluator_protocol_version": EVALUATOR_PROTOCOL_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": _git_commit(),
        "evaluator_code_sha256": _evaluator_code_sha256(),
        "dataset": str(dataset_path.resolve()),
        "dataset_sha256": _file_sha256(dataset_path),
        "panel": str(Path(panel_path).resolve()) if panel_path else None,
        "panel_sha256": _file_sha256(Path(panel_path)) if panel_path else None,
        "settings": asdict(settings),
        "endpoints": {label: endpoints[label].public_dict() for label in sorted(endpoints)},
        "defenders": defenders,
        "attacks": [asdict(attack) for attack in attacks],
        "case_counts": {"clean": clean_count, "attacked": attacked_count},
        "case_selection": {
            "suites": sorted(args.suites or []),
            "limit_per_suite": args.limit_per_suite,
            "selection_seed": args.selection_seed,
            "selected_case_ids_sha256": _identifier_set_sha256(
                [case.case_id for case in cases]
            ),
        },
        "planned_episodes": planned,
        "planned_episode_ids_sha256": _identifier_set_sha256(
            planned_episode_ids
        ),
    }
    fingerprint_fields = {
        key: manifest[key]
        for key in (
            "benchmark_kind",
            "evaluator_protocol_version",
            "evaluator_code_sha256",
            "dataset_sha256",
            "panel_sha256",
            "settings",
            "endpoints",
            "defenders",
            "attacks",
            "case_counts",
            "case_selection",
            "planned_episodes",
            "planned_episode_ids_sha256",
        )
    }
    manifest["evaluation_fingerprint"] = hashlib.sha256(
        json.dumps(fingerprint_fields, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest_path = output_dir / "eval_manifest.json"
    if args.resume:
        if not manifest_path.exists():
            raise FileNotFoundError(f"cannot resume without {manifest_path}")
        with manifest_path.open(encoding="utf-8") as handle:
            old_manifest = json.load(handle)
        if old_manifest.get("evaluation_fingerprint") != manifest["evaluation_fingerprint"]:
            raise ValueError(
                "resume configuration/code/data/checkpoints differ from the existing manifest; "
                "use a new output directory"
            )
    else:
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)

    runner = AgentDynEpisodeRunner(endpoints, settings)
    rows, newly_written = asyncio.run(
        _run_jobs(
            benchmark_kind=benchmark_kind,
            runner=runner,
            cases=cases,
            defenders=defenders,
            attacks=attacks,
            output_path=output_path,
            resume=args.resume,
        )
    )
    write_summary(output_dir, rows, benchmark_kind)
    logger.info("finished new=%d total=%d output=%s", newly_written, len(rows), output_dir)
    return 0


def defender_main(argv: list[str] | None = None) -> int:
    return run_cli("defender", argv)


def attacker_main(argv: list[str] | None = None) -> int:
    return run_cli("attacker", argv)
