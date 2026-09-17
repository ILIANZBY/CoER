"""Historical checkpoint populations for adversarial co-evolution.

Population members are frozen vLLM replicas.  New checkpoints first occupy a
dedicated probation slot and must collect stratified evaluation samples before
they can enter fitness-weighted elite selection.
"""

from __future__ import annotations

import collections
import logging
import math
import os
import random
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_REQUIRED_BUCKETS = (
    "slack",
    "shopping",
    "workspace",
    "dailylife",
    "banking",
    "github",
    "travel",
)


def _normalise_required_buckets(value: Any) -> tuple[str, ...]:
    if value is None:
        buckets = DEFAULT_REQUIRED_BUCKETS
    elif isinstance(value, str):
        buckets = tuple(part.strip() for part in value.split(",") if part.strip())
    else:
        buckets = tuple(str(part).strip() for part in value if str(part).strip())
    if not buckets:
        raise ValueError("population required bucket set must not be empty")
    if len(set(buckets)) != len(buckets):
        raise ValueError(f"population required buckets must be unique: {buckets}")
    return buckets


def _get_hf_path(ckpt_dir: str, step: int) -> str:
    return os.path.join(ckpt_dir, f"global_step_{step}", "actor", "huggingface")


@dataclass
class CandidateCheckpoint:
    """One historical checkpoint and its immutable-opponent evaluation data."""

    step: int
    hf_path: str
    fitness: float = 0.0
    reward_history: collections.deque = field(
        default_factory=lambda: collections.deque(maxlen=200)
    )
    n_samples: int = 0
    bucket_counts: collections.Counter = field(default_factory=collections.Counter)
    bucket_reward_history: dict[str, collections.deque] = field(default_factory=dict)

    def record(self, score: float, task_bucket: str) -> None:
        score = float(score)
        if not math.isfinite(score):
            raise ValueError(f"population score must be finite, got {score}")
        self.reward_history.append(score)
        self.n_samples += 1
        if task_bucket and task_bucket != "unknown":
            bucket = str(task_bucket)
            self.bucket_counts[bucket] += 1
            history = self.bucket_reward_history.setdefault(
                bucket, collections.deque(maxlen=200)
            )
            history.append(score)
        self.fitness = sum(self.reward_history) / len(self.reward_history)

    def evaluated_bucket_count(
        self,
        min_samples_per_bucket: int,
        required_buckets: tuple[str, ...] | None = None,
    ) -> int:
        buckets = required_buckets or tuple(self.bucket_counts)
        return sum(
            self.bucket_counts.get(bucket, 0) >= min_samples_per_bucket
            for bucket in buckets
        )

    def is_eligible(
        self,
        min_samples_per_bucket: int,
        min_eval_buckets: int,
        required_buckets: tuple[str, ...] | None = None,
    ) -> bool:
        return (
            self.evaluated_bucket_count(min_samples_per_bucket, required_buckets)
            >= min_eval_buckets
        )

    def recompute_macro_fitness(self, required_buckets: tuple[str, ...]) -> float:
        """Equal-weight suite score; high-frequency suites cannot dominate selection."""

        bucket_means = []
        for bucket in required_buckets:
            history = self.bucket_reward_history.get(bucket)
            if history:
                bucket_means.append(sum(history) / len(history))
        if bucket_means:
            self.fitness = sum(bucket_means) / len(bucket_means)
        elif self.reward_history:
            # Backward-compatible fallback for v2 states that did not persist
            # per-suite reward histories.
            self.fitness = sum(self.reward_history) / len(self.reward_history)
        else:
            self.fitness = 0.0
        return self.fitness

    def min_required_bucket_samples(self, required_buckets: tuple[str, ...]) -> int:
        return min((self.bucket_counts.get(bucket, 0) for bucket in required_buckets), default=0)

    def state_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "hf_path": self.hf_path,
            "fitness": self.fitness,
            "reward_history": list(self.reward_history),
            "reward_history_maxlen": self.reward_history.maxlen,
            "n_samples": self.n_samples,
            "bucket_counts": dict(self.bucket_counts),
            "bucket_reward_history": {
                bucket: list(history)
                for bucket, history in self.bucket_reward_history.items()
            },
            "bucket_reward_history_maxlen": 200,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any], ckpt_dir: str) -> "CandidateCheckpoint":
        step = int(state["step"])
        history = collections.deque(
            (float(value) for value in state.get("reward_history", [])),
            maxlen=int(state.get("reward_history_maxlen") or 200),
        )
        candidate = cls(
            step=step,
            # Always resolve against the current run directory.  Checkpoints
            # are often resumed after the experiment root is relocated.
            hf_path=_get_hf_path(ckpt_dir, step),
            fitness=float(state.get("fitness", 0.0)),
            reward_history=history,
            n_samples=int(state.get("n_samples", len(history))),
            bucket_counts=collections.Counter(
                {str(key): int(value) for key, value in state.get("bucket_counts", {}).items()}
            ),
            bucket_reward_history={
                str(bucket): collections.deque(
                    (float(value) for value in values),
                    maxlen=int(state.get("bucket_reward_history_maxlen") or 200),
                )
                for bucket, values in state.get("bucket_reward_history", {}).items()
            },
        )
        if any(not math.isfinite(value) for value in candidate.reward_history) or any(
            not math.isfinite(value)
            for bucket_history in candidate.bucket_reward_history.values()
            for value in bucket_history
        ):
            raise ValueError(f"population checkpoint step={step} contains non-finite rewards")
        if history and not candidate.bucket_reward_history:
            candidate.fitness = sum(history) / len(history)
        return candidate


class PopulationManager:
    """Manage historical populations and restart their frozen vLLM replicas."""

    STATE_VERSION = 4
    REWARD_CONTRACT_VERSION = "agentdojo_stratified_macro_v2"

    def __init__(
        self,
        attacker_ckpt_dir: str,
        defender_ckpt_dir: str,
        population_size: int = 4,
        update_interval: int = 40,
        top_fraction: float = 0.5,
        save_freq: int = 20,
        min_eval_samples_per_bucket: int = 32,
        min_eval_buckets: int = 7,
        required_buckets: Any = DEFAULT_REQUIRED_BUCKETS,
        probation_sampling_prob: float = 0.75,
        candidate_lag_steps: int = 0,
        random_seed: int = 0,
    ):
        if population_size <= 0:
            raise ValueError(f"population_size must be positive, got {population_size}")
        if update_interval <= 0 or save_freq <= 0:
            raise ValueError("population update_interval and save_freq must be positive")
        if not 0 < top_fraction <= 1:
            raise ValueError(f"top_fraction must be in (0, 1], got {top_fraction}")
        if min_eval_samples_per_bucket <= 0 or min_eval_buckets <= 0:
            raise ValueError("population evaluation quotas must be positive")
        required_buckets = _normalise_required_buckets(required_buckets)
        if min_eval_buckets > len(required_buckets):
            raise ValueError(
                "population min_eval_buckets cannot exceed the required bucket set: "
                f"required={required_buckets}, min_eval_buckets={min_eval_buckets}"
            )
        if not 0.0 <= probation_sampling_prob <= 1.0:
            raise ValueError(
                "population probation_sampling_prob must be in [0, 1], "
                f"got {probation_sampling_prob}"
            )
        if candidate_lag_steps < 0:
            raise ValueError(
                "population candidate_lag_steps must be non-negative, "
                f"got {candidate_lag_steps}"
            )
        if candidate_lag_steps % update_interval != 0:
            raise ValueError(
                "population candidate_lag_steps must be a multiple of the "
                f"update interval: lag={candidate_lag_steps}, interval={update_interval}"
            )

        self.attacker_ckpt_dir = attacker_ckpt_dir
        self.defender_ckpt_dir = defender_ckpt_dir
        self.population_size = int(population_size)
        self.update_interval = int(update_interval)
        self.top_fraction = float(top_fraction)
        self.save_freq = int(save_freq)
        self.min_eval_samples_per_bucket = int(min_eval_samples_per_bucket)
        self.min_eval_buckets = int(min_eval_buckets)
        self.required_buckets = required_buckets
        self.probation_sampling_prob = float(probation_sampling_prob)
        self.candidate_lag_steps = int(candidate_lag_steps)

        self._attacker_candidates: dict[int, CandidateCheckpoint] = {}
        self._defender_candidates: dict[int, CandidateCheckpoint] = {}
        self._attacker_loaded_steps: list[int] = []
        self._defender_loaded_steps: list[int] = []
        self._attacker_llm_mgr = None
        self._defender_llm_mgr = None
        self._base_model_paths: dict[str, str] = {}
        self._last_attacker_update_step = 0
        self._last_defender_update_step = 0
        self._refreshing_count = 0
        self._pending_restore_steps: dict[str, list[int]] = {}
        # Replica slots are replaced sequentially because no spare GPUs are
        # available. Keep one immutable target plan across partial failures so
        # the next retry completes it instead of drawing a different target.
        self._pending_target_steps: dict[str, list[int]] = {}
        self._rng = random.Random(random_seed)

    def set_llm_managers(self, attacker_mgr=None, defender_mgr=None) -> None:
        """Attach the historical vLLM managers used by this training mode.

        Normal co-evolution supplies both managers.  Defender-only hardening
        deliberately supplies only the frozen attacker manager; requiring a
        dummy historical defender would waste an entire serving node.  Every
        *present* role still has to provide exactly one replica per population
        slot, so an accidentally undersized challenger pool fails before
        rollout starts.
        """

        if attacker_mgr is None and defender_mgr is None:
            raise ValueError("at least one historical LLM manager is required")

        managers = {"attacker": attacker_mgr, "defender": defender_mgr}
        replica_counts = {
            role: len(manager.get_addresses())
            for role, manager in managers.items()
            if manager is not None
        }
        mismatched = {
            role: count
            for role, count in replica_counts.items()
            if count != self.population_size
        }
        if mismatched:
            raise ValueError(
                "population_size must equal every configured historical replica "
                f"count: configured={self.population_size}, replicas={replica_counts}"
            )

        self._attacker_llm_mgr = attacker_mgr
        self._defender_llm_mgr = defender_mgr
        self._attacker_loaded_steps = (
            [0] * replica_counts["attacker"] if attacker_mgr is not None else []
        )
        self._defender_loaded_steps = (
            [0] * replica_counts["defender"] if defender_mgr is not None else []
        )
        self._base_model_paths = {
            role: str(manager.model_config.path)
            for role, manager in managers.items()
            if manager is not None
        }
        logger.info(
            "[PopulationManager] Linked historical roles=%s with %d replicas each",
            sorted(replica_counts),
            self.population_size,
        )

    def get_old_attacker_urls(self) -> list[str]:
        return self._urls(self._attacker_llm_mgr)

    def get_old_defender_urls(self) -> list[str]:
        return self._urls(self._defender_llm_mgr)

    @staticmethod
    def _urls(manager) -> list[str]:
        if manager is None:
            return []
        return [f"http://{address}/v1" for address in manager.get_addresses()]

    def get_attacker_population_info(self) -> list[dict[str, Any]]:
        return self._population_info(
            "attacker", self._attacker_loaded_steps, self._attacker_candidates
        )

    def get_defender_population_info(self) -> list[dict[str, Any]]:
        return self._population_info(
            "defender", self._defender_loaded_steps, self._defender_candidates
        )

    def _population_info(
        self,
        role: str,
        loaded_steps: list[int],
        candidates: dict[int, CandidateCheckpoint],
    ) -> list[dict[str, Any]]:
        info = []
        latest_candidate_step = max(candidates, default=0)
        latest_eligible_step = max(
            (step for step, candidate in candidates.items() if self._eligible(candidate)),
            default=0,
        )
        pending_target = self._pending_target_steps.get(role, [])
        for slot, step in enumerate(loaded_steps):
            candidate = candidates.get(step)
            is_eligible = (
                self._eligible(candidate) if candidate is not None else False
            )
            info.append(
                {
                    "slot": slot,
                    "step": step,
                    "fitness": candidate.fitness if candidate is not None else 0.0,
                    "n_samples": candidate.n_samples if candidate is not None else 0,
                    "evaluated_buckets": (
                        candidate.evaluated_bucket_count(
                            self.min_eval_samples_per_bucket,
                            self.required_buckets,
                        )
                        if candidate is not None
                        else 0
                    ),
                    "eligible": is_eligible,
                    "probation": bool(
                        candidate is not None
                        and slot == len(loaded_steps) - 1
                        and not is_eligible
                    ),
                    "min_required_bucket_samples": (
                        candidate.min_required_bucket_samples(self.required_buckets)
                        if candidate is not None
                        else 0
                    ),
                    "latest_candidate_step": latest_candidate_step,
                    "latest_eligible_step": latest_eligible_step,
                    "pending_target_step": (
                        int(pending_target[slot])
                        if slot < len(pending_target)
                        else -1
                    ),
                }
            )
        return info

    def choose_population_slot(self, role: str, task_bucket: str) -> int:
        """Prefer the probation slot while its current suite quota is incomplete."""

        loaded = self._loaded_steps(role)
        if not loaded:
            raise RuntimeError(f"{role} population has no loaded slots")
        probation_slot = len(loaded) - 1
        candidate = self._candidates(role).get(loaded[probation_slot])
        bucket = str(task_bucket or "unknown")
        if (
            candidate is not None
            and not self._eligible(candidate)
            and bucket in self.required_buckets
            and candidate.bucket_counts.get(bucket, 0) < self.min_eval_samples_per_bucket
            and self._rng.random() < self.probation_sampling_prob
        ):
            return probation_slot
        return self._rng.randrange(len(loaded))

    def update_due(self, role: str, current_param_version: int) -> bool:
        last = self._last_update_step(role)
        return self._get_update_target(current_param_version, last) is not None

    async def maybe_update_attacker(self, current_param_version: int) -> None:
        await self._maybe_update("attacker", current_param_version)

    async def maybe_update_defender(self, current_param_version: int) -> None:
        await self._maybe_update("defender", current_param_version)

    async def _maybe_update(self, role: str, current_param_version: int) -> None:
        self._register_new_checkpoints(role, current_param_version)
        target = self._get_update_target(current_param_version, self._last_update_step(role))
        if target is None:
            return
        await self._resample_population(role, max_candidate_step=target)
        if role == "attacker":
            self._last_attacker_update_step = target
        else:
            self._last_defender_update_step = target

    def report_attacker_reward(self, step: int, reward: float, task_bucket: str) -> None:
        self._report(self._attacker_candidates, step, reward, task_bucket)

    def report_defender_reward(self, step: int, reward: float, task_bucket: str) -> None:
        self._report(self._defender_candidates, step, reward, task_bucket)

    def _report(
        self,
        candidates: dict[int, CandidateCheckpoint],
        step: int,
        score: float,
        task_bucket: str,
    ) -> None:
        candidate = candidates.get(int(step))
        if candidate is not None:
            score = float(score)
            if not math.isfinite(score):
                logger.warning(
                    "[PopulationManager] Ignoring non-finite score for step=%d bucket=%s: %r",
                    step,
                    task_bucket,
                    score,
                )
                return
            candidate.record(score, task_bucket)
            candidate.recompute_macro_fitness(self.required_buckets)

    def get_step_for_url(self, url: str) -> int:
        for manager, loaded in (
            (self._attacker_llm_mgr, self._attacker_loaded_steps),
            (self._defender_llm_mgr, self._defender_loaded_steps),
        ):
            if manager is None:
                continue
            for slot, address in enumerate(manager.get_addresses()):
                if url.rstrip("/") == f"http://{address}/v1".rstrip("/"):
                    return loaded[slot]
        return 0

    def get_loaded_step(self, role: str, slot: int) -> int:
        """Return the checkpoint occupying a slot at request-dispatch time."""

        loaded = self._loaded_steps(role)
        slot = int(slot)
        if slot < 0 or slot >= len(loaded):
            raise IndexError(
                f"{role} population slot out of range: {slot} not in [0, {len(loaded)})"
            )
        return int(loaded[slot])

    @property
    def is_refreshing(self) -> bool:
        return self._refreshing_count > 0

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": self.STATE_VERSION,
            "population_size": self.population_size,
            "config": {
                "update_interval": self.update_interval,
                "top_fraction": self.top_fraction,
                "save_freq": self.save_freq,
                "min_eval_samples_per_bucket": self.min_eval_samples_per_bucket,
                "min_eval_buckets": self.min_eval_buckets,
                "required_buckets": list(self.required_buckets),
                "probation_sampling_prob": self.probation_sampling_prob,
                "candidate_lag_steps": self.candidate_lag_steps,
                "reward_contract_version": self.REWARD_CONTRACT_VERSION,
            },
            "last_attacker_update_step": self._last_attacker_update_step,
            "last_defender_update_step": self._last_defender_update_step,
            "attacker_loaded_steps": list(self._attacker_loaded_steps),
            "defender_loaded_steps": list(self._defender_loaded_steps),
            "attacker_pending_target_steps": list(
                self._pending_target_steps.get("attacker", [])
            ),
            "defender_pending_target_steps": list(
                self._pending_target_steps.get("defender", [])
            ),
            "attacker_candidates": {
                step: candidate.state_dict()
                for step, candidate in self._attacker_candidates.items()
            },
            "defender_candidates": {
                step: candidate.state_dict()
                for step, candidate in self._defender_candidates.items()
            },
            "rng_state": self._rng.getstate(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not state:
            return
        version = int(state.get("version", 1))
        if version > self.STATE_VERSION:
            raise ValueError(
                f"population state version {version} is newer than supported {self.STATE_VERSION}"
            )
        saved_size = int(state.get("population_size", self.population_size))
        if saved_size != self.population_size:
            raise ValueError(
                "cannot restore population with a different size: "
                f"checkpoint={saved_size}, configured={self.population_size}"
            )
        saved_config = state.get("config")
        if saved_config:
            expected = {
                "update_interval": self.update_interval,
                "top_fraction": self.top_fraction,
                "save_freq": self.save_freq,
                "min_eval_samples_per_bucket": self.min_eval_samples_per_bucket,
                "min_eval_buckets": self.min_eval_buckets,
                "required_buckets": list(self.required_buckets),
                "probation_sampling_prob": self.probation_sampling_prob,
                "candidate_lag_steps": self.candidate_lag_steps,
                "reward_contract_version": self.REWARD_CONTRACT_VERSION,
            }
            mismatches = {
                key: (saved_config.get(key), expected_value)
                for key, expected_value in expected.items()
                if saved_config.get(key) != expected_value
                # v3 predates the age-lag field.  Allow a recovery run to
                # adopt the safer lag, then persist it in the first v4 state.
                and not (version < 4 and key == "candidate_lag_steps")
            }
            if mismatches:
                raise ValueError(
                    "population checkpoint config/reward contract mismatch: "
                    f"{mismatches}"
                )
        self._attacker_candidates = self._restore_candidates(
            state.get("attacker_candidates", {}), self.attacker_ckpt_dir
        )
        self._defender_candidates = self._restore_candidates(
            state.get("defender_candidates", {}), self.defender_ckpt_dir
        )
        self._last_attacker_update_step = int(state.get("last_attacker_update_step", 0))
        self._last_defender_update_step = int(state.get("last_defender_update_step", 0))
        self._pending_restore_steps = {
            "attacker": [int(step) for step in state.get("attacker_loaded_steps", [])],
            "defender": [int(step) for step in state.get("defender_loaded_steps", [])],
        }
        self._pending_target_steps = {}
        for role in ("attacker", "defender"):
            pending = [
                int(step)
                for step in state.get(f"{role}_pending_target_steps", [])
            ]
            if pending:
                if len(pending) != self.population_size:
                    raise ValueError(
                        f"invalid persisted {role} refresh target size: "
                        f"{len(pending)} != {self.population_size}"
                    )
                self._pending_target_steps[role] = pending
        rng_state = state.get("rng_state")
        if rng_state is not None:
            self._rng.setstate(rng_state)
        for candidate in (
            *self._attacker_candidates.values(),
            *self._defender_candidates.values(),
        ):
            candidate.recompute_macro_fitness(self.required_buckets)
        logger.info(
            "[PopulationManager] Loaded state: attacker_candidates=%d defender_candidates=%d",
            len(self._attacker_candidates),
            len(self._defender_candidates),
        )

    @staticmethod
    def _restore_candidates(
        states: dict[Any, dict[str, Any]], ckpt_dir: str
    ) -> dict[int, CandidateCheckpoint]:
        restored = {}
        for candidate_state in states.values():
            candidate = CandidateCheckpoint.from_state_dict(candidate_state, ckpt_dir)
            restored[candidate.step] = candidate
        return restored

    async def restore_loaded_models(self) -> set[str]:
        """Restore persisted slots and isolate a role whose replica failed.

        A historical replica is optional for PPO progress.  If one model path
        disappeared or one vLLM restart fails, callers can keep current-policy
        traffic running and retry the whole affected role at the next
        population update.  Contract/config mismatches are still rejected by
        :meth:`load_state_dict` before this method is entered.
        """

        if not self._pending_restore_steps:
            return set()
        failed_roles: set[str] = set()
        self._refreshing_count += 1
        try:
            for role in ("attacker", "defender"):
                desired = self._pending_restore_steps.get(role, [])
                manager = self._llm_manager(role)
                # A role omitted by the active training mode is intentionally
                # absent, even if a dual-training source state contains saved
                # slots for it.  Do not attempt to restore or mark it failed.
                if manager is None or not desired:
                    continue
                if len(desired) != self.population_size:
                    failed_roles.add(role)
                    logger.error(
                        "[PopulationManager] Cannot restore %s population: "
                        "saved slots=%d expected=%d; keeping base replicas",
                        role,
                        len(desired),
                        self.population_size,
                    )
                    self._mark_role_for_retry(role)
                    continue
                loaded = self._loaded_steps(role)
                for slot, step in enumerate(desired):
                    if step == 0:
                        continue
                    try:
                        path = self._model_path(role, step)
                        await manager.restart_replica_with_model(slot, path)
                    except Exception:
                        failed_roles.add(role)
                        logger.exception(
                            "[PopulationManager] Cannot restore %s slot=%d step=%d; "
                            "disabling this historical role until a later refresh",
                            role,
                            slot,
                            step,
                        )
                        continue
                    loaded[slot] = step
                if role in failed_roles:
                    self._mark_role_for_retry(role)
        finally:
            self._pending_restore_steps = {}
            self._refreshing_count -= 1
        return failed_roles

    def _mark_role_for_retry(self, role: str) -> None:
        """Make the next eligible trainer checkpoint retry a failed restore."""

        if role == "attacker":
            self._last_attacker_update_step = 0
        elif role == "defender":
            self._last_defender_update_step = 0
        else:
            raise ValueError(f"unknown population role: {role}")

    def _register_new_checkpoints(self, role: str, current_param_version: int) -> None:
        candidates = self._candidates(role)
        ckpt_dir = self.attacker_ckpt_dir if role == "attacker" else self.defender_ckpt_dir
        for step in range(self.save_freq, current_param_version + 1, self.save_freq):
            if step in candidates:
                continue
            path = _get_hf_path(ckpt_dir, step)
            if os.path.exists(path):
                candidates[step] = CandidateCheckpoint(step=step, hf_path=path)
                logger.info(
                    "[PopulationManager] Registered provisional %s checkpoint step=%d",
                    role,
                    step,
                )

    def _get_update_target(self, current_param_version: int, last_update_step: int) -> int | None:
        eligible_version = current_param_version - self.candidate_lag_steps
        if eligible_version < self.update_interval:
            return None
        target = (eligible_version // self.update_interval) * self.update_interval
        return target if target > last_update_step else None

    async def _resample_population(
        self,
        role: str,
        max_candidate_step: int | None = None,
    ) -> None:
        manager = self._llm_manager(role)
        loaded = self._loaded_steps(role)
        candidates = self._candidates(role)
        if manager is None:
            raise RuntimeError(f"{role} population has no LLMServerManager")
        if len(loaded) != self.population_size:
            raise RuntimeError(
                f"{role} loaded slot count changed: {len(loaded)} != {self.population_size}"
            )

        pending_target = self._pending_target_steps.get(role)
        probation = None
        if pending_target:
            desired = list(pending_target)
        else:
            eligible = [candidate for candidate in candidates.values() if self._eligible(candidate)]
            provisional = [candidate for candidate in candidates.values() if not self._eligible(candidate)]
            probation = self._choose_probation_candidate(
                provisional,
                loaded,
                max_candidate_step=max_candidate_step,
            )
            elite_slot_count = self.population_size - int(probation is not None)
            sampled_elites = self._sample_elites(eligible, elite_slot_count)

            desired = self._arrange_steps(
                [candidate.step for candidate in sampled_elites],
                loaded[:elite_slot_count],
                elite_slot_count,
            )
            if probation is not None:
                desired.append(probation.step)
            self._pending_target_steps[role] = list(desired)

        logger.info(
            "[PopulationManager] %s target=%s current=%s probation=%s",
            role,
            desired,
            loaded,
            probation.step if probation is not None else None,
        )
        self._refreshing_count += 1
        try:
            for slot, step in enumerate(desired):
                if loaded[slot] == step:
                    continue
                path = self._model_path(role, step)
                await manager.restart_replica_with_model(slot, path)
                loaded[slot] = step
            self._pending_target_steps.pop(role, None)
        finally:
            self._refreshing_count -= 1

    def _eligible(self, candidate: CandidateCheckpoint) -> bool:
        return candidate.is_eligible(
            self.min_eval_samples_per_bucket,
            self.min_eval_buckets,
            self.required_buckets,
        )

    def _choose_probation_candidate(
        self,
        provisional: list[CandidateCheckpoint],
        loaded: list[int],
        max_candidate_step: int | None = None,
    ) -> CandidateCheckpoint | None:
        selectable = [
            candidate
            for candidate in provisional
            if max_candidate_step is None or candidate.step <= max_candidate_step
        ]
        if not selectable:
            return None
        current = next(
            (candidate for candidate in selectable if candidate.step == loaded[-1]),
            None,
        )
        # Keep an under-evaluated active probation model until it reaches its
        # quota. Once it graduates, jump to the newest checkpoint available at
        # this refresh boundary. Choosing the oldest untested checkpoint is an
        # unstable queue when save_freq < update_interval: candidates arrive
        # faster than the single probation slot can evaluate them.
        return current or max(selectable, key=lambda candidate: candidate.step)

    def _sample_elites(
        self,
        eligible: list[CandidateCheckpoint],
        n_slots: int,
    ) -> list[CandidateCheckpoint]:
        if not eligible or n_slots <= 0:
            return []
        pool = list(eligible)
        if len(pool) >= 2 * n_slots:
            keep = max(n_slots, int(len(pool) * self.top_fraction))
            pool = sorted(pool, key=lambda candidate: candidate.fitness, reverse=True)[:keep]

        selected = []
        while pool and len(selected) < n_slots:
            minimum = min(candidate.fitness for candidate in pool)
            weights = [candidate.fitness - minimum + 0.01 for candidate in pool]
            chosen = self._rng.choices(pool, weights=weights, k=1)[0]
            selected.append(chosen)
            pool.remove(chosen)
        return selected

    @staticmethod
    def _arrange_steps(selected: list[int], current: list[int], n_slots: int) -> list[int]:
        """Keep selected checkpoints in-place where possible to avoid restarts."""

        desired: list[int | None] = [None] * n_slots
        remaining = set(selected)
        for slot, step in enumerate(current[:n_slots]):
            if step in remaining:
                desired[slot] = step
                remaining.remove(step)
        remaining_iter = iter(sorted(remaining))
        for slot in range(n_slots):
            if desired[slot] is None:
                desired[slot] = next(remaining_iter, 0)
        return [int(step) for step in desired]

    def _model_path(self, role: str, step: int) -> str:
        if step == 0:
            return self._base_model_paths[role]
        candidate = self._candidates(role).get(step)
        if candidate is None:
            raise RuntimeError(f"unknown {role} candidate step={step}")
        if not os.path.exists(candidate.hf_path):
            raise FileNotFoundError(candidate.hf_path)
        return candidate.hf_path

    def _last_update_step(self, role: str) -> int:
        if role == "attacker":
            return self._last_attacker_update_step
        if role == "defender":
            return self._last_defender_update_step
        raise ValueError(f"unknown population role: {role}")

    def _candidates(self, role: str) -> dict[int, CandidateCheckpoint]:
        if role == "attacker":
            return self._attacker_candidates
        if role == "defender":
            return self._defender_candidates
        raise ValueError(f"unknown population role: {role}")

    def _loaded_steps(self, role: str) -> list[int]:
        if role == "attacker":
            return self._attacker_loaded_steps
        if role == "defender":
            return self._defender_loaded_steps
        raise ValueError(f"unknown population role: {role}")

    def _llm_manager(self, role: str):
        if role == "attacker":
            return self._attacker_llm_mgr
        if role == "defender":
            return self._defender_llm_mgr
        raise ValueError(f"unknown population role: {role}")


# Backwards-compatible import used by older orchestration code.
OldModelManager = PopulationManager
