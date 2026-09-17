"""Co-Training Rollouter — unified attacker/defender rollout routing.

Bilateral Co-PPO manages current/historical attacker and defender groups and
routes current-role training data to two MessageQueues.

Sampling ratios for injected rows (default):
  curr_curr        40% → trains both attacker and defender
  old_atk_curr_def 25% → trains defender against population attacker
  curr_atk_old_def 25% → trains attacker against population defender
  fixed_static     10% → trains defender against static templates

Native clean rows are routed by their dataset metadata, not by a second random
``prob_clean`` branch. The current training parquet contains about 25% such
rows, so they remain a real clean-task defender stream without changing the
injected-row population mix.
"""

import asyncio
import collections
import copy
import hashlib
import json
import logging
import math
import os
import random
import time
import uuid
from pprint import pformat
from typing import TYPE_CHECKING

import numpy as np
import ray
import torch

from cotrain.backpressure import both_role_queues_full
from verl.experimental.fully_async_policy.detach_utils import (
    RolloutSample,
    safe_create_task,
)
from verl.experimental.fully_async_policy.message_queue import MessageQueueClient

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from verl import DataProto
    from verl.experimental.agent_loop.agent_loop import AgentLoopOutput

POPULATION_STATE_FILENAME = "cotrain_population_state.pt"

DUAL_TRAINING_MODE = "dual"
_SUPPORTED_TRAINING_MODES = {
    DUAL_TRAINING_MODE,
}

ATTACKER_REWARD_EXTRA_KEYS = (
    "attacker/asr_success",
    "attacker/effective_asr",
    "attacker/format_violation",
    "attacker/target_tool_called",
    "attacker/target_tool_call_count",
    "attacker/target_parameter_match",
    "attacker/ground_truth_call_count",
    "attacker/utility_failed",
    "attacker/is_refusal",
    "attacker/payload_length",
    "attacker/payload_too_long",
    "attacker/reward",
    "attacker/defender_turns",
    "attacker/defender_tool_calls",
    "attacker/tool_responses",
    "attacker/num_generated_turns",
    "attacker/num_injection_points",
    "attacker/num_nonempty_injection_points",
    "attacker/has_injected_payload",
    "attacker/generated_tokens",
    "attacker/length_finish_count",
    "attacker/exact_trace_drops",
    "attacker/train_sequence_tokens",
    "attacker/train_sequence_overflow",
    "attacker/tool_responses_truncated",
    "attacker/payloads_truncated",
    "attacker/payloads_survived_truncation",
    "attacker/is_fixed_template",
    "attacker/is_clean",
    "attacker/asr_cc",
    "attacker/asr_vs_old_def",
    "attacker/reward_cc",
    "attacker/reward_vs_old_def",
    "attacker/mp_is_cc",
    "attacker/mp_is_old_def",
)

DEFENDER_REWARD_EXTRA_KEYS = (
    "defender/asr_success",
    "defender/utility_success",
    "defender/num_turns",
    "defender/tool_call_count",
    "defender/tool_call_format_violations",
    "defender/resist_rate",
    "defender/reward",
    "defender/exact_trace_drops",
    "defender/train_sequence_tokens",
    "defender/train_sequence_overflow",
    "defender/tool_responses_truncated",
    "defender/payloads_truncated",
    "defender/payloads_survived_truncation",
    "defender/asr_cc",
    "defender/asr_vs_old_atk",
    "defender/asr_vs_template",
    "defender/utility_clean",
    "defender/utility_cc",
    "defender/utility_vs_old_atk",
    "defender/resist_cc",
    "defender/resist_vs_old_atk",
    "defender/resist_vs_template",
    "defender/mp_is_cc",
    "defender/mp_is_old_atk",
    "defender/mp_is_template",
    "defender/mp_is_clean",
)


def _training_dataset_fingerprint(
    local_data_files,
    *,
    dataset_length: int,
    feed_contract: dict,
) -> str:
    """Hash the effective co-training data stream, not just its filenames."""
    digest = hashlib.sha256()
    digest.update(b"cotrain-dataset-v1\0")
    digest.update(
        json.dumps(
            {"dataset_length": int(dataset_length), **feed_contract},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    )
    for index, data_file in enumerate(local_data_files):
        path = os.path.abspath(os.path.expanduser(str(data_file)))
        digest.update(f"\0file:{index}\0".encode("ascii"))
        try:
            with open(path, "rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
        except OSError as exc:
            raise RuntimeError(
                f"cannot fingerprint co-training dataset file {path}: {exc}"
            ) from exc
    return digest.hexdigest()


def _capture_sampler_replay_state(sampler) -> dict:
    """Capture the sampler's *initial* state used to replay+skip on resume."""
    generator = getattr(sampler, "generator", None)
    if generator is not None and hasattr(generator, "get_state"):
        return {"kind": "generator", "state": generator.get_state().clone()}
    if sampler.__class__.__name__ == "SequentialSampler":
        return {"kind": "sequential"}
    if hasattr(sampler, "state_dict") and hasattr(sampler, "load_state_dict"):
        return {"kind": "state_dict", "state": copy.deepcopy(sampler.state_dict())}
    return {"kind": "unsupported", "class": sampler.__class__.__qualname__}


def _restore_sampler_replay_state(sampler, replay_state: dict) -> None:
    """Restore the stream origin before skipping a persisted feed cursor."""
    kind = (replay_state or {}).get("kind")
    if kind == "sequential":
        return
    if kind == "generator":
        generator = getattr(sampler, "generator", None)
        if generator is None or not hasattr(generator, "set_state"):
            raise RuntimeError("checkpoint expects a generator-backed training sampler")
        generator.set_state(replay_state["state"])
        return
    if kind == "state_dict" and hasattr(sampler, "load_state_dict"):
        sampler.load_state_dict(copy.deepcopy(replay_state["state"]))
        return
    raise RuntimeError(
        "training sampler cannot deterministically replay a dataset cursor: "
        f"{(replay_state or {}).get('class', type(sampler).__qualname__)}"
    )


def _resolve_dataset_resume_cursor(
    saved_fingerprint: str | None,
    current_fingerprint: str,
    saved_cursor: int,
    total_rollout_steps: int,
) -> int:
    """Continue only an identical feed; changed/legacy data starts at zero."""
    if saved_fingerprint != current_fingerprint:
        return 0
    cursor = int(saved_cursor)
    if not 0 <= cursor <= int(total_rollout_steps):
        raise ValueError(
            "rollouter dataset cursor is outside the current feed: "
            f"{cursor} not in [0, {total_rollout_steps}]"
        )
    return cursor


def _resolve_training_mode(cotrain_cfg) -> str:
    """Return the explicit co-training topology, preserving dual as default."""

    training_mode = str(
        cotrain_cfg.get("training_mode", DUAL_TRAINING_MODE)
    ).strip().lower()
    if training_mode not in _SUPPORTED_TRAINING_MODES:
        raise ValueError(
            "unsupported cotrain.training_mode: "
            f"{training_mode!r}; expected one of {sorted(_SUPPORTED_TRAINING_MODES)}"
        )
    return training_mode


def _training_step_for_mode(training_mode: str, policy_versions: dict[str, int]) -> int:
    """Choose the progress clock owned by the active trainer topology."""

    if training_mode == DUAL_TRAINING_MODE:
        return min(int(policy_versions["attacker"]), int(policy_versions["defender"]))
    raise ValueError(f"unsupported training mode: {training_mode!r}")


def _validate_resume_training_mode(
    saved_training_mode_value,
    configured_training_mode: str,
) -> bool:
    """Reject checkpoints from an incompatible trainer topology."""

    saved_training_mode = str(
        saved_training_mode_value or DUAL_TRAINING_MODE
    ).strip().lower()
    if saved_training_mode not in _SUPPORTED_TRAINING_MODES:
        raise RuntimeError(
            "rollouter checkpoint has unsupported training mode: "
            f"{saved_training_mode!r}"
        )
    if saved_training_mode == configured_training_mode:
        return False
    raise RuntimeError(
        "rollouter checkpoint training mode does not match this run: "
        f"checkpoint={saved_training_mode}, configured={configured_training_mode}"
    )


def _training_queues_full(
    training_mode: str,
    attacker_depth: int,
    defender_depth: int,
    max_queue_size: int,
) -> bool:
    """Apply backpressure only to queues that have an active trainer."""

    if training_mode == DUAL_TRAINING_MODE:
        return both_role_queues_full(
            attacker_depth, defender_depth, max_queue_size
        )
    raise ValueError(f"unsupported training mode: {training_mode!r}")


def _validate_model_pair_config(cotrain_cfg) -> tuple[bool, tuple[float, ...]]:
    """Validate routing probabilities and the population ablation contract."""

    _resolve_training_mode(cotrain_cfg)
    population_enabled = bool(cotrain_cfg.get("population_enabled", True))
    model_pair_probs = tuple(
        float(value)
        for value in (
            cotrain_cfg.get("prob_curr_curr", 0.4),
            cotrain_cfg.get("prob_old_atk_curr_def", 0.25),
            cotrain_cfg.get("prob_curr_atk_old_def", 0.25),
            cotrain_cfg.get("prob_fixed_template", 0.1),
            cotrain_cfg.get("prob_clean", 0.0),
        )
    )
    if any(
        not math.isfinite(probability) or probability < 0 or probability > 1
        for probability in model_pair_probs
    ):
        raise ValueError(f"model-pair probabilities must be in [0, 1]: {model_pair_probs}")
    probability_sum = sum(model_pair_probs)
    if abs(probability_sum - 1.0) > 1e-8:
        raise ValueError(
            "injected-row model-pair probabilities must sum to exactly 1.0; "
            f"got {probability_sum:.12f} from {model_pair_probs}"
        )
    replay_probability = float(cotrain_cfg.get("prob_dataset_replay", 0.0))
    if not math.isfinite(replay_probability) or replay_probability != 0.0:
        raise ValueError(
            "cotrain.prob_dataset_replay is reserved but not implemented; "
            "it must remain exactly 0.0 until a replay bank is wired"
        )

    old_pair_probs = model_pair_probs[1:3]
    if population_enabled:
        if any(probability <= 0 for probability in old_pair_probs):
            raise ValueError(
                "old-attacker and old-defender pair probabilities must both be "
                "positive so probation candidates can become eligible"
            )
    elif any(probability != 0 for probability in old_pair_probs):
        raise ValueError(
            "cotrain.population_enabled=False requires both historical "
            f"model-pair probabilities to be zero; got {old_pair_probs}"
        )
    return population_enabled, model_pair_probs


@ray.remote(num_cpus=10, max_concurrency=100)
class CoTrainRollouter:
    """Async rollouter that manages current and optional historical model groups."""

    def __init__(
        self,
        config,
        tokenizer,
        processor=None,
        defender_tokenizer=None,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        if defender_tokenizer is None:
            raise ValueError("CoTrainRollouter requires a dedicated defender_tokenizer")
        self.defender_tokenizer = defender_tokenizer

        cotrain_cfg = config.cotrain
        self.training_mode = _resolve_training_mode(cotrain_cfg)
        self.population_enabled, self.model_pair_probs = _validate_model_pair_config(
            cotrain_cfg
        )
        self.n_rollouts_per_prompt = int(cotrain_cfg.get("n_rollouts_per_prompt", 1))
        if self.n_rollouts_per_prompt <= 0:
            raise ValueError("cotrain.n_rollouts_per_prompt must be positive")
        if str(cotrain_cfg.get("attacker_algorithm", "ppo")).lower() != "ppo":
            raise ValueError("Co-PPO supports only the trajectory-level PPO attacker")
        inner_n = int(config.actor_rollout_ref.rollout.get("n", 1))
        if self.n_rollouts_per_prompt != 1 or inner_n != 1:
            raise ValueError(
                "adv-evo PPO requires cotrain.n_rollouts_per_prompt=1 and "
                "actor_rollout_ref.rollout.n=1"
            )
        self._routing_rng = random.Random(
            int(cotrain_cfg.get("population_random_seed", 0)) + 1_000_003
        )
        # Defender sequence lengths for _pack_defender_output padding
        self._defender_max_prompt_len = cotrain_cfg.get("defender_max_prompt_length", 15360)
        self._defender_max_resp_len = cotrain_cfg.get("defender_max_response_length", 16384)
        # MQ clients (set later by orchestrator)
        self.attacker_mq_client: MessageQueueClient | None = None
        self.defender_mq_client: MessageQueueClient | None = None

        # Current model vLLM addresses (set after LLMServerManager init)
        self.current_attacker_urls: list[str] = []
        self.current_defender_urls: list[str] = []

        # Old model vLLM addresses (from OldModelManager)
        self.old_attacker_urls: list[str] = []
        self.old_defender_urls: list[str] = []

        # Old model manager handle (Ray actor or direct ref)
        self.old_model_manager = None
        self._pending_population_state = None
        self._population_draining_roles: set[str] = set()
        self._population_failed_roles: set[str] = set()
        self._population_fallback_counts = collections.Counter()
        self._population_refresh_failures = collections.Counter()
        self._population_refresh_retry_interval = int(
            cotrain_cfg.get(
                "population_refresh_retry_interval",
                cotrain_cfg.get("population_update_interval", 20),
            )
        )
        if self._population_refresh_retry_interval <= 0:
            raise ValueError("population_refresh_retry_interval must be positive")
        self._population_next_retry_version = {"attacker": 0, "defender": 0}
        self._population_state_save_failures = 0
        self._population_state_dirty = False
        self._population_inflight = {
            "attacker": collections.Counter(),
            "defender": collections.Counter(),
        }
        self._population_drain_timeout_s = float(
            cotrain_cfg.get("population_drain_timeout_s", 1800)
        )
        if (
            not math.isfinite(self._population_drain_timeout_s)
            or self._population_drain_timeout_s <= 0
        ):
            raise ValueError("population_drain_timeout_s must be finite and positive")
        population_state_dir = cotrain_cfg.get("population_state_dir")
        if not population_state_dir:
            population_state_dir = os.path.join(
                str(config.trainer.default_local_dir), "population"
            )
        self._population_state_dir = os.path.abspath(str(population_state_dir))
        self._population_state_path = os.path.join(
            self._population_state_dir, POPULATION_STATE_FILENAME
        )

        # Agent loop manager
        self.agent_loop_manager = None

        # LLMServerManagers for current models (for NCCL weight sync)
        self.attacker_llm_server_manager = None
        self.defender_llm_server_manager = None

        # Staleness and policy versions are role-specific.  A defender sync
        # must never make attacker trajectories appear fresh (or vice versa).
        self.staleness_samples = 0
        self.role_staleness_samples = {"attacker": 0, "defender": 0}
        self.policy_versions = {"attacker": 0, "defender": 0}
        self.max_required_samples = None
        self.max_queue_size = None
        self.max_concurrent_samples = 32

        # Statistics
        self.total_generated_samples = 0
        self.dropped_samples = 0
        self.global_steps = 1
        self.training_steps = 0
        self.total_rollout_steps = None

        # Co-evolution tracking
        self._coevo_tracker = _CoEvolutionTracker(window_size=200)

        # Task-level ASR tracker: keyed by (suite_name, injection_task_id)
        self._task_asr_tracker = _TaskLevelASRTracker(window_size=100)

        # Prompt-level ASR tracker: "did any of the N rollouts from one prompt succeed?"
        self._prompt_asr_tracker = _PromptLevelASRTracker(window_size=200)
        # Compact rollout-health metrics are easier to read than raw per-sample
        # character counts and dozens of task-specific series.
        self._rollout_quality_tracker = _RolloutQualityTracker(window_size=400)

        # Concurrency control
        self.paused = False
        self.running = True
        self.pending_queue: asyncio.Queue | None = None
        self.active_tasks: set = set()

        # Timing
        self.idle_start_time = time.time()
        self.step_start_time = time.time()

        # Dataloader
        self._init_dataloader()

    def _init_dataloader(self):
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(
            self.config.data.train_files,
            self.config.data,
            self.tokenizer,
            self.processor,
            max_samples=self.config.data.get("train_max_samples", -1),
        )
        train_sampler = create_rl_sampler(self.config.data, train_dataset)
        self.train_dataset = train_dataset
        self.train_sampler = train_sampler
        self.dataset_feed_cursor = 0
        self._sampler_replay_state = _capture_sampler_replay_state(train_sampler)

        from torch.utils.data import DataLoader

        gen_batch_size = max(1, int(self.config.data.get("gen_batch_size", 1)))
        self.gen_batch_size = gen_batch_size
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=gen_batch_size,
            sampler=train_sampler,
            collate_fn=collate_fn,
            drop_last=True,
            num_workers=4,
            pin_memory=False,
        )

        # ``CoTrainRollouter`` dispatches one independent environment per
        # prompt.  The dataloader may still batch prompts to keep CPU loading
        # efficient, so count prompts rather than loader batches here.
        self.total_rollout_steps = (
            len(self.train_dataloader) * self.gen_batch_size * self.config.trainer.total_epochs
        )
        if self.config.rollout.get("total_rollout_steps") is not None:
            self.total_rollout_steps = min(self.config.rollout.total_rollout_steps, self.total_rollout_steps)
        local_data_files = getattr(
            train_dataset, "data_files", self.config.data.train_files
        )
        self.dataset_fingerprint = _training_dataset_fingerprint(
            local_data_files,
            dataset_length=len(train_dataset),
            feed_contract={
                "gen_batch_size": self.gen_batch_size,
                "drop_last": True,
                "total_epochs": int(self.config.trainer.total_epochs),
                "total_rollout_steps": int(self.total_rollout_steps),
                "train_max_samples": int(
                    self.config.data.get("train_max_samples", -1)
                ),
                "shuffle": bool(self.config.data.get("shuffle", False)),
                "seed": self.config.data.get("seed"),
                "sampler_class": (
                    f"{train_sampler.__class__.__module__}."
                    f"{train_sampler.__class__.__qualname__}"
                ),
                "prompt_key": self.config.data.get("prompt_key", "prompt"),
                "max_prompt_length": int(
                    self.config.data.get("max_prompt_length", 1024)
                ),
                "filter_overlong_prompts": bool(
                    self.config.data.get("filter_overlong_prompts", True)
                ),
                "truncation": self.config.data.get("truncation", "error"),
            },
        )
        logger.info(f"[CoTrainRollouter] Total rollout steps: {self.total_rollout_steps}")
        logger.info(
            "[CoTrainRollouter] Dataset fingerprint=%s cursor=%d",
            self.dataset_fingerprint,
            self.dataset_feed_cursor,
        )

    def _init_async_objects(self):
        self.lock = asyncio.Lock()
        # Serialize population mutations/checkpoints, while the narrower gate
        # below protects only URL selection + in-flight reservation.  Keeping
        # these separate lets unrelated/current traffic continue while one
        # historical role drains.
        self._population_update_lock = asyncio.Lock()
        self._population_refresh_lock = asyncio.Lock()
        self._resume_event = asyncio.Event()
        self._resume_event.set()
        self.pending_queue = asyncio.Queue(maxsize=128)

    async def init_workers(self):
        """Initialize agent loop workers. Called after LLMServerManagers are set."""
        self._init_async_objects()

        from verl.experimental.agent_loop.agent_loop import AgentLoopManager
        from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncAgentLoopManager

        # Import cotrain agent loop to register it
        import cotrain.agent_loop  # noqa: F401

        # Create agent loop manager using current attacker URLs as the primary LLM client
        # The actual model routing happens inside CoTrainAgentLoop via kwargs
        self.agent_loop_manager = await FullyAsyncAgentLoopManager.create(
            config=self.config,
            llm_client=None,  # Agent loop uses OpenAI clients directly
            reward_loop_worker_handles=None,
        )
        logger.info("[CoTrainRollouter] Agent loop workers initialized")

    async def set_mq_clients(
        self,
        attacker_mq_client: MessageQueueClient | None,
        defender_mq_client: MessageQueueClient | None,
    ):
        async with self.lock:
            self.attacker_mq_client = attacker_mq_client
            self.defender_mq_client = defender_mq_client

    async def set_current_model_urls(self, attacker_urls: list[str], defender_urls: list[str]):
        if not defender_urls or ((not attacker_urls)):
            raise RuntimeError(
                "CoTrainRollouter cannot start with empty current vLLM routes: "
                f"attacker={len(attacker_urls)}, defender={len(defender_urls)}"
            )
        async with self.lock:
            self.current_attacker_urls = list(attacker_urls)
            self.current_defender_urls = list(defender_urls)
        logger.info(
            "[CoTrainRollouter] Current vLLM routes ready: attacker=%d defender=%d",
            len(attacker_urls),
            len(defender_urls),
        )

    async def set_old_model_urls(self, attacker_urls: list[str], defender_urls: list[str]):
        if not self.population_enabled:
            raise RuntimeError("population model URLs cannot be set when population is disabled")
        routes_missing = not attacker_urls or not defender_urls
        if routes_missing:
            raise RuntimeError(
                "CoTrainRollouter cannot start with empty population vLLM routes: "
                f"attacker={len(attacker_urls)}, defender={len(defender_urls)}"
            )
        async with self.lock:
            self.old_attacker_urls = list(attacker_urls)
            self.old_defender_urls = list(defender_urls)
        logger.info(
            "[CoTrainRollouter] Population vLLM routes ready: attacker=%d defender=%d",
            len(attacker_urls),
            len(defender_urls),
        )

    async def set_old_model_manager(self, manager):
        if not self.population_enabled:
            raise RuntimeError("PopulationManager cannot be set when population is disabled")
        self.old_model_manager = manager
        failed_roles: set[str] = set()
        if self._pending_population_state:
            self.old_model_manager.load_state_dict(self._pending_population_state)
            failed_roles = await self.old_model_manager.restore_loaded_models()
            self._population_failed_roles.update(failed_roles)
            self._pending_population_state = None
        self.old_attacker_urls = self.old_model_manager.get_old_attacker_urls()
        self.old_defender_urls = (
            self.old_model_manager.get_old_defender_urls()
        )
        if "attacker" in failed_roles:
            self.old_attacker_urls = []
        if "defender" in failed_roles:
            self.old_defender_urls = []
        if failed_roles:
            logger.error(
                "[CoTrainRollouter] PopulationManager linked with disabled historical "
                "roles=%s; current-policy routes remain available",
                sorted(failed_roles),
            )
        else:
            logger.info("[CoTrainRollouter] PopulationManager linked")
        self._try_save_authoritative_state_locked("population-manager restore")
        return self.old_attacker_urls, self.old_defender_urls

    async def reconcile_policy_versions(
        self,
        attacker_param_version: int,
        defender_param_version: int,
    ) -> dict[str, int]:
        """Make rollout freshness match the checkpoints actually loaded by trainers."""

        versions = {
            "attacker": int(attacker_param_version),
            "defender": int(defender_param_version),
        }
        if any(version < 0 for version in versions.values()):
            raise ValueError(f"negative restored policy version: {versions}")
        async with self._population_refresh_lock:
            if self.policy_versions != versions:
                logger.warning(
                    "[CoTrainRollouter] Reconciling saved rollout versions %s to "
                    "trainer checkpoints %s",
                    self.policy_versions,
                    versions,
                )
            self.policy_versions = versions
            # Message queues are newly created on every launch, so no old
            # in-flight sample depth can survive the restart.
            self.role_staleness_samples = {"attacker": 0, "defender": 0}
            self.staleness_samples = 0
            self.training_steps = _training_step_for_mode(
                self.training_mode, versions
            )
            self._try_save_authoritative_state_locked("policy-version reconciliation")
        return dict(self.policy_versions)

    async def set_max_required_samples(self):
        async with self.lock:
            staleness_threshold = float(
                self.config.async_training.get("staleness_threshold", 1)
            )
            trigger_step = int(
                self.config.async_training.get("trigger_parameter_sync_step", 4)
            )
            required = int(self.config.actor_rollout_ref.actor.ppo_mini_batch_size)
            if (
                not math.isfinite(staleness_threshold)
                or staleness_threshold < 0
                or trigger_step <= 0
                or required <= 0
            ):
                raise ValueError(
                    "invalid async buffer settings: "
                    f"staleness_threshold={staleness_threshold}, "
                    f"trigger_parameter_sync_step={trigger_step}, "
                    f"ppo_mini_batch_size={required}"
                )
            # Bound the real message queues.  ``role_staleness_samples`` is only
            # queue-depth telemetry; it must never be used as a cumulative
            # generation counter because trainers consume samples asynchronously.
            self.max_required_samples = math.ceil(
                required * (staleness_threshold + 1.0) * trigger_step
            )
            self.max_queue_size = self.max_required_samples
            # With parallel n_rollouts_per_prompt, each prompt dispatches N concurrent
            # agent loop calls. Cap concurrent prompts to avoid overloading vLLM.
            n_rollouts = self.n_rollouts_per_prompt
            agent_workers = int(
                self.config.actor_rollout_ref.rollout.agent.get("num_workers", 128)
            )
            if agent_workers <= 0:
                raise ValueError("actor_rollout_ref.rollout.agent.num_workers must be positive")
            self.max_concurrent_samples = max(
                1,
                min(agent_workers // n_rollouts * 2, self.max_required_samples),
            )
            logger.info(
                f"[CoTrainRollouter] max_required_samples={self.max_required_samples} "
                f"(per-role on-policy buffer, staleness_threshold={staleness_threshold}) "
                f"max_queue_size={self.max_queue_size} "
                f"max_concurrent_samples={self.max_concurrent_samples} "
                f"agent_workers={agent_workers}"
            )

    def get_max_queue_size(self):
        return self.max_queue_size

    def get_total_train_steps(self):
        required = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
        trigger_step = self.config.async_training.get("trigger_parameter_sync_step", 4)
        return int(self.total_rollout_steps / (required * trigger_step))

    def _population_metrics(self) -> dict[str, float]:
        """Return compact population-health metrics for W&B."""
        if self.old_model_manager is None:
            return {}
        metrics: dict[str, float] = {}
        for role in ("attacker", "defender"):
            try:
                info = getattr(self.old_model_manager, f"get_{role}_population_info")()
            except Exception:
                continue
            steps = [int(item.get("step", 0)) for item in info]
            fitness = [float(item.get("fitness", 0.0)) for item in info]
            samples = [int(item.get("n_samples", 0)) for item in info]
            eligible = [bool(item.get("eligible", False)) for item in info]
            probation = [bool(item.get("probation", False)) for item in info]
            min_bucket_samples = [
                int(item.get("min_required_bucket_samples", 0)) for item in info
            ]
            latest_candidate_step = max(
                (int(item.get("latest_candidate_step", 0)) for item in info),
                default=0,
            )
            latest_eligible_step = max(
                (int(item.get("latest_eligible_step", 0)) for item in info),
                default=0,
            )
            pending_target_steps = [
                int(item.get("pending_target_step", -1)) for item in info
            ]
            prefix = f"population/{role}"
            metrics[f"{prefix}/slots"] = float(len(steps))
            metrics[f"{prefix}/unique_steps"] = float(len(set(steps)))
            metrics[f"{prefix}/mean_step"] = float(sum(steps) / len(steps)) if steps else 0.0
            metrics[f"{prefix}/mean_fitness"] = float(sum(fitness) / len(fitness)) if fitness else 0.0
            metrics[f"{prefix}/mean_samples"] = float(sum(samples) / len(samples)) if samples else 0.0
            metrics[f"{prefix}/eligible_slots"] = float(sum(eligible))
            metrics[f"{prefix}/probation_slots"] = float(sum(probation))
            metrics[f"{prefix}/min_required_bucket_samples"] = float(
                max((value for value, is_probation in zip(min_bucket_samples, probation) if is_probation), default=0)
            )
            metrics[f"{prefix}/latest_candidate_step"] = float(latest_candidate_step)
            metrics[f"{prefix}/latest_eligible_step"] = float(latest_eligible_step)
            metrics[f"{prefix}/freshness_lag"] = float(
                max(0, latest_candidate_step - max(steps, default=0))
            )
            metrics[f"{prefix}/pending_target_slots"] = float(
                sum(step >= 0 for step in pending_target_steps)
            )
        metrics["population/is_refreshing"] = float(
            bool(getattr(self.old_model_manager, "is_refreshing", False))
        )
        metrics["population/state_dirty"] = float(self._population_state_dirty)
        metrics["population/state_save_failures"] = float(
            self._population_state_save_failures
        )
        for role in ("attacker", "defender"):
            metrics[f"population/{role}/routing_disabled"] = float(
                role in self._population_draining_roles
                or role in self._population_failed_roles
            )
            metrics[f"population/{role}/refresh_failures"] = float(
                self._population_refresh_failures[role]
            )
            metrics[f"population/{role}/next_retry_version"] = float(
                self._population_next_retry_version[role]
            )
            metrics[f"population/{role}/fallback_requests"] = float(
                self._population_fallback_counts[role]
            )
        return metrics

    def _coevo_metrics(self) -> dict[str, float]:
        """Return comparisons only while their historical route is trustworthy."""

        metrics = self._coevo_tracker.get_metrics()
        comparison_roles = {
            "attacker": "coevo/attacker_improvement",
            "defender": "coevo/defender_improvement",
        }
        for role, metric_name in comparison_roles.items():
            route_available = (
                role not in self._population_draining_roles
                and role not in self._population_failed_roles
            )
            valid = route_available and metric_name in metrics
            if not valid:
                metrics.pop(metric_name, None)
            metrics[f"{metric_name}_valid"] = float(valid)
        return metrics

    def get_replicas(self):
        """CoTrainRollouter does not own rollout replicas directly; return empty list.

        FullyAsyncTrainer.set_rollouter() calls this to initialise CheckpointEngineManager.
        Returning [] means the manager is created but performs no NCCL sync (weight sync
        to current vLLM is handled separately via LLMServerManager / OldModelManager).
        """
        return []

    def get_attacker_replicas(self):
        if self.attacker_llm_server_manager:
            return self.attacker_llm_server_manager.get_replicas()
        return []

    def get_defender_replicas(self):
        if self.defender_llm_server_manager:
            return self.defender_llm_server_manager.get_replicas()
        return []

    # ==================== Model Pair Selection ====================

    def _select_model_pair(self) -> str:
        r = self._routing_rng.random()
        p1, p2, p3, p4, p5 = self.model_pair_probs
        if r < p1:
            selected = "curr_curr"
        elif r < p1 + p2:
            selected = "old_atk_curr_def"
        elif r < p1 + p2 + p3:
            selected = "curr_atk_old_def"
        elif r < p1 + p2 + p3 + p4:
            selected = "fixed_template"
        else:
            selected = "clean"

        unavailable_role = None
        if selected == "old_atk_curr_def":
            unavailable_role = "attacker"
        elif selected == "curr_atk_old_def":
            unavailable_role = "defender"
        if unavailable_role and (
            (unavailable_role in self._population_draining_roles) or (unavailable_role in self._population_failed_roles)
        ):
            self._population_fallback_counts[unavailable_role] += 1
            return "curr_curr"
        return selected

    def _get_urls_for_pair(
        self,
        model_pair: str,
        task_bucket: str = "unknown",
    ) -> tuple[str | None, str, str, str, dict]:
        """Return model endpoints plus immutable population attribution.

        For fixed_template and clean, atk_url is None (no attacker LLM generation needed).
        URL strings (not client objects) allow safe Ray serialization across worker processes.
        """
        if not self.current_defender_urls:
            raise RuntimeError("current defender vLLM routes are empty")
        def_idx = self._routing_rng.randrange(len(self.current_defender_urls))
        attribution = {
            "attacker_population_slot": -1,
            "defender_population_slot": -1,
            "attacker_population_step": -1,
            "defender_population_step": -1,
        }

        if model_pair in ("fixed_template", "clean"):
            def_url = self.current_defender_urls[def_idx]
            def_model = self.config.cotrain.defender_model_path
            return None, def_url, "", def_model, attribution

        if model_pair == "curr_curr":
            if not self.current_attacker_urls:
                raise RuntimeError("current attacker vLLM routes are empty")
            atk_idx = self._routing_rng.randrange(len(self.current_attacker_urls))
            atk_url = self.current_attacker_urls[atk_idx]
            def_url = self.current_defender_urls[def_idx]
            atk_model = self.config.cotrain.attacker_model_path
            def_model = self.config.cotrain.defender_model_path
        elif model_pair == "old_atk_curr_def":
            if not self.old_attacker_urls:
                raise RuntimeError("old attacker vLLM routes are empty")
            old_atk_idx = (
                self.old_model_manager.choose_population_slot("attacker", task_bucket)
                if self.old_model_manager is not None
                else self._routing_rng.randrange(len(self.old_attacker_urls))
            )
            atk_url = self.old_attacker_urls[old_atk_idx]
            def_url = self.current_defender_urls[def_idx]
            atk_model = "old_attacker"
            def_model = self.config.cotrain.defender_model_path
            attribution["attacker_population_slot"] = old_atk_idx
            if self.old_model_manager is not None:
                attribution["attacker_population_step"] = (
                    self.old_model_manager.get_loaded_step("attacker", old_atk_idx)
                )
        elif model_pair == "curr_atk_old_def":
            if not self.current_attacker_urls:
                raise RuntimeError("current attacker vLLM routes are empty")
            if not self.old_defender_urls:
                raise RuntimeError("old defender vLLM routes are empty")
            atk_idx = self._routing_rng.randrange(len(self.current_attacker_urls))
            atk_url = self.current_attacker_urls[atk_idx]
            old_def_idx = (
                self.old_model_manager.choose_population_slot("defender", task_bucket)
                if self.old_model_manager is not None
                else self._routing_rng.randrange(len(self.old_defender_urls))
            )
            def_url = self.old_defender_urls[old_def_idx]
            atk_model = self.config.cotrain.attacker_model_path
            def_model = "old_defender"
            attribution["defender_population_slot"] = old_def_idx
            if self.old_model_manager is not None:
                attribution["defender_population_step"] = (
                    self.old_model_manager.get_loaded_step("defender", old_def_idx)
                )
        else:
            raise ValueError(f"unsupported model pair: {model_pair!r}")

        return atk_url, def_url, atk_model, def_model, attribution

    def _reserve_population_requests(
        self,
        model_pair: str,
        attribution: dict,
        count: int,
    ) -> tuple[str, int] | None:
        if model_pair == "old_atk_curr_def":
            key = ("attacker", int(attribution["attacker_population_slot"]))
        elif model_pair == "curr_atk_old_def":
            key = ("defender", int(attribution["defender_population_slot"]))
        else:
            return None
        role, slot = key
        if slot < 0:
            raise RuntimeError(f"missing immutable population attribution for {model_pair}")
        self._population_inflight[role][slot] += int(count)
        return key

    def _release_population_requests(self, reservation: tuple[str, int] | None, count: int) -> None:
        if reservation is None:
            return
        role, slot = reservation
        remaining = self._population_inflight[role][slot] - int(count)
        if remaining < 0:
            raise RuntimeError(f"negative in-flight count for {role} population slot {slot}")
        if remaining:
            self._population_inflight[role][slot] = remaining
        else:
            del self._population_inflight[role][slot]

    # ==================== Sample Processing ====================

    @staticmethod
    def _batch_extra_info(batch) -> dict:
        """Decode one dataset row's extra_info without mutating the batch."""
        value = batch.non_tensor_batch.get("extra_info", [{}])
        if isinstance(value, np.ndarray):
            value = value[0] if value.size else {}
        if hasattr(value, "item") and not isinstance(value, (dict, str)):
            try:
                value = value.item()
            except Exception:
                pass
        if isinstance(value, str):
            try:
                import json
                value = json.loads(value)
            except Exception:
                return {}
        return value if isinstance(value, dict) else {}

    @classmethod
    def _is_native_clean_sample(cls, rollout_sample: RolloutSample) -> bool:
        """Return whether the parquet row is one of the native clean tasks."""
        extra = cls._batch_extra_info(rollout_sample.full_batch)
        return not extra.get("injection_task_id") and not extra.get("injections")

    async def _process_single_sample(self, rollout_sample: RolloutSample):
        """Process one prompt: run n_rollouts_per_prompt rollouts with same model pair."""
        # Clean-vs-injected is a property of the training row. Randomly
        # assigning a clean pair to an injected row (or dropping native clean
        # rows in AgentLoop) makes the RL distribution differ from SFT/data.
        row_extra_info = self._batch_extra_info(rollout_sample.full_batch)
        rollout_timestamp = time.time()
        start_policy_versions = dict(self.policy_versions)
        task_bucket = str(row_extra_info.get("suite_name", "unknown"))
        # Select, attribute, and reserve a historical slot atomically with
        # refresh.  Without this critical section a refresh could observe zero
        # in-flight requests after URL selection but before reservation, then
        # restart the exact replica that this sample was about to call.
        async with self._population_refresh_lock:
            model_pair = (
                "clean"
                if self._is_native_clean_sample(rollout_sample)
                else self._select_model_pair()
            )
            uses_current_attacker = model_pair in ("curr_curr", "curr_atk_old_def")
            uses_current_defender = model_pair in (
                "curr_curr",
                "old_atk_curr_def",
                "fixed_template",
                "clean",
            )
            atk_url, def_url, atk_model, def_model, population_attribution = (
                self._get_urls_for_pair(model_pair, task_bucket=task_bucket)
            )

            # Shallow-copy batch and inject routing metadata as plain strings so they survive
            # Ray cloudpickle serialization when dispatched to AgentLoopWorker actors.
            batch_data = rollout_sample.full_batch
            routing_batch = copy.copy(batch_data)
            routing_batch.non_tensor_batch = dict(batch_data.non_tensor_batch)
            routing_batch.non_tensor_batch["model_pair"] = np.array(
                [model_pair], dtype=object
            )
            routing_batch.non_tensor_batch["attacker_openai_url"] = np.array(
                [atk_url or ""], dtype=object
            )
            routing_batch.non_tensor_batch["defender_openai_url"] = np.array(
                [def_url], dtype=object
            )
            routing_batch.non_tensor_batch["attacker_model_name"] = np.array(
                [atk_model], dtype=object
            )
            routing_batch.non_tensor_batch["defender_model_name"] = np.array(
                [def_model], dtype=object
            )
            for metadata_key, metadata_value in population_attribution.items():
                routing_batch.non_tensor_batch[metadata_key] = np.array(
                    [metadata_value], dtype=np.int64
                )
            population_reservation = self._reserve_population_requests(
                model_pair,
                population_attribution,
                self.n_rollouts_per_prompt,
            )

        _ROUTING_KEYS = (
            "model_pair", "attacker_openai_url", "defender_openai_url",
            "attacker_model_name", "defender_model_name",
            "attacker_population_slot", "defender_population_slot",
            "attacker_population_step", "defender_population_step",
        )

        attacker_results: list = []
        # Generate a shared uid for all rollouts from this prompt (required by GRPO grouping)
        prompt_uid = str(uuid.uuid4())
        defender_results: list = []
        evaluated_asr: list[bool] = []

        # Dispatch all n_rollouts_per_prompt rollouts in parallel.
        # Each rollout still has a fresh environment and independent generation.
        rollout_requests = []
        try:
            for _ in range(self.n_rollouts_per_prompt):
                rollout_requests.append(
                    self.agent_loop_manager.generate_sequences_single(routing_batch)
                )
            raw_outputs = await asyncio.gather(*rollout_requests, return_exceptions=True)
        finally:
            # Once the OpenAI requests have returned, the old vLLM slot is safe
            # to drain.  Reward processing below uses immutable attribution and
            # no longer needs the live endpoint.
            self._release_population_requests(population_reservation, len(rollout_requests))

        for output in raw_outputs:
            if isinstance(output, Exception):
                logger.warning(f"[CoTrainRollouter] Rollout failed: {output}", exc_info=output)
                continue
            if output is None:
                continue

            # Remove routing keys from output so they don't pollute the training batch.
            for key in _ROUTING_KEYS:
                output.non_tensor_batch.pop(key, None)

            end_policy_versions = dict(self.policy_versions)
            output.non_tensor_batch["min_global_steps"] = np.array(
                [start_policy_versions["attacker"]], dtype=object
            )
            output.non_tensor_batch["max_global_steps"] = np.array(
                [end_policy_versions["attacker"]], dtype=object
            )
            output.non_tensor_batch["attacker_policy_version"] = np.array(
                [start_policy_versions["attacker"]], dtype=np.int64
            )
            output.non_tensor_batch["defender_policy_version"] = np.array(
                [start_policy_versions["defender"]], dtype=np.int64
            )
            output.non_tensor_batch["attacker_policy_version_end"] = np.array(
                [end_policy_versions["attacker"]], dtype=np.int64
            )
            output.non_tensor_batch["defender_policy_version_end"] = np.array(
                [end_policy_versions["defender"]], dtype=np.int64
            )
            output.non_tensor_batch["rollout_timestamp"] = np.array([rollout_timestamp], dtype=np.float64)
            output.non_tensor_batch["model_pair"] = np.array([model_pair], dtype=object)
            for metadata_key, metadata_value in population_attribution.items():
                output.non_tensor_batch[metadata_key] = np.array([metadata_value], dtype=np.int64)
            output.non_tensor_batch["uid"] = np.array([prompt_uid], dtype=object)
            self._normalize_reward_extra_dataproto(output, ATTACKER_REWARD_EXTRA_KEYS)

            # The adaptive loop may discover that the defender never read an
            # injection vector.  Such a trajectory still trains the defender,
            # but contains no attacker action and must not enter attacker PPO.
            output_uses_current_attacker = bool(
                output.non_tensor_batch.get("uses_current_attacker", [uses_current_attacker])[0]
            )
            output_uses_current_defender = bool(
                output.non_tensor_batch.get("uses_current_defender", [uses_current_defender])[0]
            )
            has_attacker_action = bool(
                output.non_tensor_batch.get(
                    "has_attacker_action", [model_pair not in ("fixed_template", "clean")]
                )[0]
            )
            has_injected_payload = bool(
                output.non_tensor_batch.get("has_injected_payload", [has_attacker_action])[0]
            )
            attacker_sample_valid = bool(
                output.non_tensor_batch.get("attacker_sample_valid", [False])[0]
            )
            injection_reached = bool(
                float(
                    output.non_tensor_batch.get(
                        "attacker/num_injection_points", [0.0]
                    )[0]
                ) > 0.0
            )

            # Defender AgentLoopOutput was stored in extra_fields["_cotrain_defender_output"]
            # by CoTrainAgentLoop.run(), then transferred to non_tensor_batch by _postprocess.
            def_arr = output.non_tensor_batch.pop("_cotrain_defender_output", None)
            def_alo = def_arr[0] if def_arr is not None else None
            defender_sample_valid = bool(
                def_alo
                and getattr(def_alo, "extra_fields", {}).get("defender_sample_valid", False)
            )
            def_reward_info = getattr(def_alo, "extra_fields", {}).get("reward_extra_info", {}) if def_alo else {}
            self._rollout_quality_tracker.record(
                model_pair=model_pair,
                learned_attacker=model_pair not in ("fixed_template", "clean"),
                attacker_attempted=injection_reached,
                has_injected_payload=bool(output.non_tensor_batch.get("has_injected_payload", [False])[0]),
                attacker_sample_valid=bool(output.non_tensor_batch.get("attacker_sample_valid", [False])[0]),
                defender_sample_valid=defender_sample_valid,
                format_violation=float(output.non_tensor_batch.get("attacker/format_violation", [float("nan")])[0]),
                action_asr=float(output.non_tensor_batch.get("attacker/asr_success", [float("nan")])[0]),
                effective_asr=float(output.non_tensor_batch.get("attacker/effective_asr", [float("nan")])[0]),
                tool_responses=float(output.non_tensor_batch.get("attacker/tool_responses", [float("nan")])[0]),
                attacker_generated_turns=float(output.non_tensor_batch.get("attacker/num_generated_turns", [float("nan")])[0]),
                attacker_length_finishes=float(output.non_tensor_batch.get("attacker/length_finish_count", [float("nan")])[0]),
                tool_responses_truncated=float(output.non_tensor_batch.get("attacker/tool_responses_truncated", [float("nan")])[0]),
                payloads_survived=float(output.non_tensor_batch.get("attacker/payloads_survived_truncation", [float("nan")])[0]),
                payloads_truncated=float(output.non_tensor_batch.get("attacker/payloads_truncated", [float("nan")])[0]),
                attacker_trace_drops=float(output.non_tensor_batch.get("attacker/exact_trace_drops", [float("nan")])[0]),
                defender_trace_drops=float(def_reward_info.get("defender/exact_trace_drops", float("nan"))),
                attacker_train_sequence_tokens=float(
                    output.non_tensor_batch.get(
                        "attacker/train_sequence_tokens", [float("nan")]
                    )[0]
                ),
                attacker_train_sequence_overflow=float(
                    output.non_tensor_batch.get(
                        "attacker/train_sequence_overflow", [float("nan")]
                    )[0]
                ),
                defender_train_sequence_tokens=float(
                    def_reward_info.get("defender/train_sequence_tokens", float("nan"))
                ),
                defender_train_sequence_overflow=float(
                    def_reward_info.get("defender/train_sequence_overflow", float("nan"))
                ),
            )
            # Evaluation validity is intentionally separate from PPO validity.
            # Once an injection point was reached and the defender trajectory
            # is evaluable, malformed/empty/overlong attacker output is a real
            # failed attack (ASR=0), not a missing population sample.  Only the
            # PPO queue still requires an exactly aligned attacker trace.
            interaction_eval_valid = defender_sample_valid and injection_reached
            defender_population_eval_valid = (
                interaction_eval_valid
                and attacker_sample_valid
                and has_injected_payload
            )

            # Record to co-evolution tracker
            _asr = bool(output.non_tensor_batch.get("asr_success", [False])[0])
            _util_fail = False
            if def_alo and hasattr(def_alo, "extra_fields"):
                _rei = def_alo.extra_fields.get("reward_extra_info", {})
                _util_fail = not bool(_rei.get("defender/utility_success", 1.0))
            # Effective ASR is reached-only: after the environment exposes an
            # injection point, malformed/empty attacker output counts as a
            # failed attempt.  This prevents format collapse from disappearing
            # out of the metric denominator.
            if (
                interaction_eval_valid
                and (output_uses_current_attacker or output_uses_current_defender)
            ):
                self._coevo_tracker.record(model_pair, _asr, _util_fail)
                evaluated_asr.append(_asr)
            # Task-level ASR tracking
            _extra = output.non_tensor_batch.get("extra_info", [None])[0]
            if isinstance(_extra, str):
                import json as _json
                try:
                    _extra = _json.loads(_extra)
                except Exception:
                    _extra = {}
            _suite = _extra.get("suite_name", "unknown") if isinstance(_extra, dict) else "unknown"
            _inj_id = (
                _extra.get("injection_task_id", "unknown")
                if isinstance(_extra, dict)
                else "unknown"
            )
            if (
                interaction_eval_valid
                and isinstance(_extra, dict)
                and (output_uses_current_attacker or output_uses_current_defender)
            ):
                self._task_asr_tracker.record(_suite, _inj_id, _asr)

            # Report actual reward to population manager for fitness-weighted selection
            if self.old_model_manager is not None:
                if model_pair == "old_atk_curr_def":
                    if interaction_eval_valid:
                        old_step = int(population_attribution["attacker_population_step"])
                        # Population strength is authoritative ASR only.  The
                        # PPO-only 0.05 format bonus must not select elites.
                        self.old_model_manager.report_attacker_reward(
                            old_step, float(_asr), _suite
                        )
                elif model_pair == "curr_atk_old_def":
                    # Defender strength is defined only when it actually saw a
                    # non-empty adversarial payload.  Crediting an old defender
                    # for a malformed/empty current-attacker action would make
                    # its fitness mostly measure opponent failure.
                    if defender_population_eval_valid:
                        old_step = int(population_attribution["defender_population_step"])
                        def_reward_info = def_alo.extra_fields.get("reward_extra_info", {}) if def_alo else {}
                        def_reward = float(def_reward_info.get("defender/reward", 0.0))
                        self.old_model_manager.report_defender_reward(old_step, def_reward, _suite)

            if (
                uses_current_attacker
                and output_uses_current_attacker
                and has_attacker_action
                and attacker_sample_valid
            ):
                attacker_results.append(output)
            if (
                uses_current_defender
                and output_uses_current_defender
                and def_alo is not None
                and defender_sample_valid
            ):
                # A trajectory that reached no injection point still carries
                # authoritative utility supervision for the defender.  Keeping
                # it is important: otherwise a defender that fails before the
                # injectable tool call receives no gradient and can permanently
                # starve the attacker of future actions.  It remains excluded
                # from ASR/population fitness above and never enters attacker PPO.
                def_output = self._pack_defender_output(
                    def_alo,
                    start_policy_versions=start_policy_versions,
                    end_policy_versions=end_policy_versions,
                    rollout_timestamp=rollout_timestamp,
                    model_pair=model_pair,
                    population_attribution=population_attribution,
                )
                def_output.non_tensor_batch["uid"] = np.array([prompt_uid], dtype=object)
                self._normalize_reward_extra_dataproto(def_output, DEFENDER_REWARD_EXTRA_KEYS)
                defender_results.append(def_output)

        # Prompt-level ASR: if ANY rollout in this group succeeded, prompt is "passed"
        group_any_asr = any(evaluated_asr) if evaluated_asr else False
        _extra = rollout_sample.full_batch.non_tensor_batch.get("extra_info", [None])[0]
        if isinstance(_extra, str):
            import json as _json
            try:
                _extra = _json.loads(_extra)
            except Exception:
                _extra = {}
        if evaluated_asr and isinstance(_extra, dict):
            _suite = _extra.get("suite_name", "unknown")
            self._prompt_asr_tracker.record(_suite, model_pair, group_any_asr)

        await self._push_to_queues(attacker_results, defender_results, rollout_sample)

    async def _push_to_queues(
        self,
        attacker_results: list,
        defender_results: list,
        rollout_sample: RolloutSample,
    ):
        """Concatenate DataProto lists and push to respective message queues."""
        from verl.protocol import DataProto

        if (
            (attacker_results) and (self.attacker_mq_client)
        ):
            atk_batch = DataProto.concat(attacker_results) if len(attacker_results) > 1 else attacker_results[0]
            atk_sample = RolloutSample(
                full_batch=atk_batch,
                sample_id=f"atk_{rollout_sample.sample_id}",
                epoch=rollout_sample.epoch,
                rollout_status={"model_pair_dist": "attacker_training_data"},
            )
            success = await self.attacker_mq_client.put_sample(sample=ray.cloudpickle.dumps(atk_sample))
            if success:
                self.total_generated_samples += 1
            else:
                self.dropped_samples += 1

        if defender_results and self.defender_mq_client:
            def_batch = DataProto.concat(defender_results) if len(defender_results) > 1 else defender_results[0]
            def_sample = RolloutSample(
                full_batch=def_batch,
                sample_id=f"def_{rollout_sample.sample_id}",
                epoch=rollout_sample.epoch,
                rollout_status={"model_pair_dist": "defender_training_data"},
            )
            success = await self.defender_mq_client.put_sample(sample=ray.cloudpickle.dumps(def_sample))
            if success:
                self.total_generated_samples += 1
            else:
                self.dropped_samples += 1

    def _normalize_reward_extra_dataproto(self, data, reward_extra_keys: tuple[str, ...]):
        """Ensure reward extra fields have a stable schema before DataProto.concat."""
        reward_extra_keys = list(reward_extra_keys)
        prefix = reward_extra_keys[0].split("/", 1)[0] + "/" if reward_extra_keys else ""
        batch_size = data.batch.batch_size[0] if data.batch is not None else 1

        # Drop role-specific reward keys outside the canonical schema, then fill
        # missing keys. DataProto.concat requires both non_tensor keys and
        # meta_info values to be consistent across rollout samples.
        for key in list(data.non_tensor_batch.keys()):
            if prefix and key.startswith(prefix) and key not in reward_extra_keys:
                data.non_tensor_batch.pop(key, None)
        for key in reward_extra_keys:
            if key not in data.non_tensor_batch:
                data.non_tensor_batch[key] = np.full(batch_size, float("nan"), dtype=np.float32)

        data.meta_info["reward_extra_keys"] = reward_extra_keys

    def _pack_defender_output(
        self,
        alo: "AgentLoopOutput",
        *,
        start_policy_versions: dict[str, int],
        end_policy_versions: dict[str, int],
        rollout_timestamp: float,
        model_pair: str,
        population_attribution: dict[str, int],
    ) -> "DataProto":
        """Convert a defender AgentLoopOutput into a padded DataProto for the defender trainer.

        Mirrors _agent_loop_postprocess / _postprocess: left-pad prompt, right-pad response,
        and place the terminal reward in rm_scores at the last non-padding response position.
        """
        from tensordict import TensorDict
        from verl.protocol import DataProto

        pad_id = self.defender_tokenizer.pad_token_id or 0
        max_prompt = self._defender_max_prompt_len
        max_resp = self._defender_max_resp_len

        prompt_ids = list(alo.prompt_ids)
        resp_ids = list(alo.response_ids)
        resp_mask_raw = list(alo.response_mask)

        # Left-pad prompt
        prompt_len = len(prompt_ids)
        if prompt_len > max_prompt:
            raise ValueError(
                f"defender prompt exceeds configured limit: {prompt_len} > {max_prompt}"
            )
        prompt_ids = [pad_id] * (max_prompt - prompt_len) + prompt_ids
        actual_prompt_len = prompt_len

        # Right-pad response
        resp_len = len(resp_ids)
        if resp_len > max_resp:
            raise ValueError(
                f"defender response exceeds configured limit: {resp_len} > {max_resp}"
            )
        resp_ids = resp_ids + [pad_id] * (max_resp - resp_len)
        resp_mask_raw = resp_mask_raw + [0] * (max_resp - resp_len)
        actual_resp_len = resp_len

        # Handle response logprobs (right-pad with 0.0)
        resp_logprobs_raw = list(alo.response_logprobs) if alo.response_logprobs else None
        if resp_logprobs_raw is not None:
            if len(resp_logprobs_raw) != resp_len:
                raise ValueError(
                    "defender response token/logprob length mismatch before padding: "
                    f"{resp_len} != {len(resp_logprobs_raw)}"
                )
            resp_logprobs_raw = resp_logprobs_raw + [0.0] * (max_resp - len(resp_logprobs_raw))

        prompt_t = torch.tensor([prompt_ids], dtype=torch.long)
        resp_t = torch.tensor([resp_ids], dtype=torch.long)
        resp_mask_t = torch.tensor([resp_mask_raw], dtype=torch.float32)
        input_ids = torch.cat([prompt_t, resp_t], dim=1)

        # attention_mask: 0 for left-pad tokens, 1 for actual prompt + actual response tokens
        prompt_attn = torch.zeros(1, max_prompt, dtype=torch.long)
        prompt_attn[0, max_prompt - actual_prompt_len:] = 1
        resp_attn = torch.zeros(1, max_resp, dtype=torch.long)
        resp_attn[0, :actual_resp_len] = 1
        attention_mask = torch.cat([prompt_attn, resp_attn], dim=1)

        position_ids = (attention_mask.cumsum(dim=1) - 1).clamp(min=0)

        # rm_scores: place each step reward at its token position
        rm_scores = torch.zeros(1, max_resp, dtype=torch.float32)
        step_rm_scores = alo.extra_fields.get("step_rm_scores", [])
        if step_rm_scores:
            for token_idx, reward in step_rm_scores:
                clamped = min(int(token_idx), max_resp - 1)
                rm_scores[0, clamped] = reward
        elif alo.reward_score is not None and actual_resp_len > 0:
            rm_scores[0, actual_resp_len - 1] = alo.reward_score

        batch = TensorDict(
            {
                "prompts": prompt_t,
                "responses": resp_t,
                "response_mask": resp_mask_t,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "rm_scores": rm_scores,
                "rollout_log_probs": torch.tensor(
                    [resp_logprobs_raw if resp_logprobs_raw is not None else [0.0] * max_resp],
                    dtype=torch.float32,
                ),
            },
            batch_size=1,
        )

        reward_extra_info = dict(alo.extra_fields.get("reward_extra_info", {}))
        if any(key.startswith("attacker/") for key in reward_extra_info):
            reward_extra_keys = list(ATTACKER_REWARD_EXTRA_KEYS)
        elif any(key.startswith("defender/") for key in reward_extra_info):
            reward_extra_keys = list(DEFENDER_REWARD_EXTRA_KEYS)
        else:
            reward_extra_keys = sorted(reward_extra_info.keys())

        non_tensor_batch = {
            "__num_turns__": np.array([alo.num_turns], dtype=np.int32),
            "min_global_steps": np.array([start_policy_versions["defender"]], dtype=object),
            "max_global_steps": np.array([end_policy_versions["defender"]], dtype=object),
            "attacker_policy_version": np.array([start_policy_versions["attacker"]], dtype=np.int64),
            "defender_policy_version": np.array([start_policy_versions["defender"]], dtype=np.int64),
            "attacker_policy_version_end": np.array([end_policy_versions["attacker"]], dtype=np.int64),
            "defender_policy_version_end": np.array([end_policy_versions["defender"]], dtype=np.int64),
            "rollout_timestamp": np.array([rollout_timestamp], dtype=np.float64),
            "model_pair": np.array([model_pair], dtype=object),
        }
        for metadata_key, metadata_value in population_attribution.items():
            non_tensor_batch[metadata_key] = np.array([metadata_value], dtype=np.int64)
        for key in reward_extra_keys:
            non_tensor_batch[key] = np.array([reward_extra_info.get(key, float("nan"))])

        meta_info = {
            "metrics": [alo.metrics.model_dump()],
            "reward_extra_keys": reward_extra_keys,
        }

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)

    # ==================== Data Feed ====================

    def _create_continuous_iterator(self):
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                yield epoch, batch_dict

    async def _feed_samples(self):
        # Rebuild the exact original sampler stream, then advance to the
        # dataset-relative cursor persisted in the rollouter checkpoint.
        _restore_sampler_replay_state(
            self.train_sampler, self._sampler_replay_state
        )
        continuous_iterator = self._create_continuous_iterator()
        resume_cursor = int(self.dataset_feed_cursor)
        stream_position = 0
        reached_limit = False
        for epoch, batch_dict in continuous_iterator:
            # Do not use prepare_single_generation_data here: it repeats each
            # prompt for batched vLLM generation, while CoTrainRollouter owns
            # rollout multiplicity in _process_single_sample.  Keep batched
            # dataloading for CPU efficiency, then dispatch each row as an
            # independent environment so per-sample routing metadata stays
            # aligned with the DataProto batch dimension.
            from verl.protocol import DataProto
            loader_batch = DataProto.from_single_dict(batch_dict)

            for full_batch in loader_batch.split(1):
                if stream_position < resume_cursor:
                    stream_position += 1
                    continue
                if self.dataset_feed_cursor >= self.total_rollout_steps:
                    reached_limit = True
                    break

                sample_id = f"cotrain_{epoch}_{self.global_steps}"
                rollout_sample = RolloutSample(
                    full_batch=full_batch,
                    sample_id=sample_id,
                    epoch=epoch,
                    rollout_status={},
                )
                await self.pending_queue.put(rollout_sample)
                stream_position += 1
                self.dataset_feed_cursor = stream_position
                self.global_steps += 1

            if reached_limit:
                logger.info(f"[CoTrainRollouter] Reached max rollout steps: {self.total_rollout_steps}")
                break

        await self.pending_queue.put(None)
        logger.info(
            "[CoTrainRollouter] Feed complete, dataset cursor=%d/%d, global step=%d",
            self.dataset_feed_cursor,
            self.total_rollout_steps,
            self.global_steps,
        )

    # ==================== Processor Worker ====================

    async def _processor_worker(self):
        while True:
            if self.paused or await self._should_pause_generation():
                async with self.lock:
                    self.paused = True
                    self._resume_event.clear()

                resume_future = asyncio.ensure_future(self._resume_event.wait())
                try:
                    while self.active_tasks and not resume_future.done():
                        wait_set = set(self.active_tasks) | {resume_future}
                        done, _ = await asyncio.wait(wait_set, return_when=asyncio.FIRST_COMPLETED)
                        actual_done = done - {resume_future}
                        if actual_done:
                            async with self.lock:
                                for task in actual_done:
                                    self.active_tasks.discard(task)
                                    await task
                        if resume_future in done:
                            break

                    if not resume_future.done():
                        self.idle_start_time = time.time()
                        await resume_future
                finally:
                    if not resume_future.done():
                        resume_future.cancel()
                        await asyncio.gather(resume_future, return_exceptions=True)
                continue

            rollout_sample = await self.pending_queue.get()
            self.pending_queue.task_done()
            if rollout_sample is None:
                while self.active_tasks:
                    async with self.lock:
                        if self.active_tasks:
                            done_tasks, self.active_tasks = await asyncio.wait(
                                self.active_tasks, return_when=asyncio.FIRST_COMPLETED
                            )
                            for task in done_tasks:
                                await task
                break

            while len(self.active_tasks) >= self.max_concurrent_samples:
                async with self.lock:
                    if self.active_tasks:
                        done_tasks, self.active_tasks = await asyncio.wait(
                            self.active_tasks, return_when=asyncio.FIRST_COMPLETED
                        )
                        for task in done_tasks:
                            await task

            if self.paused:
                await self._resume_event.wait()

            async with self.lock:
                task = safe_create_task(
                    self._process_single_sample(rollout_sample),
                    name=rollout_sample.sample_id,
                    task_set=self.active_tasks,
                )

    # ==================== Pause / Resume ====================

    async def _refresh_queue_depth_metrics(self) -> tuple[int, int]:
        """Read authoritative queue depths and refresh legacy metric names.

        Older checkpoints called these values ``staleness_samples`` and stored
        cumulative enqueue counts.  Keeping the names avoids a W&B/state schema
        break, while making the values represent the only quantity useful for
        backpressure: samples that are still waiting to be consumed.
        """

        atk_size = (
            await self.attacker_mq_client.get_queue_size()
            if (self.attacker_mq_client)
            else 0
        )
        def_size = (
            await self.defender_mq_client.get_queue_size()
            if self.defender_mq_client
            else 0
        )
        self.role_staleness_samples = {
            "attacker": int(atk_size),
            "defender": int(def_size),
        }
        self.staleness_samples = max(self.role_staleness_samples.values())
        return int(atk_size), int(def_size)

    async def _should_pause_generation(self) -> bool:
        atk_size, def_size = await self._refresh_queue_depth_metrics()

        return _training_queues_full(
            self.training_mode,
            atk_size,
            def_size,
            self.max_queue_size,
        )

    async def reset_staleness(self, trainer_role: str, param_version: int):
        """Advance exactly one role's rollout version after its weight sync."""
        if trainer_role not in self.policy_versions:
            raise ValueError(f"unknown co-training role: {trainer_role!r}")
        async with self.lock:
            self.paused = False
            self._resume_event.set()
            self.policy_versions[trainer_role] = max(
                self.policy_versions[trainer_role], int(param_version)
            )
            self.training_steps = _training_step_for_mode(
                self.training_mode, self.policy_versions
            )

            await self._refresh_queue_depth_metrics()

            timing_raw = {}
            version_time = max(time.time() - self.step_start_time, 1e-6)
            if self.idle_start_time > self.step_start_time:
                active_time = self.idle_start_time - self.step_start_time
                idle_ratio = 1 - active_time / version_time
            else:
                active_time = version_time
                idle_ratio = 0

            timing_raw["fully_async/rollouter/active_time"] = active_time
            timing_raw["fully_async/rollouter/version_time"] = version_time
            timing_raw["fully_async/rollouter/idle_ratio"] = idle_ratio
            timing_raw[f"fully_async/{trainer_role}/policy_version"] = self.policy_versions[trainer_role]
            timing_raw[f"fully_async/{trainer_role}/staleness_samples"] = self.role_staleness_samples[trainer_role]

            # Include co-evolution metrics so they reach wandb
            timing_raw.update(self._coevo_metrics())
            timing_raw.update(self._task_asr_tracker.get_metrics())
            timing_raw.update(self._prompt_asr_tracker.get_metrics())
            timing_raw.update(self._rollout_quality_tracker.get_metrics())
            timing_raw.update(self._population_metrics())

            logger.info(
                f"[CoTrainRollouter] reset_staleness role={trainer_role} "
                f"version={self.policy_versions[trainer_role]} "
                f"role_depths={self.role_staleness_samples}, "
                f"idle_ratio: {idle_ratio:.4f}"
            )
            self.step_start_time = time.time()

        return timing_raw

    # ==================== Old Model Refresh ====================

    async def maybe_refresh_old_model(self, trainer_role: str, param_version: int):
        """Drain one historical role, then refresh it without killing requests."""
        if self.old_model_manager is None or param_version <= 0:
            return
        if trainer_role not in ("attacker", "defender"):
            raise ValueError(f"Unknown co-training role: {trainer_role}")
        param_version = int(param_version)
        if param_version < self._population_next_retry_version[trainer_role]:
            return False
        if not self.old_model_manager.update_due(trainer_role, param_version):
            return

        async with self._population_update_lock:
            async with self._population_refresh_lock:
                if not self.old_model_manager.update_due(trainer_role, param_version):
                    return
                if param_version < self._population_next_retry_version[trainer_role]:
                    return False
                # New samples can continue through the routing gate, but this
                # role's historical pair is redirected to current-current.
                self._population_draining_roles.add(trainer_role)
            refreshed = False
            try:
                drained = await self._wait_for_population_role_idle(trainer_role)
                if not drained:
                    self._population_refresh_failures[trainer_role] += 1
                    self._population_next_retry_version[trainer_role] = (
                        param_version + self._population_refresh_retry_interval
                    )
                    logger.error(
                        "[CoTrainRollouter] Refusing to refresh %s population: "
                        "requests did not drain within %.1fs; next retry is "
                        "version >= %d",
                        trainer_role,
                        self._population_drain_timeout_s,
                        self._population_next_retry_version[trainer_role],
                    )
                    return False
                if trainer_role == "attacker":
                    await self.old_model_manager.maybe_update_attacker(param_version)
                    self.old_attacker_urls = (
                        self.old_model_manager.get_old_attacker_urls()
                    )
                else:
                    await self.old_model_manager.maybe_update_defender(param_version)
                    self.old_defender_urls = (
                        self.old_model_manager.get_old_defender_urls()
                    )
                comparison_pair = (
                    "old_atk_curr_def"
                    if trainer_role == "attacker"
                    else "curr_atk_old_def"
                )
                # The opponent mixture changed.  Do not compare the new
                # population against samples retained from the previous one;
                # wait for a fresh comparison window instead.
                self._coevo_tracker.reset_pair(comparison_pair)
                self._population_failed_roles.discard(trainer_role)
                self._population_next_retry_version[trainer_role] = 0
                refreshed = True
            except Exception:
                # A partial restart can leave a dead endpoint in that historical
                # role.  Keep routing on current-current until a later refresh
                # successfully repairs the population.
                self._population_failed_roles.add(trainer_role)
                self._population_refresh_failures[trainer_role] += 1
                self._population_next_retry_version[trainer_role] = (
                    param_version + self._population_refresh_retry_interval
                )
                logger.exception(
                    "[CoTrainRollouter] Population refresh failed for %s at "
                    "version=%d; disabling only that historical route and "
                    "continuing current-policy training; next retry is "
                    "version >= %d",
                    trainer_role,
                    param_version,
                    self._population_next_retry_version[trainer_role],
                )
            finally:
                async with self._population_refresh_lock:
                    self._population_draining_roles.discard(trainer_role)
                    # The trainer checkpoint is written before refresh.  Persist
                    # the post-refresh slot mapping (or failed status telemetry)
                    # immediately so resume never rolls population membership back.
                    self._try_save_authoritative_state_locked(
                        f"post-refresh role={trainer_role} version={param_version}"
                    )
            return refreshed

    async def _wait_for_population_role_idle(self, role: str) -> bool:
        deadline = time.monotonic() + self._population_drain_timeout_s
        while sum(self._population_inflight[role].values()) > 0:
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(0.1)
        return True

    # ==================== Main Loop ====================

    async def fit(self):
        """Main entry point — start async generation loop."""
        logger.info("[CoTrainRollouter] Starting fit()...")

        if not self.defender_mq_client:
            raise ValueError("Defender MQ client not set. Call set_mq_clients() first.")
        if (not self.attacker_mq_client):
            raise ValueError("Attacker MQ client not set for dual training.")
        if not self.current_defender_urls:
            raise RuntimeError("current defender vLLM routes are empty")
        if (not self.current_attacker_urls):
            raise RuntimeError("current attacker vLLM routes are empty")

        async with self.lock:
            self.paused = False
            self.running = True
            self._resume_event.set()

        generation_task = safe_create_task(self._streaming_generation_main(), name="generation_task")
        monitor_task = safe_create_task(self._async_monitor_loop(), name="monitor_task")

        try:
            results = await asyncio.gather(
                generation_task,
                monitor_task,
                return_exceptions=True,
            )
            failures = [result for result in results if isinstance(result, BaseException)]
            if failures:
                raise RuntimeError(
                    f"CoTrainRollouter background task failed: {failures[0]}"
                ) from failures[0]
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("[CoTrainRollouter] fit() failed")
            raise
        finally:
            if not generation_task.done():
                generation_task.cancel()
            if not monitor_task.done():
                monitor_task.cancel()
            await asyncio.gather(generation_task, monitor_task, return_exceptions=True)

        logger.info("[CoTrainRollouter] fit() completed")

    async def _streaming_generation_main(self):
        logger.info(f"[CoTrainRollouter] Starting streaming, max_concurrent={self.max_concurrent_samples}")

        feed_task = safe_create_task(self._feed_samples(), name="feed_task")
        processor_task = safe_create_task(self._processor_worker(), name="processor_task")

        try:
            done, pending = await asyncio.wait(
                [feed_task, processor_task], return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                if task.exception():
                    raise task.exception()

            if feed_task not in done:
                raise RuntimeError("Processor exited before feed completed")

            await processor_task
            await self.pending_queue.join()

        except Exception as e:
            logger.error(f"[CoTrainRollouter] Streaming error: {e}")
            raise
        finally:
            if feed_task and not feed_task.done():
                feed_task.cancel()
                await asyncio.gather(feed_task, return_exceptions=True)
            if processor_task and not processor_task.done():
                processor_task.cancel()
                await asyncio.gather(processor_task, return_exceptions=True)

            # Only active trainers receive end-of-stream sentinels.
            if (self.attacker_mq_client):
                await self.attacker_mq_client.put_sample(sample=None)
            if self.defender_mq_client:
                await self.defender_mq_client.put_sample(sample=None)

            async with self.lock:
                self.running = False

    async def _async_monitor_loop(self):
        stats_interval = 60.0
        check_interval = 10.0
        last_stats_time = time.time()

        while True:
            async with self.lock:
                if not self.running:
                    break
            await asyncio.sleep(check_interval)

            current_time = time.time()
            if current_time - last_stats_time >= stats_interval:
                stats = await self.get_statistics()
                logger.info(f"[CoTrainRollouter] Stats: {pformat(stats)}")
                last_stats_time = current_time

            if self.paused and not await self._should_pause_generation():
                async with self.lock:
                    self.paused = False
                    self._resume_event.set()

    async def get_statistics(self) -> dict:
        atk_stats = (
            await self.attacker_mq_client.get_statistics()
            if (self.attacker_mq_client)
            else {}
        )
        def_stats = await self.defender_mq_client.get_statistics() if self.defender_mq_client else {}

        self.role_staleness_samples = {
            "attacker": int(atk_stats.get("queue_size", 0)),
            "defender": int(def_stats.get("queue_size", 0)),
        }
        self.staleness_samples = max(self.role_staleness_samples.values())

        return {
            "active_tasks": len(self.active_tasks),
            "pending_queue_size": self.pending_queue.qsize() if self.pending_queue else 0,
            "attacker_mq_size": atk_stats.get("queue_size", 0),
            "defender_mq_size": def_stats.get("queue_size", 0),
            "total_generated": self.total_generated_samples,
            "dropped": self.dropped_samples,
            "staleness_samples": self.staleness_samples,
            "attacker_staleness_samples": self.role_staleness_samples["attacker"],
            "defender_staleness_samples": self.role_staleness_samples["defender"],
            "attacker_policy_version": self.policy_versions["attacker"],
            "defender_policy_version": self.policy_versions["defender"],
            "global_steps": self.global_steps,
            **self._coevo_metrics(),
            **self._task_asr_tracker.get_metrics(),
            **self._prompt_asr_tracker.get_metrics(),
            **self._rollout_quality_tracker.get_metrics(),
            **self._population_metrics(),
        }

    # ==================== Checkpoint ====================

    def _rollouter_state_dict(self) -> dict:
        population_state = (
            self.old_model_manager.state_dict()
            if self.old_model_manager is not None
            else (
                self._pending_population_state
                if self.population_enabled
                else None
            )
        )
        return {
            "state_version": 5,
            "saved_at": time.time(),
            "training_mode": self.training_mode,
            "population_enabled": self.population_enabled,
            "global_steps": self.global_steps,
            "dataset_fingerprint": self.dataset_fingerprint,
            "dataset_feed_cursor": int(self.dataset_feed_cursor),
            "sampler_replay_state": copy.deepcopy(
                self._sampler_replay_state
            ),
            "total_generated": self.total_generated_samples,
            "policy_versions": dict(self.policy_versions),
            "role_staleness_samples": dict(self.role_staleness_samples),
            "routing_rng_state": self._routing_rng.getstate(),
            "population_state": population_state,
        }

    @staticmethod
    def _atomic_torch_save(state: dict, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        temporary_path = f"{path}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        try:
            torch.save(state, temporary_path)
            os.replace(temporary_path, path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)

    def _save_authoritative_state_locked(self, state: dict | None = None) -> None:
        self._atomic_torch_save(
            state if state is not None else self._rollouter_state_dict(),
            self._population_state_path,
        )
        self._population_state_dirty = False

    def _try_save_authoritative_state_locked(self, reason: str) -> bool:
        """Best-effort state write outside a trainer checkpoint boundary.

        Population refresh is optional serving infrastructure.  A transient
        filesystem error must not terminate both PPO trainers after the role
        was already drained safely.  The next explicit trainer checkpoint
        still performs a strict write and will surface persistent storage
        failures.
        """

        try:
            self._save_authoritative_state_locked()
            return True
        except Exception:
            self._population_state_dirty = True
            self._population_state_save_failures += 1
            logger.exception(
                "[CoTrainRollouter] Failed to persist authoritative population "
                "state after %s; continuing and retrying at the next checkpoint",
                reason,
            )
            return False

    async def save_checkpoint(self, path: str):
        from verl.utils.fs import local_mkdir_safe
        local_mkdir_safe(path)
        dataloader_path = os.path.join(path, "cotrain_rollouter_data.pt")
        async with self._population_update_lock:
            async with self._population_refresh_lock:
                state = self._rollouter_state_dict()
                # Keep a role-local copy for forensic/backward compatibility, but
                # resume always uses the independent authoritative state below.
                self._atomic_torch_save(state, dataloader_path)
                self._save_authoritative_state_locked(state)
        logger.info(
            "[CoTrainRollouter] Saved checkpoint to %s and authoritative state to %s",
            dataloader_path,
            self._population_state_path,
        )

    async def do_validate(self):
        """No-op validation stub. CoTrainRollouter does not run its own validation.

        Called by FullyAsyncTrainer._fit_validate() when test_freq > 0.
        """
        from dataclasses import dataclass, field

        @dataclass
        class ValidateMetrics:
            metrics: dict = field(default_factory=dict)
            timing_raw: dict = field(default_factory=dict)
            val_generations: list = field(default_factory=list)

        return ValidateMetrics()

    def load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0
        from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
        # Prefer the independent atomic state, then fall back to the newest
        # role-local forensic copy if the authoritative file is absent or
        # unreadable.  Atomic replace prevents partial normal writes; this
        # fallback covers storage corruption and pre-v7 layouts.
        state_paths = []
        if os.path.exists(self._population_state_path):
            state_paths.append(self._population_state_path)
        role_local_paths = []
        roots = [
            str(self.config.trainer.default_local_dir),
            str(self.config.cotrain.get("attacker_ckpt_dir") or ""),
            str(self.config.cotrain.get("defender_ckpt_dir") or ""),
        ]
        for root in roots:
            if not root:
                continue
            global_step_folder = find_latest_ckpt_path(root)
            if global_step_folder is None:
                continue
            candidate = os.path.join(global_step_folder, "cotrain_rollouter_data.pt")
            if os.path.exists(candidate):
                role_local_paths.append(candidate)
        state_paths.extend(
            path
            for path in sorted(set(role_local_paths), key=os.path.getmtime, reverse=True)
            if path not in state_paths
        )
        if not state_paths:
            return 0
        state = None
        state_path = None
        load_errors = []
        for candidate_path in state_paths:
            try:
                candidate_state = torch.load(candidate_path, weights_only=False)
                if not isinstance(candidate_state, dict):
                    raise TypeError(
                        f"rollouter checkpoint must be a dict, got {type(candidate_state).__name__}"
                    )
                state_version = int(candidate_state.get("state_version", 1))
                if state_version > 5:
                    raise ValueError(
                        f"rollouter state version {state_version} is newer than supported 5"
                    )
            except Exception as exc:
                load_errors.append(f"{candidate_path}: {exc}")
                logger.exception(
                    "[CoTrainRollouter] Cannot load population state candidate %s",
                    candidate_path,
                )
                continue
            state = candidate_state
            state_path = candidate_path
            break
        if state is None:
            raise RuntimeError(
                "all co-training population state candidates are unreadable: "
                + "; ".join(load_errors)
            )

        _validate_resume_training_mode(state.get("training_mode"), self.training_mode)

        if (
            not self.population_enabled
            and state.get("population_state") is not None
        ):
            raise RuntimeError(
                "refusing to resume the no-population ablation from a checkpoint "
                "that contains population state"
            )
        saved_population_enabled = state.get("population_enabled")
        if (
            saved_population_enabled is not None
            and bool(saved_population_enabled) != self.population_enabled
        ):
            raise RuntimeError(
                "rollouter checkpoint population mode does not match this run: "
                f"checkpoint={saved_population_enabled}, configured={self.population_enabled}"
            )

        self.global_steps = int(state.get("global_steps", 1))
        saved_dataset_fingerprint = state.get("dataset_fingerprint")
        if saved_dataset_fingerprint == self.dataset_fingerprint:
            dataset_feed_cursor = _resolve_dataset_resume_cursor(
                saved_dataset_fingerprint,
                self.dataset_fingerprint,
                state.get("dataset_feed_cursor", 0),
                self.total_rollout_steps,
            )
            saved_sampler_state = state.get("sampler_replay_state")
            if dataset_feed_cursor and not saved_sampler_state:
                raise ValueError(
                    "rollouter checkpoint has a dataset cursor but no sampler replay state"
                )
            if saved_sampler_state:
                # Validate compatibility now. _feed_samples restores it again
                # immediately before constructing the first iterator.
                _restore_sampler_replay_state(
                    self.train_sampler, saved_sampler_state
                )
                self._sampler_replay_state = copy.deepcopy(saved_sampler_state)
            self.dataset_feed_cursor = dataset_feed_cursor
            logger.info(
                "[CoTrainRollouter] Continuing dataset %s at cursor %d/%d",
                self.dataset_fingerprint,
                self.dataset_feed_cursor,
                self.total_rollout_steps,
            )
        else:
            # Old checkpoints have no fingerprint. Replaying from zero is the
            # only safe choice: inferring a cursor from global_steps could skip
            # unrelated rows after validators/parquet were regenerated.
            self.dataset_feed_cursor = 0
            logger.warning(
                "[CoTrainRollouter] Training dataset changed or checkpoint has "
                "no fingerprint (saved=%s current=%s); restarting dataset feed "
                "at cursor 0 while preserving policy/global step",
                saved_dataset_fingerprint,
                self.dataset_fingerprint,
            )
        self.total_generated_samples = int(state.get("total_generated", 0))
        self.policy_versions.update(state.get("policy_versions", {}))
        self.training_steps = _training_step_for_mode(
            self.training_mode, self.policy_versions
        )
        self.role_staleness_samples.update(state.get("role_staleness_samples", {}))
        self.staleness_samples = max(self.role_staleness_samples.values())
        routing_rng_state = state.get("routing_rng_state")
        if routing_rng_state is not None:
            self._routing_rng.setstate(routing_rng_state)
        self._pending_population_state = (
            state.get("population_state") if self.population_enabled else None
        )
        logger.info(
            "[CoTrainRollouter] Resumed from %s at step %s with policy versions %s",
            state_path,
            self.global_steps,
            self.policy_versions,
        )
        return self.global_steps


class _RolloutQualityTracker:
    """Rolling, compact health metrics for the rollout-to-PPO data path."""

    _FIELDS = (
        "attacker_attempted",
        "has_injected_payload",
        "attacker_sample_valid",
        "defender_sample_valid",
        "format_violation",
        "action_asr",
        "effective_asr",
        "tool_responses",
        "attacker_generated_turns",
        "attacker_length_finishes",
        "tool_responses_truncated",
        "payloads_survived",
        "payloads_truncated",
        "attacker_trace_drops",
        "defender_trace_drops",
        "attacker_train_sequence_tokens",
        "attacker_train_sequence_overflow",
        "defender_train_sequence_tokens",
        "defender_train_sequence_overflow",
    )

    def __init__(self, window_size: int = 400):
        self._records = collections.deque(maxlen=window_size)

    def record(self, **values) -> None:
        record = {key: values.get(key, float("nan")) for key in self._FIELDS}
        record["model_pair"] = values.get("model_pair", "unknown")
        record["learned_attacker"] = bool(values.get("learned_attacker", False))
        self._records.append(record)

    @staticmethod
    def _finite(value) -> bool:
        try:
            return bool(np.isfinite(float(value)))
        except (TypeError, ValueError):
            return False

    def get_metrics(self) -> dict[str, float]:
        records = list(self._records)
        if not records:
            return {}

        metrics: dict[str, float] = {
            "rollout_quality/window_samples": float(len(records)),
            "rollout_quality/clean_row_rate": sum(
                row["model_pair"] == "clean" for row in records
            ) / len(records),
            "rollout_quality/injected_row_rate": sum(
                row["model_pair"] != "clean" for row in records
            ) / len(records),
            "rollout_quality/injected_payload_rate": sum(
                bool(row["has_injected_payload"]) for row in records
            ) / len(records),
            "rollout_quality/attacker_trainable_rate": sum(
                bool(row["attacker_sample_valid"]) for row in records if row["learned_attacker"]
            ) / max(1, sum(row["learned_attacker"] for row in records)),
            "rollout_quality/defender_trainable_rate": sum(
                bool(row["defender_sample_valid"]) for row in records
            ) / len(records),
        }

        learned = [row for row in records if row["learned_attacker"]]
        learned_valid_defender = [
            row for row in learned if bool(row["defender_sample_valid"])
        ]
        if learned_valid_defender:
            metrics["rollout_quality/defender_utility_only_rate"] = sum(
                not bool(row["has_injected_payload"])
                for row in learned_valid_defender
            ) / len(learned_valid_defender)
        # Rows where the defender never exposed an injection point contain no
        # attacker generation.  Counting their default zero as a valid format
        # inflated this metric and hid the v6 generation failure.
        valid_format = [
            row for row in learned
            if bool(row["attacker_attempted"])
            and self._finite(row["format_violation"])
        ]
        if valid_format:
            metrics["rollout_quality/format_valid_rate"] = 1.0 - sum(
                float(row["format_violation"]) for row in valid_format
            ) / len(valid_format)
        valid_asr = [row for row in learned if self._finite(row["action_asr"])]
        if valid_asr:
            metrics["rollout_quality/action_conditioned_asr"] = sum(
                float(row["action_asr"]) for row in valid_asr
            ) / len(valid_asr)
        valid_effective_asr = [
            row for row in learned if self._finite(row["effective_asr"])
        ]
        if valid_effective_asr:
            metrics["rollout_quality/effective_asr_reached_only"] = sum(
                float(row["effective_asr"]) for row in valid_effective_asr
            ) / len(valid_effective_asr)

        generation_rows = [
            row for row in learned
            if self._finite(row["attacker_generated_turns"])
            and self._finite(row["attacker_length_finishes"])
        ]
        total_generated_turns = sum(
            max(0.0, float(row["attacker_generated_turns"]))
            for row in generation_rows
        )
        if total_generated_turns > 0:
            metrics["rollout_quality/attacker_length_finish_rate"] = sum(
                max(0.0, float(row["attacker_length_finishes"]))
                for row in generation_rows
            ) / total_generated_turns

        tool_rows = [row for row in records if self._finite(row["tool_responses"])]
        total_tools = sum(max(0.0, float(row["tool_responses"])) for row in tool_rows)
        total_tool_truncated = sum(
            max(0.0, float(row["tool_responses_truncated"]))
            for row in tool_rows if self._finite(row["tool_responses_truncated"])
        )
        if total_tools > 0:
            metrics["rollout_quality/tool_response_truncation_rate"] = total_tool_truncated / total_tools

        payload_rows = [
            row for row in records
            if self._finite(row["payloads_survived"]) and self._finite(row["payloads_truncated"])
        ]
        payload_survived = sum(max(0.0, float(row["payloads_survived"])) for row in payload_rows)
        payload_truncated = sum(max(0.0, float(row["payloads_truncated"])) for row in payload_rows)
        if payload_survived + payload_truncated > 0:
            metrics["rollout_quality/payload_survival_rate"] = payload_survived / (payload_survived + payload_truncated)

        for role, key in (("attacker", "attacker_trace_drops"), ("defender", "defender_trace_drops")):
            rows = [row for row in records if self._finite(row[key])]
            if rows:
                metrics[f"rollout_quality/{role}_trace_drop_rate"] = sum(
                    float(row[key]) > 0 for row in rows
                ) / len(rows)
        for role in ("attacker", "defender"):
            overflow_key = f"{role}_train_sequence_overflow"
            token_key = f"{role}_train_sequence_tokens"
            overflow_rows = [
                row
                for row in records
                if self._finite(row[overflow_key]) and self._finite(row[token_key])
            ]
            if overflow_rows:
                metrics[f"rollout_quality/{role}_train_sequence_overflow_rate"] = sum(
                    bool(float(row[overflow_key])) for row in overflow_rows
                ) / len(overflow_rows)
            token_rows = [
                float(row[token_key])
                for row in records
                if self._finite(row[token_key])
            ]
            if token_rows:
                metrics[f"rollout_quality/{role}_train_sequence_tokens_max"] = max(token_rows)
        return metrics


class _CoEvolutionTracker:
    """滚动窗口追踪器，计算 per-pair ASR 和 co-evolution 指标。"""

    def __init__(self, window_size: int = 200):
        self.window_size = window_size
        self._windows: dict[str, collections.deque] = {}
        self._total_counts: dict[str, int] = {}
        self._total_asr: dict[str, int] = {}

    def record(self, model_pair: str, asr: bool, utility_failed: bool):
        if model_pair not in self._windows:
            self._windows[model_pair] = collections.deque(maxlen=self.window_size)
            self._total_counts[model_pair] = 0
            self._total_asr[model_pair] = 0
        self._windows[model_pair].append((asr, utility_failed))
        self._total_counts[model_pair] += 1
        if asr:
            self._total_asr[model_pair] += 1

    def reset_pair(self, model_pair: str) -> None:
        """Start a fresh rolling comparison after an opponent population swap."""

        self._windows.pop(model_pair, None)

    def get_metrics(self) -> dict:
        metrics: dict[str, float] = {}
        for mp, window in self._windows.items():
            if not window:
                continue
            asr_vals = [int(x[0]) for x in window]
            util_vals = [int(x[1]) for x in window]
            n = len(window)
            asr_rate = sum(asr_vals) / n
            util_fail_rate = sum(util_vals) / n
            metrics[f"coevo/{mp}/asr"] = asr_rate
            metrics[f"coevo/{mp}/resist_rate"] = 1.0 - asr_rate
            metrics[f"coevo/{mp}/utility_fail_rate"] = util_fail_rate
            metrics[f"coevo/{mp}/n_samples"] = n

        cc_window = self._windows.get("curr_curr")
        if cc_window and len(cc_window) >= 10:
            cc_asr = sum(int(x[0]) for x in cc_window) / len(cc_window)
            metrics["coevo/cc_asr"] = cc_asr
            metrics["coevo/arms_race_balance"] = 1.0 - abs(cc_asr - 0.5) * 2

        # Attacker improvement: new_atk(cc) vs old_atk — both face curr_def
        cc_w = self._windows.get("curr_curr")
        old_atk_w = self._windows.get("old_atk_curr_def")
        if cc_w and old_atk_w and len(cc_w) >= 10 and len(old_atk_w) >= 10:
            curr_atk_asr = sum(int(x[0]) for x in cc_w) / len(cc_w)
            old_atk_asr = sum(int(x[0]) for x in old_atk_w) / len(old_atk_w)
            metrics["coevo/attacker_improvement"] = curr_atk_asr - old_atk_asr

        # Defender improvement: new_def(cc) vs old_def — both face curr_atk
        curr_atk_old_def_w = self._windows.get("curr_atk_old_def")
        if cc_w and curr_atk_old_def_w and len(cc_w) >= 10 and len(curr_atk_old_def_w) >= 10:
            curr_def_resist = 1.0 - sum(int(x[0]) for x in cc_w) / len(cc_w)
            old_def_resist = 1.0 - sum(int(x[0]) for x in curr_atk_old_def_w) / len(curr_atk_old_def_w)
            metrics["coevo/defender_improvement"] = curr_def_resist - old_def_resist

        # Template baseline comparison: curr_atk vs fixed templates (both face curr_def)
        template_w = self._windows.get("fixed_template")
        if cc_w and template_w and len(cc_w) >= 10 and len(template_w) >= 10:
            curr_atk_asr = sum(int(x[0]) for x in cc_w) / len(cc_w)
            template_asr = sum(int(x[0]) for x in template_w) / len(template_w)
            metrics["coevo/template_asr"] = template_asr
            metrics["coevo/curr_atk_vs_template"] = curr_atk_asr - template_asr

        # Overall co-training health score (0-1, higher=better training dynamics)
        if cc_w and len(cc_w) >= 10:
            cc_asr = sum(int(x[0]) for x in cc_w) / len(cc_w)
            cc_util_fail = sum(int(x[1]) for x in cc_w) / len(cc_w)
            balance = 1.0 - abs(cc_asr - 0.5) * 2
            utility_health = 1.0 - cc_util_fail
            metrics["coevo/health_score"] = balance * 0.5 + utility_health * 0.5

        return metrics


class _TaskLevelASRTracker:
    """Per-task (suite × injection_task_id) ASR tracker with rolling window.

    Reports:
      - task_asr/{suite_name}/asr: per-suite ASR
      - task_asr/{suite_name}/{injection_task_id}/asr: per-task ASR
      - task_asr/overall: overall task-level ASR (fraction of unique tasks ever breached)
      - task_asr/n_tasks_seen: total unique tasks seen
      - task_asr/n_tasks_breached: tasks breached at least once
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.verbose_task_metrics = os.getenv("COTRAIN_WANDB_VERBOSE_TASK_METRICS", "0") == "1"
        # Per-suite rolling windows
        self._suite_windows: dict[str, collections.deque] = {}
        # Per-task rolling windows: key = (suite, task_id)
        self._task_windows: dict[tuple[str, str], collections.deque] = {}
        # Cumulative tracking for "ever breached" metric
        self._tasks_seen: set[tuple[str, str]] = set()
        self._tasks_breached: set[tuple[str, str]] = set()

    def record(self, suite_name: str, injection_task_id: str, asr: bool):
        # Suite level
        if suite_name not in self._suite_windows:
            self._suite_windows[suite_name] = collections.deque(maxlen=self.window_size)
        self._suite_windows[suite_name].append(asr)

        # Task level
        key = (suite_name, injection_task_id)
        if key not in self._task_windows:
            self._task_windows[key] = collections.deque(maxlen=self.window_size)
        self._task_windows[key].append(asr)

        self._tasks_seen.add(key)
        if asr:
            self._tasks_breached.add(key)

    def get_metrics(self) -> dict:
        metrics: dict[str, float] = {}

        # Per-suite ASR
        for suite, window in self._suite_windows.items():
            if len(window) >= 5:
                metrics[f"task_asr/{suite}/asr"] = sum(window) / len(window)
                metrics[f"task_asr/{suite}/n_samples"] = float(len(window))

        # Per-task series are useful for diagnosis but make the default W&B
        # run hard to read (there can be dozens of them).  Opt in explicitly.
        if self.verbose_task_metrics:
            for (suite, task_id), window in self._task_windows.items():
                if len(window) >= 5:
                    metrics[f"task_asr/{suite}/{task_id}/asr"] = sum(window) / len(window)

        all_suite_values = [value for window in self._suite_windows.values() for value in window]
        if all_suite_values:
            metrics["task_asr/rolling_overall_asr"] = sum(all_suite_values) / len(all_suite_values)

        # Overall task-level breach rate
        n_seen = len(self._tasks_seen)
        n_breached = len(self._tasks_breached)
        if n_seen > 0:
            metrics["task_asr/overall_breach_rate"] = n_breached / n_seen
            metrics["task_asr/n_tasks_seen"] = float(n_seen)
            metrics["task_asr/n_tasks_breached"] = float(n_breached)

        return metrics


class _PromptLevelASRTracker:
    """Prompt-level ASR: for each prompt group (N rollouts), did ANY succeed?

    Reports:
      - prompt_asr/overall: fraction of prompts where at least 1 rollout succeeded
      - prompt_asr/{suite}/asr: per-suite prompt-level ASR
      - prompt_asr/{model_pair}/asr: per model-pair prompt-level ASR
    """

    def __init__(self, window_size: int = 200):
        self.window_size = window_size
        self._overall = collections.deque(maxlen=window_size)
        self._by_suite: dict[str, collections.deque] = {}
        self._by_pair: dict[str, collections.deque] = {}

    def record(self, suite_name: str, model_pair: str, any_success: bool):
        self._overall.append(any_success)

        if suite_name not in self._by_suite:
            self._by_suite[suite_name] = collections.deque(maxlen=self.window_size)
        self._by_suite[suite_name].append(any_success)

        if model_pair not in self._by_pair:
            self._by_pair[model_pair] = collections.deque(maxlen=self.window_size)
        self._by_pair[model_pair].append(any_success)

    def get_metrics(self) -> dict:
        metrics: dict[str, float] = {}

        if len(self._overall) >= 10:
            metrics["prompt_asr/overall"] = sum(self._overall) / len(self._overall)

        for suite, window in self._by_suite.items():
            if len(window) >= 5:
                metrics[f"prompt_asr/{suite}/asr"] = sum(window) / len(window)

        for mp, window in self._by_pair.items():
            if len(window) >= 5:
                metrics[f"prompt_asr/{mp}/asr"] = sum(window) / len(window)

        return metrics
