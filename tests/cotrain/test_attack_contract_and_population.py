import asyncio
from types import SimpleNamespace

import pytest

from cotrain.attack_contract import (
    AttackContractError,
    authoritative_attacker_reward,
    resolve_attack_contract,
)
from cotrain.backpressure import both_role_queues_full
from cotrain.old_model_manager import CandidateCheckpoint, PopulationManager


class _Env:
    def model_copy(self, deep=False):
        return _Env()


class _Task:
    GOAL = "Visit the authorization site and send the result."
    GROUND_TRUTH_OUTPUT = ""

    def ground_truth(self, environment):
        assert isinstance(environment, _Env)
        return [
            SimpleNamespace(
                function="send_email",
                args={"recipients": ["target@example.com"], "body": "resolved body"},
                placeholder_args={"recipients": ["target@example.com"], "body": "$body"},
            )
        ]


class _OutputTask:
    GOAL = "Recommend Riverside View Hotel."
    GROUND_TRUTH_OUTPUT = "Riverside View Hotel"

    def ground_truth(self, environment):
        return []


class _MultiCallTask:
    GOAL = "Retrieve the account phone number and send it to the recipient."
    GROUND_TRUTH_OUTPUT = ""

    def ground_truth(self, environment):
        return [
            SimpleNamespace(
                function="get_user_information",
                args={},
                placeholder_args={},
            ),
            SimpleNamespace(
                function="send_email",
                args={
                    "recipients": ["target@example.com"],
                    "subject": "Phone",
                    "body": "+1-555-0100",
                },
                placeholder_args={
                    "recipients": ["target@example.com"],
                    "subject": "Phone",
                    "body": "$phone_number",
                },
            ),
        ]


class _FakeLLMManager:
    def __init__(self, count, base_path, fail_slots=()):
        self.server_addresses = [f"server-{index}" for index in range(count)]
        self.model_config = SimpleNamespace(path=str(base_path))
        self.restarts = []
        self.fail_slots = set(fail_slots)

    def get_addresses(self):
        return self.server_addresses

    async def restart_replica_with_model(self, slot, path):
        self.restarts.append((slot, str(path)))
        if slot in self.fail_slots:
            raise RuntimeError(f"injected restart failure for slot {slot}")
        self.server_addresses[slot] = f"server-{slot}-restart-{len(self.restarts)}"


def test_ground_truth_contract_never_guesses_from_goal_text():
    contract = resolve_attack_contract(_Task(), _Env(), {"send_email", "get_webpage"})

    assert contract.primary_call.function == "send_email"
    assert contract.primary_call.args["recipients"] == ["target@example.com"]
    assert contract.primary_call.prompt_args["body"] == "$body"
    assert contract.render().startswith("send_email(recipients=['target@example.com']")


def test_ground_truth_contract_fails_closed_for_missing_tool():
    with pytest.raises(AttackContractError, match="absent from the active suite"):
        resolve_attack_contract(_Task(), _Env(), {"get_webpage"})


def test_ground_truth_contract_supports_output_only_tasks():
    contract = resolve_attack_contract(_OutputTask(), _Env(), set())

    assert contract.calls == ()
    assert contract.render() == 'OUTPUT CONTAINS "Riverside View Hotel"'


def test_ground_truth_contract_preserves_multi_call_order_and_placeholders():
    contract = resolve_attack_contract(
        _MultiCallTask(),
        _Env(),
        {"get_user_information", "send_email"},
    )

    assert [call.function for call in contract.calls] == [
        "get_user_information",
        "send_email",
    ]
    assert contract.calls[1].args["body"] == "+1-555-0100"
    assert contract.calls[1].prompt_args["body"] == "$phone_number"
    assert contract.render().startswith("get_user_information() -> send_email(")


def test_attacker_reward_has_no_target_tool_or_parameter_shaping():
    assert authoritative_attacker_reward(
        asr_success=False,
        format_violation=False,
        has_injected_payload=True,
    ) == 0.05
    assert authoritative_attacker_reward(
        asr_success=True,
        format_violation=True,
        has_injected_payload=True,
    ) == 1.0


def _candidate(step, path, bucket_rewards):
    candidate = CandidateCheckpoint(step=step, hf_path=str(path))
    for bucket, rewards in bucket_rewards.items():
        for reward in rewards:
            candidate.record(reward, bucket)
    return candidate


def test_provisional_candidate_gets_probation_slot_before_elite_selection(tmp_path):
    attacker_dir = tmp_path / "attacker"
    defender_dir = tmp_path / "defender"
    step10 = attacker_dir / "global_step_10" / "actor" / "huggingface"
    step20 = attacker_dir / "global_step_20" / "actor" / "huggingface"
    step10.mkdir(parents=True)
    step20.mkdir(parents=True)

    manager = PopulationManager(
        str(attacker_dir),
        str(defender_dir),
        population_size=2,
        min_eval_samples_per_bucket=2,
        min_eval_buckets=2,
    )
    attacker_llm = _FakeLLMManager(2, tmp_path / "attacker-base")
    defender_llm = _FakeLLMManager(2, tmp_path / "defender-base")
    manager.set_llm_managers(attacker_llm, defender_llm)
    manager._attacker_candidates = {
        10: _candidate(10, step10, {"travel": [1.0]}),
        20: _candidate(20, step20, {"travel": [1.0, 1.0], "github": [0.0, 1.0]}),
    }

    asyncio.run(manager._resample_population("attacker"))

    assert manager._attacker_loaded_steps == [20, 10]
    assert manager._attacker_candidates[10].is_eligible(2, 2) is False
    assert manager._attacker_candidates[20].is_eligible(2, 2) is True


def test_population_triggers_on_update_interval_and_advances_probation(tmp_path):
    attacker_dir = tmp_path / "attacker"
    defender_dir = tmp_path / "defender"
    for step in (10, 20, 30, 40):
        (attacker_dir / f"global_step_{step}" / "actor" / "huggingface").mkdir(
            parents=True
        )

    manager = PopulationManager(
        str(attacker_dir),
        str(defender_dir),
        population_size=2,
        save_freq=10,
        update_interval=20,
        min_eval_samples_per_bucket=1,
        min_eval_buckets=1,
        required_buckets=("travel",),
        probation_sampling_prob=1.0,
    )
    attacker_llm = _FakeLLMManager(2, tmp_path / "attacker-base")
    manager.set_llm_managers(
        attacker_llm,
        _FakeLLMManager(2, tmp_path / "defender-base"),
    )

    asyncio.run(manager.maybe_update_attacker(19))
    assert manager._attacker_loaded_steps == [0, 0]

    asyncio.run(manager.maybe_update_attacker(20))
    assert set(manager._attacker_candidates) == {10, 20}
    # The single probation slot evaluates the newest checkpoint at the refresh
    # boundary; step 10 is intentionally skipped instead of creating backlog.
    assert manager._attacker_loaded_steps == [0, 20]
    assert manager.update_due("attacker", 20) is False

    manager.report_attacker_reward(20, 1.0, "travel")
    asyncio.run(manager.maybe_update_attacker(40))
    assert set(manager._attacker_candidates) == {10, 20, 30, 40}
    assert manager._attacker_loaded_steps == [20, 40]
    assert attacker_llm.restarts == [
        (1, str(attacker_dir / "global_step_20" / "actor" / "huggingface")),
        (0, str(attacker_dir / "global_step_20" / "actor" / "huggingface")),
        (1, str(attacker_dir / "global_step_40" / "actor" / "huggingface")),
    ]


def test_rollout_backpressure_does_not_pause_when_only_one_queue_is_full():
    assert both_role_queues_full(384, 0, 384) is False
    assert both_role_queues_full(64, 0, 384) is False


def test_rollout_backpressure_pauses_only_when_both_live_queues_are_full():
    assert both_role_queues_full(384, 384, 384) is True
    assert both_role_queues_full(383, 384, 384) is False


def test_population_state_round_trip_restores_fitness_buckets_and_slots(tmp_path):
    attacker_dir = tmp_path / "attacker"
    defender_dir = tmp_path / "defender"
    step10 = attacker_dir / "global_step_10" / "actor" / "huggingface"
    step10.mkdir(parents=True)

    manager = PopulationManager(
        str(attacker_dir),
        str(defender_dir),
        population_size=2,
        min_eval_samples_per_bucket=2,
        min_eval_buckets=2,
    )
    manager.set_llm_managers(
        _FakeLLMManager(2, tmp_path / "attacker-base"),
        _FakeLLMManager(2, tmp_path / "defender-base"),
    )
    manager._attacker_candidates[10] = CandidateCheckpoint(10, str(step10))
    manager.report_attacker_reward(10, 1.0, "travel")
    manager.report_attacker_reward(10, 0.0, "github")
    manager._attacker_loaded_steps = [10, 0]
    state = manager.state_dict()

    restored = PopulationManager(
        str(attacker_dir),
        str(defender_dir),
        population_size=2,
        min_eval_samples_per_bucket=2,
        min_eval_buckets=2,
    )
    attacker_llm = _FakeLLMManager(2, tmp_path / "attacker-base")
    restored.set_llm_managers(attacker_llm, _FakeLLMManager(2, tmp_path / "defender-base"))
    restored.load_state_dict(state)
    asyncio.run(restored.restore_loaded_models())

    candidate = restored._attacker_candidates[10]
    assert candidate.fitness == 0.5
    assert candidate.bucket_counts == {"travel": 1, "github": 1}
    assert restored._attacker_loaded_steps == [10, 0]
    assert attacker_llm.restarts == [(0, str(step10))]


def test_population_restore_failure_disables_only_role_and_retries_later(tmp_path):
    attacker_dir = tmp_path / "attacker"
    defender_dir = tmp_path / "defender"
    step10 = attacker_dir / "global_step_10" / "actor" / "huggingface"
    step10.mkdir(parents=True)

    original = PopulationManager(
        str(attacker_dir),
        str(defender_dir),
        population_size=2,
        update_interval=20,
    )
    original.set_llm_managers(
        _FakeLLMManager(2, tmp_path / "attacker-base"),
        _FakeLLMManager(2, tmp_path / "defender-base"),
    )
    original._attacker_candidates[10] = CandidateCheckpoint(10, str(step10))
    original._attacker_loaded_steps = [10, 0]
    original._last_attacker_update_step = 40

    restored = PopulationManager(
        str(attacker_dir),
        str(defender_dir),
        population_size=2,
        update_interval=20,
    )
    failing_attacker = _FakeLLMManager(
        2,
        tmp_path / "attacker-base",
        fail_slots={0},
    )
    restored.set_llm_managers(
        failing_attacker,
        _FakeLLMManager(2, tmp_path / "defender-base"),
    )
    restored.load_state_dict(original.state_dict())

    failed_roles = asyncio.run(restored.restore_loaded_models())

    assert failed_roles == {"attacker"}
    assert restored._attacker_loaded_steps == [0, 0]
    assert restored._last_attacker_update_step == 0
    assert restored.update_due("attacker", 40) is True
    assert restored.update_due("defender", 40) is True


def test_partial_refresh_retries_the_same_persisted_target(tmp_path):
    attacker_dir = tmp_path / "attacker"
    defender_dir = tmp_path / "defender"
    for step in (10, 20, 40):
        (attacker_dir / f"global_step_{step}" / "actor" / "huggingface").mkdir(
            parents=True
        )

    manager = PopulationManager(
        str(attacker_dir),
        str(defender_dir),
        population_size=2,
        min_eval_samples_per_bucket=1,
        min_eval_buckets=1,
        required_buckets=("travel",),
    )
    attacker_llm = _FakeLLMManager(
        2, tmp_path / "attacker-base", fail_slots={1}
    )
    manager.set_llm_managers(
        attacker_llm,
        _FakeLLMManager(2, tmp_path / "defender-base"),
    )
    manager._attacker_candidates = {
        10: _candidate(10, attacker_dir / "global_step_10", {"travel": [1.0]}),
        20: _candidate(20, attacker_dir / "global_step_20", {}),
    }

    with pytest.raises(RuntimeError, match="injected restart failure"):
        asyncio.run(manager._resample_population("attacker", max_candidate_step=20))

    assert manager._attacker_loaded_steps == [10, 0]
    assert manager.state_dict()["attacker_pending_target_steps"] == [10, 20]

    # A newer checkpoint appears before retry. The in-progress plan must still
    # finish [10, 20] rather than redraw and churn slot 0 again.
    manager._attacker_candidates[40] = _candidate(
        40, attacker_dir / "global_step_40", {}
    )
    attacker_llm.fail_slots.clear()
    asyncio.run(manager._resample_population("attacker", max_candidate_step=40))

    assert manager._attacker_loaded_steps == [10, 20]
    assert manager.state_dict()["attacker_pending_target_steps"] == []
    assert attacker_llm.restarts == [
        (0, str(attacker_dir / "global_step_10")),
        (1, str(attacker_dir / "global_step_20")),
        (1, str(attacker_dir / "global_step_20")),
    ]


def test_v3_population_state_remains_loadable(tmp_path):
    original = PopulationManager(
        str(tmp_path / "attacker"),
        str(tmp_path / "defender"),
        population_size=2,
    )
    original.set_llm_managers(
        _FakeLLMManager(2, tmp_path / "attacker-base"),
        _FakeLLMManager(2, tmp_path / "defender-base"),
    )
    state = original.state_dict()
    state["version"] = 3
    state.pop("attacker_pending_target_steps")
    state.pop("defender_pending_target_steps")
    state["config"].pop("candidate_lag_steps")

    restored = PopulationManager(
        str(tmp_path / "attacker"),
        str(tmp_path / "defender"),
        population_size=2,
        candidate_lag_steps=40,
    )
    restored.load_state_dict(state)

    assert restored._pending_target_steps == {}
    assert restored.candidate_lag_steps == 40


def test_population_candidate_lag_keeps_one_refresh_of_age(tmp_path):
    manager = PopulationManager(
        str(tmp_path / "attacker"),
        str(tmp_path / "defender"),
        population_size=1,
        update_interval=20,
        candidate_lag_steps=20,
    )

    assert manager._get_update_target(500, 480) is None
    assert manager._get_update_target(510, 480) is None
    assert manager._get_update_target(520, 480) == 500
    assert manager._get_update_target(600, 560) == 580


def test_population_size_must_match_both_historical_replica_counts(tmp_path):
    manager = PopulationManager(str(tmp_path / "atk"), str(tmp_path / "def"), population_size=2)

    with pytest.raises(ValueError, match="population_size must equal"):
        manager.set_llm_managers(
            _FakeLLMManager(1, tmp_path / "attacker-base"),
            _FakeLLMManager(2, tmp_path / "defender-base"),
        )


def test_attacker_only_population_restores_without_dummy_defender(tmp_path):
    attacker_dir = tmp_path / "attacker"
    step10 = attacker_dir / "global_step_10" / "actor" / "huggingface"
    step10.mkdir(parents=True)

    source = PopulationManager(
        str(attacker_dir),
        str(tmp_path / "defender"),
        population_size=2,
    )
    source.set_llm_managers(
        _FakeLLMManager(2, tmp_path / "attacker-base"),
        _FakeLLMManager(2, tmp_path / "defender-base"),
    )
    source._attacker_candidates[10] = CandidateCheckpoint(10, str(step10))
    source._attacker_loaded_steps = [10, 0]
    source._defender_loaded_steps = [20, 30]

    restored = PopulationManager(
        str(attacker_dir),
        str(tmp_path / "defender"),
        population_size=2,
    )
    attacker = _FakeLLMManager(2, tmp_path / "attacker-base")
    restored.set_llm_managers(attacker, None)
    restored.load_state_dict(source.state_dict())

    assert asyncio.run(restored.restore_loaded_models()) == set()
    assert restored._attacker_loaded_steps == [10, 0]
    assert restored._defender_loaded_steps == []
    assert attacker.restarts == [(0, str(step10))]


def test_attacker_only_population_still_requires_exact_replica_count(tmp_path):
    manager = PopulationManager(
        str(tmp_path / "attacker"),
        str(tmp_path / "defender"),
        population_size=2,
    )

    with pytest.raises(ValueError, match="every configured historical replica"):
        manager.set_llm_managers(
            _FakeLLMManager(1, tmp_path / "attacker-base"),
            None,
        )


def test_population_url_attribution_requires_exact_endpoint_match(tmp_path):
    manager = PopulationManager(
        str(tmp_path / "atk"),
        str(tmp_path / "def"),
        population_size=2,
    )
    attacker = _FakeLLMManager(2, tmp_path / "attacker-base")
    defender = _FakeLLMManager(2, tmp_path / "defender-base")
    attacker.server_addresses = ["host:900", "host:9000"]
    manager.set_llm_managers(attacker, defender)
    manager._attacker_loaded_steps = [10, 20]

    assert manager.get_step_for_url("http://host:9000/v1") == 20
    assert manager.get_step_for_url("http://host:900/v1/") == 10
    assert manager.get_step_for_url("http://host:9000/v1/chat/completions") == 0


def test_population_slot_attribution_is_direct_and_immutable(tmp_path):
    manager = PopulationManager(
        str(tmp_path / "atk"),
        str(tmp_path / "def"),
        population_size=2,
    )
    manager.set_llm_managers(
        _FakeLLMManager(2, tmp_path / "attacker-base"),
        _FakeLLMManager(2, tmp_path / "defender-base"),
    )
    manager._defender_loaded_steps = [10, 40]

    assert manager.get_loaded_step("defender", 1) == 40
    with pytest.raises(IndexError, match="slot out of range"):
        manager.get_loaded_step("defender", 2)


def test_population_eligibility_counts_only_authoritative_buckets(tmp_path):
    manager = PopulationManager(
        str(tmp_path / "atk"),
        str(tmp_path / "def"),
        population_size=1,
        min_eval_samples_per_bucket=2,
        min_eval_buckets=3,
        required_buckets=("travel", "github", "slack"),
    )
    candidate = _candidate(
        10,
        tmp_path / "step10",
        {
            "travel": [1.0, 1.0],
            "github": [0.0, 1.0],
            "not_a_real_suite": [1.0, 1.0],
        },
    )

    assert manager._eligible(candidate) is False
    assert candidate.evaluated_bucket_count(2, manager.required_buckets) == 2


def test_population_rejects_impossible_bucket_quota(tmp_path):
    with pytest.raises(ValueError, match="cannot exceed"):
        PopulationManager(
            str(tmp_path / "atk"),
            str(tmp_path / "def"),
            min_eval_buckets=3,
            required_buckets=("travel", "github"),
        )


def test_population_fitness_is_suite_macro_average(tmp_path):
    manager = PopulationManager(
        str(tmp_path / "atk"),
        str(tmp_path / "def"),
        population_size=1,
        min_eval_samples_per_bucket=1,
        min_eval_buckets=2,
        required_buckets=("travel", "github"),
    )
    candidate = CandidateCheckpoint(10, str(tmp_path / "step10"))
    manager._attacker_candidates[10] = candidate
    for _ in range(100):
        manager.report_attacker_reward(10, 1.0, "travel")
    manager.report_attacker_reward(10, 0.0, "github")

    assert candidate.fitness == pytest.approx(0.5)
    assert sum(candidate.reward_history) / len(candidate.reward_history) > 0.98


def test_population_ignores_nonfinite_fitness_samples(tmp_path):
    manager = PopulationManager(
        str(tmp_path / "atk"),
        str(tmp_path / "def"),
        population_size=1,
        min_eval_samples_per_bucket=1,
        min_eval_buckets=1,
        required_buckets=("travel",),
    )
    candidate = CandidateCheckpoint(10, str(tmp_path / "step10"))
    manager._attacker_candidates[10] = candidate

    manager.report_attacker_reward(10, float("nan"), "travel")
    manager.report_attacker_reward(10, float("inf"), "travel")

    assert candidate.n_samples == 0
    assert candidate.fitness == 0.0


def test_probation_slot_is_targeted_for_missing_suite(tmp_path):
    attacker_dir = tmp_path / "attacker"
    defender_dir = tmp_path / "defender"
    manager = PopulationManager(
        str(attacker_dir),
        str(defender_dir),
        population_size=2,
        min_eval_samples_per_bucket=2,
        min_eval_buckets=2,
        required_buckets=("travel", "github"),
        probation_sampling_prob=1.0,
    )
    manager.set_llm_managers(
        _FakeLLMManager(2, tmp_path / "attacker-base"),
        _FakeLLMManager(2, tmp_path / "defender-base"),
    )
    manager._attacker_candidates[10] = _candidate(
        10,
        tmp_path / "step10",
        {"travel": [1.0], "github": [0.0, 1.0]},
    )
    manager._attacker_loaded_steps = [0, 10]

    assert manager.choose_population_slot("attacker", "travel") == 1


def test_population_restore_rejects_reward_contract_or_quota_mismatch(tmp_path):
    original = PopulationManager(
        str(tmp_path / "atk"),
        str(tmp_path / "def"),
        population_size=1,
        min_eval_samples_per_bucket=2,
        min_eval_buckets=2,
        required_buckets=("travel", "github"),
    )
    state = original.state_dict()
    changed = PopulationManager(
        str(tmp_path / "atk"),
        str(tmp_path / "def"),
        population_size=1,
        min_eval_samples_per_bucket=1,
        min_eval_buckets=2,
        required_buckets=("travel", "github"),
    )

    with pytest.raises(ValueError, match="config/reward contract mismatch"):
        changed.load_state_dict(state)
