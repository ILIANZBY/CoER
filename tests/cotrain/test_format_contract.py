import numpy as np
import pytest

# The head-node SciPy wheel still references NumPy's removed scalar aliases.
# Ray's submitted runtime pins a compatible stack, but keep this CPU-only test
# importable on the head node as well.
if not hasattr(np, "long"):
    np.long = np.int64
if not hasattr(np, "ulong"):
    np.ulong = np.uint64

from cotrain.adv_evo_agent_loop import extract_complete_attack_payload
from cotrain.adaptive_attack_context import action_visible_text
from cotrain.agent_loop import extract_payload, has_malformed_tool_call, parse_tool_calls
from cotrain.rollouter import (
    _CoEvolutionTracker,
    _RolloutQualityTracker,
    _validate_model_pair_config,
)


COMPLETE_CALL = """<tool_call>
<function=send_email>
<parameter=recipients>["target@example.com"]</parameter>
<parameter=urgent>true</parameter>
</function>
</tool_call>"""


def test_no_population_ablation_accepts_only_current_and_template_routes():
    enabled, probabilities = _validate_model_pair_config(
        {
            "population_enabled": False,
            "prob_curr_curr": 0.9,
            "prob_old_atk_curr_def": 0.0,
            "prob_curr_atk_old_def": 0.0,
            "prob_fixed_template": 0.1,
            "prob_clean": 0.0,
        }
    )

    assert enabled is False
    assert probabilities == (0.9, 0.0, 0.0, 0.1, 0.0)


def test_no_population_ablation_rejects_any_historical_route():
    with pytest.raises(ValueError, match="historical model-pair probabilities"):
        _validate_model_pair_config(
            {
                "population_enabled": False,
                "prob_curr_curr": 0.89,
                "prob_old_atk_curr_def": 0.01,
                "prob_curr_atk_old_def": 0.0,
                "prob_fixed_template": 0.1,
                "prob_clean": 0.0,
            }
        )


def test_complete_qwen_tool_call_is_parsed_and_typed():
    assert parse_tool_calls(COMPLETE_CALL) == [
        {
            "name": "send_email",
            "arguments": {
                "recipients": ["target@example.com"],
                "urgent": True,
            },
        }
    ]
    assert has_malformed_tool_call(COMPLETE_CALL) is False


def test_truncated_tool_calls_fail_closed():
    incomplete_variants = [
        COMPLETE_CALL.removesuffix("</tool_call>"),
        COMPLETE_CALL.replace("</function>", ""),
        COMPLETE_CALL.replace("</parameter>", "", 1),
        COMPLETE_CALL + "\n<tool_call><function=send_email>",
    ]
    for text in incomplete_variants:
        assert has_malformed_tool_call(text) is True
        assert parse_tool_calls(text) == []


def test_attacker_payload_requires_complete_sft_tags():
    assert extract_complete_attack_payload("<think>x</think><ATTACK> payload </ATTACK>") == "payload"
    assert extract_complete_attack_payload("<think>x</think><attack>partial") == ""
    assert extract_complete_attack_payload("payload without tags") == ""
    assert extract_complete_attack_payload(
        "<think><attack>private draft</attack></think>"
        "<attack>public payload</attack>"
    ) == "public payload"
    assert extract_complete_attack_payload(
        "<think><attack>private draft</attack></think>"
    ) == ""
    malformed_nested = (
        "<think>outer</think></think>"
        "<attack>private nested draft</attack></think>"
    )
    assert extract_complete_attack_payload(malformed_nested) == ""
    assert extract_payload(malformed_nested) == ""


def test_private_reasoning_tool_xml_is_not_an_executable_action():
    text = (
        f"<think>{COMPLETE_CALL}</think>"
        "<tool_call><function=lookup>"
        "<parameter=query>balance</parameter>"
        "</function></tool_call>"
    )
    calls = parse_tool_calls(action_visible_text(text))
    assert calls == [{"name": "lookup", "arguments": {"query": "balance"}}]


def test_format_and_effective_asr_metrics_use_reached_attempts_only():
    tracker = _RolloutQualityTracker(window_size=10)
    common = {
        "model_pair": "curr_curr",
        "learned_attacker": True,
        "has_injected_payload": False,
        "attacker_sample_valid": True,
        "defender_sample_valid": True,
        "action_asr": float("nan"),
        "tool_responses": 1.0,
        "attacker_generated_turns": 1.0,
        "attacker_length_finishes": 0.0,
    }
    tracker.record(
        **(common | {
            "attacker_attempted": False,
            "format_violation": 0.0,
            "effective_asr": float("nan"),
        })
    )
    tracker.record(
        **(common | {
            "attacker_attempted": True,
            "format_violation": 0.0,
            "effective_asr": 1.0,
        })
    )
    tracker.record(
        **(common | {
            "attacker_attempted": True,
            "format_violation": 1.0,
            "effective_asr": 0.0,
            "attacker_length_finishes": 1.0,
        })
    )

    metrics = tracker.get_metrics()
    assert metrics["rollout_quality/format_valid_rate"] == 0.5
    assert metrics["rollout_quality/effective_asr_reached_only"] == 0.5
    assert metrics["rollout_quality/attacker_length_finish_rate"] == 1 / 3


def test_sequence_overflow_rate_uses_only_trajectories_with_known_token_lengths():
    tracker = _RolloutQualityTracker(window_size=10)
    common = {
        "model_pair": "curr_curr",
        "learned_attacker": True,
        "attacker_attempted": True,
        "has_injected_payload": True,
        "attacker_sample_valid": True,
        "defender_sample_valid": True,
    }
    tracker.record(
        **(
            common
            | {
                "attacker_train_sequence_tokens": 12000.0,
                "attacker_train_sequence_overflow": 0.0,
                "defender_train_sequence_tokens": 24000.0,
                "defender_train_sequence_overflow": 0.0,
            }
        )
    )
    tracker.record(
        **(
            common
            | {
                "attacker_sample_valid": False,
                "attacker_train_sequence_tokens": 45606.0,
                "attacker_train_sequence_overflow": 1.0,
                "defender_sample_valid": False,
                "defender_train_sequence_tokens": 33000.0,
                "defender_train_sequence_overflow": 1.0,
            }
        )
    )
    # A generation failure has no exact sequence length and must not dilute
    # the overflow denominator.
    tracker.record(
        **(
            common
            | {
                "attacker_sample_valid": False,
                "attacker_train_sequence_tokens": float("nan"),
                "attacker_train_sequence_overflow": 0.0,
                "defender_sample_valid": False,
                "defender_train_sequence_tokens": float("nan"),
                "defender_train_sequence_overflow": 0.0,
            }
        )
    )

    metrics = tracker.get_metrics()
    assert metrics["rollout_quality/attacker_train_sequence_overflow_rate"] == 0.5
    assert metrics["rollout_quality/defender_train_sequence_overflow_rate"] == 0.5
    assert metrics["rollout_quality/attacker_train_sequence_tokens_max"] == 45606.0
    assert metrics["rollout_quality/defender_train_sequence_tokens_max"] == 33000.0


def test_population_swap_resets_only_the_affected_comparison_window():
    tracker = _CoEvolutionTracker(window_size=20)
    for _ in range(10):
        tracker.record("curr_curr", True, False)
        tracker.record("curr_atk_old_def", False, False)

    assert "coevo/defender_improvement" in tracker.get_metrics()
    tracker.reset_pair("curr_atk_old_def")

    metrics = tracker.get_metrics()
    assert "coevo/defender_improvement" not in metrics
    assert metrics["coevo/curr_curr/asr"] == 1.0
