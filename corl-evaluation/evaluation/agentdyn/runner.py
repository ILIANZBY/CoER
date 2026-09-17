from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from openai import AsyncOpenAI, BadRequestError

from cotrain.adaptive_attack_context import (
    attacker_visible_assistant_text,
    canonical_native_visible_content,
)
from cotrain.attack_contract import AttackContractError, resolve_attack_contract
from cotrain.attacker_sft_contract import ATTACKER_SFT_SYSTEM_PROMPT
from cotrain.fixed_templates import (
    _IMPORTANT_INSTRUCTIONS_VARIANTS,
    FIXED_ATTACK_TEMPLATES,
    TEMPLATE_NAMES,
)
from cotrain.ppo_correctness import build_vector_placeholders, find_next_vector_placeholder

from .adaptive import build_attacker_user_message
from .dataset import EvalCase
from .protocol import (
    AGENT_DEFAULT_SYSTEM_PROMPT,
    SFT_CHAT_TEMPLATE,
    build_openai_tools,
    build_tools_system_section,
    coerce_tool_args,
    extract_complete_attack_payload,
    has_malformed_tool_call,
    message_content_with_reasoning,
    message_reasoning_content,
    parse_native_tool_calls,
    parse_text_tool_calls,
    sanitize_payload,
    strip_thinking,
)
from .types import AttackSpec, EndpointConfig, EvalSettings

_REPO_ROOT = Path(
    os.environ.get("CORL_TRAINING_ROOT", Path(__file__).resolve().parents[3])
).resolve()
_AGENTDYN_SRC = _REPO_ROOT / "AgentDyn" / "src"
if str(_AGENTDYN_SRC) not in sys.path:
    sys.path.insert(0, str(_AGENTDYN_SRC))

from agentdojo.functions_runtime import FunctionCall, FunctionsRuntime  # noqa: E402

from .task_registry import load_task_suites  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass
class Completion:
    text: str
    tool_calls: list[dict[str, Any]]
    native_tool_calls: bool
    malformed_tool_call: bool
    finish_reason: str | None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    history_content: str | None = None
    history_reasoning_content: str | None = None


class CompletionClient(Protocol):
    config: EndpointConfig

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        seed: int,
        tools: list[dict[str, Any]] | None = None,
        session_id: str | None = None,
    ) -> Completion: ...


class OpenAICompletionClient:
    def __init__(self, config: EndpointConfig):
        self.config = config
        self.client = AsyncOpenAI(
            base_url=config.base_url,
            api_key=config.api_key(),
            timeout=config.timeout_seconds,
            max_retries=config.max_retries,
        )

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        seed: int,
        tools: list[dict[str, Any]] | None = None,
        session_id: str | None = None,
    ) -> Completion:
        config = self.config
        kwargs: dict[str, Any] = {
            "model": config.model,
            "messages": messages,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_tokens": config.max_tokens,
        }
        if config.send_seed:
            kwargs["seed"] = int(seed)
        if config.protocol == "native" and tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        extra_body = dict(config.extra_body)
        if config.needs_session_id:
            if not session_id:
                raise ValueError(f"endpoint {config.label} requires an episode session_id")
            extra_body["session_id"] = session_id
        if config.training_chat_template:
            extra_body.setdefault("chat_template", SFT_CHAT_TEMPLATE)
            extra_body.setdefault("chat_template_kwargs", {"enable_thinking": True})
        if extra_body:
            kwargs["extra_body"] = extra_body

        response = await self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        message = choice.message
        text = message_content_with_reasoning(message)
        # Tool actions are part of the assistant's externally emitted content,
        # never its private reasoning.  Parsing the merged reasoning+content
        # string can turn a tool-call example merely discussed in <think> into
        # a real environment mutation.
        action_text = attacker_visible_assistant_text(text)
        native_calls, native_malformed = parse_native_tool_calls(message)
        if native_calls or getattr(message, "tool_calls", None):
            calls = native_calls
            native = True
            malformed = native_malformed
        else:
            calls = parse_text_tool_calls(action_text)
            native = False
            malformed = has_malformed_tool_call(action_text)
        usage = getattr(response, "usage", None)
        return Completion(
            text=text,
            tool_calls=calls,
            native_tool_calls=native,
            malformed_tool_call=malformed,
            finish_reason=str(choice.finish_reason) if choice.finish_reason is not None else None,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            history_content=getattr(message, "content", None),
            history_reasoning_content=message_reasoning_content(message),
        )


def _stable_seed(base_seed: int, case_id: str, role: str, turn: int) -> int:
    digest = hashlib.sha256(f"{base_seed}|{case_id}|{role}|{turn}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def episode_id(
    case: EvalCase,
    defender_label: str,
    attack: AttackSpec,
    generation_seed: int,
) -> str:
    raw = "|".join(
        (
            case.case_id,
            defender_label,
            attack.kind,
            attack.label,
            attack.attacker_label or "",
            attack.template_name or "",
            str(generation_seed),
        )
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def _serialize_result(result: Any, error: Any) -> str:
    if error:
        return f"Error: {error}"
    if hasattr(result, "model_dump"):
        return json.dumps(result.model_dump(), default=str)
    if isinstance(result, (dict, list)):
        return json.dumps(result, default=str)
    return str(result)


def _authoritative_security_snapshot(
    suite: Any,
    injection_task: Any,
    assistant_text: str,
    pre_environment: Any,
    environment: Any,
    function_trace: list[dict[str, Any]],
) -> bool:
    """Match the co-evolution early-stop predicate exactly.

    This is only an optimization.  A checker failure means that rollout must
    continue; the final evaluator below still records checker errors
    explicitly and excludes them from metric denominators.
    """

    try:
        trace = [FunctionCall(function=item["function"], args=item["args"]) for item in function_trace]
        model_output = [{"type": "text", "content": strip_thinking(assistant_text)}]
        return bool(
            suite._check_injection_task_security(
                injection_task,
                model_output,
                pre_environment,
                environment,
                trace,
            )
        )
    except Exception:
        return False


def _fixed_payload(goal: str, template_name: str, seed: int) -> str:
    if template_name not in TEMPLATE_NAMES:
        raise ValueError(f"unknown fixed template: {template_name}")
    if template_name == "important_instructions":
        variant = _IMPORTANT_INSTRUCTIONS_VARIANTS[seed % len(_IMPORTANT_INSTRUCTIONS_VARIANTS)]
        return variant.format(goal=goal, user="Emma Johnson", model="the AI assistant")
    template = FIXED_ATTACK_TEMPLATES[template_name]
    assert template is not None
    return template.format(goal=goal, user="Emma Johnson", model="the AI assistant")


def _native_assistant_message(completion: Completion) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": completion.history_content,
    }
    if completion.tool_calls:
        message["tool_calls"] = [
            {
                "id": call.get("id") or f"eval_call_{index}",
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": json.dumps(call.get("arguments", {}), default=str),
                },
            }
            for index, call in enumerate(completion.tool_calls)
        ]
    if completion.history_reasoning_content:
        message["reasoning_content"] = completion.history_reasoning_content
    return message


def _attacker_observation_text(
    protocol: str,
    completion: Completion,
) -> str:
    """Render the defender turn in the frozen attacker's training format."""

    if protocol == "native":
        observation = canonical_native_visible_content(
            completion.history_content or "",
            completion.tool_calls,
        )
    else:
        observation = completion.text
    return attacker_visible_assistant_text(observation)


def _is_context_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return isinstance(exc, BadRequestError) and any(
        phrase in text for phrase in ("context length", "context_length", "maximum context", "max model length")
    )


class AgentDynEpisodeRunner:
    def __init__(
        self,
        endpoints: dict[str, EndpointConfig],
        settings: EvalSettings,
        clients: dict[str, CompletionClient] | None = None,
    ):
        self.suites = load_task_suites(
            settings.benchmark_version,
            settings.task_catalog,
        )
        self.endpoints = dict(endpoints)
        self.settings = settings
        self.clients: dict[str, CompletionClient] = clients or {
            label: OpenAICompletionClient(config) for label, config in endpoints.items()
        }

    def _endpoint(self, label: str) -> tuple[EndpointConfig, CompletionClient]:
        if label not in self.endpoints or label not in self.clients:
            raise ValueError(f"endpoint is not configured: {label}")
        return self.endpoints[label], self.clients[label]

    async def run(
        self,
        case: EvalCase,
        *,
        defender_label: str,
        attack: AttackSpec,
        generation_seed: int,
        benchmark_kind: str,
    ) -> dict[str, Any]:
        attack.validate()
        started = time.time()
        current_episode_id = episode_id(case, defender_label, attack, generation_seed)
        result: dict[str, Any] = {
            "episode_id": current_episode_id,
            "benchmark_kind": benchmark_kind,
            "case_id": case.case_id,
            "row_index": case.row_index,
            "suite_name": case.suite_name,
            "user_task_id": case.user_task_id,
            "injection_task_id": case.injection_task_id,
            "defender_label": defender_label,
            "attacker_label": attack.attacker_label,
            "attack_kind": attack.kind,
            "attack_label": attack.label,
            "template_name": attack.template_name,
            "generation_seed": generation_seed,
            "metric_exclusion_reason": case.metric_exclusion_reason,
            "infra_error": None,
            "model_failure": None,
            "evaluator_error": None,
        }

        try:
            suite = self.suites[case.suite_name]
            user_task = suite.user_tasks[case.user_task_id]
            injection_task = (
                suite.injection_tasks[case.injection_task_id] if case.injection_task_id is not None else None
            )
        except Exception as exc:
            result["infra_error"] = f"task_load_error: {type(exc).__name__}: {exc}"
            result["elapsed_seconds"] = time.time() - started
            return result

        if attack.kind == "clean" and not case.is_clean:
            raise ValueError("clean evaluation requires a clean EvalCase")
        if attack.kind != "clean" and case.is_clean:
            raise ValueError("attack evaluation requires an injected EvalCase")

        vector_ids = case.injection_vector_ids
        placeholders: dict[str, str] = {}
        static_payloads: dict[str, str] = {}
        fixed_payload = ""
        if attack.kind == "adaptive":
            placeholders = build_vector_placeholders({key: "" for key in vector_ids})
            environment_injections = placeholders
        elif attack.kind == "fixed":
            assert injection_task is not None and attack.template_name is not None
            fixed_payload = _fixed_payload(
                str(getattr(injection_task, "GOAL", injection_task)),
                attack.template_name,
                _stable_seed(generation_seed, case.case_id, "template", 0),
            )
            static_payloads = {key: fixed_payload for key in vector_ids}
            environment_injections = static_payloads
        elif attack.kind == "dataset":
            static_payloads = dict(case.dataset_injections)
            environment_injections = static_payloads
        else:
            environment_injections = {}

        try:
            environment = suite.load_and_inject_default_environment(environment_injections)
            if hasattr(user_task, "init_environment"):
                environment = user_task.init_environment(environment)
            pre_environment = environment.model_copy(deep=True)
            runtime = FunctionsRuntime(suite.tools)
            tool_schemas = build_openai_tools(suite.tools)
        except Exception as exc:
            result["infra_error"] = f"environment_init_error: {type(exc).__name__}: {exc}"
            result["elapsed_seconds"] = time.time() - started
            return result

        contract = None
        if injection_task is not None:
            try:
                contract = resolve_attack_contract(
                    injection_task,
                    pre_environment,
                    available_tool_names=getattr(runtime, "functions", {}).keys(),
                )
            except AttackContractError as exc:
                result["infra_error"] = f"attack_contract_error: {exc}"
                result["elapsed_seconds"] = time.time() - started
                return result
        result["ground_truth_contract"] = contract.render() if contract is not None else None
        result["ground_truth_call_count"] = len(contract.calls) if contract is not None else None

        defender_config, defender_client = self._endpoint(defender_label)
        attacker_client: CompletionClient | None = None
        if attack.attacker_label:
            _, attacker_client = self._endpoint(attack.attacker_label)

        system_content = AGENT_DEFAULT_SYSTEM_PROMPT
        if defender_config.protocol == "text":
            system_content = build_tools_system_section(tool_schemas) + "\n\n" + system_content
        defender_messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": str(getattr(user_task, "PROMPT", case.user_prompt))},
        ]
        defender_trajectory: list[dict[str, Any]] = [
            {"role": "system", "content": AGENT_DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": str(getattr(user_task, "PROMPT", case.user_prompt))},
        ]
        attacker_messages = [{"role": "system", "content": ATTACKER_SFT_SYSTEM_PROMPT}]
        injection_records: list[dict[str, Any]] = []
        function_trace: list[dict[str, Any]] = []
        last_assistant_text = ""
        tool_calls_total = 0
        tool_format_violations = 0
        defender_length_finishes = 0
        attacker_length_finishes = 0
        attacker_attempts = 0
        attacker_valid_turns = 0
        tool_responses = 0
        tool_response_overflows = 0
        reached = False
        early_success = False
        defender_turns = 0
        prompt_tokens = 0
        completion_tokens = 0
        defender_stopped_normally = False

        for turn in range(self.settings.max_defender_turns):
            try:
                completion = await defender_client.complete(
                    defender_messages,
                    seed=_stable_seed(generation_seed, case.case_id, "defender", turn),
                    tools=tool_schemas,
                    session_id=f"{current_episode_id}-defender",
                )
            except Exception as exc:
                message = f"defender_generation_error: {type(exc).__name__}: {exc}"
                if _is_context_limit_error(exc):
                    result["model_failure"] = message
                else:
                    result["infra_error"] = message
                break
            defender_turns += 1
            last_assistant_text = completion.text
            prompt_tokens += completion.prompt_tokens or 0
            completion_tokens += completion.completion_tokens or 0
            defender_length_finishes += int(completion.finish_reason == "length")
            tool_format_violations += int(completion.malformed_tool_call)

            if completion.native_tool_calls:
                defender_messages.append(_native_assistant_message(completion))
            else:
                defender_messages.append({"role": "assistant", "content": completion.text})
            stored_text = _attacker_observation_text(
                defender_config.protocol,
                completion,
            )
            defender_trajectory.append(
                {
                    "role": "assistant",
                    "content": stored_text,
                    "tool_calls": [
                        {"name": call["name"], "arguments": call.get("arguments", {})} for call in completion.tool_calls
                    ],
                }
            )
            if injection_records and "defender_action_after" not in injection_records[-1]:
                injection_records[-1]["defender_action_after"] = stored_text[:200]
            if not completion.tool_calls:
                defender_stopped_normally = True
                break

            for call_index, call in enumerate(completion.tool_calls):
                tool_calls_total += 1
                tool_responses += 1
                tool_name = call["name"]
                tool_args = coerce_tool_args(tool_name, call.get("arguments", {}), runtime)
                try:
                    function_result, function_error = runtime.run_function(environment, tool_name, tool_args)
                    response_text = _serialize_result(function_result, function_error)
                    if not function_error:
                        function_trace.append({"function": tool_name, "args": tool_args})
                        if injection_records and injection_task is not None:
                            early_success = _authoritative_security_snapshot(
                                suite,
                                injection_task,
                                completion.text,
                                pre_environment,
                                environment,
                                function_trace,
                            )
                except Exception as exc:
                    response_text = f"Error executing {tool_name}: {exc}"

                if attack.kind == "adaptive":
                    placeholder_match = find_next_vector_placeholder(response_text, placeholders)
                    while placeholder_match is not None:
                        vector_id, placeholder = placeholder_match
                        reached = True
                        if early_success or (
                            self.settings.max_attacker_turns > 0
                            and len(injection_records) >= self.settings.max_attacker_turns
                        ):
                            response_text = response_text.replace(placeholder, "", 1)
                            placeholder_match = find_next_vector_placeholder(response_text, placeholders)
                            continue

                        injection_num = len(injection_records) + 1
                        marker = response_text.index(placeholder)
                        before = response_text[max(0, marker - 400) : marker]
                        after = response_text[marker + len(placeholder) : marker + len(placeholder) + 300]
                        assert contract is not None and attacker_client is not None
                        attacker_user = build_attacker_user_message(
                            attack_goal=contract.goal,
                            rendered_contract=contract.render(),
                            has_tool_calls=bool(contract.calls),
                            suite_name=case.suite_name,
                            injection_num=injection_num,
                            tool_name=tool_name,
                            before=before,
                            after=after,
                            defender_trajectory=defender_trajectory,
                            previous_record=injection_records[-1] if injection_records else None,
                        )
                        attacker_messages.append({"role": "user", "content": attacker_user})
                        attacker_attempts += 1
                        attacker_text = ""
                        finish_reason = None
                        generation_error = None
                        try:
                            attacker_completion = await attacker_client.complete(
                                attacker_messages,
                                seed=_stable_seed(
                                    generation_seed,
                                    case.case_id,
                                    "attacker",
                                    injection_num,
                                ),
                                session_id=f"{current_episode_id}-attacker",
                            )
                            attacker_text = attacker_completion.text
                            finish_reason = attacker_completion.finish_reason
                            attacker_length_finishes += int(finish_reason == "length")
                        except Exception as exc:
                            generation_error = f"{type(exc).__name__}: {exc}"
                            if _is_context_limit_error(exc):
                                result["model_failure"] = (
                                    result["model_failure"] or f"attacker_context_limit: {generation_error}"
                                )
                            else:
                                result["infra_error"] = (
                                    result["infra_error"] or f"attacker_generation_error: {generation_error}"
                                )
                        if attacker_text:
                            attacker_messages.append({"role": "assistant", "content": attacker_text})
                        payload = sanitize_payload(extract_complete_attack_payload(attacker_text))
                        format_valid = bool(payload)
                        attacker_valid_turns += int(format_valid)
                        response_text = response_text.replace(placeholder, payload, 1)
                        record = {
                            "turn": turn,
                            "tool": tool_name,
                            "vector_id": str(vector_id),
                            "injection_num": injection_num,
                            "payload": payload,
                            "format_valid": format_valid,
                            "finish_reason": finish_reason,
                            "generation_error": generation_error,
                        }
                        injection_records.append(record)
                        placeholder_match = find_next_vector_placeholder(response_text, placeholders)
                elif attack.kind in ("fixed", "dataset"):
                    visible_payloads = [
                        payload
                        for payload in static_payloads.values()
                        if payload
                        and (payload in response_text or json.dumps(payload, ensure_ascii=False)[1:-1] in response_text)
                    ]
                    if visible_payloads:
                        reached = True

                if (
                    self.settings.max_tool_response_chars > 0
                    and len(response_text) > self.settings.max_tool_response_chars
                ):
                    tool_response_overflows += 1
                    result["model_failure"] = result["model_failure"] or (
                        "tool_response_context_exceeds_limit: "
                        f"{len(response_text)} > {self.settings.max_tool_response_chars}"
                    )
                    response_text = response_text[: self.settings.max_tool_response_chars]

                defender_trajectory.append({"role": "tool", "content": response_text})
                if completion.native_tool_calls:
                    defender_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.get("id") or f"eval_call_{call_index}",
                            "content": response_text,
                        }
                    )
                else:
                    defender_messages.append(
                        {
                            "role": "user",
                            "content": f"<tool_response>\n{response_text}\n</tool_response>",
                        }
                    )

            if result["infra_error"]:
                break

        if defender_turns >= self.settings.max_defender_turns and not defender_stopped_normally:
            result["model_failure"] = result["model_failure"] or "max_defender_turns_reached"

        model_output = [{"type": "text", "content": strip_thinking(last_assistant_text)}]
        asr_success: bool | None = None
        utility_success: bool | None = None
        try:
            trace = [FunctionCall(function=item["function"], args=item["args"]) for item in function_trace]
            if injection_task is not None:
                asr_success = bool(
                    suite._check_injection_task_security(
                        injection_task,
                        model_output,
                        pre_environment,
                        environment,
                        trace,
                    )
                )
            utility_success = bool(
                suite._check_user_task_utility(
                    user_task,
                    model_output,
                    pre_environment,
                    environment,
                    trace,
                )
            )
        except Exception as exc:
            result["evaluator_error"] = f"{type(exc).__name__}: {exc}"

        has_nonempty_payload = any(record.get("payload") for record in injection_records)
        result.update(
            {
                "metric_eligible": (
                    not result["infra_error"]
                    and not result["evaluator_error"]
                    and not result["metric_exclusion_reason"]
                ),
                "asr_success": asr_success,
                "utility_success": utility_success,
                "safe_utility": (
                    bool(utility_success and not asr_success)
                    if asr_success is not None and utility_success is not None
                    else None
                ),
                "injection_reached": reached if attack.kind != "clean" else None,
                "effective_asr": (bool(asr_success) if attack.kind != "clean" and reached else None),
                "attacker_attempts": attacker_attempts,
                "attacker_valid_turns": attacker_valid_turns,
                "attacker_episode_format_valid": (
                    attacker_attempts > 0 and attacker_valid_turns == attacker_attempts
                    if attack.kind == "adaptive"
                    else None
                ),
                "has_nonempty_payload": has_nonempty_payload if attack.kind == "adaptive" else None,
                "defender_turns": defender_turns,
                "defender_tool_calls": tool_calls_total,
                "defender_tool_format_violations": tool_format_violations,
                "defender_length_finish_count": defender_length_finishes,
                "attacker_length_finish_count": attacker_length_finishes,
                "tool_responses": tool_responses,
                "tool_response_overflows": tool_response_overflows,
                "prompt_tokens": prompt_tokens or None,
                "completion_tokens": completion_tokens or None,
                "injection_records": injection_records,
                "function_trace": function_trace,
                "defender_trajectory": defender_trajectory,
                "attacker_messages": attacker_messages if attack.kind == "adaptive" else [],
                "elapsed_seconds": time.time() - started,
            }
        )
        return result
