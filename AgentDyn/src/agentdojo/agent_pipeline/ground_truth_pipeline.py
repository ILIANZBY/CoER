from collections.abc import Mapping, Sequence

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.tool_execution import tool_result_to_str
from agentdojo.base_tasks import BaseInjectionTask, BaseUserTask
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionCall, FunctionsRuntime
from agentdojo.types import ChatAssistantMessage, ChatMessage, ChatToolResultMessage, text_content_block_from_string
from pydantic import BaseModel

_ARG_PLACEHOLDERS = {str: "", int: 0, float: 0.0, bool: False, dict: {}, list: []}


class GroundTruthPipeline(BasePipelineElement):
    """Pipeline that executes a task's `ground_truth`.

    It is used to identify which injection placeholders are seen by the agent during the task execution."""

    def __init__(self, task: BaseUserTask | BaseInjectionTask | None) -> None:
        self._task = task

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        if self._task is None:
            raise ValueError("No task set for the ground truth. Call `set_task` first.")
        ground_truth_tool_calls = self._task.ground_truth(env)
        new_messages = []
        for tool_call in ground_truth_tool_calls:
            tool_call = _resolve_ground_truth_placeholders(tool_call, env)
            tool_name = tool_call.function
            new_messages.append(
                ChatAssistantMessage(
                    role="assistant", tool_calls=[tool_call], content=[text_content_block_from_string("")]
                )
            )
            tool_result, _ = runtime.run_function(env, tool_name, tool_call.args, raise_on_error=True)
            new_messages.append(
                ChatToolResultMessage(
                    role="tool",
                    content=[text_content_block_from_string(tool_result_to_str(tool_result))],
                    tool_call=tool_call,
                    tool_call_id=None,
                    error=None,
                )
            )

        assistant_message_content = self._task.GROUND_TRUTH_OUTPUT

        new_messages.append(
            ChatAssistantMessage(
                role="assistant", content=[text_content_block_from_string(assistant_message_content)], tool_calls=None
            )
        )
        return query, runtime, env, [*messages, *new_messages], extra_args


def _find_pending_otp(value: object, seen: set[int] | None = None) -> str | None:
    """Return a pending OTP from the environment, if a ground-truth step created one."""
    if seen is None:
        seen = set()
    value_id = id(value)
    if value_id in seen:
        return None
    seen.add(value_id)

    if isinstance(value, Mapping):
        for key, nested in value.items():
            if key == "verification_stack" and isinstance(nested, Mapping) and nested:
                return next(iter(nested))
            found = _find_pending_otp(nested, seen)
            if found is not None:
                return found
        return None
    if isinstance(value, BaseModel):
        for service_name in ("bank_database", "github_platform", "shopping_platform"):
            service = getattr(value, service_name, None)
            for account in getattr(service, "account_list", []):
                stack = getattr(account, "verification_stack", None)
                if isinstance(stack, Mapping) and stack:
                    return next(iter(stack))
        for field_name in value.model_fields:
            field_value = getattr(value, field_name)
            if field_name == "verification_stack" and isinstance(field_value, Mapping) and field_value:
                return next(iter(field_value))
            found = _find_pending_otp(field_value, seen)
            if found is not None:
                return found
        return None
    if isinstance(value, (list, tuple, set)):
        for nested in value:
            found = _find_pending_otp(nested, seen)
            if found is not None:
                return found
    return None


def _resolve_ground_truth_value(value: object, env: Env) -> object:
    if value == "ONE-TIME-PASSWD":
        return _find_pending_otp(env) or value
    if isinstance(value, Mapping):
        return {key: _resolve_ground_truth_value(nested, env) for key, nested in value.items()}
    if isinstance(value, list):
        return [_resolve_ground_truth_value(nested, env) for nested in value]
    if isinstance(value, tuple):
        return tuple(_resolve_ground_truth_value(nested, env) for nested in value)
    return value


def _resolve_ground_truth_placeholders(tool_call: FunctionCall, env: Env) -> FunctionCall:
    resolved_args = _resolve_ground_truth_value(tool_call.args, env)
    if resolved_args == tool_call.args:
        return tool_call
    return tool_call.model_copy(update={"args": resolved_args})
