import json
import re
import uuid
from collections.abc import Sequence
from typing import overload

import openai
from openai._types import NOT_GIVEN
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionDeveloperMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallParam,
    ChatCompletionReasoningEffort,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)
from openai.types.shared_params import FunctionDefinition
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_random_exponential

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import EmptyEnv, Env, Function, FunctionCall, FunctionsRuntime
from agentdojo.types import (
    ChatAssistantMessage,
    ChatMessage,
    ChatSystemMessage,
    ChatToolResultMessage,
    ChatUserMessage,
    MessageContentBlock,
    get_text_content_as_str,
    text_content_block_from_string,
)


def _tool_call_to_openai(tool_call: FunctionCall) -> ChatCompletionMessageToolCallParam:
    if tool_call.id is None:
        raise ValueError("`tool_call.id` is required for OpenAI")
    return ChatCompletionMessageToolCallParam(
        id=tool_call.id,
        type="function",
        function={
            "name": tool_call.function,
            "arguments": json.dumps(tool_call.args),
        },
    )


_REASONING_MODELS = {"o1", "o3"}


def _is_reasoning_model(model_name: str) -> bool:
    return any(model in model_name for model in _REASONING_MODELS)


@overload
def _content_blocks_to_openai_content_blocks(
    message: ChatUserMessage | ChatSystemMessage,
) -> list[ChatCompletionContentPartTextParam]: ...


@overload
def _content_blocks_to_openai_content_blocks(
    message: ChatAssistantMessage | ChatToolResultMessage,
) -> list[ChatCompletionContentPartTextParam] | None: ...


def _content_blocks_to_openai_content_blocks(
    message: ChatUserMessage | ChatAssistantMessage | ChatSystemMessage | ChatToolResultMessage,
) -> list[ChatCompletionContentPartTextParam] | None:
    if message["content"] is None:
        return None
    return [ChatCompletionContentPartTextParam(type="text", text=el["content"] or "") for el in message["content"]]


def _message_to_openai(message: ChatMessage, model_name: str) -> ChatCompletionMessageParam:
    match message["role"]:
        case "system":
            # Use standard 'system' role for local/non-OpenAI models that don't support 'developer' role
            _openai_only_models = {"gpt-", "o1", "o3"}
            use_developer = any(model_name.startswith(p) for p in _openai_only_models)
            if not use_developer or model_name == "qwen3-max":
                return ChatCompletionSystemMessageParam(
                    role="system", content=_content_blocks_to_openai_content_blocks(message)
                )
            else:
                return ChatCompletionDeveloperMessageParam(
                    role="developer", content=_content_blocks_to_openai_content_blocks(message)
                )
        case "user":
            return ChatCompletionUserMessageParam(
                role="user", content=_content_blocks_to_openai_content_blocks(message)
            )
        case "assistant":
            if message["tool_calls"] is not None and len(message["tool_calls"]) > 0:
                tool_calls = [_tool_call_to_openai(tool_call) for tool_call in message["tool_calls"]]
                return ChatCompletionAssistantMessageParam(
                    role="assistant",
                    content=_content_blocks_to_openai_content_blocks(message),
                    tool_calls=tool_calls,
                )
            return ChatCompletionAssistantMessageParam(
                role="assistant",
                content=_content_blocks_to_openai_content_blocks(message),
            )
        case "tool":
            if message["tool_call_id"] is None:
                raise ValueError("`tool_call_id` should be specified for OpenAI.")
            return ChatCompletionToolMessageParam(
                content=message["error"] or _content_blocks_to_openai_content_blocks(message),
                tool_call_id=message["tool_call_id"],
                role="tool",
                name=message["tool_call"].function,  # type: ignore -- this is actually used, and is important!
            )
        case _:
            raise ValueError(f"Invalid message type: {message}")


def _openai_to_tool_call(tool_call: ChatCompletionMessageToolCall) -> FunctionCall:
    return FunctionCall(
        function=tool_call.function.name,
        args=json.loads(tool_call.function.arguments),
        id=tool_call.id,
    )


def _assistant_message_to_content(message: ChatCompletionMessage) -> list[MessageContentBlock] | None:
    if message.content is None:
        return None
    # Strip <think>...</think> block from content
    content = re.sub(r"<think>.*?</think>\s*", "", message.content, flags=re.DOTALL).strip()
    if not content:
        return None
    return [text_content_block_from_string(content)]


def _parse_tool_calls_from_content(content: str) -> list[FunctionCall] | None:
    """Parse tool calls from content. Supports two formats:
    1. XML: <tool_call><function=name><parameter=key>value</parameter></function></tool_call>
    2. Hermes JSON: <tool_call>{"name": "func", "arguments": {...}}</tool_call>
    """
    tool_calls = []
    for tool_call_match in re.finditer(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL):
        tool_call_str = tool_call_match.group(1).strip()

        # Try XML format first: <function=name><parameter=key>value</parameter></function>
        func_match = re.search(r"<function=(\w+)>", tool_call_str)
        if func_match:
            func_name = func_match.group(1)
            args = {}
            for param_match in re.finditer(r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", tool_call_str, re.DOTALL):
                key = param_match.group(1)
                val = param_match.group(2).strip()
                try:
                    args[key] = json.loads(val)
                except (json.JSONDecodeError, ValueError):
                    args[key] = val
            tool_calls.append(FunctionCall(function=func_name, args=args, id=str(uuid.uuid4())))
            continue

        # Try Hermes JSON format: {"name": "func", "arguments": {...}}
        try:
            parsed = json.loads(tool_call_str)
            func_name = parsed.get("name") or parsed.get("function")
            arguments = parsed.get("arguments") or parsed.get("parameters") or {}
            if func_name:
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                tool_calls.append(FunctionCall(function=func_name, args=arguments, id=str(uuid.uuid4())))
                continue
        except (json.JSONDecodeError, ValueError, AttributeError):
            pass

    return tool_calls if tool_calls else None


def _openai_to_assistant_message(message: ChatCompletionMessage) -> ChatAssistantMessage:
    # Get reasoning_content (sglang --reasoning-parser mode puts think here)
    reasoning_content = getattr(message, "reasoning_content", None) or ""
    
    if message.tool_calls is not None:
        # API returned tool_calls directly
        tool_calls = []
        for tool_call in message.tool_calls:
            try:
                temp = _openai_to_tool_call(tool_call)
                tool_calls.append(temp)
            except Exception:
                continue
    elif message.content is not None and "<tool_call>" in message.content:
        # sglang / hermes format: tool calls embedded in content as XML
        tool_calls = _parse_tool_calls_from_content(message.content)
    elif reasoning_content and "<tool_call>" in reasoning_content:
        # Thinking mode: reasoning parser may have swallowed tool_call into reasoning_content
        tool_calls = _parse_tool_calls_from_content(reasoning_content)
    else:
        tool_calls = None
    
    result = ChatAssistantMessage(role="assistant", content=_assistant_message_to_content(message), tool_calls=tool_calls)
    # Capture reasoning_content for debugging (sglang --reasoning-parser mode)
    if reasoning_content:
        result["reasoning_content"] = reasoning_content  # type: ignore[typeddict-unknown-key]
    # Also capture raw content for debugging tool_call parsing issues
    if message.content is not None:
        result["raw_content"] = message.content  # type: ignore[typeddict-unknown-key]
    return result


def _function_to_openai(f: Function) -> ChatCompletionToolParam:
    function_definition = FunctionDefinition(
        name=f.name,
        description=f.description,
        parameters=f.parameters.model_json_schema(),
    )
    return ChatCompletionToolParam(type="function", function=function_definition)


@retry(
    wait=wait_random_exponential(multiplier=1, max=40),
    stop=stop_after_attempt(3),
    reraise=True,
    retry=retry_if_not_exception_type((openai.BadRequestError, openai.UnprocessableEntityError)),
)
def chat_completion_request(
    client: openai.OpenAI,
    model: str,
    messages: Sequence[ChatCompletionMessageParam],
    tools: Sequence[ChatCompletionToolParam],
    reasoning_effort: ChatCompletionReasoningEffort | None,
    temperature: float | None = 0.0,
    enable_thinking: bool | None = None,
):
    kwargs = dict(
        model=model,
        messages=messages,
        tools=tools or NOT_GIVEN,
        tool_choice="none" if tools else NOT_GIVEN,
        temperature=temperature if temperature is not None else NOT_GIVEN,
        reasoning_effort=reasoning_effort or NOT_GIVEN,
    )
    if enable_thinking is not None:
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": enable_thinking}}
    return client.chat.completions.create(**kwargs)


class OpenAILLM(BasePipelineElement):
    """LLM pipeline element that uses OpenAI's API.

    Args:
        client: The OpenAI client.
        model: The model name.
        temperature: The temperature to use for generation.
    """

    def __init__(
        self,
        client: openai.OpenAI,
        model: str,
        reasoning_effort: ChatCompletionReasoningEffort | None = None,
        temperature: float | None = 0.0,
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.reasoning_effort: ChatCompletionReasoningEffort | None = reasoning_effort

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        openai_messages = [_message_to_openai(message, self.model) for message in messages]
        openai_tools = [_function_to_openai(tool) for tool in runtime.functions.values()]
        completion = chat_completion_request(
            self.client, self.model, openai_messages, openai_tools, self.reasoning_effort, self.temperature
        )
        output = _openai_to_assistant_message(completion.choices[0].message)
        messages = [*messages, output]
        return query, runtime, env, messages, extra_args


class OpenAILLMToolFilter(BasePipelineElement):
    def __init__(self, prompt: str, client: openai.OpenAI, model: str, reasoning_effort: ChatCompletionReasoningEffort | None = None, temperature: float | None = 0.0) -> None:
        self.prompt = prompt
        self.client = client
        self.model = model
        self.temperature = temperature
        self.reasoning_effort: ChatCompletionReasoningEffort | None = reasoning_effort

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        messages = [*messages, ChatUserMessage(role="user", content=[text_content_block_from_string(self.prompt)])]
        openai_messages = [_message_to_openai(message, self.model) for message in messages]
        openai_tools = [_function_to_openai(tool) for tool in runtime.functions.values()]
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            tools=openai_tools or NOT_GIVEN,
            tool_choice="none" if openai_tools else NOT_GIVEN,
            temperature=self.temperature if self.temperature is not None else NOT_GIVEN,
            reasoning_effort=self.reasoning_effort or NOT_GIVEN,
        )
        output = _openai_to_assistant_message(completion.choices[0].message)

        new_tools = {}
        for tool_name, tool in runtime.functions.items():
            if output["content"] is not None and tool_name in get_text_content_as_str(output["content"]):
                new_tools[tool_name] = tool

        runtime.update_functions(new_tools)

        messages = [*messages, output]
        return query, runtime, env, messages, extra_args


# Backward-compatible alias
LocalLLM = OpenAILLM
