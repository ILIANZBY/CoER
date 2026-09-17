"""Re-definition with different names of the current types in the OpenAI API.
We re-define them to avoid relying on the API, and to allow for more flexibility
in the agent implementations.

The license for the original code is [here](https://github.com/openai/openai-python/blob/bba23438a63121102fe066982c91771fcca19c80/LICENSE).
"""

import re
from typing import Literal

from typing_extensions import Required, TypedDict

from agentdojo.functions_runtime import FunctionCall


class TextContentBlock(TypedDict, total=False):
    type: Required[Literal["text"]]
    content: Required[str]


class ThinkingContentBlock(TypedDict, total=False):
    type: Required[Literal["thinking"]]
    content: Required[str]
    id: Required[str | None]


class RedactedThinkingContentBlock(TypedDict, total=False):
    type: Required[Literal["redacted_thinking"]]
    content: Required[str]


MessageContentBlock = TextContentBlock | ThinkingContentBlock | RedactedThinkingContentBlock


def text_content_block_from_string(content: str) -> TextContentBlock:
    return TextContentBlock(type="text", content=content)


class ChatUserMessage(TypedDict, total=False):
    """Message from the user in the chat."""

    role: Required[Literal["user"]]
    """The role of the message. Always `"user"` for user messages."""
    content: Required[list[MessageContentBlock]]
    """The content of the message."""


class ChatAssistantMessage(TypedDict, total=False):
    """Message from the assistant in the chat."""

    role: Required[Literal["assistant"]]
    """The role of the message. Always `"assistant"` for assistant messages."""

    content: Required[list[MessageContentBlock] | None]
    """The content of the message, which can be a string or `None` (if `tool_calls` are provided)."""

    tool_calls: Required[list["FunctionCall"] | None]
    """A list of tool calls that the assistant requests, which can be `None` (if no tool calls are requires are provided)."""


class ChatToolResultMessage(TypedDict, total=False):
    """Message from the tool with the result of a function call."""

    tool_call: Required["FunctionCall"]
    """The function call that was executed."""
    content: Required[list[MessageContentBlock]]
    """The output of the tool call."""
    role: Required[Literal["tool"]]
    """The role of the message. Always `"tool"` for tool messages."""
    tool_call_id: Required[str | None]
    """The ID of the tool call. It can be `None` if the LLM does not provide or required it.
    It is used by e.g., OpenAI and Anthropic."""
    error: Required[str | None]
    """The error message of the tool call. It can be `None` if there is no error."""


class ChatSystemMessage(TypedDict, total=False):
    """System message in the chat. This can be set by using the
    [SystemMessage][agentdojo.agent_pipeline.SystemMessage] pipeline element in a pipeline."""

    role: Required[Literal["system"]]
    """The role of the message. Always `"system"` for system messages."""
    content: Required[list[MessageContentBlock]]
    """The content of the message."""


ChatMessage = ChatUserMessage | ChatAssistantMessage | ChatToolResultMessage | ChatSystemMessage
"""Union type of possible messages in the chat. These are very similar types currently supported by the OpenAI API."""


def get_text_content_as_str(content_blocks: list[MessageContentBlock]) -> str:
    non_empty_content = [c["content"] for c in content_blocks if c["content"] is not None]
    return "\n".join(non_empty_content)


def get_user_visible_text(text: str) -> str:
    """Remove model reasoning enclosed in ``<think>`` tags.

    Utility validators grade what the user receives, not hidden reasoning.  An
    unclosed think block is also hidden through the end of the response, which
    commonly happens on truncated rollouts.
    """
    text = re.sub(r"<think\b[^>]*>.*?</think\s*>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<think\b[^>]*>.*\Z", "", text, flags=re.IGNORECASE | re.DOTALL)
    return text.strip()


def get_user_visible_content_as_str(content_blocks: list[MessageContentBlock]) -> str:
    """Join only user-visible text blocks and remove inline think sections."""
    visible_content = [
        block["content"]
        for block in content_blocks
        if block.get("type") == "text" and block.get("content") is not None
    ]
    return get_user_visible_text("\n".join(visible_content))
