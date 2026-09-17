"""Dependency-free correctness primitives for co-evolution PPO rollouts."""

from dataclasses import dataclass
import math
import re
from typing import Any
from uuid import uuid4


class TokenAlignmentError(ValueError):
    """A rollout cannot be used by PPO because its behavior tokens are ambiguous."""


class TokenLengthOverflowError(TokenAlignmentError):
    """An exact rollout is longer than the training engine can accept."""

    def __init__(self, actual_length: int, max_length: int):
        self.actual_length = int(actual_length)
        self.max_length = int(max_length)
        super().__init__(
            "exact trajectory exceeds the training token limit: "
            f"total={self.actual_length}/{self.max_length}"
        )


def user_visible_assistant_text(text: str) -> str:
    """Return only natural-language text visible to the user.

    Co-training must preserve reasoning in the replayed PPO context, but that
    hidden reasoning is not the model's answer and must not satisfy utility or
    security substring verifiers. Likewise, this loop receives tool calls as
    textual XML even though a normal structured chat API exposes their
    arguments separately from assistant content. Those arguments are actions,
    not a user-facing answer, and must not satisfy output-only validators.

    An unclosed think/tool-call block is hidden through the end (the common
    truncation case); an orphan closing think tag is handled for vLLM responses
    where the opening token is returned separately.
    """

    text = str(text or "")
    text = re.sub(
        r"<think\b[^>]*>.*?</think\s*>",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if re.search(r"</think\s*>", text, flags=re.IGNORECASE):
        text = re.split(
            r"</think\s*>", text, flags=re.IGNORECASE
        )[-1]
    text = re.sub(
        r"<think\b[^>]*>.*\Z",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(
        r"<tool_call\b[^>]*>.*?</tool_call\s*>",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(
        r"<tool_call\b[^>]*>.*\Z",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return text.strip()


@dataclass(frozen=True)
class GenerationTrace:
    """Exact input/output token trace returned by one vLLM request."""

    prompt_token_ids: list[int]
    token_ids: list[int]
    logprobs: list[float]
    finish_reason: str | None = None


def _response_extra_field(obj, name: str):
    """Read a vLLM extension preserved by both old and new OpenAI SDKs."""
    value = getattr(obj, name, None)
    if value is not None:
        return value
    model_extra = getattr(obj, "model_extra", None)
    if isinstance(model_extra, dict):
        return model_extra.get(name)
    if isinstance(obj, dict):
        return obj.get(name)
    return None


def extract_generation_trace(response) -> GenerationTrace:
    """Extract exact vLLM token IDs and sampled-token logprobs, or fail closed."""
    try:
        choice = response.choices[0]
        prompt_token_ids = _response_extra_field(response, "prompt_token_ids")
        token_ids = _response_extra_field(choice, "token_ids")
        logprob_content = choice.logprobs.content if choice.logprobs is not None else None
        finish_reason = getattr(choice, "finish_reason", None)
    except (AttributeError, IndexError, TypeError) as exc:
        raise TokenAlignmentError("malformed chat-completion response") from exc

    if prompt_token_ids is None:
        raise TokenAlignmentError("vLLM response is missing prompt_token_ids")
    if token_ids is None:
        raise TokenAlignmentError("vLLM response is missing choice.token_ids")
    if not logprob_content:
        raise TokenAlignmentError("vLLM response is missing sampled-token logprobs")

    try:
        prompt_token_ids = [int(token_id) for token_id in prompt_token_ids]
        token_ids = [int(token_id) for token_id in token_ids]
        logprobs = [float(token_lp.logprob) for token_lp in logprob_content]
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise TokenAlignmentError(
            "vLLM response contains invalid token IDs or logprobs"
        ) from exc
    if any(token_id < 0 for token_id in (*prompt_token_ids, *token_ids)):
        raise TokenAlignmentError("vLLM response contains a negative token ID")
    if not all(math.isfinite(logprob) for logprob in logprobs):
        raise TokenAlignmentError("vLLM response contains a non-finite sampled-token logprob")
    if not token_ids:
        raise TokenAlignmentError("vLLM returned an empty generated-token sequence")
    if len(token_ids) != len(logprobs):
        raise TokenAlignmentError(
            f"generated token/logprob length mismatch: {len(token_ids)} != {len(logprobs)}"
        )
    return GenerationTrace(
        prompt_token_ids=prompt_token_ids,
        token_ids=token_ids,
        logprobs=logprobs,
        finish_reason=str(finish_reason) if finish_reason is not None else None,
    )


def build_exact_trajectory_tokens(
    turns: list[GenerationTrace],
    max_prompt_length: int,
    max_response_length: int,
    max_total_length: int | None = None,
) -> tuple[list[int], list[int], list[int], list[float]]:
    """Build a multi-turn trajectory entirely from server-returned token IDs.

    Later prompts must exactly extend the previous server trace. Context
    truncation, tokenizer drift, token/logprob mismatch, and post-sampling
    cropping all reject the sample rather than approximate its PPO ratio.
    """
    if not turns:
        raise TokenAlignmentError("trajectory has no generated turns")
    if max_prompt_length <= 0 or max_response_length <= 0:
        raise ValueError("trajectory length limits must be positive")
    if max_total_length is not None and max_total_length <= 0:
        raise ValueError("total trajectory length limit must be positive")

    initial_prompt_len = len(turns[0].prompt_token_ids)
    full_ids = list(turns[0].prompt_token_ids)
    full_mask = [0] * len(full_ids)
    full_logprobs = [0.0] * len(full_ids)

    for turn_index, turn in enumerate(turns):
        if len(turn.token_ids) != len(turn.logprobs):
            raise TokenAlignmentError(
                f"turn {turn_index} token/logprob length mismatch: "
                f"{len(turn.token_ids)} != {len(turn.logprobs)}"
            )
        if any(token_id < 0 for token_id in (*turn.prompt_token_ids, *turn.token_ids)):
            raise TokenAlignmentError(
                f"turn {turn_index} contains a negative token ID"
            )
        if not all(math.isfinite(float(logprob)) for logprob in turn.logprobs):
            raise TokenAlignmentError(
                f"turn {turn_index} contains a non-finite sampled-token logprob"
            )
        prompt_ids = list(turn.prompt_token_ids)
        if len(prompt_ids) < len(full_ids) or prompt_ids[: len(full_ids)] != full_ids:
            common_len = min(len(prompt_ids), len(full_ids))
            first_mismatch = common_len
            for index in range(common_len):
                if prompt_ids[index] != full_ids[index]:
                    first_mismatch = index
                    break
            raise TokenAlignmentError(
                f"turn {turn_index} prompt is not an exact extension of the previous token trace "
                f"(previous_len={len(full_ids)}, prompt_len={len(prompt_ids)}, "
                f"first_mismatch={first_mismatch})"
            )
        context_delta = prompt_ids[len(full_ids) :]
        full_ids.extend(context_delta)
        full_mask.extend([0] * len(context_delta))
        full_logprobs.extend([0.0] * len(context_delta))
        full_ids.extend(turn.token_ids)
        full_mask.extend([1] * len(turn.token_ids))
        full_logprobs.extend(turn.logprobs)

    last_trainable = max(index for index, value in enumerate(full_mask) if value) + 1
    full_ids = full_ids[:last_trainable]
    full_mask = full_mask[:last_trainable]
    full_logprobs = full_logprobs[:last_trainable]

    response_length = len(full_ids) - initial_prompt_len
    if initial_prompt_len > max_prompt_length or response_length > max_response_length:
        raise TokenAlignmentError(
            "exact trajectory exceeds configured limits: "
            f"prompt={initial_prompt_len}/{max_prompt_length}, "
            f"response={response_length}/{max_response_length}"
        )
    if max_total_length is not None and len(full_ids) > max_total_length:
        # PPO requires the exact sampled token/logprob trajectory.  Never crop
        # an overlong sample after generation; fail closed before queueing it.
        raise TokenLengthOverflowError(len(full_ids), max_total_length)

    prompt_ids = full_ids[:initial_prompt_len]
    response_ids = full_ids[initial_prompt_len:]
    response_mask = full_mask[initial_prompt_len:]
    response_logprobs = full_logprobs[initial_prompt_len:]
    if not response_ids or not any(response_mask):
        raise TokenAlignmentError("trajectory contains no trainable generated policy token")
    return prompt_ids, response_ids, response_mask, response_logprobs


def build_vector_placeholders(
    injections: dict[Any, Any], rollout_nonce: str | None = None
) -> dict[Any, str]:
    """Assign a collision-resistant marker to every injection vector."""
    nonce = rollout_nonce or uuid4().hex
    return {
        vector_id: f"___ADV_EVO_PAYLOAD_{nonce}_{index}_{uuid4().hex}___"
        for index, vector_id in enumerate(injections)
    }


def find_next_vector_placeholder(
    result_text: str, placeholders: dict[Any, str]
) -> tuple[Any, str] | None:
    """Return the vector owning the earliest marker currently visible to the defender."""
    matches = [
        (result_text.index(marker), vector_id, marker)
        for vector_id, marker in placeholders.items()
        if marker in result_text
    ]
    if not matches:
        return None
    _, vector_id, marker = min(matches, key=lambda item: item[0])
    return vector_id, marker
