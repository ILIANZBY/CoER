from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def expand_env(value: Any) -> Any:
    """Expand ``${NAME}`` in configuration values and fail on missing names."""

    if isinstance(value, dict):
        return {key: expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_env(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in os.environ:
            raise ValueError(f"configuration references unset environment variable {name}")
        return os.environ[name]

    return _ENV_PATTERN.sub(replace, value)


@dataclass(frozen=True)
class EndpointConfig:
    """One OpenAI-compatible model endpoint.

    ``protocol=text`` reproduces the training XML tool interface. Use
    ``protocol=native`` for endpoints that expose OpenAI-native tool calls. Secrets are
    named by ``api_key_env`` and are never serialized into the evaluation
    manifest.
    """

    label: str
    base_url: str
    model: str
    api_key_env: str | None = None
    protocol: Literal["text", "native"] = "text"
    temperature: float = 0.6
    top_p: float = 1.0
    max_tokens: int = 8192
    timeout_seconds: float = 600.0
    max_retries: int = 3
    send_seed: bool = True
    needs_session_id: bool = False
    training_chat_template: bool = False
    extra_body: dict[str, Any] = field(default_factory=dict)
    checkpoint: str | None = None

    @classmethod
    def from_dict(cls, label: str, raw: dict[str, Any]) -> EndpointConfig:
        values = expand_env(dict(raw))
        values["label"] = label
        endpoint = cls(**values)
        endpoint.validate()
        return endpoint

    def validate(self) -> None:
        if not self.label or not self.base_url or not self.model:
            raise ValueError("endpoint label, base_url and model must be non-empty")
        if self.protocol not in ("text", "native"):
            raise ValueError(f"unsupported protocol for {self.label}: {self.protocol}")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"temperature for {self.label} must be in [0, 2]")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p for {self.label} must be in (0, 1]")
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens for {self.label} must be positive")
        if self.needs_session_id and "session_id" in self.extra_body:
            raise ValueError(
                f"endpoint {self.label} must not configure a static session_id; "
                "the evaluator assigns one per episode"
            )
        if self.api_key_env and self.api_key_env not in os.environ:
            raise ValueError(f"endpoint {self.label} requires unset API key environment variable {self.api_key_env}")

    def api_key(self) -> str:
        return os.environ[self.api_key_env] if self.api_key_env else "dummy"

    def public_dict(self) -> dict[str, Any]:
        public = asdict(self)
        public.pop("api_key_env", None)
        public["api_key_env_name"] = self.api_key_env
        return public


@dataclass(frozen=True)
class EvalSettings:
    max_defender_turns: int = 20
    max_attacker_turns: int = 0
    max_tool_response_chars: int = 0
    concurrency: int = 8
    benchmark_version: str = "v1"
    task_catalog: Literal["v2_training", "canonical_v1"] = "v2_training"
    generation_seeds: tuple[int, ...] = (0,)
    trace_policy: Literal["all", "failures", "none"] = "failures"

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> EvalSettings:
        values = dict(raw or {})
        if "generation_seeds" in values:
            values["generation_seeds"] = tuple(int(item) for item in values["generation_seeds"])
        settings = cls(**values)
        if settings.max_defender_turns <= 0:
            raise ValueError("max_defender_turns must be positive")
        if settings.max_attacker_turns < 0:
            raise ValueError("max_attacker_turns must be non-negative")
        if settings.max_tool_response_chars < 0:
            raise ValueError("max_tool_response_chars must be non-negative")
        if settings.concurrency <= 0:
            raise ValueError("concurrency must be positive")
        if settings.task_catalog not in ("v2_training", "canonical_v1"):
            raise ValueError("task_catalog must be v2_training or canonical_v1")
        if settings.task_catalog == "canonical_v1" and settings.benchmark_version != "v1":
            raise ValueError("canonical_v1 task_catalog requires benchmark_version=v1")
        if not settings.generation_seeds:
            raise ValueError("generation_seeds must not be empty")
        if settings.trace_policy not in ("all", "failures", "none"):
            raise ValueError("trace_policy must be all, failures or none")
        return settings


@dataclass(frozen=True)
class AttackSpec:
    kind: Literal["clean", "dataset", "fixed", "adaptive"]
    label: str
    template_name: str | None = None
    attacker_label: str | None = None

    def validate(self) -> None:
        if self.kind == "fixed" and not self.template_name:
            raise ValueError("fixed attacks require template_name")
        if self.kind == "adaptive" and not self.attacker_label:
            raise ValueError("adaptive attacks require attacker_label")
        if self.kind != "adaptive" and self.attacker_label is not None:
            raise ValueError(f"{self.kind} attacks cannot specify attacker_label")
