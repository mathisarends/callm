from collections.abc import Sequence
from enum import StrEnum
from typing import Any

from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ServiceTier, ThinkingLevel

from llmify.pydantic_ai_adapter import PydanticAIModel, credentials_required
from llmify.retries import RetryCallback


class ReasoningEffort(StrEnum):
    """How much a reasoning model thinks before it answers.

    Which levels a model accepts differs, and an unsupported level comes back as
    a request error rather than being silently downgraded.
    """

    NONE = "none"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"


class ChatOpenAI(PydanticAIModel):
    """OpenAI's Chat Completions API.

    `api_key` falls back to `OPENAI_API_KEY`. Some reasoning models accept
    function tools here only with `reasoning_effort="none"`; prefer
    `ChatOpenAIResponses` for them.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        reasoning_effort: ReasoningEffort | str | None = None,
        # Output
        max_tokens: int | None = None,
        stop_sequences: Sequence[str] | None = None,
        # Sampling
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        seed: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        logit_bias: dict[str, int] | None = None,
        # Model behavior
        thinking: ThinkingLevel | None = None,
        parallel_tool_calls: bool | None = None,
        service_tier: ServiceTier | None = None,
        # Request and retries
        timeout: float | None = 60.0,
        extra_headers: dict[str, str] | None = None,
        extra_body: object | None = None,
        max_retries: int = 2,
        on_retry: RetryCallback | None = None,
        **settings: Any,
    ) -> None:
        with credentials_required():
            provider = OpenAIProvider(api_key=api_key, base_url=base_url)
        super().__init__(
            OpenAIChatModel(model, provider=provider),
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            logit_bias=logit_bias,
            thinking=thinking,
            parallel_tool_calls=parallel_tool_calls,
            service_tier=service_tier,
            timeout=timeout,
            extra_headers=extra_headers,
            extra_body=extra_body,
            max_retries=max_retries,
            on_retry=on_retry,
            **openai_settings(settings, reasoning_effort=reasoning_effort),
        )


class ChatOpenAIResponses(PydanticAIModel):
    """OpenAI's Responses API.

    Use this over `ChatOpenAI` for reasoning models: the Responses API carries
    reasoning state between turns, which a tool loop replays through
    `AssistantMessage.provider_state`.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        reasoning_effort: ReasoningEffort | str | None = None,
        reasoning_summary: str | None = None,
        # Output
        max_tokens: int | None = None,
        stop_sequences: Sequence[str] | None = None,
        # Sampling
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        seed: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        logit_bias: dict[str, int] | None = None,
        # Model behavior
        thinking: ThinkingLevel | None = None,
        parallel_tool_calls: bool | None = None,
        service_tier: ServiceTier | None = None,
        # Request and retries
        timeout: float | None = 60.0,
        extra_headers: dict[str, str] | None = None,
        extra_body: object | None = None,
        max_retries: int = 2,
        on_retry: RetryCallback | None = None,
        **settings: Any,
    ) -> None:
        with credentials_required():
            provider = OpenAIProvider(api_key=api_key, base_url=base_url)
        super().__init__(
            OpenAIResponsesModel(model, provider=provider),
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            logit_bias=logit_bias,
            thinking=thinking,
            parallel_tool_calls=parallel_tool_calls,
            service_tier=service_tier,
            timeout=timeout,
            extra_headers=extra_headers,
            extra_body=extra_body,
            max_retries=max_retries,
            on_retry=on_retry,
            **openai_settings(
                settings,
                reasoning_effort=reasoning_effort,
                reasoning_summary=reasoning_summary,
            ),
        )


def openai_settings(
    settings: dict[str, Any],
    *,
    reasoning_effort: ReasoningEffort | str | None = None,
    reasoning_summary: str | None = None,
) -> dict[str, Any]:
    """Fold the named OpenAI options into the pass-through model settings.

    Shared by every provider that reaches OpenAI's wire format, Azure and Codex
    included.
    """
    if reasoning_effort is not None:
        settings.setdefault(
            "openai_reasoning_effort", ReasoningEffort(reasoning_effort).value
        )
    if reasoning_summary is not None:
        settings.setdefault("openai_reasoning_summary", reasoning_summary)
    return settings
