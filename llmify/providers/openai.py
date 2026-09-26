from enum import StrEnum
from typing import Any

from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider

from llmify._adapter import PydanticAIModel


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

    `api_key` falls back to `OPENAI_API_KEY`.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        **settings: Any,
    ) -> None:
        super().__init__(
            OpenAIChatModel(
                model, provider=OpenAIProvider(api_key=api_key, base_url=base_url)
            ),
            **openai_settings(settings, default_headers=default_headers),
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
        default_headers: dict[str, str] | None = None,
        **settings: Any,
    ) -> None:
        super().__init__(
            OpenAIResponsesModel(
                model, provider=OpenAIProvider(api_key=api_key, base_url=base_url)
            ),
            **openai_settings(
                settings,
                default_headers=default_headers,
                reasoning_effort=reasoning_effort,
                reasoning_summary=reasoning_summary,
            ),
        )


class OpenAICompatible(PydanticAIModel):
    """Any endpoint that implements OpenAI's Chat Completions API."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str,
        api_key: str | None = None,
        default_headers: dict[str, str] | None = None,
        **settings: Any,
    ) -> None:
        super().__init__(
            OpenAIChatModel(
                model, provider=OpenAIProvider(api_key=api_key, base_url=base_url)
            ),
            **openai_settings(settings, default_headers=default_headers),
        )


def openai_settings(
    settings: dict[str, Any],
    *,
    default_headers: dict[str, str] | None = None,
    reasoning_effort: ReasoningEffort | str | None = None,
    reasoning_summary: str | None = None,
) -> dict[str, Any]:
    """Fold the named OpenAI options into the pass-through model settings.

    Shared by every provider that reaches OpenAI's wire format, Azure and Codex
    included.
    """
    if default_headers is not None:
        settings.setdefault("extra_headers", default_headers)
    if reasoning_effort is not None:
        settings.setdefault(
            "openai_reasoning_effort", ReasoningEffort(reasoning_effort).value
        )
    if reasoning_summary is not None:
        settings.setdefault("openai_reasoning_summary", reasoning_summary)
    return settings
