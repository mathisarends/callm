"""Anthropic's Messages API."""

from typing import Any

from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

from llmify._adapter import PydanticAIModel


class ChatAnthropic(PydanticAIModel):
    """Anthropic's Messages API. `api_key` falls back to `ANTHROPIC_API_KEY`."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        **settings: Any,
    ) -> None:
        if default_headers is not None:
            settings.setdefault("extra_headers", default_headers)
        super().__init__(
            AnthropicModel(
                model, provider=AnthropicProvider(api_key=api_key, base_url=base_url)
            ),
            **settings,
        )
