"""Google's Gemini API."""

from typing import Any

from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

from llmify._adapter import PydanticAIModel


class ChatGoogle(PydanticAIModel):
    """Google's Gemini API.

    `api_key` falls back to `GOOGLE_API_KEY`, then to the older `GEMINI_API_KEY`.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        **settings: Any,
    ) -> None:
        super().__init__(
            GoogleModel(
                model, provider=GoogleProvider(api_key=api_key, base_url=base_url)
            ),
            **settings,
        )
