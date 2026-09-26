from typing import Any

from pydantic_ai.models.cerebras import CerebrasModel
from pydantic_ai.providers.cerebras import CerebrasProvider

from llmify._adapter import PydanticAIModel
from llmify.providers.openai import openai_settings


class ChatCerebras(PydanticAIModel):
    """Cerebras' inference API. `api_key` falls back to `CEREBRAS_API_KEY`."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        default_headers: dict[str, str] | None = None,
        **settings: Any,
    ) -> None:
        super().__init__(
            CerebrasModel(model, provider=CerebrasProvider(api_key=api_key)),
            **openai_settings(settings, default_headers=default_headers),
        )
