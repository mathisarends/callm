from typing import Any, Literal, Self

from pydantic_ai.models.openai_codex import OpenAICodexModel
from pydantic_ai.providers.openai_codex import (
    OpenAICodexCredentials,
    OpenAICodexCredentialSource,
    OpenAICodexProvider,
)

from llmify.base import PydanticAIChatModel

ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]


class ChatCodex(PydanticAIChatModel):
    """Codex subscription model backed by Pydantic AI's Codex Responses provider."""

    def __init__(
        self,
        model: str,
        *,
        credentials: OpenAICodexCredentials | None = None,
        credential_source: OpenAICodexCredentialSource | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        timeout: float | None = 60.0,
        **model_settings: Any,
    ) -> None:
        provider = OpenAICodexProvider(credentials, credential_source=credential_source)
        backend = OpenAICodexModel(model, provider=provider)
        defaults = {
            "openai_reasoning_effort": reasoning_effort,
            "openai_store": False,
            "timeout": timeout,
            **model_settings,
        }
        super().__init__(model, backend, default_settings=defaults)
        self._provider = provider

    @classmethod
    def from_cli(
        cls,
        model: str,
        *,
        reasoning_effort: ReasoningEffort | None = None,
        timeout: float | None = 60.0,
        **model_settings: Any,
    ) -> Self:
        return cls(
            model,
            reasoning_effort=reasoning_effort,
            timeout=timeout,
            **model_settings,
        )

    async def aclose(self) -> None:
        await self._provider.__aexit__(None, None, None)


__all__ = ["ChatCodex"]
