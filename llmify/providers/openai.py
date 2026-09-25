from collections.abc import Awaitable, Callable
from typing import Any, Literal

from openai import AsyncOpenAI
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider

from llmify.base import PydanticAIChatModel
from llmify.providers.websocket import WebSocketChatOpenAI

ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]
Transport = Literal["http", "websocket"]


class ChatOpenAI(PydanticAIChatModel):
    """OpenAI chat model implemented exclusively through the Responses API."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | Callable[[], Awaitable[str]] | None = None,
        base_url: str | None = None,
        transport: Transport = "http",
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        verbosity: Literal["low", "medium", "high"] | None = None,
        store: bool = False,
        timeout: float | None = 60.0,
        max_retries: int = 2,
        default_headers: dict[str, str] | None = None,
        **model_settings: Any,
    ) -> None:
        if transport not in {"http", "websocket"}:
            raise ValueError("transport must be 'http' or 'websocket'.")

        client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
        )
        provider = OpenAIProvider(openai_client=client)
        backend = OpenAIResponsesModel(model, provider=provider)
        defaults = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "openai_reasoning_effort": reasoning_effort,
            "openai_text_verbosity": verbosity,
            "openai_store": store,
            **model_settings,
        }
        super().__init__(model, backend, default_settings=defaults)
        self._client = client
        self._websocket = WebSocketChatOpenAI(self) if transport == "websocket" else None

    async def call(self, *args: Any, **kwargs: Any):
        if self._websocket is not None:
            return await self._websocket.call(*args, **kwargs)
        return await super().call(*args, **kwargs)

    def stream(self, *args: Any, **kwargs: Any):
        if self._websocket is not None:
            return self._websocket.stream(*args, **kwargs)
        return super().stream(*args, **kwargs)

    async def aclose(self) -> None:
        if self._websocket is not None:
            await self._websocket.aclose()
        await self._client.close()


__all__ = ["ChatOpenAI", "ReasoningEffort", "Transport"]
