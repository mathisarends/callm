from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from types import TracebackType
from typing import Any, Literal, overload

from pydantic import BaseModel
from pydantic_ai.messages import PartDeltaEvent, PartStartEvent, TextPart, TextPartDelta
from pydantic_ai.models import Model

from ._adapters import from_pydantic_response, normalize_messages, request_parameters, to_pydantic_messages
from .messages import Message
from .views import ChatCompletion, StreamEnd, StreamEvent, StreamTextDelta, StreamToolCall

ToolChoice = Literal["auto", "required", "none"]


class ChatModel(ABC):
    @property
    @abstractmethod
    def model(self) -> str: ...

    @overload
    async def call[OutputT: BaseModel](
        self,
        messages: str | Sequence[Message],
        output_format: type[OutputT],
        **kwargs: Any,
    ) -> ChatCompletion[OutputT]: ...

    @overload
    async def call(
        self,
        messages: str | Sequence[Message],
        output_format: None = None,
        **kwargs: Any,
    ) -> ChatCompletion[str]: ...

    @abstractmethod
    async def call[OutputT: BaseModel](
        self,
        messages: str | Sequence[Message],
        output_format: type[OutputT] | None = None,
        **kwargs: Any,
    ) -> ChatCompletion[OutputT] | ChatCompletion[str]: ...

    @abstractmethod
    def stream(
        self,
        messages: str | Sequence[Message],
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]: ...

    async def aclose(self) -> None:
        return None

    async def __aenter__(self):
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()


class PydanticAIChatModel(ChatModel):
    def __init__(
        self,
        model: str,
        backend: Model[Any],
        *,
        default_settings: dict[str, Any] | None = None,
    ) -> None:
        self._model_name = model
        self._backend = backend
        self._default_settings = default_settings or {}

    @property
    def model(self) -> str:
        return self._model_name

    async def call[OutputT: BaseModel](
        self,
        messages: str | Sequence[Message],
        output_format: type[OutputT] | None = None,
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        tool_choice: ToolChoice = "auto",
        **kwargs: Any,
    ) -> ChatCompletion[OutputT] | ChatCompletion[str]:
        converted = to_pydantic_messages(normalize_messages(messages))
        params = request_parameters(tools, output_format)
        settings = self._settings(tool_choice=tool_choice, overrides=kwargs)
        response = await self._backend.request(converted, settings, params)
        return from_pydantic_response(response, output_format)

    async def stream(
        self,
        messages: str | Sequence[Message],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        tool_choice: ToolChoice = "auto",
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        if kwargs.pop("output_format", None) is not None:
            raise TypeError("Structured output is supported by call(), not stream().")

        converted = to_pydantic_messages(normalize_messages(messages))
        params = request_parameters(tools, None)
        settings = self._settings(tool_choice=tool_choice, overrides=kwargs)

        async with self._backend.request_stream(converted, settings, params) as response:
            async for event in response:
                delta = _text_delta(event)
                if delta:
                    yield StreamTextDelta(delta=delta)
            final = response.get()

        completion = from_pydantic_response(final)
        for tool_call in completion.tool_calls:
            yield StreamToolCall(tool_call=tool_call)
        yield StreamEnd(**completion.model_dump())

    def _settings(self, *, tool_choice: ToolChoice, overrides: dict[str, Any]) -> dict[str, Any]:
        settings = {**self._default_settings, **_normalize_settings(overrides)}
        settings["tool_choice"] = tool_choice
        return {key: value for key, value in settings.items() if value is not None}


def _text_delta(event: object) -> str | None:
    if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
        return event.part.content or None
    if isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
        return event.delta.content_delta or None
    return None


def _normalize_settings(settings: dict[str, Any]) -> dict[str, Any]:
    aliases = {
        "reasoning_effort": "openai_reasoning_effort",
        "store": "openai_store",
        "verbosity": "openai_text_verbosity",
    }
    normalized: dict[str, Any] = {}
    for key, value in settings.items():
        if key == "stream":
            raise TypeError("Do not pass 'stream'; use stream() instead.")
        normalized[aliases.get(key, key)] = value
    return normalized


__all__ = ["ChatModel", "PydanticAIChatModel", "ToolChoice"]
