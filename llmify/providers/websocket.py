import asyncio
import importlib.util
from collections.abc import AsyncIterator, Sequence
from typing import TYPE_CHECKING, Any

from openai import omit
from pydantic import BaseModel
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings

from llmify._adapters import from_pydantic_response, normalize_messages, request_parameters, to_pydantic_messages
from llmify.base import _text_delta
from llmify.messages import Message
from llmify.views import ChatCompletion, StreamEnd, StreamEvent, StreamTextDelta, StreamToolCall

if TYPE_CHECKING:
    from openai.resources.responses.responses import AsyncResponsesConnection

    from llmify.providers.openai import ChatOpenAI


class WebSocketChatOpenAI:
    """Persistent Responses WebSocket transport for a ``ChatOpenAI`` instance.

    Pydantic AI currently has no public WebSocket hook for ``OpenAIResponsesModel``.
    This adapter deliberately pins Pydantic AI to one minor release and reuses its
    request builder and response mapper so HTTP and WebSocket have identical message,
    tool, usage, and structured-output behavior.
    """

    def __init__(self, owner: "ChatOpenAI") -> None:
        self._owner = owner
        self._connection: AsyncResponsesConnection | None = None
        self._lock = asyncio.Lock()

    async def call[OutputT: BaseModel](
        self,
        messages: str | Sequence[Message],
        output_format: type[OutputT] | None = None,
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
        **kwargs: Any,
    ) -> ChatCompletion[OutputT] | ChatCompletion[str]:
        final = None
        async for item in self._run(messages, output_format, tools, tool_choice, kwargs):
            if not isinstance(item, str):
                final = item
        if final is None:
            raise RuntimeError("The Responses WebSocket ended without a final response.")
        return from_pydantic_response(final, output_format)

    async def stream(
        self,
        messages: str | Sequence[Message],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        if kwargs.pop("output_format", None) is not None:
            raise TypeError("Structured output is supported by call(), not stream().")

        final = None
        async for item in self._run(messages, None, tools, tool_choice, kwargs):
            if isinstance(item, str):
                if item:
                    yield StreamTextDelta(delta=item)
            else:
                final = item

        if final is None:
            raise RuntimeError("The Responses WebSocket ended without a final response.")
        completion = from_pydantic_response(final)
        for tool_call in completion.tool_calls:
            yield StreamToolCall(tool_call=tool_call)
        yield StreamEnd(**completion.model_dump())

    async def _run(
        self,
        messages: str | Sequence[Message],
        output_format: type[BaseModel] | None,
        tools: Sequence[dict[str, Any]] | None,
        tool_choice: str,
        overrides: dict[str, Any],
    ) -> AsyncIterator[str | ModelResponse]:
        backend = self._owner._backend
        assert isinstance(backend, OpenAIResponsesModel)
        converted = to_pydantic_messages(normalize_messages(messages))
        params = request_parameters(tools, output_format)
        settings = self._owner._settings(tool_choice=tool_choice, overrides=overrides)
        settings, params = backend.prepare_request(settings, params)
        openai_settings = OpenAIResponsesModelSettings(**(settings or {}))

        async with self._lock:
            connection = await self._connect()
            finished = False
            try:
                request = await backend._build_responses_request_params(  # noqa: SLF001
                    converted,
                    openai_settings,
                    params,
                    backend.profile,
                )
                include = backend._build_include(openai_settings)  # noqa: SLF001
                await connection.response.create(
                    model=request.model,
                    input=request.input,
                    instructions=request.instructions,
                    parallel_tool_calls=request.parallel_tool_calls,
                    tools=request.tools,
                    tool_choice=request.tool_choice,
                    previous_response_id=request.previous_response_id,
                    conversation=request.conversation,
                    reasoning=request.reasoning,
                    text=request.text,
                    truncation=request.truncation,
                    context_management=request.context_management,
                    max_output_tokens=openai_settings.get("max_tokens", omit),
                    temperature=openai_settings.get("temperature", omit),
                    top_p=openai_settings.get("top_p", omit),
                    store=openai_settings.get("openai_store", omit),
                    include=include or omit,
                    prompt_cache_key=openai_settings.get("openai_prompt_cache_key", omit),
                    prompt_cache_retention=openai_settings.get("openai_prompt_cache_retention", omit),
                    user=openai_settings.get("openai_user", omit),
                )

                streamed = await backend._process_streamed_response(  # noqa: SLF001
                    _response_events(connection), openai_settings, params
                )
                async for event in streamed:
                    if delta := _text_delta(event):
                        yield delta
                final = streamed.get()
                yield final
                finished = True
            finally:
                if not finished:
                    await self._discard_connection()

    async def _connect(self) -> "AsyncResponsesConnection":
        if self._connection is not None:
            return self._connection
        if importlib.util.find_spec("websockets") is None:
            raise ImportError(
                "WebSocket transport requires the optional dependency: pip install 'py-llmify[websocket]'"
            )
        self._connection = await self._owner._client.responses.connect().enter()
        return self._connection

    async def _discard_connection(self) -> None:
        connection, self._connection = self._connection, None
        if connection is not None:
            await connection.close()

    async def aclose(self) -> None:
        async with self._lock:
            await self._discard_connection()


async def _response_events(connection: "AsyncResponsesConnection") -> AsyncIterator[Any]:
    while True:
        event = await connection.recv()
        if event.type == "error":
            raise RuntimeError(f"Responses WebSocket error: {event}")
        if event.type == "response.failed":
            message = event.response.error.message if event.response.error else "Response failed."
            raise RuntimeError(message)
        yield event
        if event.type in {"response.completed", "response.incomplete"}:
            return


__all__ = ["WebSocketChatOpenAI"]
