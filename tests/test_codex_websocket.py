import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from openai import AsyncOpenAI
from openai.types.responses import ResponseCompletedEvent
from pydantic_ai.messages import ModelResponse as PydanticResponse
from pydantic_ai.messages import TextPart
from pydantic_ai.providers.openai_codex import OpenAICodexCredentials

from llmify.base import UserMessage
from llmify.providers.codex import ChatCodex
from llmify.providers.codex_transport import (
    CodexResponsesResource,
    WebSocketInterrupted,
    WebSocketUnavailable,
)


def _model(transport: str = "websocket") -> ChatCodex:
    credentials = OpenAICodexCredentials(
        access_token="access", refresh_token="refresh", account_id="account"
    )
    return ChatCodex("gpt-5.6-terra", credentials=credentials, transport=transport)


def test_codex_transport_selection_and_http_fallback() -> None:
    async def run() -> None:
        model = _model()
        resource = model._responses_resource
        assert resource is not None
        http_calls = []

        async def request(*_args: object) -> PydanticResponse:
            http_calls.append(resource._http_only.get())
            if not resource._http_only.get():
                raise WebSocketUnavailable("socket unavailable")
            return PydanticResponse(parts=[TextPart(content="HTTP answer")])

        model._model.request = request
        response = await model.call([UserMessage(content="question")])
        assert response.completion == "HTTP answer"
        assert http_calls == [False, True]
        await model.aclose()

        http_model = _model("http")
        assert http_model._responses_resource is None
        await http_model.aclose()

    asyncio.run(run())


def test_stream_falls_back_only_before_response_starts() -> None:
    async def run() -> None:
        model = _model()
        resource = model._responses_resource
        assert resource is not None
        calls = []

        class EmptyStream:
            async def __aiter__(self):
                if False:
                    yield None

            def get(self) -> PydanticResponse:
                return PydanticResponse(parts=[TextPart(content="HTTP answer")])

        @asynccontextmanager
        async def request_stream(*_args: object):
            calls.append(resource._http_only.get())
            if not resource._http_only.get():
                raise WebSocketUnavailable("socket unavailable")
            yield EmptyStream()

        model._model.request_stream = request_stream
        events = [
            event async for event in model.stream([UserMessage(content="question")])
        ]
        assert events[-1].completion == "HTTP answer"
        assert calls == [False, True]
        await model.aclose()

    asyncio.run(run())


def test_started_websocket_response_is_never_replayed() -> None:
    async def run() -> None:
        client = AsyncOpenAI(api_key="test")
        resource = CodexResponsesResource(client, object())  # type: ignore[arg-type]
        connection = SimpleNamespace(
            response=SimpleNamespace(create=AsyncMock()),
            recv=AsyncMock(
                side_effect=[
                    SimpleNamespace(type="response.created"),
                    ConnectionError("lost"),
                ]
            ),
        )
        resource._ensure_connection = AsyncMock(return_value=connection)
        stream = await resource.create(
            model="gpt-5.6-terra", input="question", stream=True
        )
        with pytest.raises(WebSocketInterrupted):
            async with stream:
                await anext(stream)
                await anext(stream)
        await client.close()

    asyncio.run(run())


def test_send_failure_allows_http_fallback() -> None:
    async def run() -> None:
        client = AsyncOpenAI(api_key="test")
        resource = CodexResponsesResource(client, object())  # type: ignore[arg-type]
        connection = SimpleNamespace(
            response=SimpleNamespace(
                create=AsyncMock(side_effect=ConnectionError("send failed"))
            ),
        )
        resource._ensure_connection = AsyncMock(return_value=connection)
        stream = await resource.create(
            model="gpt-5.6-terra", input="question", stream=True
        )
        with pytest.raises(WebSocketUnavailable):
            async with stream:
                pass
        await client.close()

    asyncio.run(run())


def test_completed_websocket_response_is_returned() -> None:
    async def run() -> None:
        client = AsyncOpenAI(api_key="test")
        resource = CodexResponsesResource(client, object())  # type: ignore[arg-type]
        response = SimpleNamespace(id="response-1")
        event = ResponseCompletedEvent.model_construct(
            type="response.completed", sequence_number=1, response=response
        )
        connection = SimpleNamespace(
            response=SimpleNamespace(create=AsyncMock()),
            recv=AsyncMock(return_value=event),
        )
        resource._ensure_connection = AsyncMock(return_value=connection)
        result = await resource.create(
            model="gpt-5.6-terra", input="question", stream=False
        )
        assert result is response
        await client.close()

    asyncio.run(run())
