import asyncio
import sys
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from openai import AsyncOpenAI
from openai.types.responses import ResponseCompletedEvent
from pydantic_ai.messages import ModelResponse as PydanticResponse
from pydantic_ai.messages import TextPart

from llmify import (
    ChatCodex,
    ModelResponse,
    OpenAICodexCredentials,
    ResponseInterruptedError,
    Transport,
    UserMessage,
)
from llmify.providers.codex.websocket import (
    CodexResponsesResource,
    WebSocketInterrupted,
    WebSocketUnavailable,
)


def _model(transport: Transport = "websocket") -> ChatCodex:
    credentials = OpenAICodexCredentials(
        access_token="access", refresh_token="refresh", account_id="account"
    )
    return ChatCodex("gpt-5.6-terra", credentials=credentials, transport=transport)


def test_codex_defaults_to_http() -> None:
    credentials = OpenAICodexCredentials(
        access_token="access", refresh_token="refresh", account_id="account"
    )
    model = ChatCodex("gpt-5.6-terra", credentials=credentials)

    assert model._responses_resource is None
    asyncio.run(model.aclose())


def test_websocket_transport_without_websockets_installed_names_the_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "websockets", None)

    with pytest.raises(ImportError, match=r"py-llmify\[websocket\]"):
        _model("websocket")


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

        cast(Any, model._model).request = request
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

        cast(Any, model._model).request_stream = request_stream
        events = [
            event async for event in model.stream([UserMessage(content="question")])
        ]
        assert isinstance(events[-1], ModelResponse)
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
        resource.ensure_connection = AsyncMock(return_value=connection)
        stream = await resource.create(
            model="gpt-5.6-terra", input="question", stream=True
        )
        with pytest.raises(WebSocketInterrupted):
            async with stream:
                await anext(stream)
                await anext(stream)
        await client.close()

    asyncio.run(run())


def test_interrupted_call_is_classified_without_replay() -> None:
    async def run() -> None:
        model = _model()
        attempts = 0

        async def request(*_args: object) -> PydanticResponse:
            nonlocal attempts
            attempts += 1
            raise WebSocketInterrupted("connection lost after response started")

        cast(Any, model._model).request = request

        with pytest.raises(ResponseInterruptedError) as caught:
            await model.call([UserMessage(content="question")])

        assert attempts == 1
        assert isinstance(caught.value.__cause__, WebSocketInterrupted)
        assert caught.value.code == "model_response_interrupted"
        assert not caught.value.retryable
        await model.aclose()

    asyncio.run(run())


def test_interrupted_stream_is_classified_without_replay() -> None:
    async def run() -> None:
        model = _model()
        attempts = 0

        @asynccontextmanager
        async def request_stream(*_args: object):
            nonlocal attempts
            attempts += 1
            raise WebSocketInterrupted("connection lost after response started")
            yield  # pragma: no cover

        cast(Any, model._model).request_stream = request_stream

        with pytest.raises(ResponseInterruptedError) as caught:
            async for _event in model.stream([UserMessage(content="question")]):
                pass

        assert attempts == 1
        assert isinstance(caught.value.__cause__, WebSocketInterrupted)
        await model.aclose()

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
        resource.ensure_connection = AsyncMock(return_value=connection)
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
        resource.ensure_connection = AsyncMock(return_value=connection)
        result = await resource.create(
            model="gpt-5.6-terra", input="question", stream=False
        )
        assert result is response
        await client.close()

    asyncio.run(run())
