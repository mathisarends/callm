import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

from openai import AsyncOpenAI
from openai.types.responses import ResponseCompletedEvent
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse as PydanticResponse,
    TextPart,
)
from pydantic_ai.providers.openai_codex import OpenAICodexCredentials

from llmify.base import ModelTool
from llmify.messages import AssistantMessage, SystemMessage, UserMessage
from llmify.providers.codex import ChatCodex, TransportFallbackEvent
from llmify.providers.codex_transport import (
    CodexResponsesResource,
    WebSocketUnavailable,
)


def _model(transport: str = "websocket", **kwargs) -> ChatCodex:
    credentials = OpenAICodexCredentials(
        access_token="access", refresh_token="refresh", account_id="account"
    )
    return ChatCodex(
        "gpt-5.6-terra", credentials=credentials, transport=transport, **kwargs
    )


def _connected(model: ChatCodex) -> CodexResponsesResource:
    resource = model._responses_resource
    assert resource is not None
    resource._connection = SimpleNamespace(close=AsyncMock())
    resource._connection_generation = 1
    return resource


def test_prepare_reuses_matching_prefix_once() -> None:
    async def run() -> None:
        fallback_events: list[TransportFallbackEvent] = []

        async def on_fallback(event: TransportFallbackEvent) -> None:
            fallback_events.append(event)

        model = _model(on_transport_fallback=on_fallback)
        resource = _connected(model)
        calls = []

        async def request(history, settings, parameters):
            calls.append(
                (history, settings, parameters, resource._warmup_request.get())
            )
            if resource._warmup_request.get():
                return PydanticResponse(parts=[], provider_response_id="warmup-id")
            return PydanticResponse(parts=[TextPart(content="answer")])

        model._model.request = request
        system = SystemMessage(content="Answer briefly.")
        user = UserMessage(content="Question")
        tools = [ModelTool(name="lookup")]

        await model.prepare([system], tools=tools)
        assert (await model.call([system, user], tools=tools)).completion == "answer"
        assert (await model.call([system, user], tools=tools)).completion == "answer"

        warmup, prepared, ordinary = calls
        assert warmup[3] is True
        assert "openai_previous_response_id" not in warmup[1]
        assert len(warmup[0]) == 1
        assert isinstance(warmup[0][0], ModelRequest)
        assert warmup[0][0].instructions == "Answer briefly."
        assert warmup[0][0].parts == []
        assert [tool.name for tool in warmup[2].function_tools] == ["lookup"]

        assert prepared[3] is False
        assert prepared[1]["openai_previous_response_id"] == "warmup-id"
        assert len(prepared[0]) == 1
        assert isinstance(prepared[0][0], ModelRequest)
        assert prepared[0][0].instructions == "Answer briefly."
        assert prepared[0][0].parts[0].content == "Question"

        assert "openai_previous_response_id" not in ordinary[1]
        assert len(ordinary[0]) == 1
        assert fallback_events == []
        await model.aclose()

    asyncio.run(run())


def test_prepare_after_http_response_does_not_chain_to_http_response_id() -> None:
    async def run() -> None:
        model = _model()
        resource = _connected(model)
        calls = []

        async def request(history, settings, _parameters):
            calls.append((history, settings, resource._warmup_request.get()))
            return PydanticResponse(
                parts=[], provider_response_id="websocket-warmup-id"
            )

        model._model.request = request
        http_response = PydanticResponse(
            parts=[TextPart(content="HTTP answer")],
            provider_response_id="http-response-id",
            provider_name="openai-codex",
        )
        history = [
            UserMessage(content="First question"),
            AssistantMessage(content="HTTP answer", provider_state=http_response),
        ]

        await model.prepare(history)

        assert len(calls[0][0]) == 3
        assert calls[0][0][1] is http_response
        assert "openai_previous_response_id" not in calls[0][1]
        assert model._prepared_request is not None
        await model.aclose()

    asyncio.run(run())


def test_prepare_uses_full_history_when_context_or_connection_changes() -> None:
    async def run() -> None:
        model = _model()
        resource = _connected(model)
        calls = []

        async def request(history, settings, _parameters):
            if resource._warmup_request.get():
                return PydanticResponse(parts=[], provider_response_id="warmup-id")
            calls.append((history, settings))
            return PydanticResponse(parts=[TextPart(content="answer")])

        model._model.request = request
        system = SystemMessage(content="System")
        user = UserMessage(content="Question")
        prepared_tools = [ModelTool(name="one")]

        await model.prepare([system], tools=prepared_tools)
        await model.call([system, user], tools=[ModelTool(name="two")])
        assert "openai_previous_response_id" not in calls[-1][1]
        assert calls[-1][0][0].instructions == "System"

        await model.prepare([system], tools=prepared_tools)
        resource._connection_generation += 1
        await model.call([system, user], tools=prepared_tools)
        assert "openai_previous_response_id" not in calls[-1][1]
        await model.aclose()

    asyncio.run(run())


def test_stream_consumes_prepared_prefix() -> None:
    async def run() -> None:
        model = _model()
        _connected(model)
        calls = []

        async def request(_history, _settings, _parameters):
            return PydanticResponse(parts=[], provider_response_id="warmup-id")

        class EmptyStream:
            async def __aiter__(self):
                if False:
                    yield None

            def get(self):
                return PydanticResponse(parts=[TextPart(content="stream answer")])

        @asynccontextmanager
        async def request_stream(history, settings, _parameters):
            calls.append((history, settings))
            yield EmptyStream()

        model._model.request = request
        model._model.request_stream = request_stream
        system = SystemMessage(content="System")
        await model.prepare([system])

        events = [
            event
            async for event in model.stream([system, UserMessage(content="Question")])
        ]

        assert events[-1].completion == "stream answer"
        assert calls[0][1]["openai_previous_response_id"] == "warmup-id"
        assert calls[0][0][0].parts[0].content == "Question"
        await model.aclose()

    asyncio.run(run())


def test_http_fallback_does_not_reuse_websocket_preparation() -> None:
    async def run() -> None:
        fallback_events: list[TransportFallbackEvent] = []

        async def on_fallback(event: TransportFallbackEvent) -> None:
            fallback_events.append(event)

        model = _model(on_transport_fallback=on_fallback)
        resource = _connected(model)
        calls = []

        async def request(history, settings, _parameters):
            if resource._warmup_request.get():
                return PydanticResponse(parts=[], provider_response_id="warmup-id")
            calls.append((history, settings, resource.http_only))
            if not resource.http_only:
                raise WebSocketUnavailable("socket unavailable")
            return PydanticResponse(parts=[TextPart(content="HTTP answer")])

        model._model.request = request
        system = SystemMessage(content="System")
        user = UserMessage(content="Question")
        await model.prepare([system])

        assert (await model.call([system, user])).completion == "HTTP answer"
        websocket, http = calls
        assert websocket[1]["openai_previous_response_id"] == "warmup-id"
        assert http[2] is True
        assert "openai_previous_response_id" not in http[1]
        assert http[0][0].instructions == "System"
        assert http[0][0].parts[0].content == "Question"
        assert fallback_events == [
            TransportFallbackEvent(phase="call", reason="socket unavailable")
        ]
        await model.aclose()

    asyncio.run(run())


def test_failed_prepare_reports_reason_and_leaves_no_prepared_state() -> None:
    async def run() -> None:
        fallback_events: list[TransportFallbackEvent] = []

        async def on_fallback(event: TransportFallbackEvent) -> None:
            fallback_events.append(event)

        model = _model(on_transport_fallback=on_fallback)

        async def request(_history, _settings, _parameters):
            raise WebSocketUnavailable("connection refused")

        model._model.request = request
        await model.prepare([SystemMessage(content="System")])

        assert model._prepared_request is None
        assert fallback_events == [
            TransportFallbackEvent(phase="prepare", reason="connection refused")
        ]
        await model.aclose()

    asyncio.run(run())


def test_stream_fallback_reports_phase_and_reason() -> None:
    async def run() -> None:
        fallback_events: list[TransportFallbackEvent] = []

        async def on_fallback(event: TransportFallbackEvent) -> None:
            fallback_events.append(event)

        model = _model(on_transport_fallback=on_fallback)
        resource = _connected(model)

        class EmptyStream:
            async def __aiter__(self):
                if False:
                    yield None

            def get(self):
                return PydanticResponse(parts=[TextPart(content="HTTP answer")])

        @asynccontextmanager
        async def request_stream(_history, _settings, _parameters):
            if not resource.http_only:
                raise WebSocketUnavailable("handshake failed")
            yield EmptyStream()

        model._model.request_stream = request_stream
        events = [
            event async for event in model.stream([UserMessage(content="Question")])
        ]

        assert events[-1].completion == "HTTP answer"
        assert fallback_events == [
            TransportFallbackEvent(phase="stream", reason="handshake failed")
        ]
        await model.aclose()

    asyncio.run(run())


def test_prepare_is_noop_for_http_transport() -> None:
    async def run() -> None:
        model = _model("http")
        model._model.request = AsyncMock(
            side_effect=AssertionError("unexpected request")
        )
        await model.prepare([SystemMessage(content="System")])
        model._model.request.assert_not_awaited()
        await model.aclose()

    asyncio.run(run())


def test_warmup_request_disables_generation_without_stream_id() -> None:
    async def run() -> None:
        client = AsyncOpenAI(api_key="test")
        resource = CodexResponsesResource(client, object())  # type: ignore[arg-type]
        response = SimpleNamespace(id="warmup-id")
        event = ResponseCompletedEvent.model_construct(
            type="response.completed", sequence_number=1, response=response
        )
        connection = SimpleNamespace(
            send=AsyncMock(), recv=AsyncMock(return_value=event)
        )
        resource._ensure_connection = AsyncMock(return_value=connection)

        async with resource.warmup_request():
            result = await resource.create(
                model="gpt-5.6-terra", input="", stream=False
            )

        assert result is response
        connection.send.assert_awaited_once_with(
            {
                "type": "response.create",
                "model": "gpt-5.6-terra",
                "input": "",
                "generate": False,
            }
        )
        await client.close()

    asyncio.run(run())


def test_regular_websocket_request_omits_unsupported_stream_id() -> None:
    async def run() -> None:
        client = AsyncOpenAI(api_key="test")
        resource = CodexResponsesResource(client, object())  # type: ignore[arg-type]
        response = SimpleNamespace(id="response-id")
        event = ResponseCompletedEvent.model_construct(
            type="response.completed", sequence_number=1, response=response
        )
        connection = SimpleNamespace(
            response=SimpleNamespace(create=AsyncMock()),
            recv=AsyncMock(return_value=event),
        )
        resource._ensure_connection = AsyncMock(return_value=connection)

        assert (
            await resource.create(model="gpt-5.6-terra", input="Question") is response
        )
        connection.response.create.assert_awaited_once_with(
            model="gpt-5.6-terra", input="Question"
        )
        await client.close()

    asyncio.run(run())
