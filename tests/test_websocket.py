import asyncio
from collections import deque
from types import SimpleNamespace
from unittest.mock import AsyncMock

from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseFunctionToolCall,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseTextDeltaEvent,
)

from llmify import ChatOpenAI, StreamEnd, StreamTextDelta, StreamToolCall


def _response(status: str) -> Response:
    return Response.model_construct(
        id="response-1",
        model="gpt-test",
        status=status,
        created_at=1,
        background=False,
        service_tier=None,
        conversation=None,
        moderation=None,
        incomplete_details=None,
        error=None,
        usage=None,
        output=[],
    )


def _events() -> deque:
    tool_call = ResponseFunctionToolCall.model_construct(
        id="item-1",
        type="function_call",
        call_id="call-1",
        name="weather",
        arguments='{"city":"Berlin"}',
        namespace=None,
        status="completed",
    )
    return deque(
        [
            ResponseCreatedEvent.model_construct(
                response=_response("in_progress"), sequence_number=0, type="response.created"
            ),
            ResponseTextDeltaEvent.model_construct(
                item_id="text-1", delta="Hallo", sequence_number=1, type="response.output_text.delta"
            ),
            ResponseOutputItemAddedEvent.model_construct(
                item=tool_call, output_index=1, sequence_number=2, type="response.output_item.added"
            ),
            ResponseOutputItemDoneEvent.model_construct(
                item=tool_call, output_index=1, sequence_number=3, type="response.output_item.done"
            ),
            ResponseCompletedEvent.model_construct(
                response=_response("completed"), sequence_number=4, type="response.completed"
            ),
        ]
    )


class _Connection:
    def __init__(self) -> None:
        events = _events()
        self.recv = AsyncMock(side_effect=lambda: events.popleft())
        self.close = AsyncMock()
        self.response = SimpleNamespace(create=AsyncMock())


def test_websocket_call_maps_streamed_text_and_tool_call() -> None:
    async def run():
        model = ChatOpenAI(model="gpt-test", api_key="test", transport="websocket")
        connection = _Connection()
        model._websocket._connection = connection
        try:
            result = await model.call("Wetter?", tools=[{"name": "weather", "parameters": {"type": "object"}}])
            return result, connection
        finally:
            await model.aclose()

    result, connection = asyncio.run(run())

    assert result.completion == "Hallo"
    assert [(call.id, call.function.name) for call in result.tool_calls] == [("call-1", "weather")]
    assert result.stop_reason == "tool_calls"
    connection.response.create.assert_awaited_once()
    connection.close.assert_awaited_once()


def test_websocket_stream_emits_text_tool_and_end() -> None:
    async def run():
        model = ChatOpenAI(model="gpt-test", api_key="test", transport="websocket")
        model._websocket._connection = _Connection()
        try:
            return [event async for event in model.stream("Wetter?")]
        finally:
            await model.aclose()

    events = asyncio.run(run())

    assert isinstance(events[0], StreamTextDelta)
    assert events[0].delta == "Hallo"
    assert isinstance(events[1], StreamToolCall)
    assert isinstance(events[2], StreamEnd)
    assert events[2].completion == "Hallo"
    assert events[2].tool_calls[0].id == "call-1"
