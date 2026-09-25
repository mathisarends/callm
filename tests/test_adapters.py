import asyncio
from contextlib import asynccontextmanager

from pydantic import BaseModel
from pydantic_ai.messages import ModelResponse, TextPart, ThinkingPart, ToolCallPart
from pydantic_ai.models import CompletedStreamedResponse
from pydantic_ai.usage import RequestUsage

from llmify import (
    AssistantMessage,
    ChatModel,
    Function,
    StreamEnd,
    StreamTextDelta,
    StreamToolCall,
    SystemMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from llmify._adapters import to_pydantic_messages, to_tool_definitions
from llmify.base import PydanticAIChatModel


class Person(BaseModel):
    name: str


class StubBackend:
    def __init__(self, response: ModelResponse) -> None:
        self.response = response
        self.requests = []

    async def request(self, messages, settings, params):
        self.requests.append((messages, settings, params))
        return self.response

    @asynccontextmanager
    async def request_stream(self, messages, settings, params):
        self.requests.append((messages, settings, params))
        yield CompletedStreamedResponse(
            self.response,
            model_request_parameters=params,
            replay_events=True,
        )


def test_chat_model_only_exposes_call_and_stream() -> None:
    assert hasattr(ChatModel, "call")
    assert hasattr(ChatModel, "stream")
    assert not hasattr(ChatModel, "invoke")


def test_call_maps_text_tools_thinking_and_usage() -> None:
    response = ModelResponse(
        parts=[
            ThinkingPart("brief thought"),
            TextPart("hello"),
            ToolCallPart("search", '{"query":"sky"}', "call-1"),
        ],
        usage=RequestUsage(input_tokens=10, cache_read_tokens=4, output_tokens=3),
        finish_reason="stop",
    )
    backend = StubBackend(response)
    model = PydanticAIChatModel("test", backend)  # type: ignore[arg-type]

    result = asyncio.run(model.call("hi"))

    assert result.completion == "hello"
    assert result.thinking == "brief thought"
    assert result.tool_calls[0].function.name == "search"
    assert result.stop_reason == "tool_calls"
    assert result.usage is not None
    assert result.usage.total_tokens == 13


def test_call_validates_structured_output() -> None:
    backend = StubBackend(ModelResponse(parts=[TextPart('{"name":"Ada"}')]))
    model = PydanticAIChatModel("test", backend)  # type: ignore[arg-type]

    result = asyncio.run(model.call("extract", Person))

    assert result.completion == Person(name="Ada")
    params = backend.requests[0][2]
    assert params.output_mode == "native"


def test_stream_emits_text_tool_and_end() -> None:
    response = ModelResponse(
        parts=[TextPart("hello"), ToolCallPart("search", "{}", "call-1")],
        finish_reason="stop",
    )
    backend = StubBackend(response)
    model = PydanticAIChatModel("test", backend)  # type: ignore[arg-type]

    async def collect():
        return [event async for event in model.stream([UserMessage(content="hi")])]

    events = asyncio.run(collect())

    assert isinstance(events[0], StreamTextDelta)
    assert events[0].delta == "hello"
    assert isinstance(events[1], StreamToolCall)
    assert isinstance(events[2], StreamEnd)
    assert events[2].completion == "hello"


def test_message_history_converts_tool_names() -> None:
    call = ToolCall(id="call-1", function=Function(name="weather", arguments="{}"))
    converted = to_pydantic_messages(
        [
            SystemMessage(content="help"),
            UserMessage(content="weather?"),
            AssistantMessage(tool_calls=[call]),
            ToolResultMessage(tool_call_id="call-1", content="sunny"),
        ]
    )

    tool_return = converted[-1].parts[0]
    assert tool_return.tool_name == "weather"


def test_raw_openai_tool_schema_is_supported() -> None:
    definitions = to_tool_definitions(
        [
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
    )

    assert definitions[0].name == "weather"
    assert definitions[0].description == "Get weather"
