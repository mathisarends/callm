import asyncio
import base64
import struct
import zlib
from collections.abc import Callable

import pytest
from pydantic import BaseModel

from llmify import (
    AssistantMessage,
    ChatCodex,
    ChatModel,
    ImageUrl,
    Message,
    ModelEventType,
    ModelResponse,
    ModelTool,
    SystemMessage,
    ToolCallEvent,
    ToolResultMessage,
    UserMessage,
)

POPULATION = ModelTool(
    name="city_population",
    description="Look up how many people live in a city.",
    parameters={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
)

BERLIN = "3878000"


class City(BaseModel):
    name: str
    country: str


def solid_png(rgb: tuple[int, int, int], size: int = 32) -> bytes:
    row = b"\x00" + bytes(rgb) * size

    def chunk(kind: bytes, data: bytes) -> bytes:
        checksum = zlib.crc32(kind + data)
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)

    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(row * size))
        + chunk(b"IEND", b"")
    )


def data_uri(image: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(image).decode()


async def run_tool_loop(
    model: ChatModel,
    messages: list[Message],
    *,
    keep_provider_state: bool = True,
) -> ModelResponse[str]:
    for _ in range(5):
        response = await model(messages, tools=[POPULATION])
        turn = response.as_assistant_message()
        if not keep_provider_state:
            turn = turn.model_copy(update={"provider_state": None})
        messages.append(turn)
        if not response.tool_calls:
            return response
        for call in response.tool_calls:
            assert call.parsed_arguments["city"].lower() == "berlin"
            messages.append(
                ToolResultMessage(
                    tool_call_id=call.id, tool_name=call.name, content=BERLIN
                )
            )
    pytest.fail("the tool loop did not finish")


# --- plain turns -------------------------------------------------------------


async def test_a_turn_returns_text_and_usage(model: ChatModel) -> None:
    response = await model([UserMessage(content="What is 2+2? Digits only.")])

    assert "4" in response.completion
    assert response.finish_reason == "stop"
    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens > 0
    assert response.tool_calls == ()


async def test_the_system_message_is_followed(model: ChatModel) -> None:
    response = await model(
        [
            SystemMessage(content="Reply with the single word BANANA, nothing else."),
            UserMessage(content="Hello!"),
        ]
    )

    assert "BANANA" in response.completion.upper()


async def test_a_history_without_provider_state_is_understood(
    model: ChatModel,
) -> None:
    response = await model(
        [
            UserMessage(content="My name is Zed."),
            AssistantMessage(content="Nice to meet you, Zed."),
            UserMessage(content="What is my name? One word."),
        ]
    )

    assert "zed" in response.completion.lower()


async def test_one_model_serves_concurrent_turns(model: ChatModel) -> None:
    questions = ["1+1? Digits only.", "2+3? Digits only.", "4+4? Digits only."]

    responses = await asyncio.gather(
        *(model([UserMessage(content=q)]) for q in questions)
    )

    assert ["2" in responses[0].completion, "5" in responses[1].completion] == [
        True,
        True,
    ]
    assert "8" in responses[2].completion


# --- streaming ---------------------------------------------------------------


async def test_stream_deltas_add_up_to_the_final_response(model: ChatModel) -> None:
    events = [
        event
        async for event in model.stream(
            [UserMessage(content="Count from 1 to 5, separated by spaces.")]
        )
    ]

    *deltas, final = events
    assert final.type == ModelEventType.RESPONSE
    assert all(event.type != ModelEventType.RESPONSE for event in deltas)
    text = "".join(e.delta for e in deltas if e.type == ModelEventType.TEXT_DELTA)
    assert text == final.completion
    assert "1 2 3 4 5" in final.completion
    assert final.usage.output_tokens > 0


async def test_a_streamed_tool_call_is_emitted_and_returned(model: ChatModel) -> None:
    events = [
        event
        async for event in model.stream(
            [UserMessage(content="How many people live in Berlin?")],
            tools=[POPULATION],
            tool_choice="required",
        )
    ]

    calls = [e.tool_call for e in events if isinstance(e, ToolCallEvent)]
    final = events[-1]
    assert final.type == ModelEventType.RESPONSE
    assert [call.name for call in calls] == ["city_population"]
    assert final.tool_calls == tuple(calls)


# --- structured output -------------------------------------------------------


async def test_structured_output_is_parsed(model: ChatModel) -> None:
    response = await model(
        [UserMessage(content="Which city is the capital of France?")],
        output_format=City,
    )

    assert response.completion.name == "Paris"
    assert response.completion.country == "France"


async def test_a_structured_turn_can_be_followed_by_a_plain_one(
    model: ChatModel,
) -> None:
    messages: list[Message] = [UserMessage(content="Capital of France?")]
    first = await model(messages, output_format=City)
    messages += [
        first.as_assistant_message(),
        UserMessage(content="And the capital of Germany? Just the name."),
    ]

    second = await model(messages)

    assert "Berlin" in second.completion


async def test_structured_output_can_follow_a_tool_loop(model: ChatModel) -> None:
    messages: list[Message] = [
        SystemMessage(content="Use the tools before answering."),
        UserMessage(content="How many people live in Berlin?"),
    ]
    await run_tool_loop(model, messages)
    messages.append(UserMessage(content="Now give me that city as structured data."))

    response = await model(messages, output_format=City)

    assert response.completion.name == "Berlin"


class Population(BaseModel):
    city: str
    people: int


async def test_a_structured_tool_loop_answers_once_the_tools_are_done(
    model: ChatModel,
) -> None:
    messages: list[Message] = [
        SystemMessage(content="Use the tools before answering."),
        UserMessage(content="How many people live in Berlin?"),
    ]
    completions: list[Population | None] = []

    for _ in range(5):
        response = await model(messages, tools=[POPULATION], output_format=Population)
        completions.append(response.completion)
        messages.append(response.as_assistant_message())
        if not response.tool_calls:
            break
        messages += [
            ToolResultMessage(tool_call_id=call.id, tool_name=call.name, content=BERLIN)
            for call in response.tool_calls
        ]

    *pending, answer = completions
    assert pending and all(completion is None for completion in pending)
    assert answer == Population(city="Berlin", people=int(BERLIN))


# --- images ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rgb", "colour"), [((255, 0, 0), "red"), ((0, 0, 255), "blue")]
)
async def test_a_base64_image_is_seen(
    model: ChatModel, rgb: tuple[int, int, int], colour: str
) -> None:
    image = ImageUrl(url=data_uri(solid_png(rgb)), detail="low")

    response = await model(
        [UserMessage(content=("Which colour fills this image? One word.", image))]
    )

    assert colour in response.completion.lower()


async def test_a_linked_image_is_seen(model: ChatModel) -> None:
    image = ImageUrl(
        url="https://raw.githubusercontent.com/github/explore/main/topics/python/python.png"
    )

    response = await model(
        [UserMessage(content=("Which programming language's logo is this?", image))]
    )

    assert "python" in response.completion.lower()


async def test_several_images_in_one_message_are_all_seen(model: ChatModel) -> None:
    red = ImageUrl(url=data_uri(solid_png((255, 0, 0))), detail="low")
    green = ImageUrl(url=data_uri(solid_png((0, 255, 0))), detail="low")

    response = await model(
        [UserMessage(content=("Name the colour of each image, in order.", red, green))]
    )

    answer = response.completion.lower()
    assert answer.index("red") < answer.index("green")


# --- tools -------------------------------------------------------------------


async def test_tool_choice_required_forces_a_call(model: ChatModel) -> None:
    response = await model(
        [UserMessage(content="How many people live in Berlin?")],
        tools=[POPULATION],
        tool_choice="required",
    )

    (call,) = response.tool_calls
    assert call.name == "city_population"
    assert call.parsed_arguments["city"].lower() == "berlin"


async def test_tool_choice_none_forbids_calls(model: ChatModel) -> None:
    response = await model(
        [UserMessage(content="How many people live in Berlin?")],
        tools=[POPULATION],
        tool_choice="none",
    )

    assert response.tool_calls == ()
    assert response.completion


@pytest.mark.parametrize("keep_provider_state", [True, False])
async def test_a_tool_loop_uses_the_tool_result(
    model: ChatModel, keep_provider_state: bool
) -> None:
    messages: list[Message] = [
        SystemMessage(content="Use the tools before answering."),
        UserMessage(content="How many people live in Berlin? Digits only."),
    ]

    response = await run_tool_loop(
        model, messages, keep_provider_state=keep_provider_state
    )

    assert BERLIN in response.completion.replace(",", "").replace(".", "")


async def test_a_failed_tool_result_is_reported_back(model: ChatModel) -> None:
    messages: list[Message] = [UserMessage(content="How many people live in Berlin?")]
    first = await model(messages, tools=[POPULATION], tool_choice="required")
    messages.append(first.as_assistant_message())
    messages += [
        ToolResultMessage(
            tool_call_id=call.id,
            tool_name=call.name,
            content="Service unavailable.",
            is_error=True,
        )
        for call in first.tool_calls
    ]

    second = await model(messages, tools=[POPULATION], tool_choice="none")

    assert second.completion


# --- Codex WebSocket ---------------------------------------------------------


async def test_a_prepared_websocket_conversation_continues(
    provider: str, make_model: Callable[[], ChatModel]
) -> None:
    if provider != "codex-websocket":
        pytest.skip("prepare() only matters over the WebSocket transport")
    messages: list[Message] = [SystemMessage(content="Answer in one word.")]

    async with make_model() as model:
        assert isinstance(model, ChatCodex)
        await model.prepare(messages)
        messages.append(UserMessage(content="Capital of Italy?"))
        first = await model(messages)
        messages.append(first.as_assistant_message())
        await model.prepare(messages)
        messages.append(UserMessage(content="And of Spain?"))
        second = await model(messages)

    assert "Rome" in first.completion
    assert "Madrid" in second.completion
