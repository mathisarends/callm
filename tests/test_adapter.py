import pytest
from pydantic import BaseModel
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior
from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.messages import (
    ImageUrl as PydanticImageUrl,
)
from pydantic_ai.models.function import (
    AgentInfo,
    DeltaThinkingPart,
    DeltaToolCall,
    FunctionModel,
)
from pydantic_ai.usage import RequestUsage

from llmify.exceptions import (
    AuthenticationError,
    ContextLengthExceededError,
    ModelBehaviorError,
    OutOfCreditsError,
    RateLimitError,
    RetryableError,
)
from llmify.ports import (
    AssistantMessage,
    ImageUrl,
    ModelEventType,
    ModelTool,
    SystemMessage,
    TextDelta,
    ThinkingDelta,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from llmify.pydantic_ai_adapter import (
    PydanticAIModel,
    _mapped_error,
    _model_messages,
    _usage,
)
from llmify.retries import retry_delay

PNG_PIXEL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def model_for(*parts: object) -> PydanticAIModel:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=list(parts))  # type: ignore[arg-type]

    return PydanticAIModel(FunctionModel(respond))


class Answer(BaseModel):
    value: int
    unit: str


# --- the turn ---------------------------------------------------------------


async def test_call_returns_text_thinking_and_tool_calls() -> None:
    model = model_for(
        ThinkingPart(content="let me see"),
        TextPart(content="4"),
        ToolCallPart("calc", '{"x": 1}', "tc1"),
    )

    response = await model.call([UserMessage(content="2+2?")])

    assert response.completion == "4"
    assert response.thinking == "let me see"
    assert response.tool_calls == (
        ToolCall(id="tc1", name="calc", arguments='{"x": 1}'),
    )


async def test_calling_the_model_directly_is_the_same_as_call() -> None:
    model = model_for(TextPart(content="4"))

    assert (await model([UserMessage(content="2+2?")])).completion == "4"


async def test_tools_reach_the_model_as_definitions() -> None:
    seen: list[str] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.extend(f"{t.name}:{t.description}" for t in info.function_tools)
        return ModelResponse(parts=[TextPart(content="ok")])

    model = PydanticAIModel(FunctionModel(respond))
    await model.call(
        [UserMessage(content="hi")],
        tools=[ModelTool(name="calc", description="Do maths")],
    )

    assert seen == ["calc:Do maths"]


async def test_usage_carries_cache_counters() -> None:
    assert _usage(
        RequestUsage(
            input_tokens=10, output_tokens=3, cache_read_tokens=7, cache_write_tokens=2
        )
    ) == Usage(
        input_tokens=10, output_tokens=3, cache_read_tokens=7, cache_write_tokens=2
    )


# --- history ----------------------------------------------------------------


def test_a_system_message_becomes_instructions() -> None:
    history = _model_messages(
        [SystemMessage(content="be terse"), UserMessage(content="hi")]
    )

    assert len(history) == 1
    assert history[0].instructions == "be terse"


def test_tool_results_answering_one_turn_share_a_request() -> None:
    history = _model_messages(
        [
            UserMessage(content="hi"),
            AssistantMessage(
                tool_calls=(
                    ToolCall(id="a", name="one"),
                    ToolCall(id="b", name="two"),
                )
            ),
            ToolResultMessage(tool_call_id="a", tool_name="one", content="1"),
            ToolResultMessage(tool_call_id="b", tool_name="two", content="2"),
        ]
    )

    assert [type(message).__name__ for message in history] == [
        "ModelRequest",
        "ModelResponse",
        "ModelRequest",
    ]
    assert [type(part).__name__ for part in history[2].parts] == [
        "ToolReturnPart",
        "ToolReturnPart",
    ]


def test_a_failed_tool_result_says_so() -> None:
    history = _model_messages(
        [
            ToolResultMessage(
                tool_call_id="a", tool_name="one", content="boom", is_error=True
            )
        ]
    )

    part = history[0].parts[0]
    assert isinstance(part, ToolReturnPart)
    assert part.outcome == "failed"


def test_provider_state_is_replayed_verbatim() -> None:
    original = ModelResponse(parts=[TextPart(content="kept")])
    history = _model_messages(
        [AssistantMessage(content="lossy", provider_state=original)]
    )

    assert history[0] is original


def test_an_assistant_message_without_provider_state_is_rebuilt() -> None:
    history = _model_messages(
        [
            AssistantMessage(
                content="said",
                thinking="thought",
                tool_calls=(ToolCall(id="a", name="one", arguments='{"x": 1}'),),
            )
        ]
    )

    assert [type(part).__name__ for part in history[0].parts] == [
        "ThinkingPart",
        "TextPart",
        "ToolCallPart",
    ]


def test_an_image_url_travels_as_a_url() -> None:
    message = UserMessage(
        content=(
            "look",
            ImageUrl(url="https://example.test/x.png", detail="high"),
        )
    )

    part = _model_messages([message])[0].parts[0]
    assert isinstance(part, UserPromptPart)
    image = part.content[1]
    assert isinstance(image, PydanticImageUrl)
    assert image.url == "https://example.test/x.png"
    assert image.vendor_metadata == {"detail": "high"}


def test_a_data_uri_travels_as_bytes() -> None:
    message = UserMessage(content=(ImageUrl(url=PNG_PIXEL),))

    part = _model_messages([message])[0].parts[0]
    assert isinstance(part, UserPromptPart)
    image = part.content[0]
    assert isinstance(image, BinaryContent)
    assert image.media_type == "image/png"
    assert image.data.startswith(b"\x89PNG")


def test_plain_text_stays_a_plain_string() -> None:
    part = _model_messages([UserMessage(content="hi")])[0].parts[0]

    assert isinstance(part, UserPromptPart)
    assert part.content == "hi"


# --- structured output ------------------------------------------------------


async def test_structured_output_is_parsed_and_removed_from_tool_calls() -> None:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        name = info.output_tools[0].name
        return ModelResponse(
            parts=[ToolCallPart(name, '{"value": 4, "unit": "apples"}', "o1")]
        )

    model = PydanticAIModel(FunctionModel(respond))
    response = await model.call(
        [UserMessage(content="how many?")], output_format=Answer
    )

    assert response.completion == Answer(value=4, unit="apples")
    assert response.tool_calls == ()


async def test_structured_output_keeps_the_model_s_own_tool_calls() -> None:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[
                ToolCallPart("calc", "{}", "tc1"),
                ToolCallPart(
                    info.output_tools[0].name, '{"value": 1, "unit": "x"}', "o1"
                ),
            ]
        )

    model = PydanticAIModel(FunctionModel(respond))
    response = await model.call([UserMessage(content="?")], output_format=Answer)

    assert [call.name for call in response.tool_calls] == ["calc"]


async def test_a_missing_output_call_is_a_model_behaviour_error() -> None:
    model = model_for(TextPart(content="just prose"))

    with pytest.raises(ModelBehaviorError, match="final_result"):
        await model.call([UserMessage(content="?")], output_format=Answer)


async def test_an_unparsable_output_call_is_a_model_behaviour_error() -> None:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        name = info.output_tools[0].name
        return ModelResponse(parts=[ToolCallPart(name, '{"value": "many"}', "o1")])

    model = PydanticAIModel(FunctionModel(respond))

    with pytest.raises(ModelBehaviorError, match="not a valid Answer"):
        await model.call([UserMessage(content="?")], output_format=Answer)


# --- streaming --------------------------------------------------------------


async def test_a_stream_yields_deltas_then_one_response() -> None:
    async def stream(messages: list[ModelMessage], info: AgentInfo):
        yield "Hello "
        yield "world"
        yield {0: DeltaToolCall(name="calc", json_args='{"x":', tool_call_id="tc9")}
        yield {0: DeltaToolCall(json_args="1}")}

    model = PydanticAIModel(FunctionModel(stream_function=stream))
    events = [event async for event in model.stream([UserMessage(content="hi")])]

    assert [event.type for event in events] == [
        ModelEventType.TEXT_DELTA,
        ModelEventType.TEXT_DELTA,
        ModelEventType.TOOL_CALL,
        ModelEventType.RESPONSE,
    ]
    assert events[2].tool_call == ToolCall(id="tc9", name="calc", arguments='{"x":1}')
    assert events[-1].completion == "Hello world"
    assert events[-1].tool_calls == (events[2].tool_call,)


async def test_a_stream_reports_thinking_separately_from_text() -> None:
    async def stream(messages: list[ModelMessage], info: AgentInfo):
        yield {0: DeltaThinkingPart(content="hmm")}
        yield "answer"

    model = PydanticAIModel(FunctionModel(stream_function=stream))
    events = [event async for event in model.stream([UserMessage(content="hi")])]

    assert events[0] == ThinkingDelta(delta="hmm")
    assert events[1] == TextDelta(delta="answer")
    assert events[-1].thinking == "hmm"


# --- failures ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (401, "nope", AuthenticationError),
        (403, "nope", AuthenticationError),
        (429, "slow down", RateLimitError),
        (402, "pay up", OutOfCreditsError),
        (400, "insufficient_quota", OutOfCreditsError),
        (400, "context_length_exceeded", ContextLengthExceededError),
        (400, "prompt is too long", ContextLengthExceededError),
        (500, "boom", RetryableError),
        (503, "boom", RetryableError),
        (408, "slow", RetryableError),
    ],
)
def test_http_failures_map_onto_the_taxonomy(
    status: int, body: str, expected: type[Exception]
) -> None:
    error = ModelHTTPError(status_code=status, model_name="m", body=body)

    assert isinstance(_mapped_error(error), expected)


def test_a_rate_limit_carries_the_providers_retry_after() -> None:
    error = ModelHTTPError(
        status_code=429, model_name="m", body="slow down", headers={"Retry-After": "12"}
    )

    mapped = _mapped_error(error)

    assert isinstance(mapped, RateLimitError)
    assert mapped.retry_after == 12
    assert retry_delay(mapped, 0) == 12


def test_an_unrecognised_failure_is_left_alone() -> None:
    error = ModelHTTPError(status_code=418, model_name="m", body="teapot")

    assert _mapped_error(error) is error


def test_unexpected_model_behaviour_becomes_a_model_behaviour_error() -> None:
    assert isinstance(
        _mapped_error(UnexpectedModelBehavior("nonsense")), ModelBehaviorError
    )


async def test_a_transient_failure_is_retried() -> None:
    attempts = 0

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ModelHTTPError(status_code=503, model_name="m", body="boom")
        return ModelResponse(parts=[TextPart(content="second time lucky")])

    model = PydanticAIModel(FunctionModel(respond), max_retries=1)
    response = await model.call([UserMessage(content="hi")])

    assert (attempts, response.completion) == (2, "second time lucky")


async def test_a_permanent_failure_is_not_retried() -> None:
    attempts = 0

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal attempts
        attempts += 1
        raise ModelHTTPError(status_code=401, model_name="m", body="nope")

    model = PydanticAIModel(FunctionModel(respond), max_retries=3)

    with pytest.raises(AuthenticationError):
        await model.call([UserMessage(content="hi")])
    assert attempts == 1


def test_max_retries_is_checked_up_front() -> None:
    with pytest.raises(TypeError):
        PydanticAIModel(
            FunctionModel(lambda m, i: ModelResponse(parts=[])), max_retries=True
        )
    with pytest.raises(ValueError):
        PydanticAIModel(
            FunctionModel(lambda m, i: ModelResponse(parts=[])), max_retries=-1
        )


# --- settings ---------------------------------------------------------------


async def test_settings_and_tool_choice_reach_the_request() -> None:
    seen: dict[str, object] = {}

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.update(info.model_settings or {})
        return ModelResponse(parts=[TextPart(content="ok")])

    model = PydanticAIModel(
        FunctionModel(respond),
        temperature=0.25,
        max_tokens=64,
        stop=["END"],
        openai_reasoning_effort="low",
    )
    await model.call([UserMessage(content="hi")], tool_choice="required")

    assert seen["temperature"] == 0.25
    assert seen["max_tokens"] == 64
    assert seen["stop_sequences"] == ["END"]
    assert seen["tool_choice"] == "required"
    assert seen["openai_reasoning_effort"] == "low"
