import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

from llmify import base, messages


class Answer(BaseModel):
    value: int


class Recorder(base.ChatModel):
    """The smallest thing that satisfies the contract."""

    def __init__(self) -> None:
        self.closed = False
        self.calls: list[dict[str, object]] = []

    @property
    def model(self) -> str:
        return "recorder"

    async def call(self, messages, *, tools=(), tool_choice="auto", output_format=None):
        self.calls.append({"tool_choice": tool_choice, "output_format": output_format})
        return base.ModelResponse(completion="ok")

    async def stream(self, messages, *, tools=(), tool_choice="auto"):
        yield base.TextDelta(delta="ok")

    async def aclose(self) -> None:
        self.closed = True


def test_messages_are_frozen() -> None:
    message = messages.UserMessage(content="hi")

    with pytest.raises(ValidationError):
        message.content = "there"  # type: ignore[misc]


def test_messages_discriminate_on_type() -> None:
    adapter = TypeAdapter(messages.Message)

    assert isinstance(
        adapter.validate_python({"type": "system", "content": "be terse"}),
        messages.SystemMessage,
    )
    assert isinstance(
        adapter.validate_python(
            {
                "type": "tool_result",
                "tool_call_id": "a",
                "tool_name": "t",
                "content": "1",
            }
        ),
        messages.ToolResultMessage,
    )


def test_events_discriminate_on_type() -> None:
    adapter = TypeAdapter(base.ModelEvent)

    assert adapter.validate_python({"type": "thinking_delta", "delta": "hm"}) == (
        base.ThinkingDelta(delta="hm")
    )
    assert (
        adapter.validate_python({"type": "response", "completion": "done"}).completion
        == "done"
    )


def test_user_text_reads_through_content_parts() -> None:
    message = messages.UserMessage(
        content=(
            "look at",
            messages.ImageUrl(url="https://example.test/x.png"),
            "and this",
        )
    )

    assert message.text == "look at\nand this"


def test_tool_call_arguments_default_to_an_empty_object() -> None:
    assert messages.ToolCall(id="a", name="t").parsed_arguments == {}


def test_total_tokens_counts_input_and_output() -> None:
    usage = base.Usage(input_tokens=10, output_tokens=5, cache_read_tokens=100)

    assert usage.total_tokens == 15


def test_a_text_turn_becomes_its_history_entry() -> None:
    response = base.ModelResponse(
        completion="said",
        thinking="thought",
        tool_calls=(messages.ToolCall(id="a", name="t"),),
        provider_state="opaque",
    )

    message = response.as_assistant_message()

    assert message.content == "said"
    assert message.thinking == "thought"
    assert message.tool_calls == response.tool_calls
    assert message.provider_state == "opaque"


def test_a_structured_turn_leaves_its_history_content_empty() -> None:
    response = base.ModelResponse(completion=Answer(value=4), provider_state="opaque")

    message = response.as_assistant_message()

    assert message.content == ""
    assert message.provider_state == "opaque"


def test_provider_state_stays_out_of_the_repr() -> None:
    response = base.ModelResponse(
        completion="x", provider_state="a-secret-looking-blob"
    )

    assert "a-secret-looking-blob" not in repr(response)


async def test_calling_the_model_delegates_to_call() -> None:
    model = Recorder()

    response = await model([messages.UserMessage(content="hi")], tool_choice="required")

    assert response.completion == "ok"
    assert model.calls == [{"tool_choice": "required", "output_format": None}]


async def test_the_context_manager_closes_the_model() -> None:
    model = Recorder()

    async with model as entered:
        assert entered is model
        assert model.closed is False

    assert model.closed is True
