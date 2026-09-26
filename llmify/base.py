from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from enum import StrEnum
from types import TracebackType
from typing import Annotated, Any, Literal, Self, overload

from pydantic import BaseModel, Field

from llmify.messages import AssistantMessage, Frozen, Message, ToolCall


class ModelEventType(StrEnum):
    TEXT_DELTA = "text_delta"
    THINKING_DELTA = "thinking_delta"
    TOOL_CALL = "tool_call"
    RESPONSE = "response"


class ModelTool(Frozen):
    """A tool as the model sees it: a name, a description, a JSON schema."""

    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )


type ToolChoice = Literal["auto", "required", "none"]


class Usage(Frozen):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class TextDelta(Frozen):
    type: Literal[ModelEventType.TEXT_DELTA] = ModelEventType.TEXT_DELTA
    delta: str


class ThinkingDelta(Frozen):
    type: Literal[ModelEventType.THINKING_DELTA] = ModelEventType.THINKING_DELTA
    delta: str


class ToolCallEvent(Frozen):
    """Emitted once a tool call's arguments JSON is fully assembled."""

    type: Literal[ModelEventType.TOOL_CALL] = ModelEventType.TOOL_CALL
    tool_call: ToolCall


class ModelResponse[T](Frozen):
    """A finished turn. Also the final event of every stream, emitted exactly once."""

    type: Literal[ModelEventType.RESPONSE] = ModelEventType.RESPONSE
    completion: T
    thinking: str | None = None
    finish_reason: str = "stop"
    tool_calls: tuple[ToolCall, ...] = ()
    usage: Usage = Usage()
    provider_state: object | None = Field(default=None, repr=False)

    def as_assistant_message(self) -> AssistantMessage:
        """This turn, shaped for the next request's history.

        A structured completion leaves ``content`` empty: the parsed object has
        no faithful text form, and ``provider_state`` carries what the model
        actually said.
        """
        return AssistantMessage(
            content=self.completion if isinstance(self.completion, str) else "",
            thinking=self.thinking,
            tool_calls=self.tool_calls,
            provider_state=self.provider_state,
        )


type ModelEvent = Annotated[
    TextDelta | ThinkingDelta | ToolCallEvent | ModelResponse[str],
    Field(discriminator="type"),
]


class ChatModel(ABC):
    """A chat model: call for a whole turn, stream for incremental results."""

    @property
    @abstractmethod
    def model(self) -> str:
        """The model identifier this instance talks to."""

    @overload
    async def call[T: BaseModel](
        self,
        messages: Sequence[Message],
        *,
        tools: tuple[()] = (),
        tool_choice: ToolChoice = "auto",
        output_format: type[T],
    ) -> ModelResponse[T]: ...

    @overload
    async def call[T: BaseModel](
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ModelTool],
        tool_choice: ToolChoice = "auto",
        output_format: type[T],
    ) -> ModelResponse[T | None]: ...

    @overload
    async def call(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ModelTool] = (),
        tool_choice: ToolChoice = "auto",
        output_format: None = None,
    ) -> ModelResponse[str]: ...

    @abstractmethod
    async def call[T: BaseModel](
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ModelTool] = (),
        tool_choice: ToolChoice = "auto",
        output_format: type[T] | None = None,
    ) -> ModelResponse[T] | ModelResponse[T | None] | ModelResponse[str]:
        """Run one turn and return it whole."""

    @abstractmethod
    def stream(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ModelTool] = (),
        tool_choice: ToolChoice = "auto",
    ) -> AsyncIterator[ModelEvent]:
        """Run one turn, yielding deltas as they arrive and a ``ModelResponse`` last."""

    @abstractmethod
    async def aclose(self) -> None:
        """Release the underlying HTTP client."""

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()
