"""The provider-neutral contract llmify speaks.

Everything public flows through these types. A provider translates them to and
from whatever its SDK wants; nothing above this module needs to know which
backend answered.
"""

import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from enum import StrEnum
from types import TracebackType
from typing import Annotated, Any, Literal, Self, overload

from pydantic import BaseModel, ConfigDict, Field


class MessageType(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_RESULT = "tool_result"


class ModelEventType(StrEnum):
    TEXT_DELTA = "text_delta"
    THINKING_DELTA = "thinking_delta"
    TOOL_CALL = "tool_call"
    RESPONSE = "response"


class Frozen(BaseModel):
    """Base for every value in the contract.

    Immutable, so a message list can be shared between requests without one of
    them editing another's history.
    """

    model_config = ConfigDict(frozen=True)


class ToolCall(Frozen):
    id: str
    name: str
    arguments: str = "{}"
    """The model's arguments as raw JSON, exactly as it emitted them."""

    @property
    def parsed_arguments(self) -> dict[str, Any]:
        return json.loads(self.arguments)


SupportedImageMediaType = Literal["image/jpeg", "image/png", "image/gif", "image/webp"]


class ImageUrl(Frozen):
    """An image, either by URL or inline as a ``data:`` URI."""

    url: str
    media_type: SupportedImageMediaType = "image/png"
    detail: Literal["auto", "low", "high"] = "auto"


class SystemMessage(Frozen):
    type: Literal[MessageType.SYSTEM] = MessageType.SYSTEM
    content: str


class UserMessage(Frozen):
    type: Literal[MessageType.USER] = MessageType.USER
    content: str | tuple[str | ImageUrl, ...]

    @property
    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        return "\n".join(part for part in self.content if isinstance(part, str))


class AssistantMessage(Frozen):
    type: Literal[MessageType.ASSISTANT] = MessageType.ASSISTANT
    content: str = ""
    thinking: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    provider_state: object | None = Field(default=None, repr=False)
    """The provider's own response object, replayed verbatim when this turn is sent back.

    Reasoning models carry state across a tool round-trip that no text field can
    reconstruct (encrypted reasoning items, thinking signatures). Keeping the
    original object here and handing it back means a tool loop does not silently
    degrade the model's reasoning. Providers that need nothing ignore it.
    """


class ToolResultMessage(Frozen):
    type: Literal[MessageType.TOOL_RESULT] = MessageType.TOOL_RESULT
    tool_call_id: str
    tool_name: str
    content: str
    is_error: bool = False


type Message = Annotated[
    SystemMessage | UserMessage | AssistantMessage | ToolResultMessage,
    Field(discriminator="type"),
]


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
    """A chat model: awaited for a whole turn, iterated for a stream."""

    @property
    @abstractmethod
    def model(self) -> str:
        """The model identifier this instance talks to."""

    @overload
    async def call[T: BaseModel](
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ModelTool] = (),
        tool_choice: ToolChoice = "auto",
        output_format: type[T],
    ) -> ModelResponse[T]: ...

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
    ) -> ModelResponse[T] | ModelResponse[str]:
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

    @overload
    async def __call__[T: BaseModel](
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ModelTool] = (),
        tool_choice: ToolChoice = "auto",
        output_format: type[T],
    ) -> ModelResponse[T]: ...

    @overload
    async def __call__(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ModelTool] = (),
        tool_choice: ToolChoice = "auto",
        output_format: None = None,
    ) -> ModelResponse[str]: ...

    async def __call__[T: BaseModel](
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ModelTool] = (),
        tool_choice: ToolChoice = "auto",
        output_format: type[T] | None = None,
    ) -> ModelResponse[T] | ModelResponse[str]:
        """Convenience alias for ``call``, so ``await model(messages)`` reads as a call."""
        return await self.call(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            output_format=output_format,
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()
