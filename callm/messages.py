import json
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class MessageType(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_RESULT = "tool_result"


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
