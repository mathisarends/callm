from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ContentPartText(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    type: Literal["text"] = "text"


class ImageURL(BaseModel):
    model_config = ConfigDict(frozen=True)

    url: str
    detail: Literal["auto", "low", "high"] = "auto"


class ContentPartImage(BaseModel):
    model_config = ConfigDict(frozen=True)

    image_url: ImageURL
    type: Literal["image_url"] = "image_url"


class Function(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    arguments: str = "{}"


class ToolCall(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    function: Function
    type: Literal["function"] = "function"
    provider_metadata: dict[str, object] = Field(default_factory=dict, repr=False)


class MessageRole(StrEnum):
    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"
    TOOL = "tool"


class _Message(BaseModel):
    role: MessageRole


class UserMessage(_Message):
    role: Literal[MessageRole.USER] = MessageRole.USER
    content: str | list[ContentPartText | ContentPartImage]
    name: str | None = None

    @property
    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        return "\n".join(part.text for part in self.content if isinstance(part, ContentPartText))


class SystemMessage(_Message):
    role: Literal[MessageRole.SYSTEM] = MessageRole.SYSTEM
    content: str | list[ContentPartText]
    name: str | None = None

    @property
    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        return "\n".join(part.text for part in self.content)


class AssistantMessage(_Message):
    role: Literal[MessageRole.ASSISTANT] = MessageRole.ASSISTANT
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)

    @property
    def text(self) -> str:
        return self.content or ""


class ToolResultMessage(_Message):
    role: Literal[MessageRole.TOOL] = MessageRole.TOOL
    tool_call_id: str
    content: str
    tool_name: str | None = None


type Message = UserMessage | SystemMessage | AssistantMessage | ToolResultMessage
type MessageInput = str | list[Message]

# The longer names existed in the pre-1.0 API and remain harmless data-model aliases.
ContentPartTextParam = ContentPartText
ContentPartImageParam = ContentPartImage


__all__ = [
    "AssistantMessage",
    "ContentPartImage",
    "ContentPartImageParam",
    "ContentPartText",
    "ContentPartTextParam",
    "Function",
    "ImageURL",
    "Message",
    "MessageInput",
    "MessageRole",
    "SystemMessage",
    "ToolCall",
    "ToolResultMessage",
    "UserMessage",
]
