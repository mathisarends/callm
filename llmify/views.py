from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .messages import ToolCall


class ChatUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_cached_tokens: int | None = None


class ChatCompletion[OutputT](BaseModel):
    completion: OutputT
    thinking: str | None = None
    usage: ChatUsage | None = None
    stop_reason: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)


class StreamEventType(StrEnum):
    TEXT = "text"
    TOOL_CALL = "tool_call"
    END = "end"


class StreamTextDelta(BaseModel):
    type: Literal[StreamEventType.TEXT] = StreamEventType.TEXT
    delta: str


class StreamToolCall(BaseModel):
    type: Literal[StreamEventType.TOOL_CALL] = StreamEventType.TOOL_CALL
    tool_call: ToolCall


class StreamEnd(BaseModel):
    type: Literal[StreamEventType.END] = StreamEventType.END
    completion: str = ""
    thinking: str | None = None
    usage: ChatUsage | None = None
    stop_reason: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)


StreamEvent = StreamTextDelta | StreamToolCall | StreamEnd


__all__ = [
    "ChatCompletion",
    "ChatUsage",
    "StreamEnd",
    "StreamEvent",
    "StreamEventType",
    "StreamTextDelta",
    "StreamToolCall",
]
