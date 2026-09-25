from collections.abc import Sequence
from typing import Any

from pydantic_ai.messages import ImageUrl as PydanticImageUrl
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.output import OutputObjectDefinition
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RequestUsage

from .messages import (
    AssistantMessage,
    ContentPartImage,
    ContentPartText,
    Function,
    Message,
    SystemMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from .views import ChatCompletion, ChatUsage


def normalize_messages(messages: str | Sequence[Message]) -> list[Message]:
    if isinstance(messages, str):
        return [UserMessage(content=messages)]
    return list(messages)


def to_pydantic_messages(messages: Sequence[Message]) -> list[ModelMessage]:
    converted: list[ModelMessage] = []
    tool_names: dict[str, str] = {}

    for message in messages:
        if isinstance(message, SystemMessage):
            converted.append(ModelRequest(parts=[SystemPromptPart(message.text)]))
        elif isinstance(message, UserMessage):
            converted.append(ModelRequest(parts=[UserPromptPart(_user_content(message))]))
        elif isinstance(message, AssistantMessage):
            parts = []
            if message.text:
                parts.append(TextPart(message.text))
            for call in message.tool_calls:
                tool_names[call.id] = call.function.name
                parts.append(
                    ToolCallPart(
                        call.function.name,
                        call.function.arguments,
                        call.id,
                    )
                )
            converted.append(ModelResponse(parts=parts))
        elif isinstance(message, ToolResultMessage):
            name = message.tool_name or tool_names.get(message.tool_call_id) or "tool"
            converted.append(
                ModelRequest(
                    parts=[ToolReturnPart(name, message.content, message.tool_call_id)],
                )
            )

    return converted


def _user_content(message: UserMessage) -> str | list[str | PydanticImageUrl]:
    if isinstance(message.content, str):
        return message.content

    content: list[str | PydanticImageUrl] = []
    for part in message.content:
        if isinstance(part, ContentPartText):
            content.append(part.text)
        elif isinstance(part, ContentPartImage):
            content.append(
                PydanticImageUrl(
                    part.image_url.url,
                    vendor_metadata={"detail": part.image_url.detail},
                )
            )
    return content


def to_tool_definitions(tools: Sequence[dict[str, Any]] | None) -> list[ToolDefinition]:
    definitions: list[ToolDefinition] = []
    for schema in tools or ():
        function = schema.get("function", schema)
        try:
            name = function["name"]
        except KeyError as exc:
            raise ValueError("A tool schema must contain a function name.") from exc
        definitions.append(
            ToolDefinition(
                name=name,
                description=function.get("description") or None,
                parameters_json_schema=function.get("parameters") or {"type": "object", "properties": {}},
                strict=function.get("strict"),
            )
        )
    return definitions


def request_parameters(
    tools: Sequence[dict[str, Any]] | None,
    output_format: type[Any] | None,
) -> ModelRequestParameters:
    params = ModelRequestParameters(function_tools=to_tool_definitions(tools))
    if output_format is None:
        return params
    schema = output_format.model_json_schema()
    return ModelRequestParameters(
        function_tools=params.function_tools,
        output_mode="native",
        output_object=OutputObjectDefinition(
            name=output_format.__name__,
            json_schema=schema,
            strict=True,
        ),
    )


def from_pydantic_response[OutputT](
    response: ModelResponse,
    output_format: type[OutputT] | None = None,
) -> ChatCompletion[OutputT] | ChatCompletion[str]:
    text = response.text or ""
    completion: OutputT | str = text if output_format is None else output_format.model_validate_json(text)

    tool_calls = [
        ToolCall(
            id=part.tool_call_id,
            function=Function(name=part.tool_name, arguments=part.args_as_json_str()),
            provider_metadata=part.provider_details or {},
        )
        for part in response.parts
        if isinstance(part, ToolCallPart)
    ]
    thinking_parts = [part.content for part in response.parts if isinstance(part, ThinkingPart) and part.content]

    return ChatCompletion(
        completion=completion,
        thinking="\n".join(thinking_parts) or None,
        usage=from_usage(response.usage),
        stop_reason="tool_calls" if tool_calls else response.finish_reason or response.state,
        tool_calls=tool_calls,
    )


def from_usage(usage: RequestUsage | None) -> ChatUsage | None:
    if usage is None:
        return None
    return ChatUsage(
        prompt_tokens=usage.input_tokens,
        prompt_cached_tokens=usage.cache_read_tokens or None,
        completion_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
    )


__all__ = [
    "from_pydantic_response",
    "from_usage",
    "normalize_messages",
    "request_parameters",
    "to_pydantic_messages",
    "to_tool_definitions",
]
