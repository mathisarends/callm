"""The one bridge between llmify's contract and pydantic-ai's model adapters.

Every provider in `llmify.providers` is this class plus the handful of lines it
takes to build the right `pydantic_ai` model. Translation lives here once, so a
new provider is a constructor and nothing else.
"""

import base64
import inspect
from collections.abc import AsyncIterator, Sequence
from typing import Any

from pydantic import BaseModel, ValidationError
from pydantic_ai.exceptions import (
    ModelAPIError,
    ModelHTTPError,
    UnexpectedModelBehavior,
)
from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponsePart,
    ModelResponseStreamEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolReturnPart,
    UserContent,
    UserPromptPart,
)
from pydantic_ai.messages import (
    ImageUrl as PydanticImageUrl,
)
from pydantic_ai.messages import (
    ModelResponse as PydanticModelResponse,
)
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.settings import ModelSettings, ServiceTier, ThinkingLevel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RequestUsage

from llmify.errors import (
    AuthenticationError,
    ContextLengthExceededError,
    ModelBehaviorError,
    OutOfCreditsError,
    ProviderError,
    RateLimitError,
    RetryableError,
)
from llmify.base import (
    ChatModel,
    ModelEvent,
    ModelResponse,
    ModelTool,
    TextDelta,
    ThinkingDelta,
    ToolCallEvent,
    ToolChoice,
    Usage,
)
from llmify.messages import (
    AssistantMessage,
    ImageUrl,
    Message,
    SystemMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from llmify.retries import RetryCallback, retry_call, retry_stream

OUTPUT_TOOL_NAME = "final_result"
"""The tool a model calls to deliver a structured answer.

Structured output goes through a tool on every provider rather than through each
one's native JSON mode: tool calling is the one shape all of them speak, so the
answer is parsed in one place instead of three.
"""


class PydanticAIModel(ChatModel):
    """A `ChatModel` backed by a pydantic-ai model adapter."""

    def __init__(
        self,
        model: Model,
        *,
        # Output
        max_tokens: int | None = None,
        stop_sequences: Sequence[str] | None = None,
        # Sampling
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        seed: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        logit_bias: dict[str, int] | None = None,
        # Model behavior
        thinking: ThinkingLevel | None = None,
        parallel_tool_calls: bool | None = None,
        service_tier: ServiceTier | None = None,
        # Request and retries
        timeout: float | None = 60.0,
        extra_headers: dict[str, str] | None = None,
        extra_body: object | None = None,
        max_retries: int = 2,
        on_retry: RetryCallback | None = None,
        **settings: Any,
    ) -> None:
        if "tool_choice" in settings:
            raise TypeError("'tool_choice' must be passed to call() or stream().")
        if not isinstance(max_retries, int) or isinstance(max_retries, bool):
            raise TypeError("'max_retries' must be an integer.")
        if max_retries < 0:
            raise ValueError("'max_retries' must be greater than or equal to 0.")
        if on_retry is not None and not (
            inspect.iscoroutinefunction(on_retry)
            or inspect.iscoroutinefunction(getattr(on_retry, "__call__", None))
        ):
            raise TypeError("'on_retry' must be an async callable.")

        self._model = model
        self._max_retries = max_retries
        self._on_retry = on_retry
        # The outer retry loop owns the budget and emits every on_retry event.
        # Otherwise an SDK client can silently retry several times per attempt.
        client = getattr(model, "client", None)
        if client is not None and hasattr(client, "max_retries"):
            client.max_retries = 0
        self._settings = _settings(
            {
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "seed": seed,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
                "logit_bias": logit_bias,
                "stop_sequences": list(stop_sequences)
                if stop_sequences is not None
                else None,
                "thinking": thinking,
                "parallel_tool_calls": parallel_tool_calls,
                "service_tier": service_tier,
                "timeout": timeout,
                "extra_headers": extra_headers,
                "extra_body": extra_body,
                **settings,
            }
        )

    @property
    def model(self) -> str:
        return self._model.model_name

    async def call[T: BaseModel](
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ModelTool] = (),
        tool_choice: ToolChoice = "auto",
        output_format: type[T] | None = None,
    ) -> ModelResponse[T] | ModelResponse[str]:
        parameters = _request_parameters(tools, output_format)
        history, settings = self._request_context(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            output_format=output_format,
        )

        response = await retry_call(
            lambda: self._model.request(history, settings, parameters),
            max_retries=self._max_retries,
            on_retry=self._on_retry,
            map_error=_mapped_error,
        )
        return _response(response, output_format)

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ModelTool] = (),
        tool_choice: ToolChoice = "auto",
    ) -> AsyncIterator[ModelEvent]:
        parameters = _request_parameters(tools, None)
        history, settings = self._request_context(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            output_format=None,
        )

        async def attempt() -> AsyncIterator[ModelEvent]:
            async with self._model.request_stream(
                history, settings, parameters
            ) as stream:
                async for event in stream:
                    if converted := _stream_event(event):
                        yield converted
                yield _response(stream.get(), None)

        async for event in retry_stream(
            attempt,
            max_retries=self._max_retries,
            on_retry=self._on_retry,
            map_error=_mapped_error,
        ):
            yield event

    async def aclose(self) -> None:
        await self._model.__aexit__(None, None, None)

    def _settings_for(self, tool_choice: ToolChoice) -> ModelSettings | None:
        return _settings({**(self._settings or {}), "tool_choice": tool_choice})

    def _request_context(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ModelTool],
        tool_choice: ToolChoice,
        output_format: type[BaseModel] | None,
    ) -> tuple[list[ModelMessage], ModelSettings | None]:
        return _model_messages(messages), self._settings_for(tool_choice)


def _settings(values: dict[str, Any]) -> ModelSettings | None:
    present = {key: value for key, value in values.items() if value is not None}
    return ModelSettings(**present) if present else None  # type: ignore[typeddict-item]


def _request_parameters(
    tools: Sequence[ModelTool],
    output_format: type[BaseModel] | None,
) -> ModelRequestParameters:
    function_tools = [
        ToolDefinition(
            name=tool.name,
            description=tool.description or None,
            parameters_json_schema=tool.parameters,
        )
        for tool in tools
    ]
    if output_format is None:
        return ModelRequestParameters(function_tools=function_tools)

    return ModelRequestParameters(
        function_tools=function_tools,
        output_mode="tool",
        output_tools=[
            ToolDefinition(
                name=OUTPUT_TOOL_NAME,
                description=f"The final answer, as {output_format.__name__}.",
                parameters_json_schema=output_format.model_json_schema(),
            )
        ],
        allow_text_output=False,
    )


def _model_messages(
    messages: Sequence[Message],
    *,
    initial_instructions: str | None = None,
    ensure_request: bool = False,
) -> list[ModelMessage]:
    """Fold llmify's flat message list into pydantic-ai's request/response pairs.

    Consecutive user and tool-result messages join one request rather than each
    becoming their own: several tool results answering one turn belong together,
    and providers that require strictly alternating roles reject them otherwise.
    """
    instructions = initial_instructions
    history: list[ModelMessage] = []
    pending: list[ModelRequestPart] = []

    def flush() -> None:
        if pending:
            history.append(ModelRequest(parts=list(pending), instructions=instructions))
            pending.clear()

    for message in messages:
        match message:
            case SystemMessage():
                instructions = message.content
            case UserMessage():
                pending.append(UserPromptPart(content=_user_content(message)))
            case ToolResultMessage():
                pending.append(
                    ToolReturnPart(
                        message.tool_name,
                        message.content,
                        message.tool_call_id,
                        outcome="failed" if message.is_error else "success",
                    )
                )
            case AssistantMessage(provider_state=PydanticModelResponse() as state):
                flush()
                history.append(state)
            case AssistantMessage():
                flush()
                history.append(PydanticModelResponse(parts=_assistant_parts(message)))

    flush()
    if ensure_request:
        history.append(ModelRequest(parts=[], instructions=instructions))
    return history


def _user_content(message: UserMessage) -> str | list[UserContent]:
    if isinstance(message.content, str):
        return message.content
    return [part if isinstance(part, str) else _image(part) for part in message.content]


def _image(image: ImageUrl) -> BinaryContent | PydanticImageUrl:
    detail = {"detail": image.detail}
    if not image.url.startswith("data:"):
        return PydanticImageUrl(url=image.url, vendor_metadata=detail)

    header, _, payload = image.url.partition(",")
    media_type = header.removeprefix("data:").removesuffix(";base64")
    return BinaryContent(
        data=base64.b64decode(payload),
        media_type=media_type or image.media_type,
        vendor_metadata=detail,
    )


def _assistant_parts(message: AssistantMessage) -> list[ModelResponsePart]:
    parts: list[ModelResponsePart] = []
    if message.thinking:
        parts.append(ThinkingPart(content=message.thinking))
    if message.content:
        parts.append(TextPart(content=message.content))
    parts.extend(
        ToolCallPart(call.name, call.arguments, call.id) for call in message.tool_calls
    )
    return parts


def _response[T: BaseModel](
    response: PydanticModelResponse,
    output_format: type[T] | None,
) -> ModelResponse[T] | ModelResponse[str]:
    text = "".join(
        part.content for part in response.parts if isinstance(part, TextPart)
    )
    thinking = "".join(
        part.content for part in response.parts if isinstance(part, ThinkingPart)
    )
    tool_calls = tuple(
        _tool_call(part) for part in response.parts if isinstance(part, ToolCallPart)
    )

    completion: Any = text
    if output_format is not None:
        completion, tool_calls = _structured(output_format, tool_calls)

    return ModelResponse(
        completion=completion,
        thinking=thinking or None,
        finish_reason=response.finish_reason or "stop",
        tool_calls=tool_calls,
        usage=_usage(response.usage),
        provider_state=response,
    )


def _structured[T: BaseModel](
    output_format: type[T],
    tool_calls: tuple[ToolCall, ...],
) -> tuple[T, tuple[ToolCall, ...]]:
    """Split the output tool's call off the model's real tool calls and parse it."""
    answer = next((c for c in tool_calls if c.name == OUTPUT_TOOL_NAME), None)
    if answer is None:
        raise ModelBehaviorError(
            f"The model returned no {OUTPUT_TOOL_NAME!r} call, so there is no "
            f"{output_format.__name__} to parse."
        )

    try:
        parsed = output_format.model_validate_json(answer.arguments)
    except ValidationError as error:
        raise ModelBehaviorError(
            f"The model's {OUTPUT_TOOL_NAME!r} call is not a valid "
            f"{output_format.__name__}: {error}"
        ) from error

    return parsed, tuple(c for c in tool_calls if c is not answer)


def _tool_call(part: ToolCallPart) -> ToolCall:
    return ToolCall(
        id=part.tool_call_id,
        name=part.tool_name,
        arguments=part.args_as_json_str(),
    )


def _usage(usage: RequestUsage) -> Usage:
    return Usage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=usage.cache_read_tokens,
        cache_write_tokens=usage.cache_write_tokens,
    )


def _stream_event(event: ModelResponseStreamEvent) -> ModelEvent | None:
    """Narrow pydantic-ai's event stream to the three things llmify promises."""
    match event:
        case PartStartEvent(part=TextPart(content=content)) if content:
            return TextDelta(delta=content)
        case PartStartEvent(part=ThinkingPart(content=content)) if content:
            return ThinkingDelta(delta=content)
        case PartDeltaEvent(delta=TextPartDelta(content_delta=delta)) if delta:
            return TextDelta(delta=delta)
        case PartDeltaEvent(delta=ThinkingPartDelta(content_delta=delta)) if delta:
            return ThinkingDelta(delta=delta)
        case PartEndEvent(part=ToolCallPart() as part):
            return ToolCallEvent(tool_call=_tool_call(part))
    return None


_CONTEXT_LENGTH_MARKERS = (
    "context_length_exceeded",
    "context length",
    "too many tokens",
    "prompt is too long",
)
_OUT_OF_CREDITS_MARKERS = (
    "insufficient_quota",
    "insufficient credit",
    "billing",
    "exceeded your current quota",
)


def _mapped_error(error: Exception) -> Exception:
    """Translate a provider failure into llmify's taxonomy.

    pydantic-ai normalises HTTP and connection failures into `ModelHTTPError`
    and `ModelAPIError` respectively for the providers used here.
    """
    match error:
        case ModelHTTPError():
            return _from_status(error)
        case ModelAPIError():
            return RetryableError(str(error))
        case UnexpectedModelBehavior():
            return ModelBehaviorError(str(error))
    return error


def _from_status(error: ModelHTTPError) -> Exception:
    status, body = error.status_code, str(error.body).lower()
    out_of_credits = any(marker in body for marker in _OUT_OF_CREDITS_MARKERS)

    if status in (401, 403):
        return AuthenticationError(str(error), status_code=status)
    if status == 402 or (status == 429 and out_of_credits):
        return OutOfCreditsError(str(error), status_code=status)
    if status == 429:
        return RateLimitError(str(error), retry_after=error.retry_after)
    if status >= 500 or status in (408, 409, 425):
        return RetryableError(str(error), status_code=status)
    if out_of_credits:
        return OutOfCreditsError(str(error), status_code=status)
    if any(marker in body for marker in _CONTEXT_LENGTH_MARKERS):
        return ContextLengthExceededError(str(error), status_code=status)
    return ProviderError(str(error), status_code=status)
