from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, get_args, overload
from uuid import uuid4

from pydantic import BaseModel
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.openai_codex import OpenAICodexModel
from pydantic_ai.providers.openai_codex import (
    OpenAICodexCredentials,
    OpenAICodexCredentialSource,
    OpenAICodexProvider,
)
from pydantic_ai.settings import ModelSettings, ServiceTier, ThinkingLevel

from llmkit.base import ModelEvent, ModelResponse, ModelTool, ToolChoice
from llmkit.errors import ResponseInterruptedError
from llmkit.messages import Message, SystemMessage
from llmkit.providers.codex.transport import (
    Transport,
    TransportFallbackCallback,
    TransportFallbackEvent,
    TransportPhase,
)
from llmkit.providers.codex.websocket import (
    CodexResponsesResource,
    WebSocketInterrupted,
    WebSocketUnavailable,
)
from llmkit.providers.openai import ReasoningEffort, openai_settings
from llmkit.pydantic_ai_adapter import (
    PydanticAIModel,
    credentials_required,
    is_async_callable,
    model_messages,
)
from llmkit.retries import RetryCallback

_SESSION_HEADERS = ("session-id", "thread-id", "x-client-request-id")


@dataclass(frozen=True, slots=True)
class _PreparedRequest:
    messages: tuple[Message, ...]
    tools: tuple[ModelTool, ...]
    tool_choice: ToolChoice
    output_format: type[BaseModel] | None
    response_id: str
    connection_generation: int

    def continues(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ModelTool],
        tool_choice: ToolChoice,
        output_format: type[BaseModel] | None,
    ) -> bool:
        prefix = len(self.messages)
        return (
            len(messages) > prefix
            and tuple(messages[:prefix]) == self.messages
            and tuple(tools) == self.tools
            and tool_choice == self.tool_choice
            and output_format is self.output_format
        )


class ChatCodex(PydanticAIModel):
    """OpenAI's Codex endpoint.

    With neither `credentials` nor `credential_source`, the Codex CLI's
    `auth.json` is read once and never written, so refreshed tokens live only as
    long as the process. Pass a `credential_source` to persist them.
    """

    def __init__(
        self,
        model: str,
        *,
        credentials: OpenAICodexCredentials | None = None,
        credential_source: OpenAICodexCredentialSource | None = None,
        reasoning_effort: ReasoningEffort | str | None = None,
        reasoning_summary: str | None = None,
        transport: Transport = "http",
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
        on_transport_fallback: TransportFallbackCallback | None = None,
        **settings: Any,
    ) -> None:
        if transport not in get_args(Transport.__value__):
            raise ValueError("'transport' must be 'websocket' or 'http'.")
        if on_transport_fallback is not None and not is_async_callable(
            on_transport_fallback
        ):
            raise TypeError("'on_transport_fallback' must be an async callable.")
        with credentials_required():
            provider = OpenAICodexProvider(
                credentials, credential_source=credential_source
            )

        self._responses_resource: CodexResponsesResource | None = None
        self._prepared_request: _PreparedRequest | None = None
        self._on_transport_fallback = on_transport_fallback
        if transport == "websocket":
            self._responses_resource = CodexResponsesResource.install(provider)
            session_id = str(uuid4())
            extra_headers = _with_session_headers(extra_headers, session_id)
            settings.setdefault("openai_prompt_cache_key", session_id)

        super().__init__(
            OpenAICodexModel(model, provider=provider),
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            logit_bias=logit_bias,
            thinking=thinking,
            parallel_tool_calls=parallel_tool_calls,
            service_tier=service_tier,
            timeout=timeout,
            extra_headers=extra_headers,
            extra_body=extra_body,
            max_retries=max_retries,
            on_retry=on_retry,
            **openai_settings(
                settings,
                reasoning_effort=reasoning_effort,
                reasoning_summary=reasoning_summary,
            ),
        )

    async def prepare(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ModelTool] = (),
        tool_choice: ToolChoice = "auto",
        output_format: type[BaseModel] | None = None,
    ) -> None:
        """Prepare a WebSocket conversation prefix for the next matching turn."""
        self._prepared_request = None
        resource = self._responses_resource
        if resource is None:
            return

        settings = dict(self._settings_for(tool_choice) or {})
        settings.pop("openai_previous_response_id", None)
        try:
            with resource.warmup_request():
                response = await self._model.request(
                    model_messages(messages, ensure_request=True),
                    ModelSettings(**settings),  # type: ignore[typeddict-item]
                    self._request_parameters(tools, output_format),
                )
        except (WebSocketUnavailable, WebSocketInterrupted) as error:
            await self._report_transport_fallback("prepare", error)
            return

        response_id = response.provider_response_id
        generation = resource.connection_generation
        if response_id is not None and generation is not None:
            self._prepared_request = _PreparedRequest(
                messages=tuple(messages),
                tools=tuple(tools),
                tool_choice=tool_choice,
                output_format=output_format,
                response_id=response_id,
                connection_generation=generation,
            )

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

    async def call[T: BaseModel](
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ModelTool] = (),
        tool_choice: ToolChoice = "auto",
        output_format: type[T] | None = None,
    ) -> ModelResponse[T] | ModelResponse[T | None] | ModelResponse[str]:
        try:
            return await super().call(
                messages,
                tools=tools,
                tool_choice=tool_choice,
                output_format=output_format,
            )
        except WebSocketUnavailable as error:
            async with self._http_fallback("call", error):
                return await super().call(
                    messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    output_format=output_format,
                )
        except WebSocketInterrupted as error:
            raise ResponseInterruptedError(str(error)) from error

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ModelTool] = (),
        tool_choice: ToolChoice = "auto",
    ) -> AsyncIterator[ModelEvent]:
        emitted = False
        try:
            async for event in super().stream(
                messages, tools=tools, tool_choice=tool_choice
            ):
                emitted = True
                yield event
        except WebSocketUnavailable as error:
            if emitted:
                raise
            async with self._http_fallback("stream", error):
                async for event in super().stream(
                    messages, tools=tools, tool_choice=tool_choice
                ):
                    yield event
        except WebSocketInterrupted as error:
            raise ResponseInterruptedError(str(error)) from error

    async def aclose(self) -> None:
        if self._responses_resource is not None:
            await self._responses_resource.aclose()
        await super().aclose()

    def _request_context(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ModelTool],
        tool_choice: ToolChoice,
        output_format: type[BaseModel] | None,
    ) -> tuple[list[ModelMessage], ModelSettings | None]:
        prepared, self._prepared_request = self._prepared_request, None
        resource = self._responses_resource
        if (
            prepared is None
            or resource is None
            or resource.http_only
            or resource.connection_generation != prepared.connection_generation
            or not prepared.continues(
                messages,
                tools=tools,
                tool_choice=tool_choice,
                output_format=output_format,
            )
        ):
            return super()._request_context(
                messages,
                tools=tools,
                tool_choice=tool_choice,
                output_format=output_format,
            )

        history = model_messages(
            messages[len(prepared.messages) :],
            initial_instructions=_instructions(prepared.messages),
        )
        settings = dict(self._settings_for(tool_choice) or {})
        settings["openai_previous_response_id"] = prepared.response_id
        return history, ModelSettings(**settings)  # type: ignore[typeddict-item]

    @asynccontextmanager
    async def _http_fallback(
        self, phase: TransportPhase, error: WebSocketUnavailable
    ) -> AsyncGenerator[None]:
        # Only the WebSocket transport raises WebSocketUnavailable.
        assert self._responses_resource is not None
        await self._report_transport_fallback(phase, error)
        with self._responses_resource.use_http():
            yield

    async def _report_transport_fallback(
        self, phase: TransportPhase, error: Exception
    ) -> None:
        if self._on_transport_fallback is not None:
            await self._on_transport_fallback(
                TransportFallbackEvent(phase=phase, reason=str(error))
            )


def _with_session_headers(
    extra_headers: dict[str, str] | None, session_id: str
) -> dict[str, str]:
    headers = dict(extra_headers or {})
    supplied = {name.lower() for name in headers}
    for name in _SESSION_HEADERS:
        if name not in supplied:
            headers[name] = session_id
    return headers


def _instructions(messages: Sequence[Message]) -> str | None:
    return next(
        (
            message.content
            for message in reversed(messages)
            if isinstance(message, SystemMessage)
        ),
        None,
    )
