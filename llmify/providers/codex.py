import inspect
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel

from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.openai_codex import OpenAICodexModel
from pydantic_ai.providers.openai_codex import (
    OpenAICodexCredentials,
    OpenAICodexCredentialSource,
    OpenAICodexProvider,
)
from pydantic_ai.settings import ModelSettings, ServiceTier, ThinkingLevel

from llmify.errors import CredentialsUnavailableError, ResponseInterruptedError
from llmify.base import (
    ModelEvent,
    ModelResponse,
    ModelTool,
    ToolChoice,
)
from llmify.messages import Message, SystemMessage
from llmify.providers.codex_transport import (
    CodexResponsesResource,
    WebSocketInterrupted,
    WebSocketUnavailable,
    install_codex_responses_resource,
)
from llmify.providers.openai import ReasoningEffort, openai_settings
from llmify.pydantic_ai_adapter import (
    PydanticAIModel,
    _model_messages,
    _request_parameters,
)
from llmify.retries import RetryCallback


@dataclass(frozen=True, slots=True)
class _PreparedRequest:
    messages: tuple[Message, ...]
    tools: tuple[ModelTool, ...]
    tool_choice: ToolChoice
    output_format: type[BaseModel] | None
    response_id: str
    connection_generation: int


@dataclass(frozen=True, slots=True)
class TransportFallbackEvent:
    phase: Literal["prepare", "call", "stream"]
    reason: str


type TransportFallbackCallback = Callable[[TransportFallbackEvent], Awaitable[None]]


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
        transport: Literal["websocket", "http"] = "http",
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
        if on_transport_fallback is not None and not (
            inspect.iscoroutinefunction(on_transport_fallback)
            or inspect.iscoroutinefunction(
                getattr(on_transport_fallback, "__call__", None)
            )
        ):
            raise TypeError("'on_transport_fallback' must be an async callable.")
        if transport not in ("websocket", "http"):
            raise ValueError("'transport' must be 'websocket' or 'http'.")
        if transport == "websocket":
            try:
                import websockets  # noqa: F401
            except ImportError:
                raise ImportError(
                    "transport='websocket' requires the 'websockets' package. "
                    "Install it with: pip install 'py-llmify[websocket]'"
                ) from None
        try:
            provider = OpenAICodexProvider(
                credentials, credential_source=credential_source
            )
        except UserError as error:
            raise CredentialsUnavailableError(str(error)) from error

        self._responses_resource: CodexResponsesResource | None = None
        self._prepared_request: _PreparedRequest | None = None
        self._on_transport_fallback = on_transport_fallback
        if transport == "websocket":
            self._responses_resource = install_codex_responses_resource(provider)
            session_id = str(uuid4())
            headers = dict(extra_headers or {})
            supplied_headers = {name.lower() for name in headers}
            for name in ("session-id", "thread-id", "x-client-request-id"):
                if name not in supplied_headers:
                    headers[name] = session_id
            extra_headers = headers
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

        history = _model_messages(messages, ensure_request=True)
        settings = dict(self._settings_for(tool_choice) or {})
        settings.pop("openai_previous_response_id", None)
        try:
            async with resource.warmup_request():
                response = await self._model.request(
                    history,
                    ModelSettings(**settings),  # type: ignore[typeddict-item]
                    _request_parameters(tools, output_format),
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

    def _request_context(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ModelTool],
        tool_choice: ToolChoice,
        output_format: type[BaseModel] | None,
    ) -> tuple[list[ModelMessage], ModelSettings | None]:
        prepared = self._prepared_request
        self._prepared_request = None
        resource = self._responses_resource
        if (
            prepared is None
            or resource is None
            or resource.http_only
            or resource.connection_generation != prepared.connection_generation
            or len(messages) <= len(prepared.messages)
            or tuple(messages[: len(prepared.messages)]) != prepared.messages
            or tuple(tools) != prepared.tools
            or tool_choice != prepared.tool_choice
            or output_format is not prepared.output_format
        ):
            return super()._request_context(
                messages,
                tools=tools,
                tool_choice=tool_choice,
                output_format=output_format,
            )

        instructions = next(
            (
                message.content
                for message in reversed(prepared.messages)
                if isinstance(message, SystemMessage)
            ),
            None,
        )
        history = _model_messages(
            messages[len(prepared.messages) :],
            initial_instructions=instructions,
        )
        settings = dict(self._settings_for(tool_choice) or {})
        settings["openai_previous_response_id"] = prepared.response_id
        return history, ModelSettings(**settings)  # type: ignore[typeddict-item]

    async def _report_transport_fallback(
        self,
        phase: Literal["prepare", "call", "stream"],
        error: Exception,
    ) -> None:
        if self._on_transport_fallback is not None:
            await self._on_transport_fallback(
                TransportFallbackEvent(phase=phase, reason=str(error))
            )

    async def call[T: BaseModel](
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ModelTool] = (),
        tool_choice: ToolChoice = "auto",
        output_format: type[T] | None = None,
    ) -> ModelResponse[T] | ModelResponse[str]:
        resource = self._responses_resource
        if resource is None:
            return await super().call(
                messages,
                tools=tools,
                tool_choice=tool_choice,
                output_format=output_format,
            )
        try:
            return await super().call(
                messages,
                tools=tools,
                tool_choice=tool_choice,
                output_format=output_format,
            )
        except WebSocketUnavailable as error:
            await self._report_transport_fallback("call", error)
            async with resource.use_http():
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
        resource = self._responses_resource
        if resource is None:
            async for event in super().stream(
                messages, tools=tools, tool_choice=tool_choice
            ):
                yield event
            return
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
            await self._report_transport_fallback("stream", error)
            async with resource.use_http():
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
