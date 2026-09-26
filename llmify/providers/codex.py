from collections.abc import AsyncIterator, Sequence
from typing import Any, Literal

from pydantic import BaseModel

from pydantic_ai.exceptions import UserError
from pydantic_ai.models.openai_codex import OpenAICodexModel
from pydantic_ai.providers.openai_codex import (
    OpenAICodexCredentials,
    OpenAICodexCredentialSource,
    OpenAICodexProvider,
)
from pydantic_ai.settings import ServiceTier, ThinkingLevel

from llmify.errors import CredentialsUnavailableError
from llmify.base import Message, ModelEvent, ModelResponse, ModelTool, ToolChoice
from llmify.providers.codex_transport import (
    CodexResponsesResource,
    WebSocketUnavailable,
    install_codex_responses_resource,
)
from llmify.providers.openai import ReasoningEffort, openai_settings
from llmify.pydantic_ai_adapter import PydanticAIModel
from llmify.retries import RetryCallback


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
        stop: Sequence[str] | None = None,
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
        if transport == "websocket":
            self._responses_resource = install_codex_responses_resource(provider)
        super().__init__(
            OpenAICodexModel(model, provider=provider),
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
            stop=stop,
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
        except WebSocketUnavailable:
            async with resource.use_http():
                return await super().call(
                    messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    output_format=output_format,
                )

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
        except WebSocketUnavailable:
            if emitted:
                raise
            async with resource.use_http():
                async for event in super().stream(
                    messages, tools=tools, tool_choice=tool_choice
                ):
                    yield event

    async def aclose(self) -> None:
        if self._responses_resource is not None:
            await self._responses_resource.aclose()
        await super().aclose()
