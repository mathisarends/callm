from collections.abc import Sequence
from typing import Any

from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_ai.providers.azure import AzureProvider
from pydantic_ai.settings import ServiceTier, ThinkingLevel

from llmify.providers.openai import ReasoningEffort, openai_settings
from llmify.pydantic_ai_adapter import PydanticAIModel
from llmify.retries import RetryCallback


class ChatAzureOpenAI(PydanticAIModel):
    """Azure OpenAI's Chat Completions API.

    `model` is the deployment name. `api_key` and `azure_endpoint` fall back to
    `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_ENDPOINT`.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        azure_endpoint: str | None = None,
        api_version: str | None = None,
        default_headers: dict[str, str] | None = None,
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
        super().__init__(
            OpenAIChatModel(
                model,
                provider=AzureProvider(
                    azure_endpoint=azure_endpoint,
                    api_key=api_key,
                    api_version=api_version,
                ),
            ),
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
            extra_headers=extra_headers
            if extra_headers is not None
            else default_headers,
            extra_body=extra_body,
            max_retries=max_retries,
            on_retry=on_retry,
            **openai_settings(settings),
        )


class ChatAzureOpenAIResponses(PydanticAIModel):
    """Azure OpenAI's Responses API."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        azure_endpoint: str | None = None,
        api_version: str | None = None,
        reasoning_effort: ReasoningEffort | str | None = None,
        reasoning_summary: str | None = None,
        default_headers: dict[str, str] | None = None,
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
        super().__init__(
            OpenAIResponsesModel(
                model,
                provider=AzureProvider(
                    azure_endpoint=azure_endpoint,
                    api_key=api_key,
                    api_version=api_version,
                ),
            ),
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
            extra_headers=extra_headers
            if extra_headers is not None
            else default_headers,
            extra_body=extra_body,
            max_retries=max_retries,
            on_retry=on_retry,
            **openai_settings(
                settings,
                reasoning_effort=reasoning_effort,
                reasoning_summary=reasoning_summary,
            ),
        )
