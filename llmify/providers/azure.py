"""Azure's hosting of the OpenAI models."""

from typing import Any

from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_ai.providers.azure import AzureProvider

from llmify._adapter import PydanticAIModel
from llmify.providers.openai import ReasoningEffort, openai_settings


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
            **openai_settings(settings, default_headers=default_headers),
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
            **openai_settings(
                settings,
                default_headers=default_headers,
                reasoning_effort=reasoning_effort,
                reasoning_summary=reasoning_summary,
            ),
        )
