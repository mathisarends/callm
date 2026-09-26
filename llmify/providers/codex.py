from typing import Any

from pydantic_ai.exceptions import UserError
from pydantic_ai.models.openai_codex import OpenAICodexModel
from pydantic_ai.providers.openai_codex import (
    OpenAICodexCredentials,
    OpenAICodexCredentialSource,
    OpenAICodexProvider,
)

from llmify.exceptions import CredentialsUnavailableError
from llmify.providers.openai import ReasoningEffort, openai_settings
from llmify.pydantic_ai_adapter import PydanticAIModel


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
        **settings: Any,
    ) -> None:
        try:
            provider = OpenAICodexProvider(
                credentials, credential_source=credential_source
            )
        except UserError as error:
            raise CredentialsUnavailableError(str(error)) from error

        super().__init__(
            OpenAICodexModel(model, provider=provider),
            **openai_settings(
                settings,
                reasoning_effort=reasoning_effort,
                reasoning_summary=reasoning_summary,
            ),
        )
