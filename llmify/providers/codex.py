import json
import os
from pathlib import Path
from typing import Any, Self

from pydantic_ai.exceptions import UserError
from pydantic_ai.models.openai_codex import OpenAICodexModel
from pydantic_ai.providers.openai_codex import (
    OpenAICodexCredentials,
    OpenAICodexCredentialSource,
    OpenAICodexProvider,
)

from llmify._adapter import PydanticAIModel
from llmify.exceptions import CredentialsUnavailableError
from llmify.providers.openai import ReasoningEffort, openai_settings


def codex_home() -> Path:
    """Where the Codex CLI keeps its state, honouring `CODEX_HOME`."""
    return Path(os.getenv("CODEX_HOME") or Path.home() / ".codex")


class CodexCliCredentials(OpenAICodexCredentialSource):
    """The Codex CLI's login, kept in sync.

    Read on its own, `auth.json` gives a token that the provider will refresh in
    memory and then forget, leaving the CLI holding a refresh token that has
    already been spent — they are single-use. Writing the rotation back is what
    lets `codex` and llmify go on sharing one login.
    """

    def __init__(self, auth_path: Path | None = None) -> None:
        self.path = auth_path or codex_home() / "auth.json"

    async def load(self) -> OpenAICodexCredentials:
        try:
            return OpenAICodexCredentials.from_codex_cli_auth(self._read())
        except UserError as error:
            raise CredentialsUnavailableError(
                f"The Codex CLI login at {self.path} is missing its tokens. "
                "Run `codex login` again."
            ) from error

    async def save(self, credentials: OpenAICodexCredentials) -> None:
        stored = self._read()
        tokens = {
            **(stored.get("tokens") or {}),
            "access_token": credentials.access_token,
            "refresh_token": credentials.refresh_token,
            "account_id": credentials.account_id,
        }
        # The file exists (load ran first), so truncating it keeps its permissions.
        self.path.write_text(
            json.dumps({**stored, "tokens": tokens}, indent=2), encoding="utf-8"
        )

    def _read(self) -> dict[str, Any]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise CredentialsUnavailableError(
                f"No usable Codex CLI login at {self.path}. Run `codex login` first."
            ) from error


class ChatCodex(PydanticAIModel):
    """OpenAI's Codex endpoint.

    With neither `credentials` nor `credential_source`, the Codex CLI's
    `auth.json` is read once and never written, so refreshed tokens live only as
    long as the process. Use `ChatCodex.from_cli` to keep the CLI's login
    up to date instead.
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

    @classmethod
    def from_cli(
        cls, model: str, *, auth_path: Path | None = None, **settings: Any
    ) -> Self:
        """Borrow the Codex CLI's login, writing refreshed tokens back to it."""
        return cls(model, credential_source=CodexCliCredentials(auth_path), **settings)
