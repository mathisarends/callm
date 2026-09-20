import json
from pathlib import Path

import pytest
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.cerebras import CerebrasModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_ai.models.openai_codex import OpenAICodexModel
from pydantic_ai.providers.azure import AzureProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.openai_codex import OpenAICodexCredentials

import llmify
from llmify.exceptions import CredentialsUnavailableError
from llmify.providers.codex import ChatCodex, CodexCliCredentials
from llmify.providers.openai import ReasoningEffort

AUTH_JSON = {
    "OPENAI_API_KEY": None,
    "tokens": {
        "access_token": "access-1",
        "refresh_token": "refresh-1",
        "account_id": "acct-1",
    },
    "last_refresh": "2026-01-01T00:00:00Z",
}


@pytest.fixture
def auth_file(tmp_path: Path) -> Path:
    path = tmp_path / "auth.json"
    path.write_text(json.dumps(AUTH_JSON), encoding="utf-8")
    return path


# --- what each provider builds ----------------------------------------------


def test_openai_uses_the_chat_completions_api() -> None:
    model = llmify.ChatOpenAI("gpt-5.6", api_key="sk-test")

    assert model.model == "gpt-5.6"
    assert isinstance(model._model, OpenAIChatModel)
    assert isinstance(model._model.provider, OpenAIProvider)


def test_openai_responses_uses_the_responses_api() -> None:
    model = llmify.ChatOpenAIResponses("gpt-5.6", api_key="sk-test")

    assert isinstance(model._model, OpenAIResponsesModel)


def test_an_openai_compatible_endpoint_keeps_its_base_url() -> None:
    model = llmify.OpenAICompatible(
        "local-model", base_url="https://example.test/v1", api_key="k"
    )

    assert model._model.base_url.startswith("https://example.test/v1")


def test_azure_reaches_the_azure_provider() -> None:
    model = llmify.ChatAzureOpenAI(
        "my-deployment",
        api_key="k",
        azure_endpoint="https://example.openai.azure.com/",
        api_version="2024-10-01",
    )

    assert isinstance(model._model, OpenAIChatModel)
    assert isinstance(model._model.provider, AzureProvider)


def test_azure_responses_reaches_the_azure_provider() -> None:
    model = llmify.ChatAzureOpenAIResponses(
        "my-deployment",
        api_key="k",
        azure_endpoint="https://example.openai.azure.com/",
        api_version="2024-10-01",
    )

    assert isinstance(model._model, OpenAIResponsesModel)
    assert isinstance(model._model.provider, AzureProvider)


def test_cerebras_anthropic_and_google_build_their_own_models() -> None:
    assert isinstance(
        llmify.ChatCerebras("gpt-oss-120b", api_key="csk")._model, CerebrasModel
    )
    assert isinstance(
        llmify.ChatAnthropic("claude-sonnet-4-5", api_key="sk-ant")._model,
        AnthropicModel,
    )
    assert isinstance(
        llmify.ChatGoogle("gemini-3-pro", api_key="k")._model, GoogleModel
    )


# --- named options become model settings ------------------------------------


def test_reasoning_effort_becomes_a_model_setting() -> None:
    model = llmify.ChatOpenAIResponses(
        "gpt-5.6", api_key="k", reasoning_effort=ReasoningEffort.HIGH
    )

    assert model._settings["openai_reasoning_effort"] == "high"


def test_reasoning_effort_also_accepts_the_plain_string() -> None:
    model = llmify.ChatOpenAIResponses("gpt-5.6", api_key="k", reasoning_effort="xhigh")

    assert model._settings["openai_reasoning_effort"] == "xhigh"


def test_an_unknown_reasoning_effort_is_rejected_up_front() -> None:
    with pytest.raises(ValueError):
        llmify.ChatOpenAIResponses("gpt-5.6", api_key="k", reasoning_effort="enormous")


def test_default_headers_become_extra_headers() -> None:
    model = llmify.ChatOpenAI(
        "gpt-5.6", api_key="k", default_headers={"X-Tenant": "acme"}
    )

    assert model._settings["extra_headers"] == {"X-Tenant": "acme"}


def test_unnamed_settings_pass_straight_through() -> None:
    model = llmify.ChatOpenAI(
        "gpt-5.6", api_key="k", temperature=0.2, service_tier="flex"
    )

    assert model._settings["temperature"] == 0.2
    assert model._settings["service_tier"] == "flex"


# --- codex ------------------------------------------------------------------


def test_codex_reads_the_cli_login(
    auth_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CODEX_HOME", str(auth_file.parent))

    model = ChatCodex("gpt-5.6-terra", reasoning_effort="high")

    assert isinstance(model._model, OpenAICodexModel)
    assert model._settings["openai_reasoning_effort"] == "high"


def test_codex_says_so_when_there_is_no_login(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))

    with pytest.raises(CredentialsUnavailableError, match="codex login"):
        ChatCodex("gpt-5.6-terra")


def test_from_cli_defers_the_read_to_the_credential_source(tmp_path: Path) -> None:
    # No auth.json anywhere, yet construction succeeds: the source loads lazily.
    model = ChatCodex.from_cli("gpt-5.6-terra", auth_path=tmp_path / "auth.json")

    assert isinstance(model._model, OpenAICodexModel)


async def test_the_cli_source_loads_the_stored_tokens(auth_file: Path) -> None:
    credentials = await CodexCliCredentials(auth_file).load()

    assert credentials.account_id == "acct-1"
    assert credentials.access_token == "access-1"


async def test_the_cli_source_writes_rotated_tokens_back(auth_file: Path) -> None:
    source = CodexCliCredentials(auth_file)

    await source.save(
        OpenAICodexCredentials(
            access_token="access-2", refresh_token="refresh-2", account_id="acct-1"
        )
    )

    stored = json.loads(auth_file.read_text(encoding="utf-8"))
    assert stored["tokens"]["access_token"] == "access-2"
    assert stored["tokens"]["refresh_token"] == "refresh-2"
    assert stored["last_refresh"] == AUTH_JSON["last_refresh"]


async def test_the_cli_source_reports_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(CredentialsUnavailableError, match="codex login"):
        await CodexCliCredentials(tmp_path / "nothing.json").load()


async def test_the_cli_source_reports_a_file_without_tokens(tmp_path: Path) -> None:
    path = tmp_path / "auth.json"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(CredentialsUnavailableError, match="missing its tokens"):
        await CodexCliCredentials(path).load()


def test_codex_home_follows_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    from llmify.providers.codex import codex_home

    monkeypatch.setenv("CODEX_HOME", "/somewhere/else")
    assert codex_home() == Path("/somewhere/else")

    monkeypatch.delenv("CODEX_HOME")
    assert codex_home() == Path.home() / ".codex"


# --- lazy exports -----------------------------------------------------------


def test_every_exported_name_resolves() -> None:
    assert [name for name in llmify.__all__ if not hasattr(llmify, name)] == []


def test_exported_names_are_discoverable() -> None:
    assert set(llmify.__all__) <= set(dir(llmify))


def test_an_unknown_name_is_an_attribute_error() -> None:
    with pytest.raises(AttributeError, match="ChatSomethingElse"):
        llmify.ChatSomethingElse  # type: ignore[attr-defined]


def test_importing_llmify_does_not_import_every_sdk() -> None:
    import subprocess
    import sys

    loaded = subprocess.run(
        [
            sys.executable,
            "-c",
            "import llmify, sys; print(int(any(m in sys.modules for m in "
            "('anthropic', 'google.genai'))))",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert loaded.stdout.strip() == "0"
