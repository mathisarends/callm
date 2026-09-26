import json
from inspect import Parameter, signature
from operator import attrgetter
from pathlib import Path

import pytest
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_ai.models.openai_codex import OpenAICodexModel
from pydantic_ai.providers.azure import AzureProvider
from pydantic_ai.providers.openai import OpenAIProvider

import llmkit
from llmkit.errors import CredentialsUnavailableError
from llmkit.providers.codex import ChatCodex
from llmkit.providers.openai import ReasoningEffort

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
    model = llmkit.ChatOpenAI("gpt-5.6", api_key="sk-test")

    assert model.model == "gpt-5.6"
    assert isinstance(model._model, OpenAIChatModel)
    assert isinstance(model._model.provider, OpenAIProvider)
    assert model._model.client.max_retries == 0


def test_openai_responses_uses_the_responses_api() -> None:
    model = llmkit.ChatOpenAIResponses("gpt-5.6", api_key="sk-test")

    assert isinstance(model._model, OpenAIResponsesModel)
    assert model._model.client.max_retries == 0


def test_azure_reaches_the_azure_provider() -> None:
    model = llmkit.ChatAzureOpenAI(
        "my-deployment",
        api_key="k",
        azure_endpoint="https://example.openai.azure.com/",
        api_version="2024-10-01",
    )

    assert isinstance(model._model, OpenAIChatModel)
    assert isinstance(model._model.provider, AzureProvider)
    assert model._model.client.max_retries == 0


def test_azure_responses_reaches_the_azure_provider() -> None:
    model = llmkit.ChatAzureOpenAIResponses(
        "my-deployment",
        api_key="k",
        azure_endpoint="https://example.openai.azure.com/",
        api_version="2024-10-01",
    )

    assert isinstance(model._model, OpenAIResponsesModel)
    assert isinstance(model._model.provider, AzureProvider)
    assert model._model.client.max_retries == 0


# --- named options become model settings ------------------------------------


@pytest.mark.parametrize(
    "provider",
    [
        llmkit.ChatOpenAI,
        llmkit.ChatOpenAIResponses,
        llmkit.ChatAzureOpenAI,
        llmkit.ChatAzureOpenAIResponses,
        ChatCodex,
    ],
)
def test_common_options_are_explicit_keyword_parameters(provider: type) -> None:
    parameters = signature(provider).parameters

    for name in (
        "max_tokens",
        "stop_sequences",
        "temperature",
        "top_p",
        "top_k",
        "seed",
        "frequency_penalty",
        "presence_penalty",
        "logit_bias",
        "thinking",
        "parallel_tool_calls",
        "service_tier",
        "timeout",
        "extra_headers",
        "extra_body",
        "max_retries",
        "on_retry",
    ):
        assert parameters[name].kind is Parameter.KEYWORD_ONLY
    assert "stop" not in parameters
    assert "default_headers" not in parameters


def test_public_provider_forwards_named_settings() -> None:
    model = llmkit.ChatOpenAI(
        "gpt-5.6",
        api_key="k",
        top_k=8,
        stop_sequences=("END",),
        extra_headers={"X-Tenant": "acme"},
    )

    settings = dict(model._settings or {})
    assert settings["top_k"] == 8
    assert settings["stop_sequences"] == ["END"]
    assert settings["extra_headers"] == {"X-Tenant": "acme"}


def test_reasoning_effort_becomes_a_model_setting() -> None:
    model = llmkit.ChatOpenAIResponses(
        "gpt-5.6", api_key="k", reasoning_effort=ReasoningEffort.HIGH
    )

    assert dict(model._settings or {})["openai_reasoning_effort"] == "high"


def test_reasoning_effort_also_accepts_the_plain_string() -> None:
    model = llmkit.ChatOpenAIResponses("gpt-5.6", api_key="k", reasoning_effort="xhigh")

    assert dict(model._settings or {})["openai_reasoning_effort"] == "xhigh"


def test_an_unknown_reasoning_effort_is_rejected_up_front() -> None:
    with pytest.raises(ValueError):
        llmkit.ChatOpenAIResponses("gpt-5.6", api_key="k", reasoning_effort="enormous")


def test_unnamed_settings_pass_straight_through() -> None:
    model = llmkit.ChatOpenAI(
        "gpt-5.6", api_key="k", temperature=0.2, service_tier="flex"
    )

    settings = dict(model._settings or {})
    assert settings["temperature"] == 0.2
    assert settings["service_tier"] == "flex"


# --- codex ------------------------------------------------------------------


def test_codex_reads_the_cli_login(
    auth_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CODEX_HOME", str(auth_file.parent))

    model = ChatCodex("gpt-5.6-terra", reasoning_effort="high")

    assert isinstance(model._model, OpenAICodexModel)
    assert dict(model._settings or {})["openai_reasoning_effort"] == "high"
    assert model._model.client.max_retries == 0


def test_codex_says_so_when_there_is_no_login(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))

    with pytest.raises(CredentialsUnavailableError, match="codex login"):
        ChatCodex("gpt-5.6-terra")


# --- public exports ---------------------------------------------------------


def test_every_exported_name_resolves() -> None:
    assert [name for name in llmkit.__all__ if not hasattr(llmkit, name)] == []


def test_exported_names_are_discoverable() -> None:
    assert set(llmkit.__all__) <= set(dir(llmkit))


def test_an_unknown_name_is_an_attribute_error() -> None:
    with pytest.raises(AttributeError, match="ChatSomethingElse"):
        attrgetter("ChatSomethingElse")(llmkit)


def test_provider_package_reexports_the_public_providers() -> None:
    from llmkit import providers

    assert providers.ChatOpenAI is llmkit.ChatOpenAI
    assert providers.ChatCodex is llmkit.ChatCodex
    assert set(providers.__all__) <= set(dir(providers))


def test_importing_llmkit_does_not_require_websockets() -> None:
    import subprocess
    import sys

    loaded = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['websockets'] = None; import llmkit; print(llmkit.ChatCodex.__name__)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert loaded.stdout.strip() == "ChatCodex"


@pytest.mark.parametrize(
    ("make", "variables"),
    [
        (lambda: llmkit.ChatOpenAI("gpt-5.6"), ["OPENAI_API_KEY"]),
        (lambda: llmkit.ChatOpenAIResponses("gpt-5.6"), ["OPENAI_API_KEY"]),
        (
            lambda: llmkit.ChatAzureOpenAI("deployment", api_version="2024-10-01"),
            ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY"],
        ),
        (
            lambda: llmkit.ChatAzureOpenAIResponses(
                "deployment", api_version="2024-10-01"
            ),
            ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY"],
        ),
    ],
)
def test_missing_credentials_are_reported_as_llmkit_errors(
    make, variables: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    for variable in variables:
        monkeypatch.delenv(variable, raising=False)

    with pytest.raises(CredentialsUnavailableError):
        make()


def test_chat_completions_take_a_reasoning_effort() -> None:
    model = llmkit.ChatOpenAI("gpt-5.6", api_key="k", reasoning_effort="none")

    assert model._settings is not None
    assert model._settings.get("openai_reasoning_effort") == "none"
