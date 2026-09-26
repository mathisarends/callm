"""Live tests against real provider APIs.

Excluded from the default run; start them with ``pytest -m live``. Each provider
is skipped when its credentials are missing: ``OPENAI_API_KEY`` for OpenAI, the
Codex CLI's ``auth.json`` for Codex.
"""

import os
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import pytest
from dotenv import load_dotenv

from llmify import (
    ChatCodex,
    ChatModel,
    ChatOpenAI,
    ChatOpenAIResponses,
    CredentialsUnavailableError,
)

load_dotenv(override=True)

OPENAI_MODEL = os.getenv("LLMIFY_OPENAI_MODEL", "gpt-5.6")
CODEX_MODEL = os.getenv("CODEX_MODEL", "gpt-5.6-terra")

type Factory = Callable[[], ChatModel]

PROVIDERS: dict[str, Factory] = {
    # Chat Completions rejects function tools for gpt-5.6 at any other effort.
    "chat": lambda: ChatOpenAI(OPENAI_MODEL, reasoning_effort="none"),
    "responses": lambda: ChatOpenAIResponses(OPENAI_MODEL, reasoning_effort="low"),
    "codex-http": lambda: ChatCodex(CODEX_MODEL, reasoning_effort="low"),
    "codex-websocket": lambda: ChatCodex(
        CODEX_MODEL, reasoning_effort="low", transport="websocket"
    ),
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    here = Path(__file__).parent
    for item in items:
        if item.path.is_relative_to(here):
            item.add_marker(pytest.mark.live)


@pytest.fixture(params=list(PROVIDERS))
def provider(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture
async def make_model(provider: str) -> Factory:
    make = PROVIDERS[provider]
    if provider in ("chat", "responses") and not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not set")
    try:
        async with make():
            pass
    except CredentialsUnavailableError as error:
        pytest.skip(str(error))
    return make


@pytest.fixture
async def model(make_model: Factory) -> AsyncIterator[ChatModel]:
    async with make_model() as model:
        yield model
