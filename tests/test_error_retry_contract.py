"""End-to-end retry decisions at the pydantic-ai adapter boundary."""

import httpx2
import pytest
from openai import AsyncOpenAI
from pydantic_ai.exceptions import (
    ModelAPIError,
    ModelHTTPError,
    UnexpectedModelBehavior,
)
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from llmify.base import UserMessage
from llmify.errors import (
    AuthenticationError,
    ContextLengthExceededError,
    ModelBehaviorError,
    OutOfCreditsError,
    ProviderError,
    RateLimitError,
    RetryableError,
)
from llmify.pydantic_ai_adapter import PydanticAIModel
from llmify.retries import RetryEvent


@pytest.mark.parametrize(
    ("original", "expected", "retry"),
    [
        (ModelHTTPError(401, "m", "invalid key"), AuthenticationError, False),
        (ModelHTTPError(403, "m", "forbidden"), AuthenticationError, False),
        (ModelHTTPError(402, "m", "payment required"), OutOfCreditsError, False),
        (ModelHTTPError(429, "m", "insufficient_quota"), OutOfCreditsError, False),
        (
            ModelHTTPError(400, "m", "context_length_exceeded"),
            ContextLengthExceededError,
            False,
        ),
        (ModelHTTPError(400, "m", "bad request"), ProviderError, False),
        (ModelHTTPError(429, "m", "slow down"), RateLimitError, True),
        (ModelHTTPError(408, "m", "timeout"), RetryableError, True),
        (ModelHTTPError(409, "m", "conflict"), RetryableError, True),
        (ModelHTTPError(425, "m", "too early"), RetryableError, True),
        (ModelHTTPError(500, "m", "server error"), RetryableError, True),
        (ModelHTTPError(503, "m", "unavailable"), RetryableError, True),
        (ModelAPIError("m", "connection failed"), RetryableError, True),
        (UnexpectedModelBehavior("invalid response"), ModelBehaviorError, False),
    ],
)
async def test_classified_errors_control_attempts_and_hooks(
    monkeypatch: pytest.MonkeyPatch,
    original: Exception,
    expected: type[Exception],
    retry: bool,
) -> None:
    attempts = 0
    events: list[RetryEvent] = []

    async def no_sleep(_delay: float) -> None:
        pass

    async def on_retry(event: RetryEvent) -> None:
        events.append(event)

    def respond(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise original
        return ModelResponse(parts=[TextPart(content="ok")])

    monkeypatch.setattr("llmify.retries.asyncio.sleep", no_sleep)
    model = PydanticAIModel(FunctionModel(respond), max_retries=1, on_retry=on_retry)

    if retry:
        assert (await model.call([UserMessage(content="hi")])).completion == "ok"
        assert attempts == 2
        assert len(events) == 1
        assert isinstance(events[0].error, expected)
        assert events[0].next_attempt == 2
    else:
        with pytest.raises(expected) as caught:
            await model.call([UserMessage(content="hi")])
        assert attempts == 1
        assert not events
        assert caught.value.__cause__ is original


@pytest.mark.parametrize(
    ("status", "body", "expected", "requests_expected"),
    [
        (503, {"error": {"message": "unavailable"}}, RetryableError, 3),
        (
            429,
            {"error": {"code": "insufficient_quota", "message": "quota exhausted"}},
            OutOfCreditsError,
            1,
        ),
        (401, {"error": {"message": "invalid key"}}, AuthenticationError, 1),
    ],
)
async def test_openai_sdk_sends_only_visible_attempts(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    body: dict[str, object],
    expected: type[Exception],
    requests_expected: int,
) -> None:
    requests = 0
    events: list[RetryEvent] = []

    def respond(_request: httpx2.Request) -> httpx2.Response:
        nonlocal requests
        requests += 1
        return httpx2.Response(status, json=body)

    async def no_sleep(_delay: float) -> None:
        pass

    async def on_retry(event: RetryEvent) -> None:
        events.append(event)

    monkeypatch.setattr("llmify.retries.asyncio.sleep", no_sleep)
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(respond)
    ) as http_client:
        client = AsyncOpenAI(api_key="test", http_client=http_client)
        model = PydanticAIModel(
            OpenAIChatModel(
                "gpt-4o-mini", provider=OpenAIProvider(openai_client=client)
            ),
            max_retries=2,
            on_retry=on_retry,
        )

        with pytest.raises(expected):
            await model.call([UserMessage(content="hi")])

    assert client.max_retries == 0
    assert requests == requests_expected
    assert len(events) == requests_expected - 1


async def test_terminal_stream_failure_never_calls_retry_hook() -> None:
    attempts = 0
    events: list[RetryEvent] = []

    async def on_retry(event: RetryEvent) -> None:
        events.append(event)

    async def stream(_messages: list[ModelMessage], _info: AgentInfo):
        nonlocal attempts
        attempts += 1
        raise ModelHTTPError(401, "m", "invalid key")
        yield "unreachable"

    model = PydanticAIModel(
        FunctionModel(stream_function=stream), max_retries=2, on_retry=on_retry
    )

    with pytest.raises(AuthenticationError):
        async for _event in model.stream([UserMessage(content="hi")]):
            pass

    assert attempts == 1
    assert not events


async def test_stream_503_uses_exact_retry_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = 0
    events: list[RetryEvent] = []

    def respond(_request: httpx2.Request) -> httpx2.Response:
        nonlocal requests
        requests += 1
        return httpx2.Response(503, json={"error": {"message": "unavailable"}})

    async def no_sleep(_delay: float) -> None:
        pass

    async def on_retry(event: RetryEvent) -> None:
        events.append(event)

    monkeypatch.setattr("llmify.retries.asyncio.sleep", no_sleep)
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(respond)
    ) as http_client:
        client = AsyncOpenAI(api_key="test", http_client=http_client)
        model = PydanticAIModel(
            OpenAIChatModel(
                "gpt-4o-mini", provider=OpenAIProvider(openai_client=client)
            ),
            max_retries=2,
            on_retry=on_retry,
        )

        with pytest.raises(RetryableError):
            async for _event in model.stream([UserMessage(content="hi")]):
                pass

    assert requests == 3
    assert [event.next_attempt for event in events] == [2, 3]


@pytest.mark.parametrize("failure", [httpx2.ConnectError, httpx2.ReadTimeout])
async def test_sdk_connection_and_timeout_failures_reach_retry_hook(
    monkeypatch: pytest.MonkeyPatch,
    failure: type[httpx2.RequestError],
) -> None:
    requests = 0
    events: list[RetryEvent] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        nonlocal requests
        requests += 1
        raise failure("transport failed", request=request)

    async def no_sleep(_delay: float) -> None:
        pass

    async def on_retry(event: RetryEvent) -> None:
        events.append(event)

    monkeypatch.setattr("llmify.retries.asyncio.sleep", no_sleep)
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(respond)
    ) as http_client:
        client = AsyncOpenAI(api_key="test", http_client=http_client)
        model = PydanticAIModel(
            OpenAIChatModel(
                "gpt-4o-mini", provider=OpenAIProvider(openai_client=client)
            ),
            max_retries=2,
            on_retry=on_retry,
        )

        with pytest.raises(RetryableError):
            await model.call([UserMessage(content="hi")])

    assert requests == 3
    assert [event.next_attempt for event in events] == [2, 3]
