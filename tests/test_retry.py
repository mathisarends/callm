import pytest

from llmify.errors import RateLimitError, RetryableError
from llmify.retries import RetryEvent, retry_call, retry_delay


def test_uses_retry_after_from_rate_limit() -> None:
    assert retry_delay(RateLimitError(retry_after=3.5), 0) == 3.5


def test_negative_retry_after_is_immediate() -> None:
    assert retry_delay(RateLimitError(retry_after=-1), 0) == 0.0


@pytest.mark.parametrize(
    ("retry_number", "expected"),
    [(0, 0.5), (1, 1.0), (4, 8.0), (10, 8.0)],
)
def test_uses_capped_exponential_backoff(
    monkeypatch: pytest.MonkeyPatch,
    retry_number: int,
    expected: float,
) -> None:
    monkeypatch.setattr("llmify.retries.random.uniform", lambda _start, _end: 1.0)

    assert retry_delay(RetryableError("transient"), retry_number) == expected


def test_retry_event_exposes_attempt_numbers() -> None:
    event = RetryEvent(
        retry_number=2,
        max_retries=4,
        delay=1.0,
        error=RetryableError("transient"),
    )

    assert event.failed_attempt == 2
    assert event.next_attempt == 3
    assert event.max_attempts == 5


async def test_async_hook_gets_retry_after_and_next_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0
    events: list[RetryEvent] = []

    async def no_sleep(_delay: float) -> None:
        pass

    async def on_retry(event: RetryEvent) -> None:
        events.append(event)

    async def operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RateLimitError(retry_after=3.4)
        return "ok"

    monkeypatch.setattr("llmify.retries.asyncio.sleep", no_sleep)

    assert await retry_call(operation, max_retries=1, on_retry=on_retry) == "ok"
    assert attempts == 2
    assert len(events) == 1
    assert events[0].delay == 3.4
    assert events[0].failed_attempt == 1
    assert events[0].next_attempt == 2
    assert events[0].max_attempts == 2
    assert events[0].error.code == "model_rate_limited"


async def test_exhausted_budget_has_no_extra_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0
    events: list[RetryEvent] = []

    async def no_sleep(_delay: float) -> None:
        pass

    async def on_retry(event: RetryEvent) -> None:
        events.append(event)

    async def operation() -> None:
        nonlocal attempts
        attempts += 1
        raise RetryableError("still unavailable")

    monkeypatch.setattr("llmify.retries.asyncio.sleep", no_sleep)

    with pytest.raises(RetryableError, match="still unavailable"):
        await retry_call(operation, max_retries=2, on_retry=on_retry)

    assert attempts == 3
    assert [event.next_attempt for event in events] == [2, 3]
