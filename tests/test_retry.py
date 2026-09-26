import pytest

from llmify.errors import RateLimitError, RetryableError
from llmify.retries import RetryEvent, retry_delay


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
