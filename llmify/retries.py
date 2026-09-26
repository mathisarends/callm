import asyncio
import random
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Never

from llmify.errors import RateLimitError, RetryableError


@dataclass(frozen=True, slots=True)
class RetryEvent:
    retry_number: int
    max_retries: int
    delay: float
    error: RetryableError

    @property
    def failed_attempt(self) -> int:
        return self.retry_number

    @property
    def next_attempt(self) -> int:
        return self.retry_number + 1

    @property
    def max_attempts(self) -> int:
        return self.max_retries + 1


type RetryCallback = Callable[[RetryEvent], Awaitable[None]]
type ErrorMapper = Callable[[Exception], Exception]


def retry_delay(error: RetryableError, retry_number: int) -> float:
    if isinstance(error, RateLimitError) and error.retry_after is not None:
        return max(error.retry_after, 0.0)

    exponential_delay = min(0.5 * 2**retry_number, 8.0)
    return exponential_delay * random.uniform(0.75, 1.0)


async def sleep_before_retry(
    error: RetryableError,
    retry_number: int,
    max_retries: int,
    on_retry: RetryCallback | None,
) -> None:
    delay = retry_delay(error, retry_number)
    event = RetryEvent(
        retry_number=retry_number + 1,
        max_retries=max_retries,
        delay=delay,
        error=error,
    )

    if on_retry is not None:
        await on_retry(event)

    await asyncio.sleep(delay)


async def retry_call[T](
    operation: Callable[[], Awaitable[T]],
    *,
    max_retries: int,
    on_retry: RetryCallback | None = None,
    map_error: ErrorMapper | None = None,
) -> T:
    """Run an idempotent operation, retrying mapped transient failures."""
    for retry_number in range(max_retries + 1):
        try:
            return await operation()
        except Exception as exc:  # noqa: BLE001 - provider SDK errors vary
            error = map_error(exc) if map_error is not None else exc
            if not isinstance(error, RetryableError):
                _raise_mapped(error, exc)
            if retry_number == max_retries:
                _raise_mapped(error, exc)
            await sleep_before_retry(error, retry_number, max_retries, on_retry)

    raise RuntimeError("Retry loop exhausted without returning or raising.")


async def retry_stream[T](
    stream_factory: Callable[[], AsyncIterator[T]],
    *,
    max_retries: int,
    on_retry: RetryCallback | None = None,
    map_error: ErrorMapper | None = None,
) -> AsyncIterator[T]:
    """Retry a stream only while doing so cannot replay already emitted output."""
    for retry_number in range(max_retries + 1):
        emitted = False
        try:
            async for event in stream_factory():
                emitted = True
                yield event
            return
        except Exception as exc:  # noqa: BLE001 - provider SDK errors vary
            error = map_error(exc) if map_error is not None else exc
            if not isinstance(error, RetryableError):
                _raise_mapped(error, exc)
            if emitted or retry_number == max_retries:
                _raise_mapped(error, exc)
            await sleep_before_retry(error, retry_number, max_retries, on_retry)


def _raise_mapped(error: Exception, original: Exception) -> Never:
    if error is original:
        raise error
    raise error from original
