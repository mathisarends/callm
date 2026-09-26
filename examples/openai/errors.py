import asyncio

from dotenv import load_dotenv

from llmify import (
    AuthenticationError,
    ChatOpenAI,
    LLMifyError,
    RetryEvent,
    UserMessage,
)

load_dotenv(override=True)


async def log_retry(event: RetryEvent) -> None:
    print(
        f"  attempt {event.failed_attempt}/{event.max_attempts} failed "
        f"({event.error.code}), retrying in {event.delay:.1f}s"
    )


async def main() -> None:
    question = [UserMessage(content="Hi")]

    # Rejected credentials are not retried.
    async with ChatOpenAI("gpt-6-sol", api_key="sk-invalid") as model:
        try:
            await model.call(question)
        except AuthenticationError as error:
            print(f"{error.code}: {error.user_message}")

    # An unreachable endpoint is retried, and every retry is reported.
    async with ChatOpenAI(
        "gpt-6-sol",
        base_url="http://127.0.0.1:9/v1",
        max_retries=2,
        on_retry=log_retry,
    ) as model:
        try:
            await model.call(question)
        except LLMifyError as error:
            print(f"{error.code} (retryable={error.retryable}): {error.user_message}")


if __name__ == "__main__":
    asyncio.run(main())
