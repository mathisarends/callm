import asyncio
import time
from typing import Literal

from llmify import ChatCodex, CredentialsUnavailableError, Message, UserMessage

QUESTIONS = [
    "Name one European capital. Just the name.",
    "Roughly how many people live there?",
    "And what river runs through it?",
]


async def conversation(transport: Literal["websocket", "http"]) -> None:
    print(f"--- {transport}")
    messages: list[Message] = []

    # One model for the whole conversation: over "websocket" the first turn
    # opens the connection and every later turn reuses it.
    async with ChatCodex("gpt-5.6-terra", transport=transport) as model:
        for question in QUESTIONS:
            messages.append(UserMessage(content=question))
            started = time.perf_counter()

            response = await model(messages)
            messages.append(response.as_assistant_message())

            elapsed = time.perf_counter() - started
            print(f"{elapsed:5.2f}s  {response.completion.strip()}")


async def main() -> None:
    try:
        await conversation("websocket")
        await conversation("http")
    except CredentialsUnavailableError as error:
        print(error)


if __name__ == "__main__":
    asyncio.run(main())
