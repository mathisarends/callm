import asyncio
import time

from llmkit import (
    ChatCodex,
    CredentialsUnavailableError,
    Message,
    SystemMessage,
    Transport,
    TransportFallbackEvent,
    UserMessage,
)

QUESTIONS = [
    "Name one European capital. Just the name.",
    "Roughly how many people live there?",
    "And what river runs through it?",
]


async def show_fallback(event: TransportFallbackEvent) -> None:
    print(f"  WebSocket fallback during {event.phase}: {event.reason}")


async def conversation(transport: Transport) -> None:
    print(f"--- {transport}")
    messages: list[Message] = [SystemMessage(content="Answer briefly.")]

    # One model for the whole conversation: over "websocket" the first turn
    # opens the connection and every later turn reuses it.
    async with ChatCodex(
        "gpt-6-sol",
        transport=transport,
        on_transport_fallback=show_fallback,
    ) as model:
        if transport == "websocket":
            await model.prepare(messages)

        for index, question in enumerate(QUESTIONS):
            messages.append(UserMessage(content=question))
            started = time.perf_counter()

            response = await model.call(messages)
            messages.append(response.as_assistant_message())

            elapsed = time.perf_counter() - started
            print(f"{elapsed:5.2f}s  {response.completion.strip()}")

            # Prepare the conversation prefix while waiting for the next turn.
            if transport == "websocket" and index + 1 < len(QUESTIONS):
                await model.prepare(messages)


async def main() -> None:
    try:
        await conversation("websocket")
        await conversation("http")
    except CredentialsUnavailableError as error:
        print(error)


if __name__ == "__main__":
    asyncio.run(main())
