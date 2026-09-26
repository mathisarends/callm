import asyncio

from dotenv import load_dotenv

from llmify import ChatOpenAIResponses, ModelEventType, UserMessage

load_dotenv(override=True)


async def main() -> None:
    model = ChatOpenAIResponses("gpt-5.6", reasoning_effort="low")

    async with model:
        async for event in model.stream(
            [UserMessage(content="Explain prompt caching in two sentences.")]
        ):
            match event.type:
                case ModelEventType.THINKING_DELTA:
                    print(f"\033[2m{event.delta}\033[0m", end="", flush=True)
                case ModelEventType.TEXT_DELTA:
                    print(event.delta, end="", flush=True)
                case ModelEventType.RESPONSE:
                    print(
                        f"\n\n[{event.finish_reason}] {event.usage.total_tokens} tokens"
                    )


if __name__ == "__main__":
    asyncio.run(main())
