import asyncio

from llmify import (
    ChatCodex,
    CredentialsUnavailableError,
    ModelEventType,
    ModelTool,
    UserMessage,
)

WEATHER = ModelTool(
    name="weather",
    description="Current weather for a city.",
    parameters={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
)


async def main() -> None:
    try:
        model = ChatCodex("gpt-6-sol", reasoning_effort="low")
    except CredentialsUnavailableError as error:
        print(error)
        return

    async with model:
        async for event in model.stream(
            [UserMessage(content="Say hello, then check the weather in Oslo.")],
            tools=[WEATHER],
        ):
            match event.type:
                case ModelEventType.TEXT_DELTA:
                    print(event.delta, end="", flush=True)
                case ModelEventType.TOOL_CALL:
                    call = event.tool_call
                    print(f"\n-> {call.name}({call.arguments})")
                case ModelEventType.RESPONSE:
                    print(f"[{event.finish_reason}] {event.usage.total_tokens} tokens")


if __name__ == "__main__":
    asyncio.run(main())
