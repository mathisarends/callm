import asyncio

from dotenv import load_dotenv

from llmify import (
    ChatOpenAIResponses,
    Message,
    ModelTool,
    SystemMessage,
    ToolResultMessage,
    UserMessage,
)

load_dotenv(override=True)

POPULATIONS = {"berlin": 3_878_000, "vienna": 2_005_000, "zurich": 421_000}

CITY_POPULATION = ModelTool(
    name="city_population",
    description="Look up how many people live in a city.",
    parameters={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
)


async def main() -> None:
    messages: list[Message] = [
        SystemMessage(content="Use the tools before answering."),
        UserMessage(content="How many more people live in Berlin than in Zurich?"),
    ]

    async with ChatOpenAIResponses("gpt-5.6") as model:
        while True:
            response = await model(messages, tools=[CITY_POPULATION])
            messages.append(response.as_assistant_message())

            if not response.tool_calls:
                print(response.completion)
                return

            for call in response.tool_calls:
                print(f"  → {call.name}({call.arguments})")
                city = call.parsed_arguments["city"]
                messages.append(
                    ToolResultMessage(
                        tool_call_id=call.id,
                        tool_name=call.name,
                        content=str(POPULATIONS[city.lower()]),
                    )
                )


if __name__ == "__main__":
    asyncio.run(main())
