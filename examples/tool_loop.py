"""A complete tool loop.

The whole loop is: ask, append the turn, run whatever the model called, repeat.
`as_assistant_message` keeps the provider's own state on the history, so a
reasoning model does not lose its train of thought across the round-trip.
"""

import asyncio

from dotenv import load_dotenv

from llmify import ChatOpenAIResponses, Message, SystemMessage, UserMessage, tool

load_dotenv(override=True)

POPULATIONS = {"berlin": 3_878_000, "vienna": 2_005_000, "zurich": 421_000}


@tool
def city_population(city: str) -> int:
    """Look up how many people live in a city."""
    return POPULATIONS[city.lower()]


async def main() -> None:
    tools = [city_population]
    by_name = {each.name: each for each in tools}
    messages: list[Message] = [
        SystemMessage(content="Use the tools before answering."),
        UserMessage(content="How many more people live in Berlin than in Zurich?"),
    ]

    async with ChatOpenAIResponses("gpt-5.6") as model:
        while True:
            response = await model(messages, tools=tools)
            messages.append(response.as_assistant_message())

            if not response.tool_calls:
                print(response.completion)
                return

            for call in response.tool_calls:
                print(f"  → {call.name}({call.arguments})")
                messages.append(await by_name[call.name].execute(call))


if __name__ == "__main__":
    asyncio.run(main())
