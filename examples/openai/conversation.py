import asyncio

from dotenv import load_dotenv
from pydantic import BaseModel

from llmify import ChatOpenAIResponses, Message, SystemMessage, UserMessage

load_dotenv(override=True)


class City(BaseModel):
    name: str
    country: str


async def main() -> None:
    messages: list[Message] = [SystemMessage(content="Answer briefly.")]

    async with ChatOpenAIResponses("gpt-5.6", reasoning_effort="low") as model:
        # A structured turn and a plain turn share one history.
        messages.append(UserMessage(content="Pick a city in Japan."))
        picked = await model(messages, output_format=City)
        messages.append(picked.as_assistant_message())
        print(f"picked: {picked.completion.name}, {picked.completion.country}")

        messages.append(UserMessage(content="What is it famous for? One sentence."))
        answer = await model(messages)
        messages.append(answer.as_assistant_message())
        print(f"famous for: {answer.completion.strip()}")


if __name__ == "__main__":
    asyncio.run(main())
