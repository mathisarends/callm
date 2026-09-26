import asyncio

from dotenv import load_dotenv

from llmkit import ChatOpenAI, SystemMessage, UserMessage

load_dotenv(override=True)


async def main() -> None:
    async with ChatOpenAI("gpt-6-sol") as model:
        response = await model.call(
            [
                SystemMessage(content="You are a helpful assistant."),
                UserMessage(content="What is 2+2?"),
            ]
        )

    print(response.completion)
    print(f"{response.usage.total_tokens} tokens")


if __name__ == "__main__":
    asyncio.run(main())
