import asyncio

from llmify import ChatCodex, CredentialsUnavailableError, UserMessage


async def main() -> None:
    try:
        model = ChatCodex("gpt-6-sol", reasoning_effort="high")
    except CredentialsUnavailableError as error:
        print(error)
        return

    async with model:
        response = await model.call([UserMessage(content="What is 2+2?")])

    print(response.completion)


if __name__ == "__main__":
    asyncio.run(main())
