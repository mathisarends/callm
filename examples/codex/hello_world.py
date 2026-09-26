import asyncio

from llmify import ChatCodex, CredentialsUnavailableError, UserMessage


async def main() -> None:
    try:
        model = ChatCodex("gpt-5.6-terra", reasoning_effort="high")
    except CredentialsUnavailableError as error:
        print(error)
        return

    async with model:
        response = await model([UserMessage(content="What is 2+2?")])

    print(response.completion)


if __name__ == "__main__":
    asyncio.run(main())
