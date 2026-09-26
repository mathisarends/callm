"""Codex: talking to the endpoint with a ChatGPT subscription.

Borrows the Codex CLI's login (`codex login`) read-only: refreshed tokens live
only as long as the process. To persist them, pass a `credential_source`; see
pydantic-ai's docs on persisting Codex credentials.

This is a reverse-engineered endpoint. OpenAI neither documents nor supports it.
"""

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
