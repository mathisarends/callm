"""Codex: talking to the endpoint with a ChatGPT subscription.

`from_cli` borrows the login the Codex CLI already has (`codex login`) and
writes refreshed tokens back to `~/.codex/auth.json`, so the CLI and llmify go
on sharing one session. Refresh tokens are single-use, which is why the
write-back matters.

This is a reverse-engineered endpoint. OpenAI neither documents nor supports it.
"""

import asyncio

from llmify import ChatCodex, CredentialsUnavailableError, UserMessage


async def main() -> None:
    try:
        model = ChatCodex.from_cli("gpt-5.6-terra", reasoning_effort="high")
    except CredentialsUnavailableError as error:
        print(error)
        return

    async with model:
        response = await model([UserMessage(content="What is 2+2?")])

    print(response.completion)


if __name__ == "__main__":
    asyncio.run(main())
