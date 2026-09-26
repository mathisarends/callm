"""Every provider, side by side.

They differ only in what it takes to reach them: past the constructor each one
is the same `ChatModel`, so the loop below does not care which it is holding.
"""

import asyncio

from dotenv import load_dotenv

from llmify import (
    ChatAzureOpenAI,
    ChatModel,
    ChatOpenAI,
    ChatOpenAIResponses,
    OpenAICompatible,
    UserMessage,
)

load_dotenv(override=True)


def every_provider() -> dict[str, ChatModel]:
    return {
        # api_key defaults to OPENAI_API_KEY
        "openai": ChatOpenAI("gpt-5.6"),
        # the Responses API: preferred for reasoning models
        "openai-responses": ChatOpenAIResponses("gpt-5.6", reasoning_effort="low"),
        # model is the deployment name; endpoint defaults to AZURE_OPENAI_ENDPOINT
        "azure": ChatAzureOpenAI("my-deployment", api_version="2024-10-01"),
        # anything else that speaks OpenAI's Chat Completions API
        "local": OpenAICompatible(
            "llama-3.3-70b", base_url="http://localhost:11434/v1"
        ),
    }


async def main() -> None:
    question = [UserMessage(content="Name one city. Just the name.")]

    for name, model in every_provider().items():
        async with model:
            try:
                response = await model(question)
            except Exception as error:  # noqa: BLE001 - a demo, not a library
                print(f"{name:<18} skipped: {type(error).__name__}")
            else:
                print(f"{name:<18} {response.completion.strip()}")


if __name__ == "__main__":
    asyncio.run(main())
