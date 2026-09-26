import asyncio
from collections.abc import Callable

from dotenv import load_dotenv

from llmify import (
    ChatAzureOpenAI,
    ChatModel,
    ChatOpenAI,
    ChatOpenAIResponses,
    LLMifyError,
    OpenAICompatible,
    UserMessage,
)

load_dotenv(override=True)


def every_provider() -> dict[str, Callable[[], ChatModel]]:
    # Factories rather than instances: a provider without credentials raises
    # CredentialsUnavailableError as soon as it is constructed.
    return {
        # api_key defaults to OPENAI_API_KEY
        "openai": lambda: ChatOpenAI("gpt-5.6"),
        # the Responses API: preferred for reasoning models
        "openai-responses": lambda: ChatOpenAIResponses(
            "gpt-5.6", reasoning_effort="low"
        ),
        # model is the deployment name; endpoint defaults to AZURE_OPENAI_ENDPOINT
        "azure": lambda: ChatAzureOpenAI("my-deployment", api_version="2024-10-01"),
        # anything else that speaks OpenAI's Chat Completions API
        "local": lambda: OpenAICompatible(
            "llama-3.3-70b", base_url="http://localhost:11434/v1", max_retries=0
        ),
    }


async def main() -> None:
    question = [UserMessage(content="Name one city. Just the name.")]

    for name, make in every_provider().items():
        try:
            async with make() as model:
                response = await model(question)
        except LLMifyError as error:
            print(f"{name:<18} skipped: {type(error).__name__}")
        else:
            print(f"{name:<18} {response.completion.strip()}")


if __name__ == "__main__":
    asyncio.run(main())
