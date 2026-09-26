"""Structured output: hand `call` a Pydantic model, get one back."""

import asyncio

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from llmify import ChatOpenAI, UserMessage

load_dotenv(override=True)


class Recipe(BaseModel):
    name: str
    minutes: int = Field(description="Total time from start to serving")
    ingredients: list[str]


async def main() -> None:
    async with ChatOpenAI("gpt-5.6") as model:
        response = await model(
            [UserMessage(content="Give me a recipe for pancakes.")],
            output_format=Recipe,
        )

    recipe = response.completion
    print(f"{recipe.name} ({recipe.minutes} min)")
    for ingredient in recipe.ingredients:
        print(f"  - {ingredient}")


if __name__ == "__main__":
    asyncio.run(main())
