import asyncio
import base64
import sys
from pathlib import Path

from dotenv import load_dotenv

from llmify import ChatOpenAI, ImageUrl, UserMessage

load_dotenv(override=True)


def data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/png;base64,{encoded}"


async def main() -> None:
    url = (
        data_uri(Path(sys.argv[1]))
        if len(sys.argv) > 1
        else "https://upload.wikimedia.org/wikipedia/commons/2/2f/Culzean_Castle.jpg"
    )
    image = ImageUrl(url=url, media_type="image/png", detail="high")

    async with ChatOpenAI("gpt-5.6") as model:
        response = await model([UserMessage(content=("What is in this image?", image))])

    print(response.completion)


if __name__ == "__main__":
    asyncio.run(main())
