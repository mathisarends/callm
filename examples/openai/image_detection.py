import asyncio
import base64
import mimetypes
import sys
from pathlib import Path

from dotenv import load_dotenv

from llmify import ChatOpenAI, ImageUrl, UserMessage

load_dotenv(override=True)

DEFAULT_IMAGE = Path(__file__).parents[2] / "static" / "banner.png"


def data_uri(path: Path) -> str:
    media_type = mimetypes.guess_type(path)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode()
    return f"data:{media_type};base64,{encoded}"


async def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IMAGE
    # A public https:// URL works the same way: ImageUrl(url="https://...")
    image = ImageUrl(url=data_uri(path), detail="high")

    async with ChatOpenAI("gpt-5.6") as model:
        response = await model([UserMessage(content=("What is in this image?", image))])

    print(response.completion)


if __name__ == "__main__":
    asyncio.run(main())
