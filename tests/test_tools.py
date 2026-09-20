from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, Field

from llmify.ports import ModelTool, ToolCall
from llmify.tools import FunctionTool, tool


class Filters(BaseModel):
    since: int | None = None


@tool
def search_web(
    query: Annotated[str, Field(description="What to look for")],
    max_results: int = 10,
    mode: Literal["fast", "deep"] = "fast",
) -> str:
    """Search the web for information."""
    return f"{query}/{max_results}/{mode}"


def test_name_and_description_come_from_the_function() -> None:
    assert search_web.name == "search_web"
    assert search_web.description == "Search the web for information."


def test_schema_describes_every_parameter() -> None:
    schema = search_web.parameters

    assert schema["type"] == "object"
    assert schema["required"] == ["query"]
    assert schema["properties"]["query"]["description"] == "What to look for"
    assert schema["properties"]["max_results"]["default"] == 10
    assert schema["properties"]["mode"]["enum"] == ["fast", "deep"]


def test_nested_models_reach_the_schema() -> None:
    @tool
    def refine(filters: Filters) -> str:
        return "ok"

    assert "$defs" in refine.parameters
    assert refine.parameters["$defs"]["Filters"]["properties"]["since"]


def test_decorator_takes_overrides() -> None:
    @tool(name="add", description="Add two numbers")
    def _add(a: int, b: int) -> int:
        return a + b

    assert (_add.name, _add.description) == ("add", "Add two numbers")


def test_the_wrapped_function_stays_callable() -> None:
    @tool
    def double(value: int) -> int:
        return value * 2

    assert double(21) == 42


def test_a_function_tool_is_a_model_tool() -> None:
    assert isinstance(search_web, ModelTool)
    assert "fn" not in search_web.model_dump()


async def test_execute_runs_the_call_and_reports_the_result() -> None:
    call = ToolCall(id="c1", name="search_web", arguments='{"query": "llmify"}')

    result = await search_web.execute(call)

    assert (result.tool_call_id, result.tool_name) == ("c1", "search_web")
    assert result.content == "llmify/10/fast"
    assert result.is_error is False


async def test_execute_awaits_async_functions() -> None:
    @tool
    async def fetch(url: str) -> str:
        return f"body of {url}"

    result = await fetch.execute(
        ToolCall(id="c2", name="fetch", arguments='{"url": "x"}')
    )

    assert result.content == "body of x"


async def test_execute_serialises_non_string_results() -> None:
    @tool
    def stats() -> dict[str, int]:
        return {"hits": 2}

    result = await stats.execute(ToolCall(id="c3", name="stats"))

    assert result.content == '{"hits": 2}'


async def test_execute_hands_a_failure_back_to_the_model() -> None:
    @tool
    def explode() -> str:
        raise RuntimeError("no luck")

    result = await explode.execute(ToolCall(id="c4", name="explode"))

    assert result.is_error is True
    assert result.content == "RuntimeError: no luck"


async def test_execute_reports_unusable_arguments_as_a_tool_error() -> None:
    call = ToolCall(id="c5", name="search_web", arguments='{"unknown": 1}')

    result = await search_web.execute(call)

    assert result.is_error is True
    assert "TypeError" in result.content


def test_from_function_is_what_the_decorator_uses() -> None:
    def ping() -> str:
        """Say hello."""
        return "pong"

    built = FunctionTool.from_function(ping)

    assert built.name == "ping"
    assert built.description == "Say hello."
    assert built.parameters["properties"] == {}


def test_a_raw_schema_needs_no_helper() -> None:
    raw = ModelTool(
        name="lookup",
        description="Look something up",
        parameters={"type": "object", "properties": {"id": {"type": "string"}}},
    )

    assert raw.name == "lookup"
    with pytest.raises(ValueError):
        raw.name = "other"  # type: ignore[misc]
