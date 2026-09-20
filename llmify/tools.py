"""Turning Python functions into tools a model can call.

A raw schema needs nothing from this module: build a
[`ModelTool`][llmify.ports.ModelTool] directly. This is for the common case
where the function already says what it takes.
"""

import inspect
import json
from collections.abc import Callable
from typing import Any, Self, overload

from pydantic import ConfigDict, Field, TypeAdapter

from llmify.ports import ModelTool, ToolCall, ToolResultMessage

type AnyCallable = Callable[..., Any]


class FunctionTool(ModelTool):
    """A tool backed by a Python function, able to run the calls it receives."""

    model_config = ConfigDict(frozen=True)

    fn: AnyCallable = Field(repr=False, exclude=True)

    @classmethod
    def from_function(
        cls,
        fn: AnyCallable,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> Self:
        """Derive name, description and JSON schema from the function itself.

        The schema comes from Pydantic, so anything Pydantic understands as a
        parameter type — nested models, literals, unions, `Annotated` field
        descriptions — is understood here too.
        """
        return cls(
            name=name or fn.__name__,
            description=description or inspect.getdoc(fn) or "",
            parameters=TypeAdapter(fn).json_schema(),
            fn=fn,
        )

    async def execute(self, call: ToolCall) -> ToolResultMessage:
        """Run one tool call, turning both its result and its failure into a message.

        A raised exception becomes a tool result flagged `is_error`, because the
        model is the one that has to recover from it: it asked for this call, and
        an exception that escapes here ends the conversation instead.
        """
        try:
            result = self.fn(**call.parsed_arguments)
            if inspect.isawaitable(result):
                result = await result
        except Exception as error:
            return ToolResultMessage(
                tool_call_id=call.id,
                tool_name=self.name,
                content=f"{type(error).__name__}: {error}",
                is_error=True,
            )

        return ToolResultMessage(
            tool_call_id=call.id,
            tool_name=self.name,
            content=_as_text(result),
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the wrapped function, so `@tool` does not take it away."""
        return self.fn(*args, **kwargs)


def _as_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result)
    except TypeError:
        return str(result)


@overload
def tool(fn: AnyCallable) -> FunctionTool: ...


@overload
def tool(
    fn: None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Callable[[AnyCallable], FunctionTool]: ...


def tool(
    fn: AnyCallable | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> FunctionTool | Callable[[AnyCallable], FunctionTool]:
    """Make a function available to the model, as `@tool` or `@tool(name=...)`."""

    def decorator(func: AnyCallable) -> FunctionTool:
        return FunctionTool.from_function(func, name=name, description=description)

    if fn is None:
        return decorator
    return decorator(fn)
