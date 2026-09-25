# llmify

A deliberately small async chat-model library built on
[Pydantic AI](https://ai.pydantic.dev/). It exposes exactly two providers:

- `ChatOpenAI`, always using the OpenAI Responses API
- `ChatCodex`, using the Codex subscription backend and the local Codex CLI login

The request API consists of `call()` and `stream()`. Both are asynchronous;
there is no synchronous API and no `invoke()` compatibility alias.

## Install

```bash
pip install py-llmify
```

WebSocket transport for `ChatOpenAI` is opt-in:

```bash
pip install "py-llmify[websocket]"
```

## Call

```python
from llmify import ChatOpenAI, SystemMessage, UserMessage

model = ChatOpenAI(model="your-openai-model")
response = await model.call(
    [
        SystemMessage(content="Answer concisely."),
        UserMessage(content="Why is the sky blue?"),
    ]
)
print(response.completion)
await model.aclose()
```

`call()` also accepts a plain string as shorthand for one `UserMessage`.

## Stream

```python
from llmify import ChatOpenAI, StreamEnd, StreamTextDelta

model = ChatOpenAI(model="your-openai-model")
async for event in model.stream("Write a haiku about Python."):
    if isinstance(event, StreamTextDelta):
        print(event.delta, end="", flush=True)
    elif isinstance(event, StreamEnd):
        print()
await model.aclose()
```

## Codex CLI login

```python
from llmify import ChatCodex

model = ChatCodex.from_cli(model="your-codex-model", reasoning_effort="high")
response = await model.call("Explain this repository.")
print(response.completion)
await model.aclose()
```

Pydantic AI reads the local Codex CLI credentials (honoring `CODEX_HOME`) and
handles refreshes. Run `codex login` first.

## WebSocket Responses

```python
from llmify import ChatOpenAI

async with ChatOpenAI(model="your-openai-model", transport="websocket") as model:
    response = await model.call("Why is a persistent transport useful for tool loops?")
    print(response.completion)
```

The connection is opened lazily and reused for sequential calls. A model
instance serializes WebSocket requests so events cannot leak between calls.
Conversation history remains explicit: pass prior messages again just like with
HTTP transport.

## Tool calls

Pass already-built OpenAI function schemas to either method. `llmify` returns
the model's tool calls; the application executes them and includes the results
in the next request.

```python
from llmify import AssistantMessage, ChatOpenAI, ToolResultMessage, UserMessage

model = ChatOpenAI(model="your-openai-model")

schema = {
    "type": "function",
    "function": {
        "name": "weather",
        "description": "Get the current weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}

messages = [UserMessage(content="What's the weather in Berlin?")]
response = await model.call(messages, tools=[schema])
if response.tool_calls:
    messages.append(AssistantMessage(content=response.completion, tool_calls=response.tool_calls))
    messages.append(
        ToolResultMessage(
            tool_call_id=response.tool_calls[0].id,
            content="19 C and sunny",
        )
    )
    answer = await model.call(messages)
```
