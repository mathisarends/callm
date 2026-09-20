"""The providers llmify ships, each imported only when it is asked for.

Every provider's SDK arrives through a pydantic-ai extra, so importing this
package must not import all of them: someone who installed only
`py-llmify[anthropic]` still has to be able to `import llmify`.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Redundant aliases: these are re-exports, resolved at runtime by __getattr__.
    from .anthropic import ChatAnthropic as ChatAnthropic
    from .azure import ChatAzureOpenAI as ChatAzureOpenAI
    from .azure import ChatAzureOpenAIResponses as ChatAzureOpenAIResponses
    from .cerebras import ChatCerebras as ChatCerebras
    from .codex import ChatCodex as ChatCodex
    from .codex import CodexCliCredentials as CodexCliCredentials
    from .codex import OpenAICodexCredentials as OpenAICodexCredentials
    from .google import ChatGoogle as ChatGoogle
    from .openai import ChatOpenAI as ChatOpenAI
    from .openai import ChatOpenAIResponses as ChatOpenAIResponses
    from .openai import OpenAICompatible as OpenAICompatible
    from .openai import ReasoningEffort as ReasoningEffort

_MODULE_BY_NAME = {
    "ChatAnthropic": "anthropic",
    "ChatAzureOpenAI": "azure",
    "ChatAzureOpenAIResponses": "azure",
    "ChatCerebras": "cerebras",
    "ChatCodex": "codex",
    "ChatGoogle": "google",
    "ChatOpenAI": "openai",
    "ChatOpenAIResponses": "openai",
    "CodexCliCredentials": "codex",
    "OpenAICodexCredentials": "codex",
    "OpenAICompatible": "openai",
    "ReasoningEffort": "openai",
}


def __getattr__(name: str) -> Any:
    module = _MODULE_BY_NAME.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f".{module}", __name__), name)


def __dir__() -> list[str]:
    # PEP 562: without this, the lazily exported names are invisible to dir()
    # and therefore to REPL and IDE completion.
    return sorted({*globals(), *__all__})


__all__ = [*_MODULE_BY_NAME]
