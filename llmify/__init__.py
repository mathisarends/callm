from typing import TYPE_CHECKING, Any

from .errors import (
    AuthenticationError,
    ContextLengthExceededError,
    CredentialsUnavailableError,
    LLMifyError,
    ModelBehaviorError,
    OutOfCreditsError,
    RateLimitError,
    RetryableError,
)
from .base import (
    AssistantMessage,
    ChatModel,
    ImageUrl,
    Message,
    MessageType,
    ModelEvent,
    ModelEventType,
    ModelResponse,
    ModelTool,
    SystemMessage,
    TextDelta,
    ThinkingDelta,
    ToolCall,
    ToolCallEvent,
    ToolChoice,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from .providers import _MODULE_BY_NAME as _PROVIDER_NAMES
from .retries import RetryCallback, RetryEvent

if TYPE_CHECKING:
    # Redundant aliases: these are re-exports, resolved at runtime by __getattr__.
    from .providers import ChatAzureOpenAI as ChatAzureOpenAI
    from .providers import ChatAzureOpenAIResponses as ChatAzureOpenAIResponses
    from .providers import ChatCodex as ChatCodex
    from .providers import ChatOpenAI as ChatOpenAI
    from .providers import ChatOpenAIResponses as ChatOpenAIResponses
    from .providers import OpenAICodexCredentials as OpenAICodexCredentials
    from .providers import OpenAICompatible as OpenAICompatible
    from .providers import ReasoningEffort as ReasoningEffort


def __getattr__(name: str) -> Any:
    if name in _PROVIDER_NAMES:
        from . import providers

        return getattr(providers, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    # PEP 562: without this, the lazily exported names are invisible to dir()
    # and therefore to REPL and IDE completion.
    return sorted({*globals(), *__all__})


__all__ = [
    "AssistantMessage",
    "AuthenticationError",
    "ChatAzureOpenAI",
    "ChatAzureOpenAIResponses",
    "ChatCodex",
    "ChatModel",
    "ChatOpenAI",
    "ChatOpenAIResponses",
    "ContextLengthExceededError",
    "CredentialsUnavailableError",
    "ImageUrl",
    "LLMifyError",
    "Message",
    "MessageType",
    "ModelBehaviorError",
    "ModelEvent",
    "ModelEventType",
    "ModelResponse",
    "ModelTool",
    "OpenAICodexCredentials",
    "OpenAICompatible",
    "OutOfCreditsError",
    "RateLimitError",
    "ReasoningEffort",
    "RetryCallback",
    "RetryEvent",
    "RetryableError",
    "SystemMessage",
    "TextDelta",
    "ThinkingDelta",
    "ToolCall",
    "ToolCallEvent",
    "ToolChoice",
    "ToolResultMessage",
    "Usage",
    "UserMessage",
]
