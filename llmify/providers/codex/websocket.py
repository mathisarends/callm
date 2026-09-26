import asyncio
import time
from collections.abc import AsyncIterator, Generator, Mapping
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from importlib.util import find_spec
from typing import Any, Protocol, Self, cast

from openai import AsyncOpenAI, NotGiven, Omit, OpenAIError
from openai.resources.responses import AsyncResponses
from openai.resources.responses.responses import AsyncResponsesConnection
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseFailedEvent,
    ResponseIncompleteEvent,
    ResponseStreamEvent,
)
from openai.types.responses.responses_client_event_param import ResponseCreate
from openai.types.responses.responses_server_event import ResponseWsError
from pydantic_ai.providers.openai_codex import (
    OpenAICodexCredentials,
    OpenAICodexProvider,
)

_COOLDOWN_SECONDS = 30.0
_ORIGINATOR = "pydantic-ai"
_TERMINAL_EVENTS = (
    ResponseCompletedEvent,
    ResponseFailedEvent,
    ResponseIncompleteEvent,
)
_HTTP_ONLY_PARAMETERS = frozenset(
    {"background", "extra_body", "extra_headers", "extra_query", "stream", "timeout"}
)


class WebSocketUnavailable(OpenAIError):
    """Raised when HTTP can safely retry a WebSocket request."""


class WebSocketInterrupted(OpenAIError):
    """Raised when a started response loses its WebSocket connection."""


class _ResponsesClient(Protocol):
    responses: AsyncResponses


class CodexResponsesResource(AsyncResponses):
    """OpenAI Responses resource preferring a persistent Codex WebSocket."""

    def __init__(self, client: AsyncOpenAI, provider: OpenAICodexProvider) -> None:
        if find_spec("websockets") is None:
            raise ImportError(
                "transport='websocket' requires the 'websockets' package. "
                "Install it with: pip install 'py-llmify[websocket]'"
            )
        super().__init__(client)
        self._provider = provider
        self._connection: AsyncResponsesConnection | None = None
        self._connection_generation = 0
        self._connection_lock = asyncio.Lock()
        self._request_lock = asyncio.Lock()
        self._retry_websocket_at = 0.0
        self._http_only = ContextVar("codex_http_only", default=False)
        self._warmup_request = ContextVar("codex_warmup_request", default=False)

    @classmethod
    def install(cls, provider: OpenAICodexProvider) -> Self:
        """Route the provider's Responses calls through a new resource."""
        resource = cls(provider.client, provider)
        cast(_ResponsesClient, provider.client).responses = resource
        return resource

    @property
    def connection_generation(self) -> int | None:
        if self._connection is None:
            return None
        return self._connection_generation

    @property
    def http_only(self) -> bool:
        return self._http_only.get()

    def use_http(self) -> AbstractContextManager[None]:
        return _enabled(self._http_only)

    def warmup_request(self) -> AbstractContextManager[None]:
        return _enabled(self._warmup_request)

    async def create(self, **request: Any) -> Any:
        if self._http_only.get():
            return await super().create(**request)
        stream = _WebSocketResponseStream(
            self, request, warmup=self._warmup_request.get()
        )
        if request.get("stream") is True:
            return stream
        return await stream.final_response()

    async def ensure_connection(
        self, extra_headers: Mapping[str, str]
    ) -> AsyncResponsesConnection:
        if self._connection is not None:
            return self._connection
        if time.monotonic() < self._retry_websocket_at:
            raise WebSocketUnavailable(
                "Codex WebSocket is cooling down after a failure"
            )

        async with self._connection_lock:
            if self._connection is not None:
                return self._connection
            try:
                connection = await self._connect(extra_headers)
            except Exception as error:
                self._retry_websocket_at = time.monotonic() + _COOLDOWN_SECONDS
                raise WebSocketUnavailable(
                    f"Could not open Codex WebSocket: {error}"
                ) from error

            self._connection = connection
            self._connection_generation += 1
            self._retry_websocket_at = 0.0
            return connection

    async def discard_connection(self) -> None:
        async with self._connection_lock:
            connection, self._connection = self._connection, None
            if connection is not None:
                await connection.close()

    async def aclose(self) -> None:
        await self.discard_connection()

    async def _connect(
        self, extra_headers: Mapping[str, str]
    ) -> AsyncResponsesConnection:
        credentials, refresh = await self._provider._prepare_request_credentials()
        try:
            return await self._open(credentials, extra_headers)
        except Exception as error:
            if not _is_unauthorized(error):
                raise
        return await self._open(await refresh(), extra_headers)

    async def _open(
        self,
        credentials: OpenAICodexCredentials,
        extra_headers: Mapping[str, str],
    ) -> AsyncResponsesConnection:
        headers = _websocket_headers(credentials, extra_headers)
        return await self.connect(extra_headers=headers).enter()


class _WebSocketResponseStream:
    """One response over the shared connection, holding it until the end."""

    def __init__(
        self,
        resource: CodexResponsesResource,
        request: dict[str, Any],
        *,
        warmup: bool,
    ) -> None:
        self._resource = resource
        self._request = request
        self._warmup = warmup
        self._connection: AsyncResponsesConnection | None = None
        self._entered = False
        self._started = False
        self._finished = False

    async def final_response(self) -> Response:
        async with self:
            async for event in self:
                if isinstance(event, ResponseCompletedEvent):
                    return event.response
                if isinstance(event, (ResponseFailedEvent, ResponseIncompleteEvent)):
                    if self._warmup:
                        raise WebSocketUnavailable(
                            "Codex WebSocket warmup did not complete successfully"
                        )
                    return event.response
        raise WebSocketInterrupted(
            "Codex WebSocket ended without a terminal response event"
        )

    async def __aenter__(self) -> Self:
        await self._resource._request_lock.acquire()
        try:
            headers = _string_headers(self._request.get("extra_headers"))
            self._connection = await self._resource.ensure_connection(headers)
            await self._send(self._connection)
        except BaseException as error:
            await self._resource.discard_connection()
            self._resource._request_lock.release()
            if isinstance(error, Exception) and not isinstance(
                error, WebSocketUnavailable
            ):
                raise WebSocketUnavailable(
                    f"Could not start a Codex WebSocket response: {error}"
                ) from error
            raise
        self._entered = True
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        try:
            await self.close()
        finally:
            if self._entered:
                self._entered = False
                self._resource._request_lock.release()

    def __aiter__(self) -> AsyncIterator[ResponseStreamEvent]:
        return self

    async def __anext__(self) -> ResponseStreamEvent:
        if self._finished:
            raise StopAsyncIteration
        assert self._connection is not None
        try:
            event = await self._connection.recv()
        except Exception as error:
            raise await self._lost(f"Codex WebSocket failed: {error}") from error
        if isinstance(event, ResponseWsError):
            raise await self._lost(f"Codex WebSocket error: {event.error.message}")

        if event.type == "response.created":
            self._started = True
        if isinstance(event, _TERMINAL_EVENTS):
            self._finished = True
        return cast(ResponseStreamEvent, event)

    async def close(self) -> None:
        if not self._finished:
            await self._resource.discard_connection()
        self._finished = True

    async def _send(self, connection: AsyncResponsesConnection) -> None:
        request = _websocket_request(self._request)
        if self._warmup:
            # The SDK event type does not include Codex's warmup-only generate flag.
            await connection.send(
                cast(
                    ResponseCreate,
                    {"type": "response.create", **request, "generate": False},
                )
            )
        else:
            await connection.response.create(**request)

    async def _lost(self, message: str) -> OpenAIError:
        """Drop the broken connection; only an unstarted response may be replayed."""
        await self._resource.discard_connection()
        if self._started:
            return WebSocketInterrupted(message)
        return WebSocketUnavailable(message)


@contextmanager
def _enabled(flag: ContextVar[bool]) -> Generator[None]:
    token = flag.set(True)
    try:
        yield
    finally:
        flag.reset(token)


def _is_unauthorized(error: Exception) -> bool:
    response = getattr(error, "response", None)
    return getattr(response, "status_code", None) == 401


def _websocket_headers(
    credentials: OpenAICodexCredentials,
    extra_headers: Mapping[str, str],
) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {credentials.access_token}",
        "chatgpt-account-id": credentials.account_id,
        "originator": _ORIGINATOR,
        **{
            name: value
            for name, value in extra_headers.items()
            if name.lower() != "user-agent"
        },
    }


def _string_headers(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {
        name: header
        for name, header in value.items()
        if isinstance(name, str) and isinstance(header, str)
    }


def _websocket_request(request: Mapping[str, Any]) -> dict[str, Any]:
    websocket_request = {
        key: value
        for key, value in request.items()
        if key not in _HTTP_ONLY_PARAMETERS and not isinstance(value, (NotGiven, Omit))
    }
    extra_body = request.get("extra_body")
    if isinstance(extra_body, Mapping):
        websocket_request.update(extra_body)
    return websocket_request
