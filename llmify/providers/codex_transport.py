import asyncio
import time
from collections.abc import AsyncIterator, Mapping
from contextlib import AbstractAsyncContextManager
from contextvars import ContextVar, Token
from typing import Any, Protocol, Self, cast

from openai import AsyncOpenAI, NotGiven, Omit, OpenAIError
from openai.resources.responses import AsyncResponses
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseFailedEvent,
    ResponseIncompleteEvent,
    ResponseStreamEvent,
)
from openai.types.responses.responses_server_event import ResponseWsError
from pydantic_ai.providers.openai_codex import (
    OpenAICodexCredentials,
    OpenAICodexProvider,
)
from websockets.exceptions import InvalidStatus

_HTTP_COOLDOWN_SECONDS = 30.0
_ORIGINATOR = "pydantic-ai"


class WebSocketUnavailable(OpenAIError):
    """Raised when HTTP can safely retry a WebSocket request."""


class WebSocketInterrupted(OpenAIError):
    """Raised when a started response loses its WebSocket connection."""


class _ResponsesClient(Protocol):
    responses: AsyncResponses


class CodexResponsesResource(AsyncResponses):
    """OpenAI Responses resource preferring a persistent Codex WebSocket."""

    def __init__(
        self,
        client: AsyncOpenAI,
        provider: OpenAICodexProvider,
    ) -> None:
        super().__init__(client)
        self._provider = provider
        self._connection: Any | None = None
        self._connection_manager: AbstractAsyncContextManager[Any] | None = None
        self._connection_lock = asyncio.Lock()
        self._request_lock = asyncio.Lock()
        self._http_only = ContextVar("codex_http_only", default=False)
        self._warmup_request = ContextVar("codex_warmup_request", default=False)
        self._connection_generation = 0
        self._retry_websocket_at = 0.0

    async def create(self, **request: Any) -> Any:
        if self._http_only.get():
            return await super().create(**request)
        stream = _WebSocketResponseStream(
            self,
            request,
            request_lock=self._request_lock,
            warmup=self._warmup_request.get(),
        )
        if request.get("stream") is True:
            return stream
        return await self._complete_response(stream)

    async def prewarm(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        await self._ensure_connection(extra_headers or {})

    async def aclose(self) -> None:
        await self._discard_connection()

    def use_http(self) -> AbstractAsyncContextManager[None]:
        return _HTTPOnly(self._http_only)

    def warmup_request(self) -> AbstractAsyncContextManager[None]:
        return _WarmupRequest(self._warmup_request)

    @property
    def connection_generation(self) -> int | None:
        if self._connection is None:
            return None
        return self._connection_generation

    async def _complete_response(self, stream: "_WebSocketResponseStream") -> Any:
        async with stream:
            async for event in stream:
                if isinstance(event, ResponseCompletedEvent):
                    return event.response
                if isinstance(event, (ResponseFailedEvent, ResponseIncompleteEvent)):
                    if stream.is_warmup:
                        raise WebSocketUnavailable(
                            "Codex WebSocket warmup did not complete successfully"
                        )
                    return event.response
        raise WebSocketInterrupted(
            "Codex WebSocket ended without a terminal response event"
        )

    async def _ensure_connection(
        self,
        extra_headers: Mapping[str, str],
    ) -> Any:
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
                (
                    credentials,
                    replay,
                ) = await self._provider._prepare_request_credentials()
                manager = self.connect(
                    extra_headers=_websocket_headers(credentials, extra_headers),
                )
                try:
                    connection = await manager.__aenter__()
                except InvalidStatus as error:
                    if error.response.status_code != 401:
                        raise
                    refreshed = await replay()
                    manager = self.connect(
                        extra_headers=_websocket_headers(refreshed, extra_headers),
                    )
                    connection = await manager.__aenter__()
            except Exception as error:
                self._retry_websocket_at = time.monotonic() + _HTTP_COOLDOWN_SECONDS
                raise WebSocketUnavailable(
                    f"Could not open Codex WebSocket: {error}"
                ) from error

            self._connection_manager = manager
            self._connection = connection
            self._connection_generation += 1
            self._retry_websocket_at = 0.0
            return connection

    async def _discard_connection(self) -> None:
        async with self._connection_lock:
            manager = self._connection_manager
            connection = self._connection
            self._connection_manager = None
            self._connection = None
            if manager is not None:
                await manager.__aexit__(None, None, None)
            elif connection is not None:
                await connection.close()


class _HTTPOnly:
    def __init__(self, state: ContextVar[bool]) -> None:
        self._state = state
        self._token: Token[bool] | None = None

    async def __aenter__(self) -> None:
        self._token = self._state.set(True)

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        assert self._token is not None
        self._state.reset(self._token)


class _WarmupRequest:
    def __init__(self, state: ContextVar[bool]) -> None:
        self._state = state
        self._token: Token[bool] | None = None

    async def __aenter__(self) -> None:
        self._token = self._state.set(True)

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        assert self._token is not None
        self._state.reset(self._token)


class _WebSocketResponseStream:
    def __init__(
        self,
        resource: CodexResponsesResource,
        request: dict[str, Any],
        *,
        request_lock: asyncio.Lock,
        warmup: bool,
    ) -> None:
        self._resource = resource
        self._request = request
        self._request_lock = request_lock
        self._warmup = warmup
        self._connection: Any | None = None
        self._entered = False
        self._started = False
        self._finished = False

    @property
    def is_warmup(self) -> bool:
        return self._warmup

    async def __aenter__(self) -> Self:
        await self._request_lock.acquire()
        try:
            headers = _string_headers(self._request.get("extra_headers"))
            self._connection = await self._resource._ensure_connection(headers)
            websocket_request = _websocket_request(self._request)
            if self._warmup:
                await self._connection.send(
                    {
                        "type": "response.create",
                        **websocket_request,
                        "generate": False,
                    }
                )
            else:
                await self._connection.response.create(**websocket_request)
            self._entered = True
            return self
        except BaseException as error:
            await self._resource._discard_connection()
            self._request_lock.release()
            if isinstance(error, asyncio.CancelledError):
                raise
            if isinstance(error, WebSocketUnavailable):
                raise
            if isinstance(error, Exception):
                raise WebSocketUnavailable(
                    f"Could not start a Codex WebSocket response: {error}"
                ) from error
            raise

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        try:
            if not self._finished:
                await self._resource._discard_connection()
        finally:
            if self._entered:
                self._entered = False
                self._request_lock.release()

    def __aiter__(self) -> AsyncIterator[ResponseStreamEvent]:
        return self

    async def __anext__(self) -> ResponseStreamEvent:
        if self._finished:
            raise StopAsyncIteration
        assert self._connection is not None
        try:
            event = await self._connection.recv()
        except Exception as error:
            await self._resource._discard_connection()
            if self._started:
                raise WebSocketInterrupted(
                    f"Codex WebSocket closed during a response: {error}"
                ) from error
            raise WebSocketUnavailable(
                f"Codex WebSocket failed before response creation: {error}"
            ) from error

        if isinstance(event, ResponseWsError):
            await self._resource._discard_connection()
            message = f"Codex WebSocket error: {event.error.message}"
            if self._started:
                raise WebSocketInterrupted(message)
            raise WebSocketUnavailable(message)

        if event.type == "response.created":
            self._started = True
        if isinstance(
            event,
            (ResponseCompletedEvent, ResponseFailedEvent, ResponseIncompleteEvent),
        ):
            self._finished = True
        return cast(ResponseStreamEvent, event)

    async def close(self) -> None:
        if not self._finished:
            await self._resource._discard_connection()
        self._finished = True


def install_codex_responses_resource(
    provider: OpenAICodexProvider,
) -> CodexResponsesResource:
    resource = CodexResponsesResource(provider.client, provider)
    responses_client = cast(_ResponsesClient, provider.client)
    responses_client.responses = resource
    return resource


def _websocket_headers(
    credentials: OpenAICodexCredentials,
    extra_headers: Mapping[str, str],
) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {credentials.access_token}",
        "chatgpt-account-id": credentials.account_id,
        "originator": _ORIGINATOR,
    }
    headers.update(
        (key, value)
        for key, value in extra_headers.items()
        if key.lower() != "user-agent"
    )
    return headers


def _string_headers(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {
        key: header
        for key, header in value.items()
        if isinstance(key, str) and isinstance(header, str)
    }


def _websocket_request(request: Mapping[str, Any]) -> dict[str, Any]:
    websocket_request = {
        key: value
        for key, value in request.items()
        if key
        not in {
            "background",
            "extra_body",
            "extra_headers",
            "extra_query",
            "stream",
            "timeout",
        }
        and not isinstance(value, (NotGiven, Omit))
    }
    extra_body = request.get("extra_body")
    if isinstance(extra_body, Mapping):
        websocket_request.update(extra_body)
    return websocket_request
