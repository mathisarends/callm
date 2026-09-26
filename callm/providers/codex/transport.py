from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

type Transport = Literal["websocket", "http"]
type TransportPhase = Literal["prepare", "call", "stream"]


@dataclass(frozen=True, slots=True)
class TransportFallbackEvent:
    """A WebSocket request that was answered over HTTP instead."""

    phase: TransportPhase
    reason: str


type TransportFallbackCallback = Callable[[TransportFallbackEvent], Awaitable[None]]
