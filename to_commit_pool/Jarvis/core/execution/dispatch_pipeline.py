"""Async pipeline primitives for dispatcher tool routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable


@dataclass(frozen=True)
class DispatchRequest:
    tool_name: str
    args: dict[str, Any]
    rationale: str


Handler = Callable[[DispatchRequest], Awaitable[Any | None]]


class DispatchPipeline:
    def __init__(self, handlers: list[Handler] | tuple[Handler, ...]) -> None:
        self._handlers = tuple(handlers)

    async def run(self, request: DispatchRequest) -> Any | None:
        for handler in self._handlers:
            response = await handler(request)
            if response is not None:
                return response
        return None


__all__ = ["DispatchPipeline", "DispatchRequest", "Handler"]
