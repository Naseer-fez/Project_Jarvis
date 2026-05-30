"""Minimal async request pipeline for controller handlers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable


@dataclass(frozen=True)
class RequestContext:
    user_input: str
    text: str
    lowered: str
    trace_id: str


Handler = Callable[[RequestContext], Awaitable[str | None]]


class RequestPipeline:
    def __init__(self, handlers: list[Handler] | tuple[Handler, ...]) -> None:
        self._handlers = tuple(handlers)

    async def run(self, context: RequestContext) -> str | None:
        for handler in self._handlers:
            response = await handler(context)
            if response is not None:
                return response
        return None


__all__ = ["Handler", "RequestContext", "RequestPipeline"]
