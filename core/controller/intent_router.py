from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

@dataclass
class IntentRoute:
    condition: Callable[[str, str, Any], bool]
    handler: Callable[[str, str, Any], Awaitable[str | None]]

class IntentRouter:
    def __init__(self):
        self._routes: list[IntentRoute] = []

    def register(
        self,
        condition: Callable[[str, str, Any], bool],
        handler: Callable[[str, str, Any], Awaitable[str | None]],
    ) -> None:
        self._routes.append(IntentRoute(condition, handler))

    async def route(self, lowered: str, user_input: str, context: Any) -> str | None:
        for route in self._routes:
            if route.condition(lowered, user_input, context):
                result = await route.handler(lowered, user_input, context)
                if result is not None:
                    return result
        return None
