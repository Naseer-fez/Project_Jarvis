"""Base contract for Jarvis dynamic integrations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


IntegrationResult = dict[str, Any]


class BaseIntegration(ABC):
    """Abstract base class for all integrations discovered at runtime."""

    name: str = ""
    description: str = ""

    def __init__(self, config: Any | None = None) -> None:
        self.config = config
        self.unavailable_reason: str = ""

    @abstractmethod
    def is_available(self) -> bool:
        """Return True when the integration is fully configured and usable."""

    @abstractmethod
    def get_tools(self) -> list[dict[str, Any]]:
        """Return planner-visible tool definitions owned by this integration."""

    @abstractmethod
    async def execute(self, tool_name: str, args: dict[str, Any]) -> IntegrationResult:
        """
        Execute one tool and return a normalized payload:
        {"success": bool, "data": Any, "error": str | None}
        """


__all__ = ["BaseIntegration", "IntegrationResult"]
