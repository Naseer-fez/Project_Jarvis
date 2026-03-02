"""Base contract for Jarvis dynamic integrations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


IntegrationResult = dict[str, Any]


class BaseIntegration(ABC):
    """Abstract base class for all integrations discovered at runtime."""

    name: str = ""
    description: str = ""
    required_config: list[str] = []

    def __init__(self, config: Any | None = None) -> None:
        self.config = config
        self.unavailable_reason: str = ""

    @abstractmethod
    def is_available(self) -> bool:
        """
        Return True if dependencies are installed and required env vars are set.
        Never raise from this method; return False silently when unavailable.
        """

    @abstractmethod
    def get_tools(self) -> list[dict[str, Any]]:
        """Return tool schema dicts in the planner SYSTEM_TOOL_SCHEMA format."""

    @abstractmethod
    async def execute(self, tool_name: str, args: dict[str, Any]) -> IntegrationResult:
        """
        Execute one tool and return a normalized payload:
        {"success": bool, "data": Any, "error": str | None}
        """


__all__ = ["BaseIntegration", "IntegrationResult"]
