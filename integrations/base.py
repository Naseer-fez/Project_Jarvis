"""Integration base contract for all dynamic Jarvis plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseIntegration(ABC):
    """
    Base class for integration plugins discovered at runtime.

    Subclasses declare metadata and expose one or more tool definitions that
    can be merged into planner tool schema prompts.
    """

    name: str = ""
    description: str = ""
    required_config: list[str] = []

    @abstractmethod
    def is_available(self) -> bool:
        """Return True when dependencies and required config are present."""

    @abstractmethod
    def get_tools(self) -> list[dict]:
        """Return tool definitions in SYSTEM_TOOL_SCHEMA-style dict format."""

    @abstractmethod
    async def execute(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        """
        Execute one integration tool by name.

        Must return:
        {"success": bool, "data": Any, "error": str | None}
        """
