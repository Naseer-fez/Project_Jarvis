"""Dynamic integration registry for Jarvis plugins."""

from __future__ import annotations

import logging
from typing import Any

from integrations.base import BaseIntegration

logger = logging.getLogger(__name__)


class IntegrationRegistry:
    """Holds active integration instances and routes tool execution."""

    def __init__(self) -> None:
        self._integrations: dict[str, BaseIntegration] = {}
        self._tool_map: dict[str, BaseIntegration] = {}

    def register(self, integration: BaseIntegration) -> None:
        """Register one integration and all declared tools."""
        if not isinstance(integration, BaseIntegration):
            raise TypeError("integration must be an instance of BaseIntegration")

        integration_name = (integration.name or integration.__class__.__name__).strip()
        tools = integration.get_tools() or []

        if not tools:
            logger.warning("Integration '%s' has no tools and will still be tracked", integration_name)

        for tool in tools:
            tool_name = str(tool.get("name", "")).strip()
            if not tool_name:
                logger.warning(
                    "Integration '%s' declared a tool without a name; skipping entry", integration_name
                )
                continue

            owner = self._tool_map.get(tool_name)
            if owner is not None and owner is not integration:
                logger.warning(
                    "Tool name collision '%s': replacing integration '%s' with '%s'",
                    tool_name,
                    owner.name,
                    integration_name,
                )
            self._tool_map[tool_name] = integration

        self._integrations[integration_name] = integration
        logger.info("Integration registered: %s (%d tools)", integration_name, len(tools))

    def get_tools(self) -> list[dict]:
        """Return all tool definitions from all active integrations."""
        merged: list[dict] = []
        for integration in self._integrations.values():
            merged.extend(integration.get_tools() or [])
        return merged

    async def execute(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Route tool execution to the owning integration."""
        integration = self._tool_map.get(tool_name)
        if integration is None:
            return {
                "success": False,
                "data": None,
                "error": f"No integration registered for tool '{tool_name}'",
            }

        try:
            result = await integration.execute(tool_name, args or {})
        except Exception as exc:  # noqa: BLE001
            logger.exception("Integration '%s' crashed for tool '%s'", integration.name, tool_name)
            return {"success": False, "data": None, "error": str(exc)}

        if not isinstance(result, dict):
            return {
                "success": False,
                "data": None,
                "error": f"Integration '{integration.name}' returned non-dict result",
            }

        return {
            "success": bool(result.get("success", False)),
            "data": result.get("data"),
            "error": result.get("error"),
        }

    def list_active(self) -> list[str]:
        """Return names of active integrations."""
        return sorted(self._integrations.keys())


integration_registry = IntegrationRegistry()

__all__ = ["IntegrationRegistry", "integration_registry"]
