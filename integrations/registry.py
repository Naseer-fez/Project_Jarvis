"""Registry and execution router for active Jarvis integrations."""

from __future__ import annotations

import logging
from typing import Any

from integrations.base import BaseIntegration

logger = logging.getLogger(__name__)


class IntegrationRegistry:
    """Stores active integrations and routes tool execution by name."""

    def __init__(self) -> None:
        self._integrations: dict[str, BaseIntegration] = {}
        self._tool_owner: dict[str, str] = {}

    def register(self, integration: BaseIntegration) -> None:
        if not isinstance(integration, BaseIntegration):
            raise TypeError("integration must inherit BaseIntegration")

        name = (integration.name or integration.__class__.__name__).strip()
        if not name:
            raise ValueError("integration name cannot be empty")

        # Replace previous tool ownership for this integration.
        stale = [tool for tool, owner in self._tool_owner.items() if owner == name]
        for tool_name in stale:
            self._tool_owner.pop(tool_name, None)

        tools = integration.get_tools() or []
        for tool in tools:
            tool_name = str(tool.get("name", "")).strip()
            if not tool_name:
                logger.warning("Integration '%s' declared a nameless tool; skipping entry", name)
                continue

            previous = self._tool_owner.get(tool_name)
            if previous and previous != name:
                logger.warning(
                    "Tool '%s' reassigned from integration '%s' to '%s'",
                    tool_name,
                    previous,
                    name,
                )
            self._tool_owner[tool_name] = name

        self._integrations[name] = integration
        logger.info("Integration registered: %s (%d tools)", name, len(tools))

    def get_tools(self) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        for integration in self._integrations.values():
            tools = integration.get_tools() or []
            for tool in tools:
                if isinstance(tool, dict):
                    merged.append(dict(tool))
        return merged

    async def execute(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        owner = self._tool_owner.get(tool_name)
        if not owner:
            return {
                "success": False,
                "data": None,
                "error": f"No integration registered for tool '{tool_name}'",
            }

        integration = self._integrations.get(owner)
        if integration is None:
            return {
                "success": False,
                "data": None,
                "error": f"Integration '{owner}' is not active",
            }

        try:
            result = await integration.execute(tool_name, args or {})
        except Exception as exc:  # noqa: BLE001
            logger.exception("Integration '%s' execution failed for tool '%s'", owner, tool_name)
            return {"success": False, "data": None, "error": str(exc)}

        if not isinstance(result, dict):
            return {
                "success": False,
                "data": None,
                "error": f"Integration '{owner}' returned non-dict result",
            }

        return {
            "success": bool(result.get("success", False)),
            "data": result.get("data"),
            "error": result.get("error"),
        }

    def list_active(self) -> list[str]:
        return sorted(self._integrations.keys())


integration_registry = IntegrationRegistry()


__all__ = ["IntegrationRegistry", "integration_registry"]
