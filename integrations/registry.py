"""Registry and execution router for active Jarvis integrations."""

from __future__ import annotations

import inspect
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

    def list_schemas(self) -> list[dict[str, Any]]:
        """Compatibility alias for planner-facing tool schemas."""
        return self.get_tools()

    def get_tool(self, tool_name: str) -> BaseIntegration | None:
        owner = self._tool_owner.get(str(tool_name).strip())
        if not owner:
            return None
        return self._integrations.get(owner)

    def list_tools(self) -> dict[str, str]:
        """Return tool name -> integration owner mapping."""
        return dict(self._tool_owner)

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
            execute_fn = getattr(integration, "execute")
            params = list(inspect.signature(execute_fn).parameters.values())
            call_args = args or {}

            if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params):
                result = execute_fn(tool_name, **call_args)
            else:
                result = execute_fn(tool_name, call_args)

            if inspect.isawaitable(result):
                result = await result
        except Exception as exc:  # noqa: BLE001
            logger.exception("Integration '%s' execution failed for tool '%s'", owner, tool_name)
            return {"success": False, "data": None, "error": str(exc)}

        if isinstance(result, dict):
            payload = result
        else:
            payload = {
                "success": bool(getattr(result, "success", False)),
                "data": getattr(result, "data", None),
                "error": getattr(result, "error", f"Integration '{owner}' returned unsupported result"),
            }

        return {
            "success": bool(payload.get("success", False)),
            "data": payload.get("data"),
            "error": payload.get("error"),
        }

    def list_active(self) -> list[str]:
        return sorted(self._integrations.keys())


integration_registry = IntegrationRegistry()
api_registry = integration_registry


__all__ = ["IntegrationRegistry", "integration_registry", "api_registry"]
