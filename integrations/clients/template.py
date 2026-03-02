"""Template integration showing the new plugin contract."""

from __future__ import annotations

from typing import Any

from integrations.base import BaseIntegration


class TemplateIntegration(BaseIntegration):
    name = "template"
    description = "Reference template for future integrations"
    required_config: list[str] = []

    def is_available(self) -> bool:
        # Keep template disabled by default so it never registers as a real plugin.
        return False

    def get_tools(self) -> list[dict]:
        return []

    async def execute(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        del tool_name, args
        return {"success": False, "data": None, "error": "Template integration is not executable"}


__all__ = ["TemplateIntegration"]
