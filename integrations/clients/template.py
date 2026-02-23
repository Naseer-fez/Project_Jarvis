"""
integrations/custom_api_2/client.py  ← TEMPLATE

Copy this file to start a new integration.
Search-and-replace "CustomApi2" / "custom_api_2_action" with your names.

Checklist before submitting a new integration
----------------------------------------------
[ ] Subclasses BaseIntegration
[ ] tool_name is unique (check api_registry._REGISTRY)
[ ] risk_level is READ_ONLY for GET, WRITE for POST/PUT/DELETE
[ ] execute() NEVER raises — all exceptions returned as ToolResult(success=False)
[ ] Offline degradation: ClientConnectorError → self._offline_result()
[ ] Cache/download files go to D:/AI/Jarvis/data/  (not C:)
[ ] All external calls logged via core.logger (before AND after)
[ ] tool_schema has _jarvis.autonomy_level_required set correctly
[ ] Registered in api_registry._load_integrations()
[ ] Tests added to tests/integrations/test_<name>.py
"""

from __future__ import annotations

from typing import Any

from integrations.base_integration import BaseIntegration, RiskLevel, ToolResult


CUSTOM_API_2_SCHEMA: dict = {
    "name": "custom_api_2_action",
    "description": "Short description for the LLM planner.",
    "parameters": {
        "type": "object",
        "properties": {
            "param1": {"type": "string", "description": "..."},
        },
        "required": ["param1"],
    },
    "_jarvis": {
        # READ_ONLY_TOOLS = 2  |  WRITE_TOOLS = 3
        "risk_level": "READ_ONLY_TOOLS",
        "autonomy_level_required": 2,
        "audit_category": "external_data_fetch",
    },
}


class CustomApi2Integration(BaseIntegration):

    tool_name  = "custom_api_2_action"
    risk_level = RiskLevel.READ_ONLY       # change to WRITE if mutating!

    @property
    def tool_schema(self) -> dict:
        return CUSTOM_API_2_SCHEMA

    async def execute(self, **kwargs: Any) -> ToolResult:
        param1 = kwargs.get("param1", "")

        try:
            import aiohttp
            # ... your API call here ...
            result = {"placeholder": param1}
            return ToolResult(success=True, data=result, tool_name=self.tool_name)

        except aiohttp.ClientConnectorError:
            return self._offline_result()
        except Exception as exc:              # noqa: BLE001
            return self._error_result(exc)
