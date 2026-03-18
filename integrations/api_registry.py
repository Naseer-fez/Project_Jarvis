"""Legacy API-registry facade."""

from __future__ import annotations

from integrations.base import RiskLevel
from integrations.weather_api.client import WeatherIntegration


_TOOLS = {
    WeatherIntegration.tool_name: WeatherIntegration(),
}


def get_tool(tool_name: str):
    return _TOOLS.get(tool_name)


def list_schemas() -> list[dict]:
    return [tool.tool_schema for tool in _TOOLS.values()]


def list_tools() -> dict[str, str]:
    return {
        tool_name: (
            tool.risk_level.value
            if isinstance(getattr(tool, "risk_level", None), RiskLevel)
            else str(getattr(tool, "risk_level", ""))
        )
        for tool_name, tool in _TOOLS.items()
    }


__all__ = ["get_tool", "list_schemas", "list_tools"]
