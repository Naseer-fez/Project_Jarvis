from __future__ import annotations

from typing import Any

class ApiRegistryFacade:
    @property
    def _weather_tool(self) -> Any:
        from integrations.weather_api.client import WeatherIntegration
        return WeatherIntegration()

    def get_tool(self, tool_name: str) -> Any:
        if tool_name == "get_current_weather":
            return self._weather_tool
        from integrations.registry import integration_registry
        modern_owner = integration_registry.get_tool(tool_name)
        if modern_owner:
            return modern_owner
        return None

    def list_schemas(self) -> list[dict]:
        from integrations.registry import integration_registry
        schemas = [self._weather_tool.tool_schema]
        for schema in integration_registry.list_schemas():
            if schema.get("name") != "get_current_weather":
                schemas.append(schema)
        return schemas

    def list_tools(self) -> dict[str, str]:
        from integrations.registry import integration_registry
        from integrations.base import IntegrationRiskLevel
        tools = {"get_current_weather": IntegrationRiskLevel.READ_ONLY.value}
        for name, owner in integration_registry.list_tools().items():
            if name != "get_current_weather":
                tools[name] = owner
        return tools

api_registry = ApiRegistryFacade()

__all__ = ["api_registry"]
