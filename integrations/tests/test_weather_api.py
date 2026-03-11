"""
tests/integrations/test_weather_api.py

Unit tests for WeatherIntegration.
All network calls are mocked; these tests run fully offline.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import integrations.weather_api.client as weather_client
from integrations.base_integration import RiskLevel, ToolResult
from integrations.weather_api.client import WeatherIntegration


_DEGREE_SIGN = "\N{DEGREE SIGN}"


def _run(awaitable: Awaitable[ToolResult]) -> ToolResult:
    return asyncio.run(awaitable)


@pytest.fixture()
def integration() -> WeatherIntegration:
    return WeatherIntegration()


_MOCK_API_RESPONSE = {
    "name": "London",
    "sys": {"country": "GB"},
    "main": {"temp": 15.2, "feels_like": 13.8, "humidity": 78},
    "weather": [{"description": "light rain"}],
    "wind": {"speed": 4.1},
}


class TestIntegrationMetadata:
    def test_tool_name(self, integration: WeatherIntegration) -> None:
        assert integration.tool_name == "get_current_weather"

    def test_risk_level_is_read_only(self, integration: WeatherIntegration) -> None:
        assert integration.risk_level == RiskLevel.READ_ONLY

    def test_schema_has_required_keys(self, integration: WeatherIntegration) -> None:
        schema = integration.tool_schema
        assert "name" in schema
        assert "description" in schema
        assert "parameters" in schema

    def test_schema_jarvis_metadata_level_2(self, integration: WeatherIntegration) -> None:
        meta = integration.tool_schema.get("_jarvis", {})
        assert meta["autonomy_level_required"] == 2
        assert meta["risk_level"] == "READ_ONLY_TOOLS"


class TestExecuteSuccess:
    @patch("integrations.weather_api.client.WEATHER_API_KEY", "fake_key")
    @patch("integrations.weather_api.client.WeatherIntegration._load_cache", return_value=None)
    @patch("integrations.weather_api.client.WeatherIntegration._save_cache")
    @patch("integrations.weather_api.client.WeatherIntegration._fetch")
    def test_returns_parsed_weather(
        self,
        mock_fetch: AsyncMock,
        mock_save: MagicMock,
        mock_load: MagicMock,
        integration: WeatherIntegration,
    ) -> None:
        mock_fetch.return_value = _MOCK_API_RESPONSE

        result = _run(integration.execute(city="London", units="metric"))

        assert result.success is True
        assert result.tool_name == "get_current_weather"
        assert result.data["city"] == "London"
        assert result.data["country"] == "GB"
        assert result.data["temperature"] == f"15.2{_DEGREE_SIGN}C"
        assert result.data["condition"] == "Light rain"
        assert "%" in result.data["humidity"]

    @patch("integrations.weather_api.client.WEATHER_API_KEY", "fake_key")
    @patch("integrations.weather_api.client.WeatherIntegration._load_cache")
    def test_cache_hit_skips_network(
        self,
        mock_load: MagicMock,
        integration: WeatherIntegration,
    ) -> None:
        cached_data = {"city": "London", "temperature": f"14.0{_DEGREE_SIGN}C"}
        mock_load.return_value = cached_data

        result = _run(integration.execute(city="London"))

        assert result.success is True
        assert result.data == cached_data

    @patch("integrations.weather_api.client.WEATHER_API_KEY", "")
    @patch("integrations.weather_api.client.WeatherIntegration._load_cache")
    def test_cache_hit_does_not_require_api_key(
        self,
        mock_load: MagicMock,
        integration: WeatherIntegration,
    ) -> None:
        cached_data = {"city": "London", "temperature": f"14.0{_DEGREE_SIGN}C"}
        mock_load.return_value = cached_data

        result = _run(integration.execute(city="London"))

        assert result.success is True
        assert result.data == cached_data


class TestExecuteFailures:
    def test_missing_city_returns_error(self, integration: WeatherIntegration) -> None:
        result = _run(integration.execute())

        assert result.success is False
        assert "city" in result.error.lower()

    @patch("integrations.weather_api.client.WEATHER_API_KEY", "")
    def test_missing_api_key_returns_error(self, integration: WeatherIntegration) -> None:
        result = _run(integration.execute(city="Paris"))

        assert result.success is False
        assert "WEATHER_API_KEY" in result.error

    @patch("integrations.weather_api.client.WEATHER_API_KEY", "fake_key")
    @patch("integrations.weather_api.client.WeatherIntegration._load_cache", return_value=None)
    @patch("integrations.weather_api.client.WeatherIntegration._fetch")
    def test_network_error_returns_offline_message(
        self,
        mock_fetch: AsyncMock,
        mock_load: MagicMock,
        integration: WeatherIntegration,
    ) -> None:
        import aiohttp

        mock_fetch.side_effect = aiohttp.ClientConnectorError(
            connection_key=MagicMock(), os_error=OSError("Network unreachable")
        )

        result = _run(integration.execute(city="Tokyo"))

        assert result.success is False
        assert "Offline mode active" in result.to_llm_string()

    @patch("integrations.weather_api.client.WEATHER_API_KEY", "fake_key")
    @patch("integrations.weather_api.client.WeatherIntegration._load_cache", return_value=None)
    @patch("integrations.weather_api.client.WeatherIntegration._fetch")
    def test_timeout_returns_clean_error(
        self,
        mock_fetch: AsyncMock,
        mock_load: MagicMock,
        integration: WeatherIntegration,
    ) -> None:
        mock_fetch.side_effect = asyncio.TimeoutError()

        result = _run(integration.execute(city="Berlin"))

        assert result.success is False
        assert "timed out" in result.error.lower()

    def test_is_available_reads_environment_dynamically(
        self,
        integration: WeatherIntegration,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("WEATHER_API_KEY", raising=False)
        assert integration.is_available() is False

        monkeypatch.setenv("WEATHER_API_KEY", "live-key")
        assert integration.is_available() is True

    def test_cache_uses_wall_clock_time(
        self,
        integration: WeatherIntegration,
        tmp_path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cache_path = tmp_path / "weather_cache.json"
        monkeypatch.setattr(weather_client, "_CACHE_PATH", cache_path)

        timestamps = iter([100.0, 100.0, 800.0])
        monkeypatch.setattr(weather_client.time, "time", lambda: next(timestamps))

        payload = {"city": "London", "temperature": f"14.0{_DEGREE_SIGN}C"}
        integration._save_cache("London", "metric", payload)

        assert integration._load_cache("London", "metric") == payload
        assert integration._load_cache("London", "metric") is None

    def test_to_llm_string_on_failure_is_clean(self, integration: WeatherIntegration) -> None:
        result = ToolResult(success=False, error="Something bad", tool_name="get_current_weather")
        llm_str = result.to_llm_string()

        assert "error" in llm_str
        assert "Something bad" in llm_str
        assert isinstance(llm_str, str)


class TestApiRegistry:
    def test_weather_tool_is_registered(self) -> None:
        from integrations import api_registry

        tool = api_registry.get_tool("get_current_weather")

        assert tool is not None
        assert isinstance(tool, WeatherIntegration)

    def test_list_schemas_includes_weather(self) -> None:
        from integrations import api_registry

        schemas = api_registry.list_schemas()
        names = [schema["name"] for schema in schemas]

        assert "get_current_weather" in names

    def test_list_tools_shows_read_only(self) -> None:
        from integrations import api_registry

        tools = api_registry.list_tools()

        assert tools.get("get_current_weather") == "READ_ONLY_TOOLS"
