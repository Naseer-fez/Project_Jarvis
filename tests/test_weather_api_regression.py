from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from integrations.weather_api.client import WeatherIntegration


_DEGREE_SIGN = "\N{DEGREE SIGN}"


@pytest.mark.asyncio
async def test_weather_execute_normalizes_units_and_formats_degree_symbol() -> None:
    payload = {
        "name": "Austin",
        "sys": {"country": "US"},
        "main": {"temp": 72.5, "humidity": 34},
        "weather": [{"description": "clear sky"}],
        "wind": {"speed": 8.0},
    }

    with (
        patch("integrations.weather_api.client.WEATHER_API_KEY", "test-key"),
        patch.object(WeatherIntegration, "_load_cache", return_value=None),
        patch.object(WeatherIntegration, "_fetch", new=AsyncMock(return_value=payload)),
        patch.object(WeatherIntegration, "_save_cache"),
    ):
        result = await WeatherIntegration().execute(city="Austin", units="Imperial")

    assert result.success is True
    assert result.data["temperature"] == f"72.5{_DEGREE_SIGN}F"
