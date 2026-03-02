"""Weather integration example using public Open-Meteo APIs."""

from __future__ import annotations

import asyncio
import json
import logging
import urllib.parse
import urllib.request
from typing import Any

from integrations.base import BaseIntegration

logger = logging.getLogger(__name__)


class WeatherIntegration(BaseIntegration):
    name = "weather"
    description = "Fetch basic current weather by city"
    required_config: list[str] = []

    def is_available(self) -> bool:
        return True

    def get_tools(self) -> list[dict]:
        return [
            {
                "name": "get_current_weather",
                "description": "Get current weather for a city.",
                "risk": "LOW",
                "args": {
                    "city": {"type": "string", "description": "City name, e.g. Delhi."},
                },
                "required_args": ["city"],
            }
        ]

    async def execute(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        if tool_name != "get_current_weather":
            return {"success": False, "data": None, "error": f"Unknown weather tool '{tool_name}'"}

        city = str(args.get("city", "")).strip()
        if not city:
            return {"success": False, "data": None, "error": "city is required"}

        try:
            data = await asyncio.to_thread(self._fetch_weather, city)
            return {"success": True, "data": data, "error": None}
        except Exception as exc:  # noqa: BLE001
            logger.warning("Weather request failed: %s", exc)
            return {"success": False, "data": None, "error": str(exc)}

    def _fetch_weather(self, city: str) -> dict[str, Any]:
        geocode_url = (
            "https://geocoding-api.open-meteo.com/v1/search?"
            + urllib.parse.urlencode({"name": city, "count": 1})
        )
        geocode_data = json.loads(urllib.request.urlopen(geocode_url, timeout=10).read().decode("utf-8"))
        results = geocode_data.get("results") or []
        if not results:
            raise ValueError(f"No geocode result for city '{city}'")

        first = results[0]
        lat = first["latitude"]
        lon = first["longitude"]

        weather_url = (
            "https://api.open-meteo.com/v1/forecast?"
            + urllib.parse.urlencode({
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
            })
        )
        weather_data = json.loads(urllib.request.urlopen(weather_url, timeout=10).read().decode("utf-8"))
        current = weather_data.get("current", {})

        return {
            "city": first.get("name", city),
            "country": first.get("country", ""),
            "temperature_c": current.get("temperature_2m"),
            "humidity": current.get("relative_humidity_2m"),
            "wind_speed_kmh": current.get("wind_speed_10m"),
        }


__all__ = ["WeatherIntegration"]
