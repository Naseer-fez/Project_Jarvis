"""Legacy weather integration wrapper."""

from __future__ import annotations

import asyncio
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from integrations.base_integration import BaseIntegration, RiskLevel, ToolResult


WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "")
_IMPORTED_WEATHER_API_KEY = WEATHER_API_KEY
_CACHE_TTL_SECONDS = 600
_CACHE_PATH = Path("data") / "weather_cache.json"
_DEGREE_SIGN = "\N{DEGREE SIGN}"
_TEMPERATURE_SUFFIXES = {
    "metric": f"{_DEGREE_SIGN}C",
    "imperial": f"{_DEGREE_SIGN}F",
}


def _configured_weather_api_key() -> str:
    if WEATHER_API_KEY != _IMPORTED_WEATHER_API_KEY:
        return WEATHER_API_KEY
    return os.getenv("WEATHER_API_KEY", "")


def _normalize_units(units: str) -> str:
    normalized = str(units or "metric").strip().lower()
    if normalized not in _TEMPERATURE_SUFFIXES:
        return "metric"
    return normalized


class WeatherIntegration(BaseIntegration):
    """Backward-compatible weather client used by older integrations code."""

    name = "weather"
    description = "Fetch current weather for a city"

    tool_name = "get_current_weather"
    risk_level = RiskLevel.READ_ONLY
    tool_schema = {
        "name": tool_name,
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "units": {"type": "string", "enum": ["metric", "imperial"]},
            },
            "required": ["city"],
        },
        "_jarvis": {
            "autonomy_level_required": 2,
            "risk_level": RiskLevel.READ_ONLY.value,
        },
    }

    def get_tools(self) -> list[dict[str, Any]]:
        return [dict(self.tool_schema)]

    def is_available(self) -> bool:
        api_key = _configured_weather_api_key()
        self.unavailable_reason = "" if api_key else "WEATHER_API_KEY is not configured"
        return bool(api_key)

    async def execute(self, city: str = "", units: str = "metric", **_: Any) -> ToolResult:
        city = str(city or "").strip()
        units = _normalize_units(units)

        if not city:
            return ToolResult(success=False, error="city is required", tool_name=self.tool_name)

        cached = self._load_cache(city, units)
        if cached is not None:
            return ToolResult(success=True, data=cached, tool_name=self.tool_name)

        api_key = _configured_weather_api_key()
        if not api_key:
            return ToolResult(
                success=False,
                error="WEATHER_API_KEY is not configured",
                tool_name=self.tool_name,
            )

        try:
            payload = await self._fetch(city, units)
            parsed = self._parse_payload(payload, units)
            self._save_cache(city, units, parsed)
            return ToolResult(success=True, data=parsed, tool_name=self.tool_name)
        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                error="Weather request timed out",
                tool_name=self.tool_name,
            )
        except OSError as exc:
            return ToolResult(
                success=False,
                error=f"Offline mode active: {exc}",
                tool_name=self.tool_name,
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult(success=False, error=str(exc), tool_name=self.tool_name)

    async def _fetch(self, city: str, units: str) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._fetch_sync, city, units)

    def _fetch_sync(self, city: str, units: str) -> dict[str, Any]:
        api_key = _configured_weather_api_key()
        query = urllib.parse.urlencode(
            {
                "q": city,
                "appid": api_key,
                "units": units,
            }
        )
        url = f"https://api.openweathermap.org/data/2.5/weather?{query}"
        with urllib.request.urlopen(url, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))

    def _load_cache(self, city: str, units: str) -> dict[str, Any] | None:
        if not _CACHE_PATH.exists():
            return None
        try:
            payload = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return None

        key = self._cache_key(city, units)
        entry = payload.get(key)
        if not isinstance(entry, dict):
            return None

        expires_at = float(entry.get("expires_at", 0))
        if expires_at < time.time():
            return None

        data = entry.get("data")
        if not isinstance(data, dict):
            return None
        return data

    def _save_cache(self, city: str, units: str, data: dict[str, Any]) -> None:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            payload = json.loads(_CACHE_PATH.read_text(encoding="utf-8")) if _CACHE_PATH.exists() else {}
        except Exception:
            payload = {}

        payload[self._cache_key(city, units)] = {
            "expires_at": time.time() + _CACHE_TTL_SECONDS,
            "data": data,
        }
        _CACHE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _cache_key(self, city: str, units: str) -> str:
        return f"{city.strip().lower()}::{units.strip().lower()}"

    def _parse_payload(self, payload: dict[str, Any], units: str) -> dict[str, Any]:
        normalized_units = _normalize_units(units)
        main = payload.get("main") or {}
        weather_items = payload.get("weather") or [{}]
        wind = payload.get("wind") or {}
        country = (payload.get("sys") or {}).get("country", "")

        temp_value = main.get("temp")
        temp_suffix = _TEMPERATURE_SUFFIXES[normalized_units]
        temperature = f"{float(temp_value):.1f}{temp_suffix}" if temp_value is not None else "unknown"

        humidity_value = main.get("humidity")
        humidity = f"{humidity_value}%" if humidity_value is not None else "unknown"

        condition = str((weather_items[0] or {}).get("description", "unknown")).strip().capitalize()

        return {
            "city": payload.get("name", ""),
            "country": country,
            "temperature": temperature,
            "condition": condition,
            "humidity": humidity,
            "wind_speed": wind.get("speed"),
        }


__all__ = ["WEATHER_API_KEY", "WeatherIntegration"]
