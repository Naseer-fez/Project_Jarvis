"""
integrations/weather_api/client.py

Weather integration for Jarvis — wraps the OpenWeatherMap free-tier API.

Design principles
-----------------
* READ_ONLY (Level 2) — only GET requests, zero mutations.
* Offline-first: catches every network exception and returns a clean
  ToolResult rather than crashing the async loop.
* All heavy cache files go to D:/AI/Jarvis/data/  (never C:).
* Uses core.logger for full audit trail of every external call.
* API key is loaded from config/ — never hard-coded.
"""

from __future__ import annotations

import asyncio
import json
import os
import pathlib
import time
from typing import Any

import aiohttp  # lightweight async HTTP; add to requirements.txt

from integrations.base_integration import BaseIntegration, RiskLevel, ToolResult
from integrations.weather_api.tool_schema import WEATHER_TOOL_SCHEMA

# ---------------------------------------------------------------------------
# Logger — fall back gracefully if running outside Jarvis
# ---------------------------------------------------------------------------
try:
    from core.logger import get_logger          # type: ignore
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
try:
    from config.settings import WEATHER_API_KEY, DATA_DIR   # type: ignore
except ImportError:
    # Fallback: read from environment for testing / standalone use
    WEATHER_API_KEY: str = os.getenv("WEATHER_API_KEY", "")
    DATA_DIR: pathlib.Path = pathlib.Path("D:/AI/Jarvis/data")

# Cache directory — must stay off C: drive
_CACHE_DIR = pathlib.Path(DATA_DIR) / "cache" / "weather"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_CACHE_TTL_SECONDS = 300          # 5-minute cache to avoid hammering the API
_API_BASE_URL      = "https://api.openweathermap.org/data/2.5/weather"
_REQUEST_TIMEOUT   = aiohttp.ClientTimeout(total=10)


# ---------------------------------------------------------------------------
# Integration class
# ---------------------------------------------------------------------------

class WeatherIntegration(BaseIntegration):
    """Fetches current weather from OpenWeatherMap."""

    tool_name  = "get_current_weather"
    risk_level = RiskLevel.READ_ONLY

    @property
    def tool_schema(self) -> dict:
        return WEATHER_TOOL_SCHEMA

    # ------------------------------------------------------------------
    # Public execute — called by dispatcher.py / tool_router.py
    # ------------------------------------------------------------------

    async def execute(self, **kwargs: Any) -> ToolResult:
        """
        Parameters
        ----------
        city  : str  — required
        units : str  — "metric" | "imperial", default "metric"
        """
        city  = kwargs.get("city", "").strip()
        units = kwargs.get("units", "metric")

        if not city:
            return ToolResult(
                success=False,
                error="Parameter 'city' is required.",
                tool_name=self.tool_name,
            )

        if not WEATHER_API_KEY:
            return ToolResult(
                success=False,
                error="WEATHER_API_KEY is not configured. Set it in config/settings.py.",
                tool_name=self.tool_name,
            )

        # 1. Check disk cache first
        cached = self._load_cache(city, units)
        if cached is not None:
            logger.info("weather_api: cache hit for '%s'", city)
            return ToolResult(success=True, data=cached, tool_name=self.tool_name)

        # 2. Attempt live API call
        logger.info(
            "weather_api: EGRESS — fetching weather for '%s' [units=%s]", city, units
        )

        try:
            data = await self._fetch(city, units)
        except aiohttp.ClientConnectorError:
            logger.warning("weather_api: network unreachable (offline?)")
            return self._offline_result()
        except asyncio.TimeoutError:
            logger.warning("weather_api: request timed out for city='%s'", city)
            return ToolResult(
                success=False,
                error=f"Request timed out fetching weather for '{city}'.",
                tool_name=self.tool_name,
            )
        except aiohttp.ClientResponseError as exc:
            logger.error("weather_api: HTTP %s — %s", exc.status, exc.message)
            return ToolResult(
                success=False,
                error=f"Weather API returned HTTP {exc.status}: {exc.message}",
                tool_name=self.tool_name,
            )
        except Exception as exc:                  # noqa: BLE001 — safety net
            logger.exception("weather_api: unexpected error")
            return self._error_result(exc)

        # 3. Parse response
        try:
            result = self._parse(data, units)
        except (KeyError, TypeError) as exc:
            logger.error("weather_api: unexpected response shape — %s", exc)
            return ToolResult(
                success=False,
                error="Unexpected response format from weather API.",
                tool_name=self.tool_name,
            )

        # 4. Cache and return
        self._save_cache(city, units, result)
        logger.info("weather_api: success for '%s' → %s", city, result)
        return ToolResult(success=True, data=result, tool_name=self.tool_name)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _fetch(self, city: str, units: str) -> dict:
        params = {
            "q":     city,
            "units": units,
            "appid": WEATHER_API_KEY,
        }
        async with aiohttp.ClientSession(timeout=_REQUEST_TIMEOUT) as session:
            async with session.get(_API_BASE_URL, params=params) as resp:
                resp.raise_for_status()
                return await resp.json()

    @staticmethod
    def _parse(raw: dict, units: str) -> dict:
        unit_symbol = "°C" if units == "metric" else "°F"
        return {
            "city":        raw["name"],
            "country":     raw["sys"]["country"],
            "temperature": f"{raw['main']['temp']}{unit_symbol}",
            "feels_like":  f"{raw['main']['feels_like']}{unit_symbol}",
            "condition":   raw["weather"][0]["description"].capitalize(),
            "humidity":    f"{raw['main']['humidity']}%",
            "wind_speed":  f"{raw['wind']['speed']} {'m/s' if units == 'metric' else 'mph'}",
        }

    # ------------------------------------------------------------------
    # Disk cache (stored in D:/AI/Jarvis/data/cache/weather/)
    # ------------------------------------------------------------------

    def _cache_path(self, city: str, units: str) -> pathlib.Path:
        safe_city = city.lower().replace(" ", "_")
        return _CACHE_DIR / f"{safe_city}_{units}.json"

    def _load_cache(self, city: str, units: str) -> dict | None:
        path = self._cache_path(city, units)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if time.time() - payload["_ts"] < _CACHE_TTL_SECONDS:
                return payload["data"]
        except (json.JSONDecodeError, KeyError):
            pass
        return None

    def _save_cache(self, city: str, units: str, data: dict) -> None:
        path = self._cache_path(city, units)
        try:
            path.write_text(
                json.dumps({"_ts": time.time(), "data": data}, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("weather_api: could not write cache — %s", exc)
