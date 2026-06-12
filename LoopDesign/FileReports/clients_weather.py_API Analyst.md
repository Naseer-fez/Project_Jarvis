# `weather.py` - API Analyst Report

## Overview
Weather integration backed by Open-Meteo public APIs. No API key required.

## Endpoints / Tools
1. `get_current_weather`
   - Description: Get current weather data for a city.
   - Risk: LOW (read-only)
   - Arguments: `city` (string, required).

## External Contracts / Dependencies
- Uses Open-Meteo geocoding API: `https://geocoding-api.open-meteo.com/v1/search`
- Uses Open-Meteo forecast API: `https://api.open-meteo.com/v1/forecast`
- Uses standard library `urllib.request` for HTTP calls.

## Assumptions
- Uses blocking `urllib.request` calls inside `loop.run_in_executor` to prevent event loop blocking.
- Fetches the first geocoding result for a given city and assumes it is the correct city.
- Hardcoded forecast parameters: `current=temperature_2m,relative_humidity_2m,wind_speed_10m`.
- Throws an error if the city fails to geolocate.
