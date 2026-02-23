"""
integrations/weather_api/tool_schema.py

JSON schema exposed to Jarvis task_planner / LLM for the weather tool.
Keep parameters minimal and strictly typed — the planner must produce
a valid call from this description alone.
"""

WEATHER_TOOL_SCHEMA: dict = {
    "name": "get_current_weather",
    "description": (
        "Fetch the current weather conditions for a given city. "
        "Returns temperature, condition summary, humidity, and wind speed. "
        "READ-ONLY — no side-effects."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "City name, e.g. 'London' or 'New York'.",
            },
            "units": {
                "type": "string",
                "enum": ["metric", "imperial"],
                "description": "Temperature unit system. Default: 'metric' (Celsius).",
                "default": "metric",
            },
        },
        "required": ["city"],
    },
    # Jarvis-specific metadata (not sent to LLM, used by tool_router / AutonomyGovernor)
    "_jarvis": {
        "risk_level": "READ_ONLY_TOOLS",
        "autonomy_level_required": 2,
        "audit_category": "external_data_fetch",
    },
}
