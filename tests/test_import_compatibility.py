from __future__ import annotations

import argparse
import asyncio
from unittest.mock import patch

import pytest


def test_legacy_hybrid_import_alias() -> None:
    from core.memory.hybrid import HybridMemory
    from core.memory.hybrid_memory import HybridMemory as NewHybridMemory

    assert HybridMemory is NewHybridMemory


def test_legacy_memory_shims_store_and_recall(tmp_path) -> None:
    from core.memory.user_memory import ConversationStore, UserMemory

    conversations = ConversationStore(db_path=str(tmp_path / "compat.db"))
    conversations.store("user", "hello")
    recalled = conversations.recall(limit=1)

    prefs = UserMemory(db_path=str(tmp_path / "compat.db"))
    prefs.store_preference("theme", "dark")

    assert recalled[0]["content"] == "hello"
    assert prefs.retrieve_preference("theme") == "dark"


def test_legacy_tool_result_to_llm_string() -> None:
    from integrations.base_integration import RiskLevel, ToolResult

    result = ToolResult(success=False, error="boom", tool_name="get_current_weather")

    assert RiskLevel.READ_ONLY.value == "READ_ONLY_TOOLS"
    assert "boom" in result.to_llm_string()


@pytest.mark.asyncio
async def test_legacy_weather_execute_uses_cache() -> None:
    from integrations.weather_api.client import WeatherIntegration

    cached_payload = {"city": "London", "temperature": "14.0°C"}

    with (
        patch("integrations.weather_api.client.WEATHER_API_KEY", "test-key"),
        patch.object(WeatherIntegration, "_load_cache", return_value=cached_payload),
    ):
        result = await WeatherIntegration().execute(city="London")

    assert result.success is True
    assert result.data == cached_payload


def test_legacy_api_registry_exposes_weather_tool() -> None:
    from integrations import api_registry
    from integrations.weather_api.client import WeatherIntegration

    tool = api_registry.get_tool("get_current_weather")

    assert isinstance(tool, WeatherIntegration)
    assert "get_current_weather" in api_registry.list_tools()


def test_legacy_main_no_voice_entrypoint_forces_headless_mode(monkeypatch) -> None:
    import Main_No_Voice as no_voice_main

    captured: dict[str, object] = {}

    async def fake_async_main(args) -> int:
        captured["voice"] = getattr(args, "voice", None)
        return 7

    monkeypatch.setattr(no_voice_main.jarvis_main, "async_main", fake_async_main)

    result = asyncio.run(
        no_voice_main.async_main(
            argparse.Namespace(
                gui=False,
                dashboard=False,
                verify=False,
                config="config/jarvis.ini",
                log_level=None,
                session_name=None,
            )
        )
    )

    assert result == 7
    assert captured["voice"] is False
