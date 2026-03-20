"""
tests/test_home_assistant_integration.py

All Home Assistant API calls are mocked.
"""

from __future__ import annotations

import os
import time
from unittest.mock import AsyncMock, patch

import pytest

from integrations.clients.home_assistant import HomeAssistantIntegration


def _env() -> dict[str, str]:
    return {
        "HOME_ASSISTANT_URL": "http://homeassistant.local:8123",
        "HOME_ASSISTANT_TOKEN": "test-token",
    }


def _state(
    entity_id: str = "light.kitchen",
    state: str = "on",
    friendly_name: str = "Kitchen Light",
) -> dict[str, object]:
    return {
        "entity_id": entity_id,
        "state": state,
        "attributes": {"friendly_name": friendly_name},
        "last_changed": "2026-03-20T08:00:00+00:00",
        "last_updated": "2026-03-20T08:00:01+00:00",
    }


def test_is_available_false_without_env():
    with patch.dict(os.environ, {}, clear=True):
        for key in ["HOME_ASSISTANT_URL", "HOME_ASSISTANT_TOKEN"]:
            os.environ.pop(key, None)
        ha = HomeAssistantIntegration()
        assert ha.is_available() is False


def test_is_available_true_with_env():
    with patch.dict(os.environ, _env()):
        ha = HomeAssistantIntegration()
        assert ha.is_available() is True


def test_get_tools_structure():
    ha = HomeAssistantIntegration()
    names = {tool["name"] for tool in ha.get_tools()}
    assert {
        "get_entity_state",
        "turn_on_entity",
        "turn_off_entity",
        "toggle_entity",
        "set_thermostat",
        "call_service",
        "list_entities",
    } == names


def test_read_tools_are_low_and_mutating_tools_require_confirm():
    ha = HomeAssistantIntegration()
    tools = {tool["name"]: tool for tool in ha.get_tools()}
    assert tools["get_entity_state"]["risk"] == "low"
    assert tools["list_entities"]["risk"] == "low"
    assert tools["turn_on_entity"]["risk"] == "confirm"
    assert tools["turn_off_entity"]["risk"] == "confirm"
    assert tools["toggle_entity"]["risk"] == "confirm"
    assert tools["set_thermostat"]["risk"] == "confirm"
    assert tools["call_service"]["risk"] == "confirm"


@pytest.mark.asyncio
async def test_get_entity_state_missing_entity_id():
    ha = HomeAssistantIntegration()
    result = await ha.execute("get_entity_state", {"entity_id": ""})
    assert result["success"] is False
    assert "required" in result["error"]


@pytest.mark.asyncio
async def test_get_entity_state_uses_cached_states_before_direct_fetch():
    ha = HomeAssistantIntegration()
    ha._get_states = AsyncMock(return_value=[_state()])  # type: ignore[method-assign]
    ha._request = AsyncMock()  # type: ignore[method-assign]

    result = await ha.execute("get_entity_state", {"entity_id": "light.kitchen"})

    assert result["success"] is True
    assert result["data"]["friendly_name"] == "Kitchen Light"
    ha._request.assert_not_awaited()


@pytest.mark.asyncio
async def test_list_entities_reuses_cache_within_ttl():
    ha = HomeAssistantIntegration()
    ha._request = AsyncMock(  # type: ignore[method-assign]
        return_value=[
            _state("light.kitchen", "on", "Kitchen Light"),
            _state("switch.fan", "off", "Ceiling Fan"),
        ]
    )

    first = await ha.execute("list_entities", {})
    second = await ha.execute("list_entities", {"domain": "light"})

    assert first["success"] is True
    assert second["success"] is True
    assert first["data"]["count"] == 2
    assert second["data"]["count"] == 1
    assert second["data"]["entities"][0]["entity_id"] == "light.kitchen"
    assert ha._request.await_count == 1


@pytest.mark.asyncio
async def test_list_entities_refreshes_after_ttl_expires():
    ha = HomeAssistantIntegration()
    ha._request = AsyncMock(return_value=[_state()])  # type: ignore[method-assign]

    first = await ha.execute("list_entities", {})
    ha._entity_cache_at = time.monotonic() - 61
    second = await ha.execute("list_entities", {})

    assert first["success"] is True
    assert second["success"] is True
    assert ha._request.await_count == 2


@pytest.mark.asyncio
async def test_turn_on_entity_blocks_sensitive_domains():
    ha = HomeAssistantIntegration()
    ha._request = AsyncMock()  # type: ignore[method-assign]

    result = await ha.execute("turn_on_entity", {"entity_id": "lock.front_door"})

    assert result["success"] is False
    assert "Sensitive" in result["error"]
    ha._request.assert_not_awaited()


@pytest.mark.asyncio
async def test_turn_off_entity_invalidates_cache_after_success():
    ha = HomeAssistantIntegration()
    ha._entity_cache = [_state()]
    ha._entity_cache_at = time.monotonic()
    ha._request = AsyncMock(return_value=[_state("light.kitchen", "off", "Kitchen Light")])  # type: ignore[method-assign]

    result = await ha.execute("turn_off_entity", {"entity_id": "light.kitchen"})

    assert result["success"] is True
    assert result["data"]["service"] == "light.turn_off"
    assert ha._entity_cache == []
    assert ha._entity_cache_at == 0.0


@pytest.mark.asyncio
async def test_call_service_requires_a_target():
    ha = HomeAssistantIntegration()
    result = await ha.execute("call_service", {"domain": "light", "service": "turn_on"})
    assert result["success"] is False
    assert "require" in result["error"].lower()


@pytest.mark.asyncio
async def test_call_service_mock_success():
    ha = HomeAssistantIntegration()
    ha._request = AsyncMock(return_value=[_state("light.kitchen", "off", "Kitchen Light")])  # type: ignore[method-assign]

    result = await ha.execute(
        "call_service",
        {
            "domain": "light",
            "service": "turn_off",
            "entity_id": "light.kitchen",
            "service_data": {"transition": 1},
        },
    )

    assert result["success"] is True
    assert result["data"]["service"] == "light.turn_off"
    ha._request.assert_awaited_once_with(
        "post",
        "/api/services/light/turn_off",
        json_payload={"transition": 1, "entity_id": "light.kitchen"},
    )


@pytest.mark.asyncio
async def test_set_thermostat_mock_success():
    ha = HomeAssistantIntegration()
    ha._request = AsyncMock(return_value=[_state("climate.living_room", "heat", "Living Room")])  # type: ignore[method-assign]

    result = await ha.execute(
        "set_thermostat",
        {
            "entity_id": "climate.living_room",
            "temperature": 22,
            "hvac_mode": "heat",
        },
    )

    assert result["success"] is True
    assert result["data"]["service"] == "climate.set_temperature"
    ha._request.assert_awaited_once_with(
        "post",
        "/api/services/climate/set_temperature",
        json_payload={
            "entity_id": "climate.living_room",
            "temperature": 22,
            "hvac_mode": "heat",
        },
    )


@pytest.mark.asyncio
async def test_unknown_tool_returns_error():
    ha = HomeAssistantIntegration()
    result = await ha.execute("no_such_tool", {})
    assert result["success"] is False
    assert "Unknown" in result["error"]
