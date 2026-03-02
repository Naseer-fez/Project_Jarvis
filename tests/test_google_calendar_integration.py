"""
tests/test_google_calendar_integration.py

All HTTP calls mocked via aiohttp.ClientSession.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from integrations.clients.google_calendar import GoogleCalendarIntegration


# ── Availability ─────────────────────────────────────────────────────────────


def test_is_available_false_without_env():
    with patch.dict(os.environ, {}, clear=True):
        for k in ["GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET", "GOOGLE_REFRESH_TOKEN"]:
            os.environ.pop(k, None)
        gc = GoogleCalendarIntegration()
        assert gc.is_available() is False


def test_is_available_true_with_env():
    env = {
        "GOOGLE_CLIENT_ID": "cid",
        "GOOGLE_CLIENT_SECRET": "csec",
        "GOOGLE_REFRESH_TOKEN": "rtoken",
    }
    with patch.dict(os.environ, env):
        gc = GoogleCalendarIntegration()
        assert gc.is_available() is True


# ── Tools structure ───────────────────────────────────────────────────────────


def test_get_tools_structure():
    gc = GoogleCalendarIntegration()
    tools = gc.get_tools()
    names = {t["name"] for t in tools}
    assert {"create_event", "list_events", "delete_event", "find_free_slot"} == names


def test_create_event_is_confirm():
    gc = GoogleCalendarIntegration()
    tool = next(t for t in gc.get_tools() if t["name"] == "create_event")
    assert tool["risk"] == "confirm"


def test_list_events_is_low():
    gc = GoogleCalendarIntegration()
    tool = next(t for t in gc.get_tools() if t["name"] == "list_events")
    assert tool["risk"] == "low"


def test_delete_event_is_confirm():
    gc = GoogleCalendarIntegration()
    tool = next(t for t in gc.get_tools() if t["name"] == "delete_event")
    assert tool["risk"] == "confirm"


# ── Helper: rfc3339 ───────────────────────────────────────────────────────────


def test_to_rfc3339_parses_naive():
    gc = GoogleCalendarIntegration()
    result = gc._to_rfc3339("2026-03-15T10:00:00")
    assert "2026-03-15" in result


def test_to_rfc3339_rejects_bad_string():
    gc = GoogleCalendarIntegration()
    with pytest.raises(ValueError):
        gc._to_rfc3339("not-a-date")


# ── Mocked execute: create_event ─────────────────────────────────────────────


def _make_mock_session(status: int, json_data: dict) -> MagicMock:
    """Build an aiohttp.ClientSession mock that returns given json_data on request."""
    mock_resp = AsyncMock()
    mock_resp.status = status
    mock_resp.json = AsyncMock(return_value=json_data)
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_resp)
    mock_session.get = MagicMock(return_value=mock_resp)
    mock_session.delete = MagicMock(return_value=mock_resp)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    return mock_session


@pytest.mark.asyncio
async def test_create_event_mock():
    env = {
        "GOOGLE_CLIENT_ID": "cid",
        "GOOGLE_CLIENT_SECRET": "csec",
        "GOOGLE_REFRESH_TOKEN": "rtoken",
    }
    token_resp = {"access_token": "fake_token"}
    event_resp = {"id": "evt123", "htmlLink": "https://calendar.google.com/event/evt123"}

    # First session call = token refresh, second = event create
    mock_token_resp = AsyncMock()
    mock_token_resp.status = 200
    mock_token_resp.json = AsyncMock(return_value=token_resp)
    mock_token_resp.__aenter__ = AsyncMock(return_value=mock_token_resp)
    mock_token_resp.__aexit__ = AsyncMock(return_value=None)

    mock_event_resp = AsyncMock()
    mock_event_resp.status = 201
    mock_event_resp.json = AsyncMock(return_value=event_resp)
    mock_event_resp.__aenter__ = AsyncMock(return_value=mock_event_resp)
    mock_event_resp.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.post = MagicMock(side_effect=[mock_token_resp, mock_event_resp])
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    with patch.dict(os.environ, env), patch("aiohttp.ClientSession", return_value=mock_session):
        gc = GoogleCalendarIntegration()
        result = await gc.execute("create_event", {
            "summary": "Team Meeting",
            "start": "2026-03-15T10:00:00",
            "end": "2026-03-15T11:00:00",
        })

    assert result["success"] is True
    assert result["data"]["event_id"] == "evt123"


@pytest.mark.asyncio
async def test_create_event_missing_summary():
    env = {
        "GOOGLE_CLIENT_ID": "cid",
        "GOOGLE_CLIENT_SECRET": "csec",
        "GOOGLE_REFRESH_TOKEN": "rtoken",
    }

    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json = AsyncMock(return_value={"access_token": "tok"})
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=None)
    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_resp)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    with patch.dict(os.environ, env), patch("aiohttp.ClientSession", return_value=mock_session):
        gc = GoogleCalendarIntegration()
        result = await gc.execute("create_event", {"summary": "", "start": "2026-03-15T10:00:00", "end": "2026-03-15T11:00:00"})

    assert result["success"] is False
    assert "required" in result["error"]


@pytest.mark.asyncio
async def test_delete_event_missing_event_id():
    env = {
        "GOOGLE_CLIENT_ID": "cid",
        "GOOGLE_CLIENT_SECRET": "csec",
        "GOOGLE_REFRESH_TOKEN": "rtoken",
    }
    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json = AsyncMock(return_value={"access_token": "tok"})
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=None)
    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_resp)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    with patch.dict(os.environ, env), patch("aiohttp.ClientSession", return_value=mock_session):
        gc = GoogleCalendarIntegration()
        result = await gc.execute("delete_event", {"event_id": ""})

    assert result["success"] is False
    assert "required" in result["error"]


@pytest.mark.asyncio
async def test_unknown_tool_returns_error():
    gc = GoogleCalendarIntegration()
    # Bypass token refresh — token needed only in real calls
    gc._refresh_access_token = AsyncMock(return_value="tok")
    result = await gc.execute("no_such_tool", {})
    assert result["success"] is False
    assert "Unknown" in result["error"]
