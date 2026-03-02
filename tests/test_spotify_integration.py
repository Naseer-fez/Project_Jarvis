"""
tests/test_spotify_integration.py

All HTTP calls mocked via aiohttp.ClientSession.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from integrations.clients.spotify import SpotifyIntegration


# ── Availability ─────────────────────────────────────────────────────────────


def test_is_available_false_without_env():
    with patch.dict(os.environ, {}, clear=True):
        for k in ["SPOTIFY_CLIENT_ID", "SPOTIFY_CLIENT_SECRET", "SPOTIFY_REFRESH_TOKEN"]:
            os.environ.pop(k, None)
        s = SpotifyIntegration()
        assert s.is_available() is False


def test_is_available_true_with_env():
    env = {
        "SPOTIFY_CLIENT_ID": "cid",
        "SPOTIFY_CLIENT_SECRET": "csec",
        "SPOTIFY_REFRESH_TOKEN": "rtoken",
    }
    with patch.dict(os.environ, env):
        s = SpotifyIntegration()
        assert s.is_available() is True


# ── Tools structure ───────────────────────────────────────────────────────────


def test_get_tools_structure():
    s = SpotifyIntegration()
    names = {t["name"] for t in s.get_tools()}
    assert {"play_track", "pause", "search_track", "get_current_track", "create_playlist"} == names


def test_play_track_is_confirm():
    s = SpotifyIntegration()
    tool = next(t for t in s.get_tools() if t["name"] == "play_track")
    assert tool["risk"] == "confirm"


def test_create_playlist_is_confirm():
    s = SpotifyIntegration()
    tool = next(t for t in s.get_tools() if t["name"] == "create_playlist")
    assert tool["risk"] == "confirm"


def test_read_tools_are_low():
    s = SpotifyIntegration()
    for tool in s.get_tools():
        if tool["name"] in ("pause", "search_track", "get_current_track"):
            assert tool["risk"] == "low", f"{tool['name']} should be low risk"


# ── Mocked execute: search_track ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_search_track_empty_query():
    s = SpotifyIntegration()
    s._refresh_access_token = AsyncMock(return_value="tok")
    result = await s.execute("search_track", {"query": ""})
    assert result["success"] is False
    assert "required" in result["error"]


@pytest.mark.asyncio
async def test_search_track_mock_success():
    env = {
        "SPOTIFY_CLIENT_ID": "cid",
        "SPOTIFY_CLIENT_SECRET": "csec",
        "SPOTIFY_REFRESH_TOKEN": "rtoken",
    }
    search_resp = {
        "tracks": {
            "items": [
                {
                    "uri": "spotify:track:abc123",
                    "name": "Test Song",
                    "artists": [{"name": "Test Artist"}],
                    "album": {"name": "Test Album"},
                    "duration_ms": 200000,
                }
            ]
        }
    }
    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json = AsyncMock(return_value=search_resp)
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.get = MagicMock(return_value=mock_resp)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    s = SpotifyIntegration()
    s._refresh_access_token = AsyncMock(return_value="tok")

    with patch.dict(os.environ, env), patch("aiohttp.ClientSession", return_value=mock_session):
        result = await s.execute("search_track", {"query": "Test Song"})

    assert result["success"] is True
    assert len(result["data"]["tracks"]) == 1
    assert result["data"]["tracks"][0]["uri"] == "spotify:track:abc123"


# ── Mocked execute: get_current_track ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_current_track_nothing_playing():
    """204 response = nothing playing. Should return success=True, playing=False."""
    mock_resp = AsyncMock()
    mock_resp.status = 204
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.get = MagicMock(return_value=mock_resp)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    s = SpotifyIntegration()
    s._refresh_access_token = AsyncMock(return_value="tok")

    env = {
        "SPOTIFY_CLIENT_ID": "cid",
        "SPOTIFY_CLIENT_SECRET": "csec",
        "SPOTIFY_REFRESH_TOKEN": "rtoken",
    }
    with patch.dict(os.environ, env), patch("aiohttp.ClientSession", return_value=mock_session):
        result = await s.execute("get_current_track", {})

    assert result["success"] is True
    assert result["data"]["playing"] is False


# ── Mocked execute: play_track — no active device ────────────────────────────


@pytest.mark.asyncio
async def test_play_track_no_active_device():
    """404 = no active device. Should return success=False with a clear message."""
    mock_resp = AsyncMock()
    mock_resp.status = 404
    mock_resp.json = AsyncMock(return_value={"error": {"message": "Device not found"}})
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.put = MagicMock(return_value=mock_resp)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    s = SpotifyIntegration()
    s._refresh_access_token = AsyncMock(return_value="tok")

    env = {
        "SPOTIFY_CLIENT_ID": "cid",
        "SPOTIFY_CLIENT_SECRET": "csec",
        "SPOTIFY_REFRESH_TOKEN": "rtoken",
    }
    with patch.dict(os.environ, env), patch("aiohttp.ClientSession", return_value=mock_session):
        result = await s.execute("play_track", {"track_uri": "spotify:track:abc"})

    assert result["success"] is False
    assert "active" in result["error"].lower() or "device" in result["error"].lower()


# ── Mocked execute: create_playlist ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_playlist_missing_name():
    s = SpotifyIntegration()
    s._refresh_access_token = AsyncMock(return_value="tok")
    result = await s.execute("create_playlist", {"name": ""})
    assert result["success"] is False
    assert "required" in result["error"]


@pytest.mark.asyncio
async def test_unknown_tool_returns_error():
    s = SpotifyIntegration()
    s._refresh_access_token = AsyncMock(return_value="tok")
    result = await s.execute("no_such_tool", {})
    assert result["success"] is False
    assert "Unknown" in result["error"]
