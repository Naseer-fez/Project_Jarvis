"""
tests/test_telegram_integration.py

All external Telegram API calls are mocked.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from integrations.clients.telegram import TelegramIntegration


# ── Availability ─────────────────────────────────────────────────────────────


def test_is_available_false_without_telegram_lib():
    with patch.dict("sys.modules", {"telegram": None}):
        tg = TelegramIntegration()
        assert tg.is_available() is False


def test_is_available_false_without_env_vars():
    with patch.dict(os.environ, {}, clear=True):
        for k in ["TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"]:
            os.environ.pop(k, None)
        # telegram lib may or may not be installed; just verify env check
        tg = TelegramIntegration()
        result = tg.is_available()
        # Must be False because env vars are missing
        assert result is False


def test_is_available_true_with_env_and_lib():
    env = {"TELEGRAM_BOT_TOKEN": "bot123:TOKEN", "TELEGRAM_CHAT_ID": "99999"}
    mock_telegram = MagicMock()
    with patch.dict(os.environ, env), patch.dict("sys.modules", {"telegram": mock_telegram}):
        tg = TelegramIntegration()
        assert tg.is_available() is True


# ── Tools structure ───────────────────────────────────────────────────────────


def test_get_tools_returns_expected_names():
    tg = TelegramIntegration()
    tools = tg.get_tools()
    names = {t["name"] for t in tools}
    assert "send_telegram" in names
    assert "get_updates" in names


def test_send_telegram_is_confirm_risk():
    tg = TelegramIntegration()
    tool = next(t for t in tg.get_tools() if t["name"] == "send_telegram")
    assert tool["risk"] == "confirm"


def test_get_updates_is_low_risk():
    tg = TelegramIntegration()
    tool = next(t for t in tg.get_tools() if t["name"] == "get_updates")
    assert tool["risk"] == "low"


# ── Execute: send_telegram ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_execute_send_telegram_empty_message():
    tg = TelegramIntegration()
    result = await tg.execute("send_telegram", {"message": ""})
    assert result["success"] is False
    assert "required" in result["error"].lower()


@pytest.mark.asyncio
async def test_execute_send_telegram_mock():
    env = {"TELEGRAM_BOT_TOKEN": "bot123:TOKEN", "TELEGRAM_CHAT_ID": "99999"}

    mock_sent = MagicMock()
    mock_sent.message_id = 42

    mock_bot = AsyncMock()
    mock_bot.send_message = AsyncMock(return_value=mock_sent)
    mock_bot.__aenter__ = AsyncMock(return_value=mock_bot)
    mock_bot.__aexit__ = AsyncMock(return_value=None)

    mock_bot_cls = MagicMock(return_value=mock_bot)
    mock_telegram_module = MagicMock()
    mock_telegram_module.Bot = mock_bot_cls

    with patch.dict(os.environ, env), patch.dict("sys.modules", {"telegram": mock_telegram_module}):
        tg = TelegramIntegration()
        result = await tg.execute("send_telegram", {"message": "Hello Jarvis!"})

    assert result["success"] is True
    assert result["data"]["message_id"] == 42


# ── Execute: get_updates ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_execute_get_updates_mock():
    env = {"TELEGRAM_BOT_TOKEN": "bot123:TOKEN", "TELEGRAM_CHAT_ID": "99999"}

    mock_update = MagicMock()
    mock_update.update_id = 1
    mock_update.message = MagicMock()
    mock_update.message.from_user = MagicMock()
    mock_update.message.from_user.username = "testuser"
    mock_update.message.text = "ping"
    mock_update.message.date = "2026-01-01"

    mock_bot = AsyncMock()
    mock_bot.get_updates = AsyncMock(return_value=[mock_update])
    mock_bot.__aenter__ = AsyncMock(return_value=mock_bot)
    mock_bot.__aexit__ = AsyncMock(return_value=None)

    mock_telegram_module = MagicMock()
    mock_telegram_module.Bot = MagicMock(return_value=mock_bot)

    with patch.dict(os.environ, env), patch.dict("sys.modules", {"telegram": mock_telegram_module}):
        tg = TelegramIntegration()
        result = await tg.execute("get_updates", {"limit": 5})

    assert result["success"] is True
    assert len(result["data"]["updates"]) == 1
    assert result["data"]["updates"][0]["text"] == "ping"


# ── Execute: unknown tool ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_execute_unknown_tool():
    tg = TelegramIntegration()
    result = await tg.execute("nonexistent_tool", {})
    assert result["success"] is False
    assert "Unknown" in result["error"]
