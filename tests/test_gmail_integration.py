"""
tests/test_gmail_integration.py

All HTTP calls mocked. No real Google API calls made.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from integrations.clients.gmail import GmailIntegration


# ── Availability ─────────────────────────────────────────────────────────────


def test_is_available_false_without_env():
    with patch.dict(os.environ, {}, clear=True):
        for k in ["GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET", "GOOGLE_REFRESH_TOKEN"]:
            os.environ.pop(k, None)
        g = GmailIntegration()
        assert g.is_available() is False


def test_is_available_true_with_env():
    env = {
        "GOOGLE_CLIENT_ID": "cid",
        "GOOGLE_CLIENT_SECRET": "csec",
        "GOOGLE_REFRESH_TOKEN": "rtoken",
    }
    with patch.dict(os.environ, env):
        g = GmailIntegration()
        assert g.is_available() is True


# ── Tools structure ───────────────────────────────────────────────────────────


def test_get_tools_structure():
    g = GmailIntegration()
    names = {t["name"] for t in g.get_tools()}
    assert {"list_unread", "send_gmail", "summarize_unread", "mark_as_read"} == names


def test_send_gmail_is_confirm():
    g = GmailIntegration()
    tool = next(t for t in g.get_tools() if t["name"] == "send_gmail")
    assert tool["risk"] == "confirm"


def test_list_unread_is_low():
    g = GmailIntegration()
    tool = next(t for t in g.get_tools() if t["name"] == "list_unread")
    assert tool["risk"] == "low"


def test_mark_as_read_is_confirm():
    g = GmailIntegration()
    tool = next(t for t in g.get_tools() if t["name"] == "mark_as_read")
    assert tool["risk"] == "confirm"


# ── Mocked execute: send_gmail ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_send_gmail_empty_to():
    g = GmailIntegration()
    g._refresh_access_token = AsyncMock(return_value="tok")
    result = await g.execute("send_gmail", {"to": "", "subject": "Hi", "body": "body"})
    assert result["success"] is False
    assert "required" in result["error"]


@pytest.mark.asyncio
async def test_send_gmail_mock_success():
    env = {
        "GOOGLE_CLIENT_ID": "cid",
        "GOOGLE_CLIENT_SECRET": "csec",
        "GOOGLE_REFRESH_TOKEN": "rtoken",
    }

    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json = AsyncMock(return_value={"id": "msg_abc123"})
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_resp)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    g = GmailIntegration()
    g._refresh_access_token = AsyncMock(return_value="tok")

    with patch.dict(os.environ, env), patch("aiohttp.ClientSession", return_value=mock_session):
        result = await g.execute("send_gmail", {
            "to": "test@example.com",
            "subject": "Hello",
            "body": "This is a test.",
        })

    assert result["success"] is True
    assert result["data"]["message_id"] == "msg_abc123"


# ── Mocked execute: mark_as_read ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_mark_as_read_empty_id():
    g = GmailIntegration()
    g._refresh_access_token = AsyncMock(return_value="tok")
    result = await g.execute("mark_as_read", {"message_id": ""})
    assert result["success"] is False
    assert "required" in result["error"]


@pytest.mark.asyncio
async def test_mark_as_read_mock_success():
    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json = AsyncMock(return_value={})
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_resp)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    g = GmailIntegration()
    g._refresh_access_token = AsyncMock(return_value="tok")

    env = {
        "GOOGLE_CLIENT_ID": "cid",
        "GOOGLE_CLIENT_SECRET": "csec",
        "GOOGLE_REFRESH_TOKEN": "rtoken",
    }
    with patch.dict(os.environ, env), patch("aiohttp.ClientSession", return_value=mock_session):
        result = await g.execute("mark_as_read", {"message_id": "msg_xyz"})

    assert result["success"] is True
    assert result["data"]["marked_read"] == "msg_xyz"


# ── summarize_unread task_type hint ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_summarize_unread_returns_task_type():
    """summarize_unread output must include task_type='synthesis' for LLM router."""
    g = GmailIntegration()
    g._refresh_access_token = AsyncMock(return_value="tok")

    # Mock _list_unread to return a pre-built result
    g._list_unread = AsyncMock(return_value={
        "success": True,
        "data": {
            "unread": [
                {"id": "m1", "from": "a@b.com", "subject": "Action needed", "date": "", "snippet": "Please review."}
            ],
            "total": 1,
        },
        "error": None,
    })

    result = await g.execute("summarize_unread", {"max_results": 1})
    assert result["success"] is True
    assert result["data"]["task_type"] == "synthesis"


@pytest.mark.asyncio
async def test_unknown_tool_returns_error():
    g = GmailIntegration()
    g._refresh_access_token = AsyncMock(return_value="tok")
    result = await g.execute("no_such_tool", {})
    assert result["success"] is False
    assert "Unknown" in result["error"]
