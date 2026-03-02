"""
tests/test_notion_integration.py

All HTTP calls mocked via aiohttp.ClientSession.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from integrations.clients.notion import NotionIntegration


# ── Availability ─────────────────────────────────────────────────────────────


def test_is_available_false_without_env():
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("NOTION_API_KEY", None)
        n = NotionIntegration()
        assert n.is_available() is False


def test_is_available_true_with_env():
    with patch.dict(os.environ, {"NOTION_API_KEY": "secret_abc123"}):
        n = NotionIntegration()
        assert n.is_available() is True


# ── Tools structure ───────────────────────────────────────────────────────────


def test_get_tools_structure():
    n = NotionIntegration()
    names = {t["name"] for t in n.get_tools()}
    assert {"create_page", "query_database", "append_block", "get_page"} == names


def test_write_tools_are_confirm():
    n = NotionIntegration()
    for tool in n.get_tools():
        if tool["name"] in ("create_page", "append_block"):
            assert tool["risk"] == "confirm", f"{tool['name']} should be confirm"


def test_read_tools_are_low():
    n = NotionIntegration()
    for tool in n.get_tools():
        if tool["name"] in ("query_database", "get_page"):
            assert tool["risk"] == "low", f"{tool['name']} should be low"


# ── Block type validation ─────────────────────────────────────────────────────


def test_validate_block_type_valid():
    assert NotionIntegration._validate_block_type("paragraph") == "paragraph"
    assert NotionIntegration._validate_block_type("bulleted_list_item") == "bulleted_list_item"


def test_validate_block_type_invalid_falls_back():
    result = NotionIntegration._validate_block_type("not_a_real_type")
    assert result == "paragraph"


def test_validate_parent_type_invalid_falls_back():
    result = NotionIntegration._validate_parent_type("bad_type")
    assert result == "page_id"


# ── Mocked execute: create_page ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_page_missing_parent_id():
    n = NotionIntegration()
    result = await n.execute("create_page", {"parent_id": "", "title": "Test"})
    assert result["success"] is False
    assert "required" in result["error"]


@pytest.mark.asyncio
async def test_create_page_missing_title():
    n = NotionIntegration()
    result = await n.execute("create_page", {"parent_id": "abc123", "title": ""})
    assert result["success"] is False
    assert "required" in result["error"]


@pytest.mark.asyncio
async def test_create_page_mock_success():
    env = {"NOTION_API_KEY": "secret_abc"}
    page_resp = {"id": "page_001", "url": "https://notion.so/page_001"}

    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json = AsyncMock(return_value=page_resp)
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_resp)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    with patch.dict(os.environ, env), patch("aiohttp.ClientSession", return_value=mock_session):
        n = NotionIntegration()
        result = await n.execute("create_page", {
            "parent_id": "parent_abc",
            "title": "Action Items",
            "content": "- Fix bug\n- Write tests",
        })

    assert result["success"] is True
    assert result["data"]["page_id"] == "page_001"


# ── Mocked execute: query_database ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_query_database_missing_id():
    n = NotionIntegration()
    result = await n.execute("query_database", {"database_id": ""})
    assert result["success"] is False
    assert "required" in result["error"]


@pytest.mark.asyncio
async def test_query_database_mock_success():
    env = {"NOTION_API_KEY": "secret_abc"}
    db_resp = {
        "results": [{"id": "r1", "url": "https://notion.so/r1", "created_time": "2026-01-01"}],
        "has_more": False,
    }

    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json = AsyncMock(return_value=db_resp)
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_resp)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    with patch.dict(os.environ, env), patch("aiohttp.ClientSession", return_value=mock_session):
        n = NotionIntegration()
        result = await n.execute("query_database", {"database_id": "db_abc"})

    assert result["success"] is True
    assert len(result["data"]["results"]) == 1


# ── Mocked execute: append_block ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_append_block_missing_page_id():
    n = NotionIntegration()
    result = await n.execute("append_block", {"page_id": "", "text": "content"})
    assert result["success"] is False
    assert "required" in result["error"]


@pytest.mark.asyncio
async def test_append_block_missing_text():
    n = NotionIntegration()
    result = await n.execute("append_block", {"page_id": "p1", "text": ""})
    assert result["success"] is False
    assert "required" in result["error"]


@pytest.mark.asyncio
async def test_unknown_tool_returns_error():
    n = NotionIntegration()
    result = await n.execute("no_such_tool", {})
    assert result["success"] is False
    assert "Unknown" in result["error"]
