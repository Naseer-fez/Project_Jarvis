"""
tests/test_integrations.py — Tests for IntegrationLoader and integration clients.
All external calls (SMTP, Twilio, etc.) are mocked.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from integrations.base import ToolResult  # noqa: E402
from integrations.clients.calendar import CalendarIntegration  # noqa: E402
from integrations.clients.computer_control import ComputerControlIntegration  # noqa: E402
from integrations.clients.email import EmailIntegration  # noqa: E402
from integrations.clients.whatsapp import WhatsAppIntegration  # noqa: E402
from integrations.loader import IntegrationLoader  # noqa: E402


# ── ToolResult ────────────────────────────────────────────────────────────────

def test_tool_result_has_success_data_error():
    tr = ToolResult(success=True, data={"foo": "bar"}, error="")
    assert tr.success is True
    assert tr.data == {"foo": "bar"}
    assert tr.error == ""


def test_tool_result_failure():
    tr = ToolResult(success=False, error="something went wrong")
    assert tr.success is False
    assert tr.error == "something went wrong"


# ── IntegrationLoader ─────────────────────────────────────────────────────────

def test_integration_loader_returns_loaded_and_skipped_keys():
    mock_registry = MagicMock()
    mock_registry.register = MagicMock()
    loader = IntegrationLoader()
    result = loader.load_all(config=None, registry=mock_registry)
    assert "loaded" in result
    assert "skipped" in result
    assert isinstance(result["loaded"], list)
    assert isinstance(result["skipped"], list)


def test_bad_plugin_in_skipped_not_crashing(tmp_path):
    """A clients/ module that raises on import must NOT crash the loader."""
    # Create a fake bad_plugin.py that raises on import
    clients_dir = PROJECT_ROOT / "integrations" / "_test_clients_tmp"
    clients_dir.mkdir(exist_ok=True)
    bad_plugin = clients_dir / "bad_plugin.py"
    bad_plugin.write_text("raise RuntimeError('intentional import error')\n", encoding="utf-8")

    mock_registry = MagicMock()
    loader = IntegrationLoader()

    # Patch the clients dir to our temp one
    with patch("integrations.loader.Path") as mock_path_cls:
        mock_clients_dir = MagicMock()
        mock_clients_dir.exists.return_value = True
        # Return a bad plugin file
        bad_file = MagicMock()
        bad_file.name = "bad_plugin.py"
        bad_file.stem = "bad_plugin"
        bad_file.__str__ = lambda self: str(bad_plugin)
        mock_clients_dir.glob.return_value = [bad_file]
        mock_path_cls.return_value.__truediv__ = MagicMock(return_value=mock_clients_dir)

        # The loader must not crash; bad plugin lands in skipped
        try:
            result = loader.load_all(config=None, registry=mock_registry)
            # Result is a dict with loaded/skipped
            assert isinstance(result, dict)
        except Exception as e:
            pytest.fail(f"Loader raised on bad plugin: {e}")
    # Cleanup
    bad_plugin.unlink(missing_ok=True)
    clients_dir.rmdir()


# ── EmailIntegration ──────────────────────────────────────────────────────────

def test_email_not_available_without_env_var():
    with patch.dict(os.environ, {}, clear=True):
        # Remove all email env vars
        for k in ["EMAIL_ADDRESS", "EMAIL_PASSWORD", "SMTP_HOST", "IMAP_HOST"]:
            os.environ.pop(k, None)
        email = EmailIntegration()
        assert email.is_available() is False


def test_email_available_with_all_env_vars():
    env = {
        "EMAIL_ADDRESS": "test@example.com",
        "EMAIL_PASSWORD": "secret",
        "SMTP_HOST": "smtp.example.com",
        "IMAP_HOST": "imap.example.com",
    }
    with patch.dict(os.environ, env):
        email = EmailIntegration()
        assert email.is_available() is True


def test_email_get_tools_returns_list():
    email = EmailIntegration()
    tools = email.get_tools()
    assert isinstance(tools, list)
    names = [t["name"] for t in tools]
    assert "send_email" in names
    assert "read_emails" in names


# ── WhatsAppIntegration ───────────────────────────────────────────────────────

def test_whatsapp_not_available_without_twilio():
    """WhatsApp is unavailable when twilio is not installed."""
    with patch.dict("sys.modules", {"twilio": None}):
        wa = WhatsAppIntegration()
        assert wa.is_available() is False


def test_whatsapp_not_available_without_env_vars():
    with patch.dict(os.environ, {}, clear=True):
        for k in ["TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_WHATSAPP_FROM"]:
            os.environ.pop(k, None)
        wa = WhatsAppIntegration()
        assert wa.is_available() is False


# ── CalendarIntegration ───────────────────────────────────────────────────────

def test_calendar_is_always_available():
    cal = CalendarIntegration()
    assert cal.is_available() is True


def test_computer_control_tools_do_not_shadow_core_planner_tools():
    from core.llm.task_planner import SYSTEM_TOOL_SCHEMA

    reserved_names = {tool["name"] for tool in SYSTEM_TOOL_SCHEMA["tools"]}
    control_names = {tool["name"] for tool in ComputerControlIntegration().get_tools()}

    assert control_names == {"move_mouse", "mouse_click", "keyboard_type", "take_screenshot"}
    assert reserved_names.isdisjoint(control_names)


@pytest.mark.asyncio
async def test_calendar_add_event_creates_ics_entry(tmp_path):
    """add_event must write a VEVENT block to the calendar file."""
    cal_path = tmp_path / "calendar.ics"

    import integrations.clients.calendar as cal_module
    original_path = cal_module.CALENDAR_PATH
    cal_module.CALENDAR_PATH = cal_path

    try:
        cal = CalendarIntegration()
        result = await cal.execute("add_event", {
            "title": "Test Meeting",
            "date": "2030-12-25",
            "time": "10:00",
            "duration_minutes": 30,
        })
        assert result["success"] is True
        content = cal_path.read_text(encoding="utf-8")
        assert "Test Meeting" in content
        assert "VEVENT" in content
    finally:
        cal_module.CALENDAR_PATH = original_path


@pytest.mark.asyncio
async def test_calendar_list_events_empty(tmp_path):
    """list_events on empty calendar returns empty list, no crash."""
    cal_path = tmp_path / "calendar.ics"

    import integrations.clients.calendar as cal_module
    original_path = cal_module.CALENDAR_PATH
    cal_module.CALENDAR_PATH = cal_path

    try:
        cal = CalendarIntegration()
        result = await cal.execute("list_events", {"days_ahead": 7})
        assert result["success"] is True
        assert "events" in result["data"]
    finally:
        cal_module.CALENDAR_PATH = original_path
