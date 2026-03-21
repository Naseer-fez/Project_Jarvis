from __future__ import annotations

import configparser
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.autonomy.autonomy_governor import AutonomyGovernor
from core.controller_v2 import JarvisControllerV2
from core.llm.task_planner import TaskPlanner
from core.tools.builtin_tools import register_all_tools
from core.tools.tool_router import ToolRouter


def _make_config(*, gui_enabled: bool) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg["execution"] = {
        "allow_gui_automation": "true" if gui_enabled else "false",
        "allow_app_launch": "true",
        "allow_web_search": "true",
    }
    cfg["ollama"] = {
        "base_url": "http://localhost:11434",
        "planner_model": "deepseek-r1:8b",
        "request_timeout_s": "5",
    }
    cfg["models"] = {"chat_model": "mistral:7b"}
    return cfg


def _make_controller(tmp_path, *, gui_enabled: bool) -> JarvisControllerV2:
    return JarvisControllerV2(
        config=_make_config(gui_enabled=gui_enabled),
        db_path=str(tmp_path / "memory.db"),
        chroma_path=str(tmp_path / "chroma"),
        model_name="deepseek-r1:8b",
        embedding_model="all-MiniLM-L6-v2",
    )


def test_task_planner_hides_gui_tools_when_disabled():
    planner = TaskPlanner(_make_config(gui_enabled=False))
    tool_names = {tool["name"] for tool in planner._tool_schema()["tools"]}

    assert "click" not in tool_names
    assert "type_text" not in tool_names
    assert "capture_screen" in tool_names


def test_register_all_tools_respects_gui_flag_and_registers_core_automation():
    router = ToolRouter()
    register_all_tools(router, config=_make_config(gui_enabled=False))
    tool_names = set(router.registered_tools())

    assert "click" not in tool_names
    assert "type_text" not in tool_names
    assert "capture_screen" in tool_names
    assert {"write_file", "launch_application", "execute_shell"}.issubset(tool_names)


def test_autonomy_governor_allows_desktop_tools_at_level_3():
    governor = AutonomyGovernor(level=3)

    allowed_click, _ = governor.can_execute("click")
    allowed_screen, _ = governor.can_execute("capture_screen")

    assert allowed_click is True
    assert allowed_screen is True
    assert governor.requires_confirmation("click") is True


@pytest.mark.asyncio
async def test_controller_returns_setup_hint_when_desktop_control_is_disabled(tmp_path):
    ctrl = _make_controller(tmp_path, gui_enabled=False)

    with patch.object(ctrl, "_dispatch_llm", new=AsyncMock(return_value="llm")) as llm_mock:
        response = await ctrl.process("please control my mouse and click the continue button")

    assert "allow_gui_automation = true" in response
    assert "requirements/desktop.txt" in response
    llm_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_controller_routes_desktop_request_into_agent_loop_when_enabled(tmp_path):
    ctrl = _make_controller(tmp_path, gui_enabled=True)
    ctrl.memory.build_context_block = MagicMock(return_value="context")

    with (
        patch.object(
            ctrl.task_planner,
            "plan",
            new=MagicMock(
                return_value={
                    "summary": "Click the continue button.",
                    "steps": [
                        {
                            "id": 1,
                            "action": "click",
                            "description": "Click continue",
                            "params": {"x": 10, "y": 20},
                        }
                    ],
                    "tools_required": ["click"],
                }
            ),
        ) as plan_mock,
        patch.object(
            ctrl.agent_loop,
            "run",
            new=AsyncMock(return_value=SimpleNamespace(final_response="Clicked continue.")),
        ) as run_mock,
        patch.object(ctrl, "_dispatch_llm", new=AsyncMock(return_value="llm")) as llm_mock,
    ):
        response = await ctrl.process("please control my mouse and click the continue button")

    assert response == "Clicked continue."
    plan_mock.assert_called_once()
    run_mock.assert_awaited_once()
    llm_mock.assert_not_awaited()
