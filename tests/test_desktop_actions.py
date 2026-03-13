from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from core.controller_v2 import JarvisControllerV2
from core.desktop_actions import DesktopCommandPlan, plan_desktop_command
from core.tools.system_automation import ToolResult


def _make_controller(tmp_path):
    return JarvisControllerV2(
        db_path=str(tmp_path / "memory.db"),
        chroma_path=str(tmp_path / "chroma"),
        model_name="deepseek-r1:8b",
        embedding_model="all-MiniLM-L6-v2",
    )


def test_plan_open_edge_accepts_fuzzy_alias():
    planned = plan_desktop_command("open Microst edge")

    assert isinstance(planned, DesktopCommandPlan)
    assert planned.app_label == "Microsoft Edge"
    assert planned.primary_target == "msedge.exe"


def test_plan_open_any_app_uses_safe_default():
    planned = plan_desktop_command("open any app which u have acces to")

    assert isinstance(planned, DesktopCommandPlan)
    assert planned.app_label == "Notepad"
    assert planned.primary_target == "notepad.exe"


def test_plan_browser_search_builds_edge_url():
    planned = plan_desktop_command("go to Microsoft Edge and search who is Virat in that browser")

    assert isinstance(planned, DesktopCommandPlan)
    assert planned.app_label == "Microsoft Edge"
    assert planned.primary_target == "msedge.exe"
    assert planned.primary_args
    assert planned.primary_args[0] == "https://www.bing.com/search?q=who+is+Virat"


def test_plan_browser_find_phrase_builds_edge_url():
    planned = plan_desktop_command("Acces the edge browser and find when is indias next cricekt match")

    assert isinstance(planned, DesktopCommandPlan)
    assert planned.app_label == "Microsoft Edge"
    assert planned.primary_target == "msedge.exe"
    assert planned.primary_args[0] == "https://www.bing.com/search?q=when+is+indias+next+cricekt+match"


def test_plan_project_folder_opens_current_repo():
    planned = plan_desktop_command("go to Jarvis , project folder")

    assert isinstance(planned, DesktopCommandPlan)
    assert planned.app_label == "Jarvis project folder"
    assert planned.primary_target == "explorer.exe"
    assert planned.primary_args
    assert Path(planned.primary_args[0]).name == "Jarvis"


@pytest.mark.asyncio
async def test_controller_executes_launch_requests_without_llm(tmp_path):
    ctrl = _make_controller(tmp_path)
    launch_mock = AsyncMock(return_value=ToolResult(True, output="opened"))

    with (
        patch("core.desktop_actions.async_launch_application", launch_mock),
        patch.object(ctrl, "_dispatch_llm", new=AsyncMock(return_value="llm")) as llm_mock,
    ):
        response = await ctrl.process("open Microsoft Edge")

    assert response == "Opened Microsoft Edge."
    launch_mock.assert_awaited_once_with("msedge.exe", None)
    llm_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_controller_executes_browser_search_without_llm(tmp_path):
    ctrl = _make_controller(tmp_path)
    launch_mock = AsyncMock(return_value=ToolResult(True, output="opened"))

    with (
        patch("core.desktop_actions.async_launch_application", launch_mock),
        patch.object(ctrl, "_dispatch_llm", new=AsyncMock(return_value="llm")) as llm_mock,
    ):
        response = await ctrl.process("go to Microsoft Edge and search who is Virat in that browser")

    assert response == 'Opened Microsoft Edge and searched for "who is Virat".'
    launch_mock.assert_awaited_once_with(
        "msedge.exe",
        ["https://www.bing.com/search?q=who+is+Virat"],
    )
    llm_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_controller_executes_browser_find_phrase_without_llm(tmp_path):
    ctrl = _make_controller(tmp_path)
    launch_mock = AsyncMock(return_value=ToolResult(True, output="opened"))

    with (
        patch("core.desktop_actions.async_launch_application", launch_mock),
        patch.object(ctrl, "_dispatch_llm", new=AsyncMock(return_value="llm")) as llm_mock,
    ):
        response = await ctrl.process("Acces the edge browser and find when is indias next cricekt match")

    assert response == 'Opened Microsoft Edge and searched for "when is indias next cricekt match".'
    launch_mock.assert_awaited_once_with(
        "msedge.exe",
        ["https://www.bing.com/search?q=when+is+indias+next+cricekt+match"],
    )
    llm_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_controller_opens_project_folder_without_llm(tmp_path):
    ctrl = _make_controller(tmp_path)
    launch_mock = AsyncMock(return_value=ToolResult(True, output="opened"))

    with (
        patch("core.desktop_actions.async_launch_application", launch_mock),
        patch.object(ctrl, "_dispatch_llm", new=AsyncMock(return_value="llm")) as llm_mock,
    ):
        response = await ctrl.process("go to Jarvis , project folder")

    assert response == "Opened the Jarvis project folder."
    launch_mock.assert_awaited_once()
    args, kwargs = launch_mock.await_args
    assert args[0] == "explorer.exe"
    assert Path(args[1][0]).name == "Jarvis"
    assert kwargs == {}
    llm_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_controller_handles_unknown_open_request_without_llm(tmp_path):
    ctrl = _make_controller(tmp_path)

    with patch.object(ctrl, "_dispatch_llm", new=AsyncMock(return_value="llm")) as llm_mock:
        response = await ctrl.process("open blender")

    assert "I can open:" in response
    assert "Microsoft Edge" in response
    llm_mock.assert_not_awaited()
