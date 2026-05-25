from __future__ import annotations

import configparser
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.controller_v2 import JarvisControllerV2


def _make_controller() -> JarvisControllerV2:
    with patch("core.controller_v2.build_controller_services") as mock_build:
        settings = MagicMock()
        settings.goal_check_interval_seconds = 60
        settings.goals_file = MagicMock()
        settings.goals_file.exists.return_value = False

        services = MagicMock()
        services.memory = MagicMock()
        services.memory.initialize.return_value = {"mode": "test"}
        services.memory.build_context_block.return_value = ""
        services.model_router = MagicMock()
        services.model_router.get_best_available.return_value = "test-model"
        services.profile = MagicMock()
        services.profile.get_communication_style.return_value = ""
        services.profile.get_system_prompt_injection.return_value = ""
        services.profile.update_from_conversation.return_value = None
        services.llm = MagicMock()
        services.synthesizer = MagicMock()
        services.synthesizer.should_run.return_value = False
        services.state_machine = MagicMock()
        services.task_planner = MagicMock()
        services.tool_router = MagicMock()
        services.risk_evaluator = MagicMock()
        services.autonomy_governor = MagicMock()
        services.autonomy_governor.describe.return_value = "LEVEL_3"
        services.agent_loop = MagicMock()
        services.goal_manager = MagicMock()
        services.goal_manager.active_goals.return_value = []
        services.scheduler = MagicMock()
        services.notifier = MagicMock()
        services.monitor = MagicMock()
        services.desktop_executor = MagicMock()
        services.desktop_observer = MagicMock()
        services.desktop_bridge = MagicMock()

        mock_build.return_value = (settings, services)

        cfg = configparser.ConfigParser()
        ctrl = JarvisControllerV2(config=cfg)
    return ctrl


@pytest.mark.asyncio
async def test_process_returns_automation_status_without_llm():
    ctrl = _make_controller()
    ctrl.live_automation = MagicMock()
    ctrl.live_automation.status_line.return_value = "Automation running"
    ctrl.live_automation.status.return_value = {
        "drop_root": "workspace/jarvis_dropbox",
        "commands_dir": "workspace/jarvis_dropbox/commands",
        "rag_dir": "workspace/jarvis_dropbox/rag",
    }

    with patch.object(ctrl, "_dispatch_llm", new=AsyncMock(return_value="llm")) as llm_mock:
        response = await ctrl.process("automation status")

    assert "Automation running" in response
    assert "commands" in response.lower()
    llm_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_process_handles_automation_scan_without_llm():
    ctrl = _make_controller()
    ctrl.live_automation = MagicMock()
    ctrl.live_automation.force_scan = AsyncMock(
        return_value={
            "commands_processed": 2,
            "files_ingested": 3,
            "chunks_ingested": 5,
            "failed_files": 0,
        }
    )
    ctrl.live_automation.status_line.return_value = "Automation running"

    with patch.object(ctrl, "_dispatch_llm", new=AsyncMock(return_value="llm")) as llm_mock:
        response = await ctrl.process("automation scan")

    assert "commands=2" in response
    assert "files=3" in response
    llm_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_process_handles_rag_search_without_llm():
    ctrl = _make_controller()
    ctrl.live_automation = MagicMock()
    ctrl.live_automation.search_rag.return_value = "RAG matches:\n1. [drop_rag] test"
    ctrl.live_automation.status_line.return_value = "Automation running"

    with patch.object(ctrl, "_dispatch_llm", new=AsyncMock(return_value="llm")) as llm_mock:
        response = await ctrl.process("rag search test")

    assert "RAG matches" in response
    llm_mock.assert_not_awaited()
