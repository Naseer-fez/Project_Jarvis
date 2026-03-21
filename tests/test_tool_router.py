from __future__ import annotations

import pytest

from core.tools.tool_router import ToolRouter
from integrations.base import ToolResult


@pytest.mark.asyncio
async def test_tool_router_marks_sync_toolresult_success_as_success():
    router = ToolRouter()

    def capture_screen():
        return ToolResult(success=True, data={"path": "outputs/screenshots/test.png"})

    router.register("capture_screen", capture_screen)
    observation = await router.execute("capture_screen", {})

    assert observation.execution_status == "success"
    assert "test.png" in observation.output_summary
    assert observation.error_message is None


@pytest.mark.asyncio
async def test_tool_router_marks_sync_toolresult_failure_as_failure():
    router = ToolRouter()

    def capture_screen():
        return ToolResult(success=False, error="pyautogui not installed")

    router.register("capture_screen", capture_screen)
    observation = await router.execute("capture_screen", {})

    assert observation.execution_status == "failure"
    assert observation.output_summary == ""
    assert "pyautogui not installed" in (observation.error_message or "")
