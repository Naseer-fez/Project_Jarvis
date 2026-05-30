from __future__ import annotations

from unittest.mock import patch

import pytest

from core.tools.screen import wait_for_text_on_screen
from integrations.base import ToolResult


@pytest.mark.asyncio
async def test_wait_for_text_on_screen_polls_until_match_is_found():
    responses = [
        ToolResult(success=True, data={"matches": [], "text": ""}),
        ToolResult(
            success=True,
            data={"matches": [{"text": "Continue", "x": 10, "y": 20, "w": 30, "h": 12}], "text": "Continue"},
        ),
    ]

    def _fake_find_text_on_screen(text: str, match_mode: str = "contains") -> ToolResult:
        del text, match_mode
        return responses.pop(0)

    with patch("core.tools.screen.find_text_on_screen", side_effect=_fake_find_text_on_screen):
        result = await wait_for_text_on_screen(
            "Continue",
            timeout_seconds=1.0,
            poll_interval_seconds=0.01,
        )

    assert result.success is True
    assert result.data["attempts"] == 2
    assert result.data["matches"][0]["text"] == "Continue"


@pytest.mark.asyncio
async def test_double_click_screen_target_resolves_and_double_clicks():
    from unittest.mock import AsyncMock
    from core.tools.gui_control import double_click_screen_target

    with (
        patch(
            "core.tools.screen.find_text_on_screen",
            return_value=ToolResult(
                success=True,
                data={
                    "matches": [
                        {
                            "text": "Submit",
                            "x": 50,
                            "y": 100,
                            "w": 60,
                            "h": 30,
                        }
                    ]
                },
            ),
        ),
        patch(
            "core.tools.gui_control.double_click",
            new=AsyncMock(return_value=ToolResult(success=True, data={"action": "double_click"})),
        ) as dclick_mock,
    ):
        result = await double_click_screen_target("Submit")

    assert result.success is True
    assert result.data["matched_text"] == "Submit"
    assert result.data["method"] == "ocr_text"
    dclick_mock.assert_awaited_once_with(80, 115)


@pytest.mark.asyncio
async def test_right_click_screen_target_falls_back_to_vision_and_right_clicks():
    from unittest.mock import AsyncMock
    from core.tools.gui_control import right_click_screen_target

    with (
        patch(
            "core.tools.screen.find_text_on_screen",
            return_value=ToolResult(success=False, error="OCR failed"),
        ),
        patch(
            "core.tools.gui_control._vision_locate_target",
            return_value=ToolResult(
                success=True,
                data={"x": 200, "y": 300, "confidence": 0.85, "target": "Cancel button"},
            ),
        ),
        patch(
            "core.tools.gui_control.right_click",
            new=AsyncMock(return_value=ToolResult(success=True, data={"action": "right_click"})),
        ) as rclick_mock,
    ):
        result = await right_click_screen_target("Cancel button")

    assert result.success is True
    assert result.data["target"] == "Cancel button"
    assert result.data["method"] == "vision"
    rclick_mock.assert_awaited_once_with(200, 300)
