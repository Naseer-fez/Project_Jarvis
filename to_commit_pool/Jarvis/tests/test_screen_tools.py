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
