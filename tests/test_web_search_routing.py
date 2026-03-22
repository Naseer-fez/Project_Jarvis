"""
tests/test_web_search_routing.py — Tests for the explicit web search fast-path.

Covers:
  1. _is_explicit_web_search() detects all trigger phrases correctly.
  2. _is_explicit_web_search() does NOT trigger on normal conversation.
  3. process() calls _handle_web_search when user says "search the web for X".
  4. process() falls through to LLM for normal conversation (no false positives).
  5. _handle_web_search() returns raw results when LLM synthesis fails.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.controller_v2 import (
    JarvisControllerV2,
    _WEB_SEARCH_EXPLICIT_PHRASES,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_minimal_controller() -> JarvisControllerV2:
    """Return a JarvisControllerV2 with all heavy deps mocked out."""
    with patch("core.controller_v2.build_controller_services") as mock_build:
        settings = MagicMock()
        settings.goal_check_interval_seconds = 60
        settings.goals_file = MagicMock()
        settings.goals_file.exists.return_value = False

        services = MagicMock()
        services.memory = MagicMock()
        services.memory.initialize.return_value = {"mode": "test"}
        services.memory.build_context_block.return_value = ""
        services.memory.store_conversation.return_value = None
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
        services.agent_loop = MagicMock()
        services.goal_manager = MagicMock()
        services.goal_manager.active_goals.return_value = []
        services.scheduler = MagicMock()
        services.notifier = MagicMock()
        services.monitor = MagicMock()

        mock_build.return_value = (settings, services)

        import configparser
        config = configparser.ConfigParser()
        ctrl = JarvisControllerV2(config=config)

    # stub persistent goal loading so we don't touch disk
    ctrl._goals_file = MagicMock()
    ctrl._goals_file.exists.return_value = False

    return ctrl


# ---------------------------------------------------------------------------
# 1. _is_explicit_web_search positive cases
# ---------------------------------------------------------------------------

class TestIsExplicitWebSearchPositive:
    """All phrases in _WEB_SEARCH_EXPLICIT_PHRASES must be detected."""

    @pytest.mark.parametrize("phrase", _WEB_SEARCH_EXPLICIT_PHRASES)
    def test_trigger_phrase_detected(self, phrase: str) -> None:
        ctrl = _make_minimal_controller()
        # test with the phrase embedded in a sentence
        assert ctrl._is_explicit_web_search(f"please {phrase} the news today"), (
            f"Expected _is_explicit_web_search to return True for phrase: '{phrase}'"
        )

    @pytest.mark.parametrize(
        "text",
        [
            "search the web for latest AI models",
            "browse the internet for Python tips",
            "search online for cheap flights",
            "web search nvidia gpu prices",
            "internet search climate change report",
            "google for best restaurants near me",
            "find on the web how to make pasta",
        ],
    )
    def test_common_user_inputs_detected(self, text: str) -> None:
        ctrl = _make_minimal_controller()
        assert ctrl._is_explicit_web_search(text.lower()), (
            f"Expected True for: '{text}'"
        )


# ---------------------------------------------------------------------------
# 2. _is_explicit_web_search negative cases (no false positives)
# ---------------------------------------------------------------------------

class TestIsExplicitWebSearchNegative:
    @pytest.mark.parametrize(
        "text",
        [
            "hello how are you",
            "what time is it",
            "remind me to drink water",
            "set a goal to exercise daily",
            "tell me a joke",
            "calculate 5 plus 10",
            "open notepad",
            "play some music",
        ],
    )
    def test_non_search_inputs_not_detected(self, text: str) -> None:
        ctrl = _make_minimal_controller()
        assert not ctrl._is_explicit_web_search(text.lower()), (
            f"Expected False for: '{text}'"
        )


# ---------------------------------------------------------------------------
# 3. process() routes to _handle_web_search for explicit web search inputs
# ---------------------------------------------------------------------------

class TestProcessRoutesToWebSearch:
    @pytest.mark.asyncio
    async def test_explicit_search_calls_handle_web_search(self) -> None:
        ctrl = _make_minimal_controller()

        # Mock _handle_web_search so we can verify it was called
        ctrl._handle_web_search = AsyncMock(return_value="Here are your search results...")

        # patch goal/preference/desktop handlers to return None (not handled)
        with (
            patch.object(ctrl, "_handle_goal_intent", return_value=None),
            patch.object(ctrl, "_handle_preference_intent", return_value=None),
            patch("core.controller_v2.handle_desktop_command", new_callable=AsyncMock, return_value=None),
        ):
            response = await ctrl.process("search the web for latest AI news")

        ctrl._handle_web_search.assert_awaited_once()
        assert "search results" in response.lower()

    @pytest.mark.asyncio
    async def test_browse_internet_calls_handle_web_search(self) -> None:
        ctrl = _make_minimal_controller()
        ctrl._handle_web_search = AsyncMock(return_value="Browsing results for you.")

        with (
            patch.object(ctrl, "_handle_goal_intent", return_value=None),
            patch.object(ctrl, "_handle_preference_intent", return_value=None),
            patch("core.controller_v2.handle_desktop_command", new_callable=AsyncMock, return_value=None),
        ):
            response = await ctrl.process("browse the internet for Python 3.13 features")

        ctrl._handle_web_search.assert_awaited_once()
        assert "browsing" in response.lower()


# ---------------------------------------------------------------------------
# 4. process() does NOT call _handle_web_search for normal conversation
# ---------------------------------------------------------------------------

class TestProcessNoFalsePositive:
    @pytest.mark.asyncio
    async def test_normal_chat_skips_web_search(self) -> None:
        ctrl = _make_minimal_controller()
        ctrl._handle_web_search = AsyncMock(return_value="should not be called")
        ctrl._dispatch_llm = AsyncMock(return_value="Hello! I'm Jarvis.")

        with (
            patch.object(ctrl, "_handle_goal_intent", return_value=None),
            patch.object(ctrl, "_handle_preference_intent", return_value=None),
            patch("core.controller_v2.handle_desktop_command", new_callable=AsyncMock, return_value=None),
        ):
            response = await ctrl.process("hello how are you")

        ctrl._handle_web_search.assert_not_awaited()
        assert "Jarvis" in response


# ---------------------------------------------------------------------------
# 5. _handle_web_search falls back to raw results when LLM synthesis fails
# ---------------------------------------------------------------------------

class TestHandleWebSearchFallback:
    @pytest.mark.asyncio
    async def test_returns_raw_results_when_llm_fails(self) -> None:
        ctrl = _make_minimal_controller()

        raw = (
            "Search query used: Python 3.13 features\n"
            "Summary: Python 3.13 adds free-threaded mode.\n"
            "Sources:\n1. Python Docs\nURL: https://docs.python.org\n"
        )

        with (
            patch("core.tools.web_tools.web_search", new_callable=AsyncMock, return_value=raw),
            patch("core.tools.web_tools._basic_query_cleanup", return_value="Python 3.13 features"),
        ):
            # Make the LLM synthesis throw
            ctrl.llm.chat_async = AsyncMock(side_effect=RuntimeError("LLM down"))

            result = await ctrl._handle_web_search(
                "search for Python 3.13 features", trace_id="test"
            )

        # Should return the raw search output instead of crashing
        assert "Python 3.13" in result

    @pytest.mark.asyncio
    async def test_falls_back_to_llm_when_search_fails(self) -> None:
        ctrl = _make_minimal_controller()
        ctrl._dispatch_llm = AsyncMock(return_value="I couldn't search right now.")

        with patch(
            "core.tools.web_tools.web_search",
            new_callable=AsyncMock,
            side_effect=RuntimeError("network error"),
        ):
            result = await ctrl._handle_web_search(
                "search the web for news", trace_id="test"
            )

        ctrl._dispatch_llm.assert_awaited_once()
        assert "couldn't search" in result
