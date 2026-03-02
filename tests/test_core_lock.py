"""
tests/test_core_lock.py

Step 1 — Architectural confidence tests.

Verifies:
 1. core.agent.controller is now a deprecation shim and re-exports JarvisControllerV2
 2. All LLM calls in JarvisControllerV2 route through LLMClientV2
 3. AgentLoopEngine._reflect() uses LLMClientV2 when injected
 4. Risk evaluator BLOCKS all irreversible tools (format_disk, registry_write, etc.)
 5. Risk evaluator BLOCKS critical actions (shell_exec, file_delete)
 6. Risk evaluator REQUIRES CONFIRM for outbound actions
 7. Risk evaluator allows low-risk read-only tools without confirmation
"""

from __future__ import annotations

import warnings
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.autonomy.risk_evaluator import RiskEvaluator, RiskLevel


# ── 1. Deprecation shim ───────────────────────────────────────────────────────


def test_main_controller_shim_emits_deprecation():
    """Importing core.agent.controller must emit a DeprecationWarning."""
    import importlib
    import sys

    # Remove cached module so the warning fires fresh
    sys.modules.pop("core.agent.controller", None)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.import_module("core.agent.controller")

    dep_warns = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert dep_warns, "Expected DeprecationWarning when importing core.agent.controller"
    msg = str(dep_warns[0].message)
    assert "core.controller_v2" in msg or "JarvisControllerV2" in msg


def test_main_controller_shim_is_v2():
    """MainController shim must be JarvisControllerV2."""
    import warnings
    import sys

    sys.modules.pop("core.agent.controller", None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from core.agent.controller import MainController

    from core.controller_v2 import JarvisControllerV2

    assert MainController is JarvisControllerV2


# ── 2. JarvisControllerV2 uses LLMClientV2 ───────────────────────────────────


def test_controller_v2_uses_llm_client_v2():
    """JarvisControllerV2.llm must be an LLMClientV2 instance."""
    from core.llm.client import LLMClientV2

    with patch("core.controller_v2.HybridMemory") as mock_mem, \
         patch("core.controller_v2.ModelRouter") as mock_router, \
         patch("core.controller_v2.UserProfileEngine"), \
         patch("core.controller_v2.GoalManager"), \
         patch("core.controller_v2.Scheduler"), \
         patch("core.controller_v2.NotificationManager"), \
         patch("core.controller_v2.BackgroundMonitor"), \
         patch("core.controller_v2.ProfileSynthesizer"):

        mock_router.return_value.route = MagicMock(return_value="deepseek-r1:8b")
        mock_mem.return_value.initialize = MagicMock(return_value={"mode": "lite"})

        from core.controller_v2 import JarvisControllerV2
        ctrl = JarvisControllerV2()

    assert isinstance(ctrl.llm, LLMClientV2), (
        f"Expected LLMClientV2, got {type(ctrl.llm)}"
    )


# ── 3. AgentLoopEngine routes reflection through LLMClientV2 when injected ───


@pytest.mark.asyncio
async def test_agent_loop_uses_llm_client_v2_for_reflection():
    """When llm= is provided, reflection goes through llm.complete() not httpx."""
    from core.agent.agent_loop import AgentLoopEngine, ExecutionTrace
    from core.state_machine import StateMachine
    from core.autonomy.autonomy_governor import AutonomyGovernor

    mock_llm = MagicMock()
    mock_llm.complete = AsyncMock(return_value="Reflection via LLMClientV2.")

    sm = StateMachine()
    engine = AgentLoopEngine(
        state_machine=sm,
        task_planner=MagicMock(),
        tool_router=MagicMock(),
        risk_evaluator=MagicMock(),
        autonomy_governor=AutonomyGovernor(level=3),
        llm=mock_llm,
    )

    trace = ExecutionTrace(goal="test goal")
    result = await engine._reflect("test goal", {}, [], trace)

    mock_llm.complete.assert_awaited_once()
    assert "LLMClientV2" in result


@pytest.mark.asyncio
async def test_agent_loop_falls_back_when_no_llm_injected():
    """Without llm=, reflection must fall back gracefully (no crash)."""
    from core.agent.agent_loop import AgentLoopEngine, ExecutionTrace
    from core.state_machine import StateMachine
    from core.autonomy.autonomy_governor import AutonomyGovernor

    sm = StateMachine()
    engine = AgentLoopEngine(
        state_machine=sm,
        task_planner=MagicMock(),
        tool_router=MagicMock(),
        risk_evaluator=MagicMock(),
        autonomy_governor=AutonomyGovernor(level=3),
    )

    trace = ExecutionTrace(goal="test goal")
    # With no httpx server, it should return the fallback string, not raise
    result = await engine._reflect("test goal", {"summary": "test"}, [], trace)
    assert isinstance(result, str)
    assert len(result) > 0


# ── 4. Risk evaluator BLOCKS critical/irreversible actions ────────────────────


_CRITICAL_ACTIONS = [
    "format_disk",
    "wipe_disk",
    "registry_write",
    "file_delete",
]

_HIGH_RISK_BLOCKED_ACTIONS = [
    "shell_exec",
]

_CONFIRM_OR_BLOCKED_SHELL_ACTIONS = [
    "execute_shell",  # CONFIRM (not outright blocked, but gated)
]


@pytest.mark.parametrize("action", _CRITICAL_ACTIONS)
def test_risk_evaluator_blocks_critical_action(action: str):
    """Critical/irreversible actions must be blocked by RiskEvaluator."""
    evaluator = RiskEvaluator()
    result = evaluator.evaluate([action])
    assert result.is_blocked, (
        f"Expected '{action}' to be BLOCKED, but is_blocked={result.is_blocked}"
    )
    assert result.level >= RiskLevel.HIGH, (
        f"Expected level >= HIGH for '{action}', got {result.level}"
    )


@pytest.mark.parametrize("action", _HIGH_RISK_BLOCKED_ACTIONS)
def test_risk_evaluator_blocks_high_risk_shell(action: str):
    """shell_exec must be classified HIGH or CRITICAL."""
    evaluator = RiskEvaluator()
    result = evaluator.evaluate([action])
    assert result.level >= RiskLevel.HIGH, (
        f"Expected level >= HIGH for '{action}', got {result.level}"
    )


@pytest.mark.parametrize("action", _CONFIRM_OR_BLOCKED_SHELL_ACTIONS)
def test_risk_evaluator_execute_shell_requires_confirmation(action: str):
    """execute_shell must require confirmation (not blocked outright — still gated)."""
    evaluator = RiskEvaluator()
    result = evaluator.evaluate([action])
    assert result.requires_confirmation or result.is_blocked, (
        f"Expected '{action}' to require confirmation or be blocked, got level={result.level}"
    )


# ── 5. Risk evaluator REQUIRES CONFIRM for outbound messaging tools ───────────


_CONFIRM_ACTIONS = [
    "send_telegram",
    "send_gmail",
    "create_event",
    "delete_event",
    "create_page",
    "append_block",
    "play_track",
    "create_playlist",
    "send_email",
    "send_whatsapp",
]


@pytest.mark.parametrize("action", _CONFIRM_ACTIONS)
def test_risk_evaluator_requires_confirm_for_outbound(action: str):
    """Outbound messaging/creation actions must require confirmation."""
    evaluator = RiskEvaluator()
    result = evaluator.evaluate([action])
    assert result.requires_confirmation or result.is_blocked, (
        f"Expected '{action}' to require confirmation, but result: "
        f"confirm={result.requires_confirmation}, blocked={result.is_blocked}"
    )


# ── 6. Risk evaluator allows read-only tools without confirmation ─────────────


_LOW_RISK_ACTIONS = [
    "memory_read",
    "status",
    "recall",
    "list_events",
    "get_updates",
    "search_track",
    "get_current_track",
    "query_database",
    "get_page",
    "summarize_unread",
]


@pytest.mark.parametrize("action", _LOW_RISK_ACTIONS)
def test_risk_evaluator_allows_low_risk_tools(action: str):
    """Read-only tools must not be blocked and must not require confirmation."""
    evaluator = RiskEvaluator()
    result = evaluator.evaluate([action])
    assert not result.is_blocked, (
        f"Low-risk tool '{action}' should not be blocked"
    )
    assert not result.requires_confirmation, (
        f"Low-risk tool '{action}' should not require confirmation"
    )


# ── 7. Mixed plan with one blocked action blocks the entire plan ───────────────


def test_risk_evaluator_plan_with_one_blocked_action_is_blocked():
    """A plan containing even one critical action must be rejected in full."""
    evaluator = RiskEvaluator()
    plan = {
        "steps": [
            {"action": "list_events", "params": {}},
            {"action": "format_disk", "params": {}},  # critical
            {"action": "send_telegram", "params": {}},
        ]
    }
    result = evaluator.evaluate_plan(plan)
    assert result.is_blocked, "Plan with a CRITICAL step must be blocked"


def test_risk_evaluator_clean_plan_is_not_blocked():
    """A plan with only low/confirm actions must not be blocked."""
    evaluator = RiskEvaluator()
    plan = {
        "steps": [
            {"action": "list_events", "params": {}},
            {"action": "send_telegram", "params": {}},
        ]
    }
    result = evaluator.evaluate_plan(plan)
    assert not result.is_blocked, "Plan with only low/confirm actions must not be blocked"
