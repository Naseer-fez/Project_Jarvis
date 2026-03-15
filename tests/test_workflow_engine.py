"""
tests/test_workflow_engine.py

Tests for WorkflowEngine, WorkflowStep, WorkflowResult, and build_steps_from_plan.
All registry and risk_evaluator calls are mocked.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
import pytest

from core.workflow.engine import (
    WorkflowEngine,
    WorkflowStep,
    build_steps_from_plan,
)
from core.autonomy.risk_evaluator import RiskLevel, RiskResult


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_registry(success: bool = True, data: dict | None = None, error: str = "") -> MagicMock:
    mock_registry = MagicMock()
    mock_registry.execute = AsyncMock(return_value={
        "success": success,
        "data": data or {"result": "ok"},
        "error": error,
    })
    return mock_registry


def _make_risk_evaluator(level: RiskLevel = RiskLevel.LOW) -> MagicMock:
    mock_eval = MagicMock()
    mock_eval.evaluate = MagicMock(return_value=RiskResult(
        level=level,
        blocking_actions=["blocked_action"] if level >= RiskLevel.CRITICAL else [],
        confirm_actions=[],
        high_risk_actions=[],
        reasons=[],
    ))
    return mock_eval


def _make_engine(tmp_path=None) -> WorkflowEngine:
    if tmp_path:
        return WorkflowEngine(log_file=tmp_path / "tool_log.jsonl")
    return WorkflowEngine()


# ── WorkflowStep ──────────────────────────────────────────────────────────────


def test_workflow_step_defaults():
    step = WorkflowStep(tool_name="search_track")
    assert step.step_id == "search_track"
    assert step.retry_count == 1
    assert step.timeout == 30.0
    assert step.depends_on == []


def test_workflow_step_custom_id():
    step = WorkflowStep(tool_name="search_track", step_id="s1")
    assert step.step_id == "s1"


# ── build_steps_from_plan ─────────────────────────────────────────────────────


def test_build_steps_from_plan_basic():
    plan = [
        {"tool": "list_unread", "args": {}},
        {"tool": "create_page", "args": {"parent_id": "p1", "title": "Emails"}},
    ]
    steps = build_steps_from_plan(plan)
    assert len(steps) == 2
    assert steps[0].tool_name == "list_unread"
    assert steps[1].tool_name == "create_page"
    assert steps[1].args == {"parent_id": "p1", "title": "Emails"}


def test_build_steps_from_plan_ignores_empty_tool():
    plan = [{"tool": "", "args": {}}, {"tool": "pause", "args": {}}]
    steps = build_steps_from_plan(plan)
    assert len(steps) == 1
    assert steps[0].tool_name == "pause"


def test_build_steps_from_plan_preserves_retry_timeout():
    plan = [{"tool": "send_telegram", "args": {}, "retry_count": 3, "timeout": 60.0}]
    steps = build_steps_from_plan(plan)
    assert steps[0].retry_count == 3
    assert steps[0].timeout == 60.0


# ── Single step success ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_single_step_success(tmp_path):
    engine = _make_engine(tmp_path)
    registry = _make_registry(success=True, data={"tracks": []})
    evaluator = _make_risk_evaluator(RiskLevel.LOW)

    steps = [WorkflowStep(tool_name="search_track", args={"query": "jazz"})]
    result = await engine.execute(steps, registry=registry, risk_evaluator=evaluator)

    assert result.success is True
    assert "search_track" in result.steps_completed
    assert result.steps_failed == []


# ── Multi-step sequential ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_multi_step_sequential(tmp_path):
    engine = _make_engine(tmp_path)

    call_order: list[str] = []

    async def mock_execute(tool_name: str, args: dict):
        call_order.append(tool_name)
        return {"success": True, "data": {"result": "ok"}, "error": None}

    registry = MagicMock()
    registry.execute = mock_execute
    evaluator = _make_risk_evaluator(RiskLevel.LOW)

    steps = [
        WorkflowStep(tool_name="list_unread"),
        WorkflowStep(tool_name="summarize_unread"),
        WorkflowStep(tool_name="create_page", args={"parent_id": "p1", "title": "Summary"}),
    ]
    result = await engine.execute(steps, registry=registry, risk_evaluator=evaluator)

    assert result.success is True
    assert len(result.steps_completed) == 3
    assert call_order == ["list_unread", "summarize_unread", "create_page"]


# ── Step failure captured ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_step_failure_stops_workflow(tmp_path):
    engine = _make_engine(tmp_path)

    call_count = {"n": 0}

    async def mock_execute(tool_name: str, args: dict):
        call_count["n"] += 1
        if tool_name == "send_gmail":
            return {"success": False, "data": None, "error": "SMTP error"}
        return {"success": True, "data": {}, "error": None}

    registry = MagicMock()
    registry.execute = mock_execute
    evaluator = _make_risk_evaluator(RiskLevel.LOW)

    steps = [
        WorkflowStep(tool_name="list_unread"),
        WorkflowStep(tool_name="send_gmail"),
        WorkflowStep(tool_name="create_page"),  # should be skipped
    ]
    result = await engine.execute(steps, registry=registry, risk_evaluator=evaluator)

    assert result.success is False
    assert "send_gmail" in result.steps_failed
    assert "create_page" in result.steps_skipped
    assert call_count["n"] == 2  # create_page never executed


# ── Risk evaluator blocks a step ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_risk_blocked_step_aborts_workflow(tmp_path):
    engine = _make_engine(tmp_path)

    registry = _make_registry(success=True)

    # Evaluator always returns CRITICAL
    evaluator = _make_risk_evaluator(RiskLevel.CRITICAL)

    steps = [WorkflowStep(tool_name="shell_exec", args={"command": "rm -rf /"})]
    result = await engine.execute(steps, registry=registry, risk_evaluator=evaluator)

    assert result.success is False
    assert "shell_exec" in result.steps_failed
    # Registry should NOT have been called (blocked before execution)
    registry.execute.assert_not_awaited()


# ── Partial completion tracking ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_partial_completion_tracking(tmp_path):
    engine = _make_engine(tmp_path)

    async def mock_execute(tool_name: str, args: dict):
        if tool_name == "second_tool":
            return {"success": False, "data": None, "error": "fail"}
        return {"success": True, "data": {}, "error": None}

    registry = MagicMock()
    registry.execute = mock_execute
    evaluator = _make_risk_evaluator(RiskLevel.LOW)

    steps = [
        WorkflowStep(tool_name="first_tool"),
        WorkflowStep(tool_name="second_tool"),
        WorkflowStep(tool_name="third_tool"),
    ]
    result = await engine.execute(steps, registry=registry, risk_evaluator=evaluator)

    assert result.partial is True
    assert "first_tool" in result.steps_completed
    assert "second_tool" in result.steps_failed
    assert "third_tool" in result.steps_skipped


# ── Empty plan ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_empty_plan_succeeds():
    engine = WorkflowEngine()
    registry = MagicMock()
    evaluator = MagicMock()
    result = await engine.execute([], registry=registry, risk_evaluator=evaluator)
    assert result.success is True


# ── Dependency check ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_unmet_dependency_skips_step(tmp_path):
    engine = _make_engine(tmp_path)
    registry = _make_registry(success=True)
    evaluator = _make_risk_evaluator(RiskLevel.LOW)

    steps = [
        WorkflowStep(tool_name="step_b", depends_on=["step_a"]),  # step_a never ran
    ]
    result = await engine.execute(steps, registry=registry, risk_evaluator=evaluator)

    assert "step_b" in result.steps_skipped


# ── Rollback hook ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rollback_hook_called_on_failure(tmp_path):
    engine = _make_engine(tmp_path)
    registry = _make_registry(success=False, error="network error")
    evaluator = _make_risk_evaluator(RiskLevel.LOW)

    rollback_called: list[str] = []

    async def rollback(tool_name: str, args: dict) -> None:
        rollback_called.append(tool_name)

    engine.register_rollback("send_telegram", rollback)

    steps = [WorkflowStep(tool_name="send_telegram", args={"message": "Hi"})]
    result = await engine.execute(steps, registry=registry, risk_evaluator=evaluator)

    assert result.success is False
    assert "send_telegram" in rollback_called


# ── JSONL log file created ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_jsonl_log_created(tmp_path):
    log_file = tmp_path / "exec.jsonl"
    engine = WorkflowEngine(log_file=log_file)
    registry = _make_registry(success=True)
    evaluator = _make_risk_evaluator(RiskLevel.LOW)

    steps = [WorkflowStep(tool_name="pause")]
    await engine.execute(steps, registry=registry, risk_evaluator=evaluator)

    assert log_file.exists()
    content = log_file.read_text()
    assert "pause" in content
