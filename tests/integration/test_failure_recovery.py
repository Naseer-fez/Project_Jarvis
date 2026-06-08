import pytest
from unittest.mock import MagicMock, AsyncMock

from core.agent.agent_loop import AgentLoopEngine
from core.state_machine import StateMachine, State
from core.context.context import TaskExecutionContext

@pytest.fixture
def mock_container():
    container = MagicMock()
    
    # We will provide a mock DAG Executor that raises errors
    executor = AsyncMock()
    container.has.return_value = True
    container.resolve.side_effect = lambda key, **kwargs: executor if key == "dag_executor" else MagicMock()
    
    return container, executor

@pytest.fixture
def mock_planner():
    planner = MagicMock()
    planner.plan = AsyncMock(return_value={
        "steps": [{"id": 1, "action": "test", "description": "test", "params": {}}]
    })
    return planner

@pytest.fixture
def mock_risk():
    risk = MagicMock()
    risk_result = MagicMock()
    risk_result.is_blocked = False
    risk_result.requires_confirmation = False
    risk_result.level.label.return_value = "low"
    risk.evaluate_plan.return_value = risk_result
    return risk

@pytest.mark.asyncio
async def test_failure_recovery_sqlite_disconnect(mock_container, mock_planner, mock_risk):
    container, executor = mock_container
    
    import sqlite3
    # Inject SQLite disconnect failure
    executor.execute.side_effect = sqlite3.OperationalError("database is locked")
    
    sm = StateMachine()
    engine = AgentLoopEngine(
        state_machine=sm,
        task_planner=mock_planner,
        risk_evaluator=mock_risk,
        container=container,
    )
    
    ctx = TaskExecutionContext(task_id="test_task_1", trace_id="trace_1", state_machine=sm)
    
    trace = await engine.run("trigger sqlite error", ctx)
    
    # Verify the loop handled it, returning a failed trace rather than raising
    assert trace.success is False
    assert "database is locked" in trace.final_response
    assert sm.state == State.IDLE

@pytest.mark.asyncio
async def test_failure_recovery_network_error(mock_container, mock_planner, mock_risk):
    container, executor = mock_container
    
    class FakeNetworkError(Exception):
        pass
    
    # Inject generic Network error
    executor.execute.side_effect = FakeNetworkError("Network unreachable")
    
    sm = StateMachine()
    engine = AgentLoopEngine(
        state_machine=sm,
        task_planner=mock_planner,
        risk_evaluator=mock_risk,
        container=container,
    )
    
    ctx = TaskExecutionContext(task_id="test_task_2", trace_id="trace_2", state_machine=sm)
    
    trace = await engine.run("trigger network error", ctx)
    
    assert trace.success is False
    assert "Network unreachable" in trace.final_response
    assert sm.state == State.IDLE

@pytest.mark.asyncio
async def test_failure_recovery_malformed_config(mock_container, mock_planner, mock_risk):
    container, executor = mock_container
    
    # Inject config error
    executor.execute.side_effect = ValueError("Malformed configuration for tool X")
    
    sm = StateMachine()
    engine = AgentLoopEngine(
        state_machine=sm,
        task_planner=mock_planner,
        risk_evaluator=mock_risk,
        container=container,
    )
    
    ctx = TaskExecutionContext(task_id="test_task_3", trace_id="trace_3", state_machine=sm)
    
    trace = await engine.run("trigger config error", ctx)
    
    assert trace.success is False
    assert "Malformed configuration" in trace.final_response
    assert sm.state == State.IDLE
