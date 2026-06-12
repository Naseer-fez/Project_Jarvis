import pytest
from unittest.mock import MagicMock, AsyncMock
from core.agent.agent_loop import AgentLoopEngine
from core.context.context import TaskExecutionContext
from core.state_machine import StateMachine, State

@pytest.fixture
def mock_planner():
    planner = MagicMock()
    planner.plan = AsyncMock()
    return planner

@pytest.fixture
def mock_risk_evaluator():
    risk = MagicMock()
    risk.evaluate_plan.return_value = MagicMock(is_blocked=False, requires_confirmation=False, level=MagicMock(label=lambda: "LOW"), blocking_actions=[], confirm_actions=[], high_risk_actions=[])
    return risk

@pytest.fixture
def mock_dag_executor():
    executor = MagicMock()
    executor.execute = AsyncMock(return_value={"status": "success", "results": {}})
    return executor

@pytest.fixture
def mock_container(mock_dag_executor):
    container = MagicMock()
    container.has.return_value = True
    container.resolve.side_effect = lambda key, **kwargs: mock_dag_executor if key == "dag_executor" else (type("MockGov", (), {"level": 1})() if key == "autonomy_governor" else MagicMock())
    return container

@pytest.fixture
def context():
    return TaskExecutionContext(task_id="test", trace_id="trace", state_machine=StateMachine())

@pytest.mark.asyncio
async def test_agent_loop_planning_failed(mock_planner, mock_risk_evaluator, mock_container, context):
    mock_planner.plan.return_value = None
    engine = AgentLoopEngine(
        task_planner=mock_planner,
        risk_evaluator=mock_risk_evaluator,
        container=mock_container,
    )
    trace = await engine.run("test goal", context)
    assert trace.success is False
    assert trace.stop_reason == "planning_failed"
    assert trace.final_response == "I couldn't generate a plan for that goal."
    assert context.state_machine.state == State.IDLE

@pytest.mark.asyncio
async def test_agent_loop_clarification_needed(mock_planner, mock_risk_evaluator, mock_container, context):
    mock_planner.plan.return_value = {"clarification_needed": True, "clarification_prompt": "What do you mean?"}
    engine = AgentLoopEngine(
        task_planner=mock_planner,
        risk_evaluator=mock_risk_evaluator,
        container=mock_container,
    )
    trace = await engine.run("test goal", context)
    assert trace.success is False
    assert trace.stop_reason == "clarification_needed"
    assert trace.final_response == "What do you mean?"
    assert context.state_machine.state == State.IDLE

@pytest.mark.asyncio
async def test_agent_loop_risk_blocked(mock_planner, mock_risk_evaluator, mock_container, context):
    mock_planner.plan.return_value = {"steps": [{"action": "format_disk"}]}
    mock_risk_evaluator.evaluate_plan.return_value.is_blocked = True
    mock_risk_evaluator.evaluate_plan.return_value.blocking_actions = ["format_disk"]
    
    engine = AgentLoopEngine(
        task_planner=mock_planner,
        risk_evaluator=mock_risk_evaluator,
        container=mock_container,
    )
    trace = await engine.run("test goal", context)
    assert trace.success is False
    assert trace.stop_reason == "risk_threshold_exceeded"
    assert "format_disk" in trace.final_response
    assert context.state_machine.state == State.IDLE

@pytest.mark.asyncio
async def test_agent_loop_user_interrupt(mock_planner, mock_risk_evaluator, mock_container, context):
    mock_planner.plan.return_value = {"steps": [{"action": "delete_file"}]}
    mock_risk_evaluator.evaluate_plan.return_value.requires_confirmation = True
    
    engine = AgentLoopEngine(
        task_planner=mock_planner,
        risk_evaluator=mock_risk_evaluator,
        container=mock_container,
    )
    # Simulate user rejecting the prompt
    async def mock_confirm(prompt):
        return False
        
    trace = await engine.run("test goal", context, confirm_callback=mock_confirm)
    assert trace.success is False
    assert trace.stop_reason == "user_interrupt"
    assert context.state_machine.state == State.IDLE

@pytest.mark.asyncio
async def test_agent_loop_happy_path(mock_planner, mock_risk_evaluator, mock_container, context):
    mock_planner.plan.return_value = {"steps": [{"action": "safe_tool"}]}
    # Setup LLM for reflection
    mock_llm = MagicMock()
    mock_llm.complete = AsyncMock(return_value="Task completed successfully.")
    
    engine = AgentLoopEngine(
        task_planner=mock_planner,
        risk_evaluator=mock_risk_evaluator,
        container=mock_container,
        llm=mock_llm
    )
    trace = await engine.run("test goal", context)
    assert trace.success is True
    assert trace.stop_reason == "goal_completed"
    assert trace.final_response == "Task completed successfully."
    assert context.state_machine.state == State.IDLE

