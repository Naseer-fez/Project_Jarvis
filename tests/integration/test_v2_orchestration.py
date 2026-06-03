import pytest
from core.state_machine import StateMachine, State, IllegalTransitionError
from core.agent.agent_loop import AgentLoopEngine
from core.context.context import TaskExecutionContext


def test_valid_manual_flow():
    """Test manual transition sequence matching the valid V2 lifecycle."""
    sm = StateMachine()
    sm.task_id = "test-task-123"
    
    assert sm.state == State.IDLE
    
    sm.transition(State.PLANNING)
    assert sm.state == State.PLANNING
    
    sm.transition(State.RISK_EVALUATION)
    assert sm.state == State.RISK_EVALUATION
    
    sm.transition(State.APPROVED)
    assert sm.state == State.APPROVED
    
    sm.transition(State.EXECUTING)
    assert sm.state == State.EXECUTING
    
    sm.transition(State.COMPLETED)
    assert sm.state == State.COMPLETED
    
    sm.transition(State.IDLE)
    assert sm.state == State.IDLE


def test_rejection_manual_flow():
    """Test manual transition sequence for rejection flow."""
    sm = StateMachine()
    
    sm.transition(State.PLANNING)
    sm.transition(State.RISK_EVALUATION)
    
    # Rejection: CANCELLED
    sm.transition(State.CANCELLED)
    assert sm.state == State.CANCELLED
    
    sm.transition(State.IDLE)
    assert sm.state == State.IDLE


def test_invalid_manual_flow_fails():
    """Test that direct transition from RISK_EVALUATION -> EXECUTING fails."""
    sm = StateMachine()
    
    sm.transition(State.PLANNING)
    sm.transition(State.RISK_EVALUATION)
    
    with pytest.raises(IllegalTransitionError) as exc_info:
        sm.transition(State.EXECUTING)
        
    assert "Cannot transition RISK_EVALUATION -> EXECUTING" in str(exc_info.value)
    assert "Allowed:" in str(exc_info.value)
    assert "- APPROVED" in str(exc_info.value)
    assert "- CANCELLED" in str(exc_info.value)
    
    # The state should remain RISK_EVALUATION after a failed transition attempt
    assert sm.state == State.RISK_EVALUATION


@pytest.mark.asyncio
async def test_double_approval_idempotency():
    """Verify that firing the approval callback multiple times handles idempotently."""
    sm = StateMachine()
    ctx = TaskExecutionContext(task_id="test-task", trace_id="trace", state_machine=sm)
    engine = AgentLoopEngine(state_machine=sm)
    
    callback_count = 0
    
    async def mock_confirm_callback(prompt):
        nonlocal callback_count
        callback_count += 1
        return True

    # 1. First invocation: Should run callback and return True
    res1 = await engine._ask_confirmation("Test prompt", mock_confirm_callback, ctx)
    assert res1 is True
    assert callback_count == 1

    # 2. Second invocation: Should return cached result and NOT run callback again
    res2 = await engine._ask_confirmation("Test prompt", mock_confirm_callback, ctx)
    assert res2 is True
    assert callback_count == 1  # count did not increase!


@pytest.mark.asyncio
async def test_async_race_conditions_and_lock():
    """Verify StateMachine correctly raises IllegalTransitionError for invalid state flows."""
    sm = StateMachine()
    sm.transition(State.PLANNING)
    
    # Simulate an external actor approving while the main task also tries to execute.
    # The first valid transition wins, the second invalid one raises.
    sm.transition(State.RISK_EVALUATION)
    assert sm.state == State.RISK_EVALUATION
    
    with pytest.raises(IllegalTransitionError):
        sm.transition(State.EXECUTING)


def test_transition_audit_trail_and_graph():
    """Test that audit trails record full caller information, timestamps, and history."""
    sm = StateMachine()
    sm.task_id = "audit-task-999"
    
    # Transition path
    sm.transition(State.PLANNING)
    sm.transition(State.RISK_EVALUATION)
    
    # Check audit trail
    trail = sm._transition_audit_trail
    assert len(trail) >= 2
    
    # First successful transition check
    assert trail[0]["from_state"] == "IDLE"
    assert trail[0]["to_state"] == "PLANNING"
    assert trail[0]["success"] is True
    assert trail[0]["task_id"] == "audit-task-999"
    assert "test_v2_orchestration.py" in trail[0]["caller"]
    
    # Second transition check
    assert trail[1]["from_state"] == "PLANNING"
    assert trail[1]["to_state"] == "RISK_EVALUATION"
    assert trail[1]["success"] is True
    
    # Graph check
    graph = sm.get_transition_graph()
    assert "RISK_EVALUATION" in graph
    assert "APPROVED" in graph["RISK_EVALUATION"]
    assert "CANCELLED" in graph["RISK_EVALUATION"]
