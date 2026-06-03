import pytest
from core.state_machine import StateMachine, State, IllegalTransitionError


def test_state_machine_initial_state():
    """Verify that state machine initializes to IDLE."""
    sm = StateMachine()
    assert sm.state == State.IDLE


@pytest.mark.parametrize("target_state, expected", [
    (State.THINKING, True),
    (State.PLANNING, True),
    (State.ACTING, False),  # IDLE -> ACTING is not allowed
])
def test_state_machine_can_transition(target_state, expected):
    """Verify that can_transition correctly checks validity of transitions."""
    sm = StateMachine()
    assert sm.can_transition(target_state) is expected


def test_state_machine_successful_transition():
    """Verify that valid transition updates state and notifies listeners."""
    sm = StateMachine()
    events = []

    def listener(old, new):
        events.append((old, new))

    sm.add_listener(listener)
    new_state = sm.transition(State.THINKING)

    assert new_state == State.THINKING
    assert sm.state == State.THINKING
    assert events == [(State.IDLE, State.THINKING)]


def test_state_machine_invalid_transition():
    """Verify that invalid transition raises IllegalTransitionError and does not update state."""
    sm = StateMachine()
    assert sm.state == State.IDLE

    with pytest.raises(IllegalTransitionError):
        sm.transition(State.ACTING)

    assert sm.state == State.IDLE


def test_state_machine_reset_allowed():
    """Verify state machine can reset from ERROR or ABORTED states to IDLE."""
    sm = StateMachine()
    
    # Transition to a path that leads to ERROR
    sm.transition(State.THINKING)
    sm.transition(State.ERROR)
    assert sm.state == State.ERROR

    # Reset should succeed
    sm.reset()
    assert sm.state == State.IDLE


def test_state_machine_reset_disallowed():
    """Verify state machine cannot reset from normal states."""
    sm = StateMachine()
    assert sm.state == State.IDLE

    with pytest.raises(IllegalTransitionError):
        sm.reset()


def test_state_machine_force_idle():
    """Verify force_idle works from any state."""
    sm = StateMachine()
    sm.transition(State.THINKING)
    
    # Force idle should change state back to IDLE
    sm.force_idle()
    assert sm.state == State.IDLE

    # Doing it from IDLE does nothing
    sm.force_idle()
    assert sm.state == State.IDLE


def test_state_machine_notification_order_under_recursion():
    """Verify that listener notifications occur in strict chronological order even with recursive transitions."""
    sm = StateMachine()
    events = []

    def listener(old_state, new_state):
        events.append((old_state, new_state))
        if new_state == State.THINKING:
            # Recursive transition inside listener
            sm.transition(State.PLANNING)

    sm.add_listener(listener)
    sm.transition(State.THINKING)

    # The expected order of events is:
    # 1. IDLE -> THINKING
    # 2. THINKING -> PLANNING
    # And there should be no recursion error or deadlock.
    assert events == [
        (State.IDLE, State.THINKING),
        (State.THINKING, State.PLANNING),
    ]


def test_state_machine_concurrent_transitions():
    """Verify that concurrent transitions from multiple threads are thread-safe and notifications are ordered."""
    import threading
    import time

    sm = StateMachine()
    events = []

    def listener(old_state, new_state):
        events.append((old_state, new_state))

    sm.add_listener(listener)
    
    # We will transition:
    # Thread 1: IDLE -> THINKING
    # Thread 2: (waits for state to be THINKING) -> PLANNING
    
    errors = []
    
    def run_t1():
        try:
            sm.transition(State.THINKING)
        except Exception as e:
            errors.append(e)

    def run_t2():
        try:
            # Spin wait until state is THINKING
            for _ in range(100):
                if sm.state == State.THINKING:
                    break
                time.sleep(0.01)
            sm.transition(State.PLANNING)
        except Exception as e:
            errors.append(e)

    t1 = threading.Thread(target=run_t1)
    t2 = threading.Thread(target=run_t2)

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert not errors, f"Unexpected transition errors: {errors}"
    assert events == [
        (State.IDLE, State.THINKING),
        (State.THINKING, State.PLANNING),
    ]


def test_state_machine_as_context_manager_success():
    """Verify that StateMachine context manager does not alter state on success."""
    sm = StateMachine()
    with sm:
        sm.transition(State.THINKING)
    assert sm.state == State.THINKING


def test_state_machine_as_context_manager_exception():
    """Verify that StateMachine context manager transitions to ERROR on exception."""
    sm = StateMachine()
    try:
        with sm:
            sm.transition(State.THINKING)
            raise ValueError("Something went wrong")
    except ValueError:
        pass
    assert sm.state == State.ERROR


def test_state_machine_as_context_manager_cancel():
    """Verify that StateMachine context manager transitions to ABORTED on CancelledError."""
    import asyncio
    sm = StateMachine()
    sm.transition(State.THINKING)
    sm.transition(State.PLANNING)
    sm.transition(State.REVIEWING)
    try:
        with sm:
            sm.transition(State.EXECUTING)
            raise asyncio.CancelledError()
    except asyncio.CancelledError:
        pass
    assert sm.state == State.ABORTED


def test_state_machine_transition_to_context_manager_success():
    """Verify StateMachine.transition_to context manager transitions on enter and reverts on exit."""
    sm = StateMachine()
    assert sm.state == State.IDLE
    with sm.transition_to(State.THINKING) as guarded_sm:
        assert guarded_sm.state == State.THINKING
    assert sm.state == State.IDLE


def test_state_machine_transition_to_context_manager_exception():
    """Verify StateMachine.transition_to context manager transitions to ERROR on exception."""
    sm = StateMachine()
    try:
        with sm.transition_to(State.THINKING):
            raise ValueError("Oops")
    except ValueError:
        pass
    assert sm.state == State.ERROR


def test_task_execution_context_as_context_manager_success():
    """Verify TaskExecutionContext context manager leaves state unchanged on success."""
    from core.context.context import TaskExecutionContext
    ctx = TaskExecutionContext()
    with ctx:
        ctx.state_machine.transition(State.THINKING)
    assert ctx.state_machine.state == State.THINKING


def test_task_execution_context_as_context_manager_exception():
    """Verify TaskExecutionContext context manager transitions state machine to ERROR on exception."""
    from core.context.context import TaskExecutionContext
    ctx = TaskExecutionContext()
    try:
        with ctx:
            ctx.state_machine.transition(State.THINKING)
            raise ValueError("Failure in execution")
    except ValueError:
        pass
    assert ctx.state_machine.state == State.ERROR


def test_task_execution_context_as_context_manager_cancel():
    """Verify TaskExecutionContext context manager transitions state machine to ABORTED on CancelledError."""
    from core.context.context import TaskExecutionContext
    import asyncio
    ctx = TaskExecutionContext()
    ctx.state_machine.transition(State.THINKING)
    ctx.state_machine.transition(State.PLANNING)
    ctx.state_machine.transition(State.REVIEWING)
    ctx.state_machine.transition(State.EXECUTING)
    try:
        with ctx:
            raise asyncio.CancelledError()
    except asyncio.CancelledError:
        pass
    assert ctx.state_machine.state == State.ABORTED


@pytest.mark.asyncio
async def test_state_machine_as_async_context_manager_success():
    """Verify StateMachine async context manager does not alter state on success."""
    sm = StateMachine()
    async with sm:
        sm.transition(State.THINKING)
    assert sm.state == State.THINKING


@pytest.mark.asyncio
async def test_state_machine_as_async_context_manager_exception():
    """Verify StateMachine async context manager transitions to ERROR on exception."""
    sm = StateMachine()
    try:
        async with sm:
            sm.transition(State.THINKING)
            raise ValueError("Async error")
    except ValueError:
        pass
    assert sm.state == State.ERROR


@pytest.mark.asyncio
async def test_state_machine_as_async_context_manager_cancel():
    """Verify StateMachine async context manager transitions to ABORTED on CancelledError."""
    import asyncio
    sm = StateMachine()
    sm.transition(State.THINKING)
    sm.transition(State.PLANNING)
    sm.transition(State.REVIEWING)
    try:
        async with sm:
            sm.transition(State.EXECUTING)
            raise asyncio.CancelledError()
    except asyncio.CancelledError:
        pass
    assert sm.state == State.ABORTED


@pytest.mark.asyncio
async def test_state_machine_transition_to_async_context_manager_success():
    """Verify StateMachine.transition_to async context manager transitions on enter and reverts on exit."""
    sm = StateMachine()
    assert sm.state == State.IDLE
    async with sm.transition_to(State.THINKING) as guarded_sm:
        assert guarded_sm.state == State.THINKING
    assert sm.state == State.IDLE


@pytest.mark.asyncio
async def test_state_machine_transition_to_async_context_manager_exception():
    """Verify StateMachine.transition_to async context manager transitions to ERROR on exception."""
    sm = StateMachine()
    try:
        async with sm.transition_to(State.THINKING):
            raise ValueError("Async error")
    except ValueError:
        pass
    assert sm.state == State.ERROR


@pytest.mark.asyncio
async def test_task_execution_context_as_async_context_manager_success():
    """Verify TaskExecutionContext async context manager leaves state unchanged on success."""
    from core.context.context import TaskExecutionContext
    ctx = TaskExecutionContext()
    async with ctx:
        ctx.state_machine.transition(State.THINKING)
    assert ctx.state_machine.state == State.THINKING


@pytest.mark.asyncio
async def test_task_execution_context_as_async_context_manager_exception():
    """Verify TaskExecutionContext async context manager transitions state machine to ERROR on exception."""
    from core.context.context import TaskExecutionContext
    ctx = TaskExecutionContext()
    try:
        async with ctx:
            ctx.state_machine.transition(State.THINKING)
            raise ValueError("Async failure in execution")
    except ValueError:
        pass
    assert ctx.state_machine.state == State.ERROR


@pytest.mark.asyncio
async def test_task_execution_context_as_async_context_manager_cancel():
    """Verify TaskExecutionContext async context manager transitions state machine to ABORTED on CancelledError."""
    from core.context.context import TaskExecutionContext
    import asyncio
    ctx = TaskExecutionContext()
    ctx.state_machine.transition(State.THINKING)
    ctx.state_machine.transition(State.PLANNING)
    ctx.state_machine.transition(State.REVIEWING)
    ctx.state_machine.transition(State.EXECUTING)
    try:
        async with ctx:
            raise asyncio.CancelledError()
    except asyncio.CancelledError:
        pass
    assert ctx.state_machine.state == State.ABORTED


