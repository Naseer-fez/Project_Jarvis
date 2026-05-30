import pytest
from core.state_machine import StateMachine, State, IllegalTransitionError


def test_state_machine_initial_state():
    """Verify that state machine initializes to IDLE."""
    sm = StateMachine()
    assert sm.state == State.IDLE


def test_state_machine_can_transition():
    """Verify that can_transition correctly checks validity of transitions."""
    sm = StateMachine()
    assert sm.can_transition(State.THINKING) is True
    assert sm.can_transition(State.PLANNING) is True
    assert sm.can_transition(State.ACTING) is False  # IDLE -> ACTING is not allowed


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
