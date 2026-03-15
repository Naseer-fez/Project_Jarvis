"""
core/state_machine.py — Compatibility shim.

Legacy test files (test_v1_acceptance.py, test_v2_acceptance.py) import:
    from core.state_machine import StateMachine, State, IllegalTransitionError

This shim provides those symbols by wrapping the canonical implementation
in core/controller/state_machine.py with the richer interface the tests expect.
"""

from __future__ import annotations

import logging
from enum import Enum, auto
from typing import Callable, List

logger = logging.getLogger("jarvis.state_machine")


class IllegalTransitionError(Exception):
    """Raised when a disallowed state transition is attempted."""


class State(Enum):
    """Canonical Jarvis agent states (V1 / V2 acceptance tests)."""
    IDLE          = auto()
    THINKING      = auto()
    PLANNING      = auto()
    RISK_EVALUATION = auto()
    AWAITING_CONFIRMATION = auto()
    ACTING        = auto()
    OBSERVING     = auto()
    REFLECTING    = auto()
    REVIEWING     = auto()
    EXECUTING     = auto()
    SPEAKING      = auto()
    LISTENING     = auto()
    TRANSCRIBING  = auto()
    ERROR         = auto()
    ABORTED       = auto()
    SHUTDOWN      = auto()


# Legal transitions table
_TRANSITIONS: dict[State, frozenset[State]] = {
    State.IDLE:         frozenset({State.THINKING, State.PLANNING, State.LISTENING, State.SHUTDOWN}),
    State.THINKING:     frozenset({State.IDLE, State.PLANNING, State.ERROR}),
    State.PLANNING:     frozenset({State.RISK_EVALUATION, State.REVIEWING, State.IDLE, State.ERROR, State.SPEAKING}),
    State.RISK_EVALUATION: frozenset({State.AWAITING_CONFIRMATION, State.ACTING, State.IDLE, State.ERROR}),
    State.AWAITING_CONFIRMATION: frozenset({State.ACTING, State.IDLE, State.ERROR}),
    State.ACTING:       frozenset({State.OBSERVING, State.IDLE, State.ERROR}),
    State.OBSERVING:    frozenset({State.ACTING, State.REFLECTING, State.IDLE, State.ERROR}),
    State.REFLECTING:   frozenset({State.SPEAKING, State.IDLE, State.ERROR}),
    State.REVIEWING:    frozenset({State.EXECUTING, State.ABORTED, State.IDLE, State.ERROR}),
    State.EXECUTING:    frozenset({State.IDLE, State.ERROR, State.ABORTED}),
    State.SPEAKING:     frozenset({State.IDLE, State.LISTENING, State.ERROR}),
    State.LISTENING:    frozenset({State.TRANSCRIBING, State.IDLE, State.ERROR}),
    State.TRANSCRIBING: frozenset({State.PLANNING, State.IDLE, State.ERROR}),
    State.ERROR:        frozenset({State.IDLE, State.SHUTDOWN}),
    State.ABORTED:      frozenset({State.IDLE, State.SHUTDOWN}),
    State.SHUTDOWN:     frozenset(),
}

_RESETTABLE = frozenset({State.ERROR, State.ABORTED})


class StateMachine:
    """
    Explicit finite-state machine for the Jarvis agent lifecycle.

    All transitions are validated; illegal ones raise IllegalTransitionError.
    Listeners receive (old_state, new_state) on every successful transition.
    """

    def __init__(self) -> None:
        self._state: State = State.IDLE
        self._listeners: List[Callable[[State, State], None]] = []

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def state(self) -> State:
        return self._state

    # ── Transition ────────────────────────────────────────────────────────────

    def transition(self, new_state: State) -> None:
        """
        Move to *new_state*.  Raises IllegalTransitionError if the transition
        is not in the allowed table.
        """
        allowed = _TRANSITIONS.get(self._state, frozenset())
        if new_state not in allowed:
            raise IllegalTransitionError(
                f"Illegal transition: {self._state.name} → {new_state.name} "
                f"(allowed: {[s.name for s in allowed]})"
            )
        old = self._state
        self._state = new_state
        logger.debug("State: %s → %s", old.name, new_state.name)
        for cb in self._listeners:
            try:
                cb(old, new_state)
            except Exception:  # noqa: BLE001
                pass

    def can_transition(self, new_state: State) -> bool:
        """Return True if *new_state* is a legal next state from the current one."""
        return new_state in _TRANSITIONS.get(self._state, frozenset())

    def reset(self) -> None:
        """
        Return to IDLE from ERROR or ABORTED.
        Raises IllegalTransitionError if called from any other state.
        """
        if self._state not in _RESETTABLE:
            raise IllegalTransitionError(
                f"reset() called from non-resettable state {self._state.name}"
            )
        self.transition(State.IDLE)

    def force_idle(self) -> None:
        """Emergency reset — always succeeds regardless of current state."""
        old = self._state
        self._state = State.IDLE
        logger.info("Force-reset: %s → IDLE", old.name)
        for cb in self._listeners:
            try:
                cb(old, State.IDLE)
            except Exception:  # noqa: BLE001
                pass

    # ── Listeners ─────────────────────────────────────────────────────────────

    def add_listener(self, callback: Callable[[State, State], None]) -> None:
        """Register a callback invoked after each successful transition."""
        self._listeners.append(callback)

    # ── Repr ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"<StateMachine state={self._state.name}>"


__all__ = ["State", "StateMachine", "IllegalTransitionError"]
