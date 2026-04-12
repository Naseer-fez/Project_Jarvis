"""Finite-state machine used across legacy and current Jarvis flows."""

from __future__ import annotations

from enum import Enum
from typing import Callable


class IllegalTransitionError(RuntimeError):
    """Raised when a state transition is not allowed."""


class State(str, Enum):
    IDLE = "IDLE"
    THINKING = "THINKING"
    PLANNING = "PLANNING"
    RISK_EVALUATION = "RISK_EVALUATION"
    AWAITING_CONFIRMATION = "AWAITING_CONFIRMATION"
    ACTING = "ACTING"
    OBSERVING = "OBSERVING"
    REFLECTING = "REFLECTING"
    REVIEWING = "REVIEWING"
    EXECUTING = "EXECUTING"
    SPEAKING = "SPEAKING"
    LISTENING = "LISTENING"
    TRANSCRIBING = "TRANSCRIBING"
    ERROR = "ERROR"
    ABORTED = "ABORTED"
    SHUTDOWN = "SHUTDOWN"


_ALLOWED_TRANSITIONS: dict[State, set[State]] = {
    State.IDLE: {State.THINKING, State.PLANNING, State.LISTENING, State.SHUTDOWN},
    State.THINKING: {State.IDLE, State.PLANNING, State.ERROR},
    State.PLANNING: {
        State.RISK_EVALUATION,
        State.REVIEWING,
        State.IDLE,
        State.ERROR,
        State.SPEAKING,
    },
    State.RISK_EVALUATION: {
        State.AWAITING_CONFIRMATION,
        State.ACTING,
        State.IDLE,
        State.ERROR,
    },
    State.AWAITING_CONFIRMATION: {State.ACTING, State.IDLE, State.ERROR},
    State.ACTING: {State.OBSERVING, State.IDLE, State.ERROR},
    State.OBSERVING: {State.ACTING, State.REFLECTING, State.IDLE, State.ERROR},
    State.REFLECTING: {State.SPEAKING, State.IDLE, State.ERROR},
    State.REVIEWING: {State.EXECUTING, State.ABORTED, State.IDLE, State.ERROR},
    State.EXECUTING: {State.IDLE, State.ERROR, State.ABORTED},
    State.SPEAKING: {State.IDLE, State.LISTENING, State.ERROR},
    State.LISTENING: {State.TRANSCRIBING, State.IDLE, State.ERROR},
    State.TRANSCRIBING: {State.PLANNING, State.IDLE, State.ERROR},
    State.ERROR: {State.IDLE, State.SHUTDOWN},
    State.ABORTED: {State.IDLE, State.SHUTDOWN},
    State.SHUTDOWN: set(),
}


class StateMachine:
    def __init__(self) -> None:
        self._state = State.IDLE
        self._listeners: list[Callable[[State, State], None]] = []

    @property
    def state(self) -> State:
        return self._state

    def add_listener(self, listener: Callable[[State, State], None]) -> None:
        self._listeners.append(listener)

    def can_transition(self, new_state: State) -> bool:
        try:
            candidate = State(new_state)
        except ValueError:
            return False
        return candidate in _ALLOWED_TRANSITIONS.get(self._state, set())

    def transition(self, new_state: State) -> State:
        candidate = State(new_state)
        if not self.can_transition(candidate):
            raise IllegalTransitionError(
                f"Illegal transition: {self._state.value} -> {candidate.value}"
            )

        old_state = self._state
        self._state = candidate
        for listener in list(self._listeners):
            listener(old_state, candidate)
        return self._state

    def reset(self) -> State:
        if self._state not in {State.ERROR, State.ABORTED}:
            raise IllegalTransitionError(
                f"Cannot reset from state {self._state.value}"
            )
        old_state = self._state
        self._state = State.IDLE
        for listener in list(self._listeners):
            listener(old_state, self._state)
        return self._state

    def force_idle(self) -> State:
        if self._state == State.IDLE:
            return self._state
        old_state = self._state
        self._state = State.IDLE
        for listener in list(self._listeners):
            listener(old_state, self._state)
        return self._state


__all__ = ["IllegalTransitionError", "State", "StateMachine"]
