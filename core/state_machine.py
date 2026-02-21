"""
core/state_machine.py - Deterministic FSM for Jarvis.
"""

from __future__ import annotations

import logging
from enum import Enum, auto
from typing import Callable

log = logging.getLogger("jarvis.fsm")


class State(Enum):
    IDLE = auto()
    PLANNING = auto()
    EXECUTING = auto()  # Enabled for V3 tool execution
    ERROR = auto()

    # Voice states
    LISTENING = auto()
    TRANSCRIBING = auto()
    SPEAKING = auto()


class IllegalTransitionError(Exception):
    pass


StateListener = Callable[[State, State], None]


class StateMachine:
    def __init__(self):
        self._state = State.IDLE
        self._listeners: list[StateListener] = []

        # Define allowed transitions to prevent illegal logic jumps
        self._valid_transitions = {
            State.IDLE: {State.PLANNING, State.LISTENING, State.ERROR},

            # Text/voice planning flow
            State.PLANNING: {State.EXECUTING, State.SPEAKING, State.IDLE, State.ERROR},

            # V3 execution flow
            State.EXECUTING: {State.SPEAKING, State.IDLE, State.ERROR},

            # Voice flow
            State.LISTENING: {State.TRANSCRIBING, State.IDLE, State.ERROR},
            State.TRANSCRIBING: {State.PLANNING, State.IDLE, State.ERROR},
            State.SPEAKING: {State.IDLE, State.ERROR},

            State.ERROR: {State.IDLE},
        }

    @property
    def state(self) -> State:
        return self._state

    def add_listener(self, listener: StateListener) -> None:
        self._listeners.append(listener)

    def can_transition(self, new_state: State) -> bool:
        return new_state in self._valid_transitions[self._state]

    def transition(self, new_state: State) -> None:
        if not self.can_transition(new_state):
            raise IllegalTransitionError(
                f"Cannot transition from {self._state.name} to {new_state.name}"
            )

        old_state = self._state
        log.debug(f"FSM Transition: {old_state.name} -> {new_state.name}")
        self._state = new_state
        self._notify_listeners(old_state, new_state)

    def force_idle(self) -> None:
        """Emergency recovery back to IDLE from any state."""
        old_state = self._state
        log.debug(f"FSM Force Recovery: {old_state.name} -> IDLE")
        self._state = State.IDLE
        self._notify_listeners(old_state, State.IDLE)

    def reset(self) -> None:
        """Clear error state."""
        if self._state == State.ERROR:
            old_state = self._state
            self._state = State.IDLE
            self._notify_listeners(old_state, State.IDLE)

    def _notify_listeners(self, old: State, new: State) -> None:
        for listener in list(self._listeners):
            try:
                listener(old, new)
            except Exception:
                log.exception("State listener failed")
