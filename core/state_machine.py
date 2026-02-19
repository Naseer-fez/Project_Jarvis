"""
core/state_machine.py — Deterministic Finite State Machine.

All state transitions are explicit. Any attempt to make an illegal transition
raises IllegalTransitionError immediately — there is no silent failure.

V1 states: IDLE, PLANNING, REVIEWING, EXECUTING, ERROR, ABORTED, SHUTDOWN
V2 adds:   LISTENING, TRANSCRIBING, SPEAKING  (voice sub-states)
"""

from __future__ import annotations

import threading
from enum import Enum, auto
from typing import Callable


class State(Enum):
    # ── V1 states ─────────────────────────────────────────────────────────────
    IDLE        = auto()   # Ready for input
    PLANNING    = auto()   # LLM generating a plan
    REVIEWING   = auto()   # Human confirming high-risk plan
    EXECUTING   = auto()   # Approved plan running (V3 only — stub in V1/V2)
    ERROR       = auto()   # Recoverable error
    ABORTED     = auto()   # User aborted — requires explicit reset
    SHUTDOWN    = auto()   # Terminal — no recovery

    # ── V2 voice states ───────────────────────────────────────────────────────
    LISTENING   = auto()   # Wake word fired — capturing audio
    TRANSCRIBING = auto()  # STT processing captured audio
    SPEAKING    = auto()   # TTS rendering response


class IllegalTransitionError(Exception):
    pass


# Allowed transitions: source → {allowed destinations}
_TRANSITIONS: dict[State, set[State]] = {
    State.IDLE:         {State.PLANNING, State.LISTENING, State.SHUTDOWN},
    State.PLANNING:     {State.REVIEWING, State.IDLE, State.ERROR},
    State.REVIEWING:    {State.EXECUTING, State.ABORTED, State.IDLE, State.ERROR},
    State.EXECUTING:    {State.IDLE, State.ERROR, State.ABORTED},
    State.ERROR:        {State.IDLE, State.SHUTDOWN},
    State.ABORTED:      {State.IDLE, State.SHUTDOWN},
    State.SHUTDOWN:     set(),  # Terminal

    # Voice transitions
    State.LISTENING:    {State.TRANSCRIBING, State.IDLE, State.ERROR},
    State.TRANSCRIBING: {State.PLANNING, State.IDLE, State.ERROR},
    State.SPEAKING:     {State.IDLE, State.LISTENING, State.ERROR},
}

# States that PLANNING can reach from (voice path adds TRANSCRIBING)
# Already covered above.

# States from which SPEAKING can be entered (after PLANNING produces output)
# PLANNING → SPEAKING is allowed so voice loop can speak before IDLE
_TRANSITIONS[State.PLANNING].add(State.SPEAKING)


class StateMachine:
    def __init__(self) -> None:
        self._state = State.IDLE
        self._lock = threading.RLock()
        self._listeners: list[Callable[[State, State], None]] = []

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def state(self) -> State:
        with self._lock:
            return self._state

    def transition(self, new_state: State) -> None:
        with self._lock:
            allowed = _TRANSITIONS.get(self._state, set())
            if new_state not in allowed:
                raise IllegalTransitionError(
                    f"Illegal transition: {self._state.name} → {new_state.name}"
                )
            old = self._state
            self._state = new_state
        # Notify outside the lock to avoid deadlocks
        for listener in self._listeners:
            try:
                listener(old, new_state)
            except Exception:
                pass

    def add_listener(self, fn: Callable[[State, State], None]) -> None:
        self._listeners.append(fn)

    def can_transition(self, new_state: State) -> bool:
        with self._lock:
            return new_state in _TRANSITIONS.get(self._state, set())

    def reset(self) -> None:
        """Recover from ERROR or ABORTED → IDLE."""
        with self._lock:
            if self._state not in (State.ERROR, State.ABORTED):
                raise IllegalTransitionError(
                    f"reset() only valid from ERROR or ABORTED, not {self._state.name}"
                )
            old = self._state
            self._state = State.IDLE
        for listener in self._listeners:
            try:
                listener(old, State.IDLE)
            except Exception:
                pass

    def force_idle(self) -> None:
        """
        Emergency return to IDLE from any voice state.
        Used by cancel-word handler and timeouts.
        Only valid from voice states or IDLE itself.
        """
        with self._lock:
            voice_states = {State.LISTENING, State.TRANSCRIBING, State.SPEAKING}
            if self._state in voice_states or self._state == State.IDLE:
                old = self._state
                self._state = State.IDLE
            else:
                raise IllegalTransitionError(
                    f"force_idle() not valid from {self._state.name}"
                )
        for listener in self._listeners:
            try:
                listener(old, State.IDLE)
            except Exception:
                pass

    def __repr__(self) -> str:
        return f"<StateMachine state={self._state.name}>"
