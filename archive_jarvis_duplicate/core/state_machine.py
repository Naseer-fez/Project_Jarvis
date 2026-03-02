"""
JARVIS State Machine - Session 5
Trusted Core: All transitions are deterministic and logged.
NO state can be bypassed. IDLE -> LISTENING -> PLANNING -> RISK_CHECK -> EXECUTING -> IDLE
"""

import logging
from enum import Enum, auto
from typing import Optional, Callable
from datetime import datetime

logger = logging.getLogger("JARVIS.StateMachine")


class State(Enum):
    IDLE = auto()
    LISTENING = auto()
    TRANSCRIBING = auto()
    PLANNING = auto()
    RISK_CHECK = auto()
    EXECUTING = auto()
    RESPONDING = auto()
    ERROR = auto()


# Legal state transitions - CANNOT be bypassed
LEGAL_TRANSITIONS = {
    State.IDLE:         [State.LISTENING, State.ERROR],
    State.LISTENING:    [State.TRANSCRIBING, State.IDLE, State.ERROR],
    State.TRANSCRIBING: [State.PLANNING, State.IDLE, State.ERROR],
    State.PLANNING:     [State.RISK_CHECK, State.IDLE, State.ERROR],
    State.RISK_CHECK:   [State.EXECUTING, State.RESPONDING, State.IDLE, State.ERROR],
    State.EXECUTING:    [State.RESPONDING, State.ERROR],
    State.RESPONDING:   [State.IDLE, State.ERROR],
    State.ERROR:        [State.IDLE],
}


class JarvisStateMachine:
    def __init__(self):
        self._state = State.IDLE
        self._history: list[dict] = []
        self._on_transition: Optional[Callable] = None
        logger.info(f"StateMachine initialized. Starting state: {self._state.name}")

    @property
    def state(self) -> State:
        return self._state

    def transition(self, new_state: State, reason: str = "") -> bool:
        """Attempt a state transition. Returns True if successful."""
        if new_state not in LEGAL_TRANSITIONS.get(self._state, []):
            logger.error(
                f"ILLEGAL TRANSITION: {self._state.name} -> {new_state.name} | "
                f"Reason: {reason} | Legal transitions: "
                f"{[s.name for s in LEGAL_TRANSITIONS.get(self._state, [])]}"
            )
            return False

        old_state = self._state
        self._state = new_state
        entry = {
            "from": old_state.name,
            "to": new_state.name,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        }
        self._history.append(entry)
        logger.info(f"State: {old_state.name} -> {new_state.name} | {reason}")

        if self._on_transition:
            self._on_transition(old_state, new_state, reason)

        return True

    def force_idle(self):
        """Emergency reset to IDLE from any state."""
        logger.warning(f"FORCE RESET to IDLE from {self._state.name}")
        self._state = State.IDLE

    def set_transition_hook(self, callback: Callable):
        self._on_transition = callback

    def get_history(self, last_n: int = 10) -> list[dict]:
        return self._history[-last_n:]

    def is_idle(self) -> bool:
        return self._state == State.IDLE
