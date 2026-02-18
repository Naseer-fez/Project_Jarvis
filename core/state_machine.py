"""
core/state_machine.py
══════════════════════
Deterministic finite state machine for Jarvis V1.

V1 States: IDLE, LISTENING, OBSERVING, THINKING, PLANNING, ERROR, ABORTED
V1 Hard Rules:
  - No ACTING states exist in V1 (ACTING_DIGITAL, ACTING_PHYSICAL blocked)
  - Illegal transitions raise StateMachineError (never silently fail)
  - Every transition is logged with timestamp
  - Machine can always transition to ERROR or ABORTED from any state
"""

from enum import Enum, auto
from datetime import datetime, timezone
from core.logger import get_logger, audit

logger = get_logger("state_machine")


class State(str, Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    OBSERVING = "OBSERVING"
    THINKING = "THINKING"
    PLANNING = "PLANNING"
    ERROR = "ERROR"
    ABORTED = "ABORTED"
    # V2+ only — listed here so we can explicitly BLOCK them
    SPEAKING = "SPEAKING"
    ACTING_DIGITAL = "ACTING_DIGITAL"
    ACTING_PHYSICAL = "ACTING_PHYSICAL"


# V1 allowed states — anything outside this is BLOCKED
V1_ALLOWED_STATES = {
    State.IDLE,
    State.LISTENING,
    State.OBSERVING,
    State.THINKING,
    State.PLANNING,
    State.ERROR,
    State.ABORTED,
}

# Legal transitions map: State -> set of reachable States
LEGAL_TRANSITIONS: dict[State, set[State]] = {
    State.IDLE: {State.LISTENING, State.OBSERVING, State.THINKING, State.ERROR, State.ABORTED},
    State.LISTENING: {State.THINKING, State.IDLE, State.ERROR, State.ABORTED},
    State.OBSERVING: {State.THINKING, State.IDLE, State.ERROR, State.ABORTED},
    State.THINKING: {State.PLANNING, State.IDLE, State.ERROR, State.ABORTED},
    State.PLANNING: {State.IDLE, State.ERROR, State.ABORTED},  # NO acting in V1
    State.ERROR: {State.IDLE, State.ABORTED},
    State.ABORTED: {State.IDLE},
}


class StateMachineError(Exception):
    """Raised when an illegal state transition is attempted."""
    pass


class JarvisStateMachine:
    """
    Deterministic state machine. Illegal transitions are impossible —
    they raise immediately and are logged before raising.
    """

    def __init__(self):
        self._state = State.IDLE
        self._history: list[dict] = []
        self._record_transition(None, State.IDLE, "boot")

    @property
    def state(self) -> State:
        return self._state

    def transition(self, target: State, reason: str = "") -> State:
        """
        Attempt a state transition.
        Raises StateMachineError on illegal transitions.
        Logs every attempt (success or failure).
        """
        # Block V2+ states in V1
        if target not in V1_ALLOWED_STATES:
            msg = f"ILLEGAL: State {target.value} is NOT available in V1. Attempted from {self._state.value}."
            logger.error(msg)
            raise StateMachineError(msg)

        # Check transition table
        allowed = LEGAL_TRANSITIONS.get(self._state, set())
        if target not in allowed:
            msg = (
                f"ILLEGAL TRANSITION: {self._state.value} → {target.value}. "
                f"Allowed from {self._state.value}: {[s.value for s in allowed]}"
            )
            logger.error(msg)
            raise StateMachineError(msg)

        # Commit transition
        previous = self._state
        self._state = target
        self._record_transition(previous, target, reason)

        audit(
            logger,
            f"STATE: {previous.value} → {target.value} | reason={reason!r}",
            state=target.value,
            action="transition"
        )
        return self._state

    def to_error(self, reason: str) -> State:
        """Emergency transition to ERROR. Always legal from any V1 state."""
        previous = self._state
        self._state = State.ERROR
        self._record_transition(previous, State.ERROR, reason)
        logger.error(f"ERROR STATE: from={previous.value} reason={reason!r}")
        return self._state

    def abort(self, reason: str) -> State:
        """Abort. Always legal from any V1 state."""
        previous = self._state
        self._state = State.ABORTED
        self._record_transition(previous, State.ABORTED, reason)
        logger.warning(f"ABORTED: from={previous.value} reason={reason!r}")
        return self._state

    def reset(self) -> State:
        """Return to IDLE. Only legal from ERROR or ABORTED."""
        if self._state not in (State.ERROR, State.ABORTED, State.IDLE):
            raise StateMachineError(f"Cannot reset from {self._state.value}. Must be ERROR or ABORTED.")
        return self.transition(State.IDLE, reason="reset")

    def _record_transition(self, frm, to: State, reason: str):
        self._history.append({
            "ts": datetime.now(timezone.utc).isoformat(),
            "from": frm.value if frm else None,
            "to": to.value,
            "reason": reason,
        })

    def history(self) -> list[dict]:
        return list(self._history)

    def __repr__(self):
        return f"<JarvisStateMachine state={self._state.value}>"
