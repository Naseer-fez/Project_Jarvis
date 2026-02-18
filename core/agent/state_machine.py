"""
State machine for Jarvis agent lifecycle.
All state transitions are explicit and logged.
"""

import logging
from enum import Enum, auto
from typing import Optional

logger = logging.getLogger("Jarvis.StateMachine")


class AgentState(Enum):
    IDLE = auto()
    LISTENING = auto()
    THINKING = auto()
    PLANNING = auto()
    RISK_EVALUATION = auto()
    AWAITING_CONFIRMATION = auto()
    ACTING = auto()
    OBSERVING = auto()
    REFLECTING = auto()
    SPEAKING = auto()
    ERROR = auto()


# Legal state transitions
TRANSITIONS: dict[AgentState, list[AgentState]] = {
    AgentState.IDLE: [AgentState.LISTENING, AgentState.THINKING],
    AgentState.LISTENING: [AgentState.THINKING, AgentState.IDLE],
    AgentState.THINKING: [AgentState.PLANNING, AgentState.SPEAKING, AgentState.IDLE, AgentState.ERROR],
    AgentState.PLANNING: [AgentState.RISK_EVALUATION, AgentState.ERROR],
    AgentState.RISK_EVALUATION: [AgentState.AWAITING_CONFIRMATION, AgentState.ACTING, AgentState.IDLE, AgentState.ERROR],
    AgentState.AWAITING_CONFIRMATION: [AgentState.ACTING, AgentState.IDLE],
    AgentState.ACTING: [AgentState.OBSERVING, AgentState.ERROR],
    AgentState.OBSERVING: [AgentState.REFLECTING, AgentState.ACTING, AgentState.ERROR],
    AgentState.REFLECTING: [AgentState.SPEAKING, AgentState.IDLE],
    AgentState.SPEAKING: [AgentState.IDLE, AgentState.LISTENING],
    AgentState.ERROR: [AgentState.IDLE],
}

# States that can be interrupted by a stop command
INTERRUPTIBLE_STATES = {
    AgentState.THINKING,
    AgentState.PLANNING,
    AgentState.RISK_EVALUATION,
    AgentState.ACTING,
    AgentState.OBSERVING,
    AgentState.SPEAKING,
}


class StateMachine:
    def __init__(self):
        self._state = AgentState.IDLE
        self._previous: Optional[AgentState] = None

    @property
    def state(self) -> AgentState:
        return self._state

    def transition(self, new_state: AgentState) -> bool:
        allowed = TRANSITIONS.get(self._state, [])
        if new_state not in allowed:
            logger.warning(
                f"Illegal transition: {self._state.name} -> {new_state.name} (allowed: {[s.name for s in allowed]})"
            )
            return False
        logger.debug(f"State: {self._state.name} -> {new_state.name}")
        self._previous = self._state
        self._state = new_state
        return True

    def force_idle(self):
        """Emergency reset — always allowed (e.g., voice stop command)."""
        logger.info(f"Force-reset: {self._state.name} -> IDLE")
        self._previous = self._state
        self._state = AgentState.IDLE

    def is_interruptible(self) -> bool:
        return self._state in INTERRUPTIBLE_STATES

    def __repr__(self):
        return f"<StateMachine state={self._state.name}>"

