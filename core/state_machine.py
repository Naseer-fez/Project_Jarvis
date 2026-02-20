"""
core/state_machine.py — Deterministic FSM for Jarvis.
"""
from enum import Enum, auto
import logging

log = logging.getLogger("jarvis.fsm")

class State(Enum):
    IDLE = auto()
    PLANNING = auto()
    EXECUTING = auto()  # Blocked in V1/V2, reserved for V3
    ERROR = auto()
    
    # --- V2 Voice States ---
    LISTENING = auto()
    TRANSCRIBING = auto()
    SPEAKING = auto()

class IllegalTransitionError(Exception):
    pass

class StateMachine:
    def __init__(self):
        self._state = State.IDLE
        
        # Define allowed transitions to prevent illegal logic jumps
        self._valid_transitions = {
            State.IDLE: {State.PLANNING, State.LISTENING, State.ERROR},
            
            # Text flow
            State.PLANNING: {State.EXECUTING, State.SPEAKING, State.IDLE, State.ERROR},
            State.EXECUTING: {State.IDLE, State.ERROR},
            
            # Voice flow
            State.LISTENING: {State.TRANSCRIBING, State.IDLE, State.ERROR},
            State.TRANSCRIBING: {State.PLANNING, State.IDLE, State.ERROR},
            State.SPEAKING: {State.IDLE, State.ERROR},
            
            State.ERROR: {State.IDLE}
        }

    @property
    def state(self) -> State:
        return self._state

    def transition(self, new_state: State) -> None:
        if new_state not in self._valid_transitions[self._state]:
            raise IllegalTransitionError(
                f"Cannot transition from {self._state.name} to {new_state.name}"
            )
        log.debug(f"FSM Transition: {self._state.name} -> {new_state.name}")
        self._state = new_state

    def force_idle(self) -> None:
        """Emergency recovery back to IDLE from any voice state."""
        log.debug(f"FSM Force Recovery: {self._state.name} -> IDLE")
        self._state = State.IDLE

    def reset(self) -> None:
        """Clear error state."""
        if self._state == State.ERROR:
            self._state = State.IDLE