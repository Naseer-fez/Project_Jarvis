"""Finite-state machine used across legacy and current Jarvis flows."""

from __future__ import annotations

import inspect
import logging
import threading
from datetime import datetime
from enum import Enum
from typing import Callable, Any

logger = logging.getLogger("Jarvis.StateMachine")


class IllegalTransitionError(RuntimeError):
    """Raised when a state transition is not allowed."""


class State(str, Enum):
    IDLE = "IDLE"
    THINKING = "THINKING"
    PLANNING = "PLANNING"
    RISK_EVALUATION = "RISK_EVALUATION"
    AWAITING_CONFIRMATION = "AWAITING_CONFIRMATION"
    APPROVED = "APPROVED"
    CANCELLED = "CANCELLED"
    ACTING = "ACTING"
    OBSERVING = "OBSERVING"
    REFLECTING = "REFLECTING"
    REVIEWING = "REVIEWING"
    EXECUTING = "EXECUTING"
    COMPLETED = "COMPLETED"
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
        State.APPROVED,
        State.CANCELLED,
        State.ACTING,
        State.IDLE,
        State.ERROR,
    },
    State.AWAITING_CONFIRMATION: {
        State.APPROVED,
        State.CANCELLED,
        State.ACTING,
        State.IDLE,
        State.ERROR,
    },
    State.APPROVED: {
        State.EXECUTING,
        State.ACTING,
        State.IDLE,
        State.ERROR,
    },
    State.ACTING: {State.OBSERVING, State.IDLE, State.ERROR},
    State.OBSERVING: {State.ACTING, State.REFLECTING, State.IDLE, State.ERROR},
    State.REFLECTING: {State.SPEAKING, State.IDLE, State.ERROR, State.COMPLETED},
    State.REVIEWING: {State.EXECUTING, State.ABORTED, State.IDLE, State.ERROR},
    State.EXECUTING: {
        State.REFLECTING,
        State.SPEAKING,
        State.COMPLETED,
        State.IDLE,
        State.ERROR,
        State.ABORTED,
    },
    State.COMPLETED: {State.IDLE, State.SHUTDOWN},
    State.CANCELLED: {State.IDLE, State.SHUTDOWN},
    State.SPEAKING: {State.IDLE, State.LISTENING, State.ERROR, State.COMPLETED},
    State.LISTENING: {State.TRANSCRIBING, State.IDLE, State.ERROR},
    State.TRANSCRIBING: {State.PLANNING, State.IDLE, State.ERROR},
    State.ERROR: {State.IDLE, State.SHUTDOWN},
    State.ABORTED: {State.IDLE, State.SHUTDOWN},
    State.SHUTDOWN: set(),
}


class StateGuard:
    """Context manager to temporarily transition to a state, reverting back on exit."""

    def __init__(self, state_machine: StateMachine, target_state: State) -> None:
        self.sm = state_machine
        self.target_state = target_state
        self.previous_state: State | None = None

    def __enter__(self) -> StateMachine:
        self.previous_state = self.sm.state
        self.sm.transition(self.target_state)
        return self.sm

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_type is not None:
            import asyncio
            error_state = State.ERROR
            if issubclass(exc_type, asyncio.CancelledError):
                error_state = State.ABORTED
            
            try:
                if self.sm.state not in {State.ERROR, State.ABORTED, State.COMPLETED, State.CANCELLED, State.SHUTDOWN}:
                    if self.sm.can_transition(error_state):
                        self.sm.transition(error_state)
                    else:
                        self.sm.force_idle()
            except Exception:
                pass
        else:
            try:
                if self.sm.state not in {State.ERROR, State.ABORTED, State.COMPLETED, State.CANCELLED, State.SHUTDOWN}:
                    if self.previous_state and self.sm.can_transition(self.previous_state):
                        self.sm.transition(self.previous_state)
                    else:
                        self.sm.force_idle()
            except Exception:
                pass

    async def __aenter__(self) -> StateMachine:
        return self.__enter__()

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.__exit__(exc_type, exc_val, exc_tb)


class StateMachine:
    def __init__(self, event_bus: Any = None) -> None:
        self._state = State.IDLE
        self._listeners: list[Callable[[State, State], None]] = []
        self.event_bus = event_bus
        self.task_id: str | None = None
        self.diagnostics_mode: bool = False
        self._transition_audit_trail: list[dict[str, Any]] = []
        self._lock = threading.RLock()
        self._pending_notifications: list[tuple[State, State]] = []
        self._notifying = False

    @property
    def state(self) -> State:
        with self._lock:
            return self._state

    def add_listener(self, listener: Callable[[State, State], None]) -> None:
        with self._lock:
            if listener not in self._listeners:
                self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[State, State], None]) -> None:
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    def can_transition(self, new_state: State) -> bool:
        with self._lock:
            try:
                candidate = State(new_state)
            except ValueError:
                return False
            return candidate in _ALLOWED_TRANSITIONS.get(self._state, set())

    def get_valid_transitions(self, state: State | None = None) -> list[State]:
        with self._lock:
            target = State(state) if state is not None else self._state
            return sorted(list(_ALLOWED_TRANSITIONS.get(target, set())), key=lambda s: s.value)

    def get_transition_graph(self) -> dict[str, list[str]]:
        with self._lock:
            return {
                state.value: sorted([t.value for t in targets])
                for state, targets in _ALLOWED_TRANSITIONS.items()
            }

    def _notify(self, old_state: State, new_state: State) -> None:
        with self._lock:
            self._pending_notifications.append((old_state, new_state))
            if self._notifying:
                return
            self._notifying = True

        try:
            while True:
                with self._lock:
                    if not self._pending_notifications:
                        self._notifying = False
                        break
                    old_s, new_s = self._pending_notifications.pop(0)
                    listeners = list(self._listeners)

                # Execute callbacks outside the lock
                for listener in listeners:
                    try:
                        listener(old_s, new_s)
                    except Exception as e:
                        logger.warning("Listener callback failed: %s", e)

                if self.event_bus:
                    try:
                        self.event_bus.publish(
                            "state_transition",
                            {"old_state": old_s.value, "new_state": new_s.value}
                        )
                    except Exception as e:
                        logger.warning("Event bus publish failed: %s", e)
        finally:
            with self._lock:
                if self._notifying:
                    self._notifying = False

    def transition(self, new_state: State) -> State:
        candidate = State(new_state)
        
        # 1. Identify the caller file/line/function
        caller = "unknown"
        try:
            stack = inspect.stack()
            for frame in stack[1:]:
                if "state_machine.py" not in frame.filename:
                    caller = f"{frame.filename}:{frame.lineno} in {frame.function}"
                    break
        except Exception:
            pass

        with self._lock:
            old_state = self._state
            history = [f"{t['from_state']}->{t['to_state']}" for t in self._transition_audit_trail if t["success"]]
            history_str = " -> ".join(history[-5:]) or "None"

            # Log attempt structured debug
            logger.debug(
                "State transition requested",
                extra={
                    "from": old_state.value,
                    "to": candidate.value,
                    "task_id": self.task_id,
                    "caller": caller,
                    "history": history_str,
                }
            )

            # 2. Check validity
            if not self.can_transition(candidate):
                # Audit trail failure entry
                self._transition_audit_trail.append({
                    "timestamp": datetime.now().isoformat(),
                    "from_state": old_state.value,
                    "to_state": candidate.value,
                    "success": False,
                    "caller": caller,
                    "task_id": self.task_id,
                })
                if len(self._transition_audit_trail) > 100:
                    self._transition_audit_trail.pop(0)
                
                # Format extremely informative error
                allowed = self.get_valid_transitions(old_state)
                allowed_str = "\n".join(f"- {s.value}" for s in allowed)
                raise IllegalTransitionError(
                    f"Cannot transition {old_state.value} -> {candidate.value}\n"
                    f"Allowed:\n{allowed_str}"
                )

            # 3. Successful transition
            self._state = candidate
            self._transition_audit_trail.append({
                "timestamp": datetime.now().isoformat(),
                "from_state": old_state.value,
                "to_state": candidate.value,
                "success": True,
                "caller": caller,
                "task_id": self.task_id,
            })
            if len(self._transition_audit_trail) > 100:
                self._transition_audit_trail.pop(0)

        # 4. Notify (outside the lock)
        self._notify(old_state, candidate)
        return candidate

    def reset(self) -> State:
        with self._lock:
            if self._state not in {State.ERROR, State.ABORTED}:
                raise IllegalTransitionError(
                    f"Cannot reset from state {self._state.value}"
                )
            old_state = self._state
            self._state = State.IDLE
            
            self._transition_audit_trail.append({
                "timestamp": datetime.now().isoformat(),
                "from_state": old_state.value,
                "to_state": State.IDLE.value,
                "success": True,
                "caller": "reset()",
                "task_id": self.task_id,
            })
            if len(self._transition_audit_trail) > 100:
                self._transition_audit_trail.pop(0)
            
        self._notify(old_state, State.IDLE)
        return State.IDLE

    def force_idle(self) -> State:
        with self._lock:
            if self._state == State.IDLE:
                return self._state
            old_state = self._state
            self._state = State.IDLE
            
            self._transition_audit_trail.append({
                "timestamp": datetime.now().isoformat(),
                "from_state": old_state.value,
                "to_state": State.IDLE.value,
                "success": True,
                "caller": "force_idle()",
                "task_id": self.task_id,
            })
            if len(self._transition_audit_trail) > 100:
                self._transition_audit_trail.pop(0)
            
        self._notify(old_state, State.IDLE)
        return State.IDLE

    def transition_to(self, target_state: State) -> StateGuard:
        """Return a context manager that temporarily transitions to target_state."""
        return StateGuard(self, target_state)

    def __enter__(self) -> StateMachine:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_type is not None:
            import asyncio
            target_state = State.ERROR
            if issubclass(exc_type, asyncio.CancelledError):
                target_state = State.ABORTED
            try:
                if self._state not in {State.ERROR, State.ABORTED, State.COMPLETED, State.CANCELLED, State.SHUTDOWN}:
                    if self.can_transition(target_state):
                        self.transition(target_state)
                    else:
                        self.force_idle()
            except Exception:
                pass

    async def __aenter__(self) -> StateMachine:
        return self.__enter__()

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.__exit__(exc_type, exc_val, exc_tb)


__all__ = ["IllegalTransitionError", "State", "StateMachine", "StateGuard"]

