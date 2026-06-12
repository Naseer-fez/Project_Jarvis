# API Analyst Report: state_machine.py

## Dependencies
- `from __future__ import annotations`
- `import inspect`
- `import logging`
- `import threading`
- `from datetime import datetime`
- `from enum import Enum`
- `from typing import Callable`
- `from typing import Any`

## Configuration Variables
- `IDLE` = `'IDLE'`
- `THINKING` = `'THINKING'`
- `PLANNING` = `'PLANNING'`
- `RISK_EVALUATION` = `'RISK_EVALUATION'`
- `AWAITING_CONFIRMATION` = `'AWAITING_CONFIRMATION'`
- `APPROVED` = `'APPROVED'`
- `CANCELLED` = `'CANCELLED'`
- `ACTING` = `'ACTING'`
- `OBSERVING` = `'OBSERVING'`
- `REFLECTING` = `'REFLECTING'`
- `REVIEWING` = `'REVIEWING'`
- `EXECUTING` = `'EXECUTING'`
- `COMPLETED` = `'COMPLETED'`
- `SPEAKING` = `'SPEAKING'`
- `LISTENING` = `'LISTENING'`
- `TRANSCRIBING` = `'TRANSCRIBING'`
- `ERROR` = `'ERROR'`
- `ABORTED` = `'ABORTED'`
- `SHUTDOWN` = `'SHUTDOWN'`

## Schemas & API Contracts (Classes)

### Class `IllegalTransitionError(RuntimeError)`
> Raised when a state transition is not allowed.



### Class `State(str, Enum)`


### Class `StateGuard`
> Context manager to temporarily transition to a state, reverting back on exit.

**Methods:**
- `def __init__(self, state_machine: StateMachine, target_state: State) -> None`
- `def __enter__(self) -> StateMachine`
- `def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None`
- `async def __aenter__(self) -> StateMachine`
- `async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None`


### Class `StateMachine`
**Methods:**
- `def __init__(self, event_bus: Any=None) -> None`
- @property
- `def state(self) -> State`
- `def add_listener(self, listener: Callable[[State, State], None]) -> None`
- `def remove_listener(self, listener: Callable[[State, State], None]) -> None`
- `def can_transition(self, new_state: State) -> bool`
- `def get_valid_transitions(self, state: State | None=None) -> list[State]`
- `def get_transition_graph(self) -> dict[str, list[str]]`
- `def _notify(self, old_state: State, new_state: State) -> None`
- `def transition(self, new_state: State) -> State`
- `def reset(self) -> State`
- `def force_idle(self) -> State`
- `def transition_to(self, target_state: State) -> StateGuard`
  - *Return a context manager that temporarily transitions to target_state.*
- `def __enter__(self) -> StateMachine`
- `def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None`
- `async def __aenter__(self) -> StateMachine`
- `async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None`

