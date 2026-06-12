# API Analyst Report: context\context.py

## Dependencies
- `from __future__ import annotations`
- `import logging`
- `import uuid`
- `from typing import Any`
- `from contextvars import Token`
- `from core.state_machine import StateMachine`
- `from core.state_machine import State`
- `from core.logging.logger import set_trace_ids`
- `from core.logging.logger import reset_trace_ids`

## Schemas & API Contracts (Classes)

### Class `TaskExecutionContext`
> Isolated execution context container for a task.

**Methods:**
- `def __init__(self, trace_id: str | None=None, task_id: str | None=None, event_bus: Any=None, state_machine: StateMachine | None=None) -> None`
- `def log(self, message: str, level: str='INFO') -> None`
  - *Log an execution trace message, enriched with correlation IDs.*
- `def get(self, key: str, default: Any=None) -> Any`
  - *Retrieve a variable value by key.*
- `def set(self, key: str, value: Any) -> None`
  - *Set a variable value by key.*
- `def __getitem__(self, key: str) -> Any`
- `def __setitem__(self, key: str, value: Any) -> None`
- `def __contains__(self, key: str) -> bool`
- `def to_dict(self) -> dict[str, Any]`
- `async def save_snapshot(self, step_id: str | None=None, metadata: dict[str, Any] | None=None) -> None`
- `def __enter__(self) -> TaskExecutionContext`
- `def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None`
- `async def __aenter__(self) -> TaskExecutionContext`
- `async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None`

