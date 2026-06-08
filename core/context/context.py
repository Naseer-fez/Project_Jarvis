"""
TaskExecutionContext — holds correlation IDs, state machine, execution logs,
and variables isolated to a single task execution flow.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any
from contextvars import Token
from core.state_machine import StateMachine, State
from core.logging.logger import set_trace_ids, reset_trace_ids

logger = logging.getLogger("Jarvis.Context")


class TaskExecutionContext:
    """Isolated execution context container for a task."""

    def __init__(
        self,
        trace_id: str | None = None,
        task_id: str | None = None,
        event_bus: Any = None,
        state_machine: StateMachine | None = None,
    ) -> None:
        self.trace_id = trace_id or uuid.uuid4().hex[:8]
        self.task_id = task_id or uuid.uuid4().hex[:8]
        self.event_bus = event_bus
        self.state_machine = state_machine or StateMachine(event_bus=event_bus)
        self.state_machine.task_id = self.task_id
        self.variables: dict[str, Any] = {}
        self.logs: list[str] = []
        self._trace_token: Token[str | None] | None = None
        self._task_token: Token[str | None] | None = None

    def log(self, message: str, level: str = "INFO") -> None:
        """Log an execution trace message, enriched with correlation IDs."""
        self.logs.append(message)
        log_level = getattr(logging, level.upper(), logging.INFO)
        logger.log(
            log_level,
            message,
            extra={"trace_id": self.trace_id, "task_id": self.task_id},
        )

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a variable value by key."""
        return self.variables.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a variable value by key."""
        self.variables[key] = value

    def __getitem__(self, key: str) -> Any:
        return self.variables[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.variables[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.variables

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "task_id": self.task_id,
            "variables": self.variables,
            "logs": self.logs,
            "state": self.state_machine.state.value if self.state_machine else "unknown",
        }

    async def save_snapshot(self, step_id: str | None = None, metadata: dict[str, Any] | None = None) -> None:
        import json
        from pathlib import Path
        import asyncio
        
        snapshot_dir = Path("logs/traces")
        
        snapshot_data = self.to_dict()
        snapshot_data["step_id"] = step_id
        snapshot_data["metadata"] = metadata or {}
        
        def _write():
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            snapshot_file = snapshot_dir / f"{self.trace_id}.json"
            try:
                snapshot_file.write_text(json.dumps(snapshot_data, indent=2, default=str), encoding="utf-8")
            except Exception as e:
                logger.warning("Failed to save trace snapshot for %s: %s", self.trace_id, e)
                
        await asyncio.to_thread(_write)


    def __enter__(self) -> TaskExecutionContext:
        self._trace_token, self._task_token = set_trace_ids(self.trace_id, self.task_id)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_type is not None:
            import asyncio
            target_state = State.ERROR
            if issubclass(exc_type, asyncio.CancelledError):
                target_state = State.ABORTED
            
            self.log(f"Context exited with exception: {exc_val} (transitioning to {target_state.value})", level="ERROR")
            if self.state_machine:
                try:
                    current_state = self.state_machine.state
                    if current_state not in {State.ERROR, State.ABORTED, State.COMPLETED, State.CANCELLED, State.SHUTDOWN}:
                        if self.state_machine.can_transition(target_state):
                            self.state_machine.transition(target_state)
                        else:
                            self.log(f"Cannot transition to {target_state.value} from {current_state}, forcing IDLE", level="WARNING")
                            self.state_machine.force_idle()
                except Exception as e:
                    self.log(f"Error during context cleanup: {e}", level="ERROR")
        
        if self._trace_token and self._task_token:
            reset_trace_ids(self._trace_token, self._task_token)

    async def __aenter__(self) -> TaskExecutionContext:
        self._trace_token, self._task_token = set_trace_ids(self.trace_id, self.task_id)
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.__exit__(exc_type, exc_val, exc_tb)


