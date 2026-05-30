"""
TaskExecutionContext — holds correlation IDs, state machine, execution logs,
and variables isolated to a single task execution flow.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any
from core.state_machine import StateMachine

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
        self.variables: dict[str, Any] = {}
        self.logs: list[str] = []

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

    def save_snapshot(self, step_id: str | None = None, metadata: dict[str, Any] | None = None) -> None:
        import json
        from pathlib import Path
        
        snapshot_dir = Path("logs/traces")
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot_file = snapshot_dir / f"{self.trace_id}.json"
        
        snapshot_data = self.to_dict()
        snapshot_data["step_id"] = step_id
        snapshot_data["metadata"] = metadata or {}
        
        try:
            snapshot_file.write_text(json.dumps(snapshot_data, indent=2, default=str), encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to save trace snapshot for %s: %s", self.trace_id, e)

    @classmethod
    def load_snapshot(cls, filepath: str, event_bus: Any = None) -> TaskExecutionContext:
        import json
        from pathlib import Path
        
        p = Path(filepath)
        if not p.exists():
            p = Path("logs/traces") / f"{filepath}.json"
            if not p.exists():
                raise FileNotFoundError(f"Snapshot not found at {filepath}")
                
        data = json.loads(p.read_text(encoding="utf-8"))
        ctx = cls(
            trace_id=data.get("trace_id"),
            task_id=data.get("task_id"),
            event_bus=event_bus,
        )
        ctx.variables = data.get("variables", {})
        ctx.logs = data.get("logs", [])
        if ctx.state_machine and "state" in data:
            from core.state_machine import State
            try:
                ctx.state_machine._state = State(data["state"])
            except ValueError:
                pass
            
        return ctx

