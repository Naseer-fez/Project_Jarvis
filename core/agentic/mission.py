"""
core/agentic/mission.py

A Mission is a bounded, multi-step execution unit spawned to pursue a Goal.
One Goal may spawn several Missions over its lifetime (e.g., retry after failure).

Responsibilities:
- Hold an ordered list of Steps
- Track overall mission status
- Record per-step results
- Expose progress for the scheduler and reflection modules
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class StepStatus(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    SUCCEEDED = "succeeded"
    FAILED    = "failed"
    SKIPPED   = "skipped"


class MissionStatus(str, Enum):
    QUEUED    = "queued"
    RUNNING   = "running"
    SUCCEEDED = "succeeded"
    FAILED    = "failed"
    ABORTED   = "aborted"


@dataclass
class Step:
    """A single atomic action inside a Mission."""

    step_id: str
    name: str
    tool: str                        # tool name (matches core/tools registry)
    params: dict = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)  # step_ids

    status: StepStatus = StepStatus.PENDING
    result: Any = None
    error: Optional[str] = None

    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    def start(self) -> None:
        self.status = StepStatus.RUNNING
        self.started_at = _utcnow()

    def succeed(self, result: Any = None) -> None:
        self.status = StepStatus.SUCCEEDED
        self.result = result
        self.finished_at = _utcnow()

    def fail(self, error: str = "") -> None:
        self.status = StepStatus.FAILED
        self.error = error
        self.finished_at = _utcnow()

    def skip(self, reason: str = "") -> None:
        self.status = StepStatus.SKIPPED
        self.error = reason
        self.finished_at = _utcnow()

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.started_at and self.finished_at:
            return (self.finished_at - self.started_at).total_seconds()
        return None

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "name": self.name,
            "tool": self.tool,
            "params": self.params,
            "depends_on": self.depends_on,
            "status": self.status.value,
            "result": str(self.result) if self.result is not None else None,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class Mission:
    """
    Multi-step execution plan for achieving (part of) a Goal.

    Create via MissionBuilder; execute steps externally (dispatcher).
    """

    mission_id: str
    goal_id: str
    description: str
    steps: list[Step] = field(default_factory=list)
    status: MissionStatus = MissionStatus.QUEUED
    attempt_number: int = 1

    created_at: datetime = field(default_factory=_utcnow)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    metadata: dict = field(default_factory=dict)

    # ── Lifecycle ────────────────────────────────────────────────────────

    def start(self) -> None:
        self.status = MissionStatus.RUNNING
        self.started_at = _utcnow()

    def succeed(self) -> None:
        self.status = MissionStatus.SUCCEEDED
        self.finished_at = _utcnow()

    def fail(self) -> None:
        self.status = MissionStatus.FAILED
        self.finished_at = _utcnow()

    def abort(self) -> None:
        self.status = MissionStatus.ABORTED
        self.finished_at = _utcnow()

    # ── Step access ──────────────────────────────────────────────────────

    def add_step(
        self,
        name: str,
        tool: str,
        params: Optional[dict] = None,
        depends_on: Optional[list[str]] = None,
    ) -> Step:
        step = Step(
            step_id=str(uuid.uuid4()),
            name=name,
            tool=tool,
            params=params or {},
            depends_on=depends_on or [],
        )
        self.steps.append(step)
        return step

    def get_step(self, step_id: str) -> Step:
        for s in self.steps:
            if s.step_id == step_id:
                return s
        raise KeyError(f"Unknown step: {step_id}")

    def next_ready_step(self) -> Optional[Step]:
        """Return first pending step whose dependencies are all satisfied."""
        succeeded_ids = {s.step_id for s in self.steps if s.status == StepStatus.SUCCEEDED}
        for step in self.steps:
            if step.status != StepStatus.PENDING:
                continue
            if all(dep in succeeded_ids for dep in step.depends_on):
                return step
        return None

    # ── Progress ─────────────────────────────────────────────────────────

    @property
    def progress(self) -> float:
        """0.0 – 1.0 fraction of steps completed (succeeded or skipped)."""
        if not self.steps:
            return 0.0
        done = sum(
            1 for s in self.steps
            if s.status in (StepStatus.SUCCEEDED, StepStatus.SKIPPED)
        )
        return done / len(self.steps)

    @property
    def has_failed_steps(self) -> bool:
        return any(s.status == StepStatus.FAILED for s in self.steps)

    @property
    def is_terminal(self) -> bool:
        return self.status in (MissionStatus.SUCCEEDED, MissionStatus.FAILED, MissionStatus.ABORTED)

    # ── Serialisation ────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "mission_id": self.mission_id,
            "goal_id": self.goal_id,
            "description": self.description,
            "status": self.status.value,
            "attempt_number": self.attempt_number,
            "progress": round(self.progress, 3),
            "steps": [s.to_dict() for s in self.steps],
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "metadata": self.metadata,
        }


class MissionBuilder:
    """Fluent builder for constructing a Mission before execution."""

    def __init__(self, goal_id: str, description: str) -> None:
        self._mission = Mission(
            mission_id=str(uuid.uuid4()),
            goal_id=goal_id,
            description=description,
        )

    def step(
        self,
        name: str,
        tool: str,
        params: Optional[dict] = None,
        depends_on: Optional[list[str]] = None,
    ) -> "MissionBuilder":
        self._mission.add_step(name, tool, params, depends_on)
        return self

    def metadata(self, **kwargs) -> "MissionBuilder":
        self._mission.metadata.update(kwargs)
        return self

    def build(self) -> Mission:
        return self._mission
