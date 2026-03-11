"""Mission and checkpoint persistence for the agentic layer."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MISSIONS_DIR = Path("data/agentic/missions")


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class MissionStatus(str, Enum):
    QUEUED = "queued"
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    SUCCEEDED = "succeeded"
    COMPLETED = "completed"
    ABORTED = "aborted"
    FAILED = "failed"


class CheckpointStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


def _mission_status_from_value(value: object) -> MissionStatus:
    normalized = str(value or MissionStatus.CREATED.value).strip().lower()
    if normalized in MissionStatus._value2member_map_:
        return MissionStatus(normalized)
    if normalized == "complete":
        return MissionStatus.COMPLETED
    return MissionStatus.CREATED


def _step_status_from_value(value: object) -> StepStatus:
    normalized = str(value or StepStatus.PENDING.value).strip().lower()
    if normalized == CheckpointStatus.DONE.value:
        normalized = StepStatus.SUCCEEDED.value
    if normalized in StepStatus._value2member_map_:
        return StepStatus(normalized)
    return StepStatus.PENDING


def _checkpoint_status_from_value(value: object) -> CheckpointStatus:
    normalized = str(value or CheckpointStatus.PENDING.value).strip().lower()
    if normalized == StepStatus.SUCCEEDED.value:
        normalized = CheckpointStatus.DONE.value
    if normalized in CheckpointStatus._value2member_map_:
        return CheckpointStatus(normalized)
    return CheckpointStatus.PENDING


@dataclass
class Step:
    """Backward-compatible step model kept for older planner/dispatcher flows."""

    step_id: str
    name: str
    tool: str
    params: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    result: Any = None
    error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None

    def start(self) -> None:
        self.status = StepStatus.RUNNING
        self.started_at = _utcnow_iso()

    def succeed(self, result: Any = None) -> None:
        self.status = StepStatus.SUCCEEDED
        self.result = result
        self.finished_at = _utcnow_iso()

    def fail(self, error: str = "") -> None:
        self.status = StepStatus.FAILED
        self.error = error
        self.finished_at = _utcnow_iso()

    def skip(self, reason: str = "") -> None:
        self.status = StepStatus.SKIPPED
        self.error = reason
        self.finished_at = _utcnow_iso()

    @property
    def duration_seconds(self) -> float | None:
        started_at = _parse_iso(self.started_at)
        finished_at = _parse_iso(self.finished_at)
        if started_at is None or finished_at is None:
            return None
        return (finished_at - started_at).total_seconds()

    def to_dict(self) -> dict[str, object]:
        return {
            "step_id": self.step_id,
            "name": self.name,
            "tool": self.tool,
            "params": self.params,
            "depends_on": self.depends_on,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": self.duration_seconds,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "Step":
        return cls(
            step_id=str(payload.get("step_id") or uuid.uuid4()),
            name=str(payload.get("name", "")),
            tool=str(payload.get("tool", "")),
            params=dict(payload.get("params") or {}),
            depends_on=[str(dep) for dep in payload.get("depends_on", [])],
            status=_step_status_from_value(payload.get("status")),
            result=payload.get("result"),
            error=payload.get("error") and str(payload["error"]),
            started_at=payload.get("started_at") and str(payload["started_at"]),
            finished_at=payload.get("finished_at") and str(payload["finished_at"]),
        )


@dataclass
class Checkpoint:
    checkpoint_id: str
    name: str
    description: str
    index: int
    status: CheckpointStatus = CheckpointStatus.PENDING
    result: Any = None
    error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "name": self.name,
            "description": self.description,
            "index": self.index,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "Checkpoint":
        return cls(
            checkpoint_id=str(payload.get("checkpoint_id") or uuid.uuid4()),
            name=str(payload.get("name", "")),
            description=str(payload.get("description", "")),
            index=int(payload.get("index", 0)),
            status=_checkpoint_status_from_value(payload.get("status")),
            result=payload.get("result"),
            error=payload.get("error") and str(payload["error"]),
            started_at=payload.get("started_at") and str(payload["started_at"]),
            finished_at=payload.get("finished_at") and str(payload["finished_at"]),
        )


def _checkpoint_from_step(step: Step, index: int) -> Checkpoint:
    return Checkpoint(
        checkpoint_id=step.step_id,
        name=step.name,
        description=step.tool,
        index=index,
        status=_checkpoint_status_from_value(step.status.value),
        result=step.result,
        error=step.error,
        started_at=step.started_at,
        finished_at=step.finished_at,
    )


@dataclass
class Mission:
    goal_id: str
    title: str = ""
    description: str | None = None
    checkpoints: list[Checkpoint] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    mission_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: MissionStatus = MissionStatus.CREATED
    attempt_number: int = 1
    created_at: str = field(default_factory=_utcnow_iso)
    updated_at: str = field(default_factory=_utcnow_iso)
    started_at: str | None = None
    finished_at: str | None = None
    abort_reason: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.title and self.description:
            self.title = self.description
        if self.description is None:
            self.description = self.title

    @classmethod
    def load(cls, mission_id: str) -> "Mission | None":
        path = MISSIONS_DIR / f"{mission_id}.json"
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            logger.warning("Could not load mission %s: %s", mission_id, exc)
            return None
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "Mission":
        steps = [
            Step.from_dict(step_payload)
            for step_payload in payload.get("steps", [])
            if isinstance(step_payload, dict)
        ]
        checkpoints = [
            Checkpoint.from_dict(checkpoint_payload)
            for checkpoint_payload in payload.get("checkpoints", [])
            if isinstance(checkpoint_payload, dict)
        ]
        if not checkpoints and steps:
            checkpoints = [_checkpoint_from_step(step, index) for index, step in enumerate(steps)]

        title = str(payload.get("title") or payload.get("description") or "")
        description = str(payload.get("description") or title)
        return cls(
            goal_id=str(payload.get("goal_id", "")),
            title=title,
            description=description,
            checkpoints=checkpoints,
            steps=steps,
            mission_id=str(payload.get("mission_id") or uuid.uuid4()),
            status=_mission_status_from_value(payload.get("status")),
            attempt_number=int(payload.get("attempt_number", 1)),
            created_at=str(payload.get("created_at") or _utcnow_iso()),
            updated_at=str(payload.get("updated_at") or payload.get("created_at") or _utcnow_iso()),
            started_at=payload.get("started_at") and str(payload["started_at"]),
            finished_at=payload.get("finished_at") and str(payload["finished_at"]),
            abort_reason=payload.get("abort_reason") and str(payload["abort_reason"]),
            metadata=dict(payload.get("metadata") or {}),
        )

    def save(self) -> Path:
        path = MISSIONS_DIR / f"{self.mission_id}.json"
        _atomic_write_json(path, self.to_dict())
        return path

    def to_dict(self) -> dict[str, object]:
        return {
            "mission_id": self.mission_id,
            "goal_id": self.goal_id,
            "title": self.title,
            "description": self.description or self.title,
            "status": self.status.value,
            "attempt_number": self.attempt_number,
            "progress": round(self.progress, 3),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "abort_reason": self.abort_reason,
            "metadata": self.metadata,
            "checkpoints": [checkpoint.to_dict() for checkpoint in self.checkpoints],
            "steps": [step.to_dict() for step in self.steps],
        }

    def add_checkpoint(self, name: str, description: str) -> Checkpoint:
        checkpoint = Checkpoint(
            checkpoint_id=str(uuid.uuid4()),
            name=name,
            description=description,
            index=len(self.checkpoints),
        )
        self.checkpoints.append(checkpoint)
        return checkpoint

    def current_checkpoint(self) -> Checkpoint | None:
        for checkpoint in self.checkpoints:
            if checkpoint.status in {CheckpointStatus.PENDING, CheckpointStatus.RUNNING}:
                return checkpoint
        return None

    def mark_checkpoint(
        self,
        checkpoint_id: str,
        status: CheckpointStatus,
        result: Any = None,
        error: str | None = None,
    ) -> bool:
        for checkpoint in self.checkpoints:
            if checkpoint.checkpoint_id != checkpoint_id:
                continue
            checkpoint.status = status
            checkpoint.result = result
            checkpoint.error = error
            if status == CheckpointStatus.RUNNING:
                checkpoint.started_at = _utcnow_iso()
                checkpoint.finished_at = None
                self.started_at = self.started_at or checkpoint.started_at
            else:
                checkpoint.finished_at = _utcnow_iso()
            self._touch()
            self.save()
            return True
        return False

    def add_step(
        self,
        name: str,
        tool: str,
        params: dict[str, Any] | None = None,
        depends_on: list[str] | None = None,
    ) -> Step:
        step = Step(
            step_id=str(uuid.uuid4()),
            name=name,
            tool=tool,
            params=dict(params or {}),
            depends_on=[str(dep) for dep in depends_on or []],
        )
        self.steps.append(step)
        return step

    def get_step(self, step_id: str) -> Step:
        for step in self.steps:
            if step.step_id == step_id:
                return step
        raise KeyError(f"Unknown step: {step_id}")

    def next_ready_step(self) -> Step | None:
        succeeded_ids = {
            step.step_id
            for step in self.steps
            if step.status == StepStatus.SUCCEEDED
        }
        for step in self.steps:
            if step.status != StepStatus.PENDING:
                continue
            if all(dep in succeeded_ids for dep in step.depends_on):
                return step
        return None

    @property
    def progress(self) -> float:
        if self.steps:
            completed = sum(
                1
                for step in self.steps
                if step.status in {StepStatus.SUCCEEDED, StepStatus.SKIPPED}
            )
            return completed / len(self.steps) if self.steps else 0.0

        if not self.checkpoints:
            return 0.0

        completed = sum(
            1
            for checkpoint in self.checkpoints
            if checkpoint.status in {CheckpointStatus.DONE, CheckpointStatus.SKIPPED}
        )
        return completed / len(self.checkpoints)

    @property
    def has_failed_steps(self) -> bool:
        return any(step.status == StepStatus.FAILED for step in self.steps) or any(
            checkpoint.status == CheckpointStatus.FAILED
            for checkpoint in self.checkpoints
        )

    @property
    def is_terminal(self) -> bool:
        return self.status in {
            MissionStatus.SUCCEEDED,
            MissionStatus.COMPLETED,
            MissionStatus.FAILED,
            MissionStatus.ABORTED,
        }

    def progress_counts(self) -> dict[str, int]:
        if self.steps:
            total = len(self.steps)
            done = sum(
                1
                for step in self.steps
                if step.status in {StepStatus.SUCCEEDED, StepStatus.SKIPPED}
            )
            failed = sum(1 for step in self.steps if step.status == StepStatus.FAILED)
        else:
            total = len(self.checkpoints)
            done = sum(
                1
                for checkpoint in self.checkpoints
                if checkpoint.status in {CheckpointStatus.DONE, CheckpointStatus.SKIPPED}
            )
            failed = sum(
                1
                for checkpoint in self.checkpoints
                if checkpoint.status == CheckpointStatus.FAILED
            )
        return {
            "total": total,
            "done": done,
            "failed": failed,
            "remaining": max(0, total - done - failed),
        }

    def start(self) -> None:
        self.status = MissionStatus.RUNNING
        self.started_at = self.started_at or _utcnow_iso()
        self._touch()
        self.save()

    def pause(self, reason: str = "") -> None:
        self.status = MissionStatus.PAUSED
        if reason:
            self.metadata["pause_reason"] = reason
        self._touch()
        self.save()

    def resume(self) -> None:
        self.status = MissionStatus.RUNNING
        self._touch()
        self.save()

    def complete(self) -> None:
        self.status = MissionStatus.COMPLETED
        self.abort_reason = None
        self.finished_at = _utcnow_iso()
        self._touch()
        self.save()

    def succeed(self) -> None:
        self.status = MissionStatus.SUCCEEDED
        self.abort_reason = None
        self.finished_at = _utcnow_iso()
        self._touch()
        self.save()

    def abort(self, reason: str) -> None:
        self.status = MissionStatus.ABORTED
        self.abort_reason = reason
        self.finished_at = _utcnow_iso()
        self._touch()
        self.save()

    def fail(self, reason: str) -> None:
        self.status = MissionStatus.FAILED
        self.abort_reason = reason
        self.finished_at = _utcnow_iso()
        self._touch()
        self.save()

    def human_summary(self) -> str:
        progress = self.progress_counts()
        return (
            f"Mission: {self.title}\n"
            f"Status: {self.status.value}\n"
            f"Progress: {progress['done']}/{progress['total']} done, "
            f"{progress['failed']} failed"
        )

    def _touch(self) -> None:
        self.updated_at = _utcnow_iso()


class MissionBuilder:
    """Fluent builder preserved for older mission-construction code."""

    def __init__(self, goal_id: str, description: str) -> None:
        self._mission = Mission(
            goal_id=goal_id,
            title=description,
            description=description,
            status=MissionStatus.QUEUED,
        )

    def step(
        self,
        name: str,
        tool: str,
        params: dict[str, Any] | None = None,
        depends_on: list[str] | None = None,
    ) -> "MissionBuilder":
        self._mission.add_step(name, tool, params, depends_on)
        return self

    def metadata(self, **kwargs: object) -> "MissionBuilder":
        self._mission.metadata.update(kwargs)
        return self

    def build(self) -> Mission:
        return self._mission


__all__ = [
    "Checkpoint",
    "CheckpointStatus",
    "Mission",
    "MissionBuilder",
    "MissionStatus",
    "MISSIONS_DIR",
    "Step",
    "StepStatus",
]
