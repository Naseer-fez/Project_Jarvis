"""Mission and checkpoint persistence for the agentic layer."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

MISSIONS_DIR = Path("data/agentic/missions")


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class CheckpointStatus(str, Enum):
    TODO = "todo"
    DONE = "done"
    FAILED = "failed"


class MissionStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass
class Step:
    name: str
    tool: str
    depends_on: list[str] = field(default_factory=list)
    step_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: StepStatus = StepStatus.PENDING


@dataclass
class Checkpoint:
    name: str
    description: str
    checkpoint_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: CheckpointStatus = CheckpointStatus.TODO
    error: str = ""


@dataclass
class Mission:
    goal_id: str
    title: str
    description: str = ""
    mission_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: MissionStatus = MissionStatus.QUEUED
    steps: list[Step] = field(default_factory=list)
    checkpoints: list[Checkpoint] = field(default_factory=list)
    error: str = ""

    def __post_init__(self) -> None:
        if not self.description:
            self.description = self.title

    def add_checkpoint(self, name: str, description: str) -> Checkpoint:
        checkpoint = Checkpoint(name=name, description=description)
        self.checkpoints.append(checkpoint)
        return checkpoint

    def mark_checkpoint(
        self,
        checkpoint_id: str,
        status: CheckpointStatus,
        error: str = "",
    ) -> Checkpoint:
        for checkpoint in self.checkpoints:
            if checkpoint.checkpoint_id == checkpoint_id:
                checkpoint.status = CheckpointStatus(status)
                checkpoint.error = error
                return checkpoint
        raise KeyError(f"Unknown checkpoint: {checkpoint_id}")

    def start(self) -> None:
        self.status = MissionStatus.RUNNING

    def abort(self, reason: str = "") -> None:
        self.status = MissionStatus.ABORTED
        self.error = reason

    def next_ready_step(self) -> Step | None:
        succeeded = {
            step.step_id for step in self.steps if step.status == StepStatus.SUCCEEDED
        }
        for step in self.steps:
            if step.status != StepStatus.PENDING:
                continue
            if all(dep in succeeded for dep in step.depends_on):
                return step
        return None

    @classmethod
    def load(cls, mission_id: str) -> "Mission | None":
        path = MISSIONS_DIR / f"{mission_id}.json"
        if not path.exists():
            return None

        payload = json.loads(path.read_text(encoding="utf-8"))
        if "checkpoints" not in payload and "steps" in payload:
            return cls._load_legacy(payload)

        mission = cls(
            mission_id=str(payload.get("mission_id", mission_id)),
            goal_id=str(payload["goal_id"]),
            title=str(payload.get("title") or payload.get("description") or mission_id),
            description=str(payload.get("description") or payload.get("title") or mission_id),
            status=MissionStatus(str(payload.get("status", MissionStatus.QUEUED.value))),
            error=str(payload.get("error", "") or ""),
        )
        mission.steps = [
            Step(
                step_id=str(step_payload.get("step_id") or uuid.uuid4()),
                name=str(step_payload.get("name", "")),
                tool=str(step_payload.get("tool", "")),
                depends_on=list(step_payload.get("depends_on", []) or []),
                status=StepStatus(str(step_payload.get("status", StepStatus.PENDING.value))),
            )
            for step_payload in payload.get("steps", [])
        ]
        mission.checkpoints = [
            Checkpoint(
                checkpoint_id=str(cp.get("checkpoint_id") or uuid.uuid4()),
                name=str(cp.get("name", "")),
                description=str(cp.get("description", "")),
                status=CheckpointStatus(str(cp.get("status", CheckpointStatus.TODO.value))),
                error=str(cp.get("error", "") or ""),
            )
            for cp in payload.get("checkpoints", [])
        ]
        return mission

    @classmethod
    def _load_legacy(cls, payload: dict[str, object]) -> "Mission":
        description = str(payload.get("description") or payload.get("title") or "")
        mission = cls(
            mission_id=str(payload.get("mission_id") or uuid.uuid4()),
            goal_id=str(payload.get("goal_id", "")),
            title=description or "Legacy mission",
            description=description or "Legacy mission",
            status=MissionStatus(str(payload.get("status", MissionStatus.QUEUED.value))),
        )

        for step_payload in payload.get("steps", []):
            step_status = StepStatus(str(step_payload.get("status", StepStatus.PENDING.value)))
            step = Step(
                step_id=str(step_payload.get("step_id") or uuid.uuid4()),
                name=str(step_payload.get("name", "")),
                tool=str(step_payload.get("tool", "")),
                status=step_status,
            )
            mission.steps.append(step)
            mission.checkpoints.append(
                Checkpoint(
                    name=step.name or step.tool,
                    description=step.name or step.tool,
                    status=CheckpointStatus.DONE if step_status == StepStatus.SUCCEEDED else CheckpointStatus.FAILED,
                    error="" if step_status == StepStatus.SUCCEEDED else mission.error,
                )
            )
        return mission


class MissionBuilder:
    def __init__(self, goal_id: str, description: str) -> None:
        self.goal_id = goal_id
        self.description = description
        self._steps: list[Step] = []

    def step(self, name: str, tool: str, depends_on: list[str] | None = None) -> "MissionBuilder":
        self._steps.append(
            Step(name=name, tool=tool, depends_on=list(depends_on or []))
        )
        return self

    def build(self) -> Mission:
        return Mission(
            goal_id=self.goal_id,
            title=self.description,
            description=self.description,
            status=MissionStatus.QUEUED,
            steps=list(self._steps),
        )


__all__ = [
    "Checkpoint",
    "CheckpointStatus",
    "MISSIONS_DIR",
    "Mission",
    "MissionBuilder",
    "MissionStatus",
    "Step",
    "StepStatus",
]
