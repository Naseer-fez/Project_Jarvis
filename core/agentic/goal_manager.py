"""Persistent goal management for the agentic layer."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

GOALS_PATH = Path("data/agentic/goals.json")


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


class GoalStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    STALLED = "stalled"
    COMPLETED = "completed"
    ABORTED = "aborted"


@dataclass
class Goal:
    goal_id: str
    description: str
    priority: int = 5
    status: GoalStatus = GoalStatus.PENDING
    created_at: str = field(default_factory=_utcnow_iso)
    started_at: str | None = None
    completed_at: str | None = None
    outcome: str | None = None
    reason: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "goal_id": self.goal_id,
            "description": self.description,
            "priority": self.priority,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "outcome": self.outcome,
            "reason": self.reason,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "Goal":
        return cls(
            goal_id=str(payload["goal_id"]),
            description=str(payload["description"]),
            priority=int(payload.get("priority", 5)),
            status=GoalStatus(str(payload.get("status", GoalStatus.PENDING.value))),
            created_at=str(payload.get("created_at") or _utcnow_iso()),
            started_at=payload.get("started_at") and str(payload["started_at"]),
            completed_at=payload.get("completed_at") and str(payload["completed_at"]),
            outcome=payload.get("outcome") and str(payload["outcome"]),
            reason=payload.get("reason") and str(payload["reason"]),
            metadata=dict(payload.get("metadata") or {}),
        )


class GoalManager:
    def __init__(self, storage_path: str | Path = GOALS_PATH) -> None:
        self.storage_path = Path(storage_path)
        self._goals: dict[str, Goal] = {}

    def load(self) -> "GoalManager":
        if not self.storage_path.exists():
            return self

        payload = json.loads(self.storage_path.read_text(encoding="utf-8"))
        self._goals = {
            goal_payload["goal_id"]: Goal.from_dict(goal_payload)
            for goal_payload in payload
        }
        return self

    def save(self) -> Path:
        payload = [goal.to_dict() for goal in self._goals.values()]
        _atomic_write_json(self.storage_path, payload)
        return self.storage_path

    def create_goal(self, description: str, priority: int = 5) -> str:
        goal_id = str(uuid.uuid4())
        self._goals[goal_id] = Goal(
            goal_id=goal_id,
            description=description,
            priority=priority,
        )
        self.save()
        return goal_id

    def get_goal(self, goal_id: str) -> Goal:
        return self._goals[goal_id]

    def start_goal(self, goal_id: str) -> Goal:
        goal = self.get_goal(goal_id)
        goal.status = GoalStatus.ACTIVE
        goal.started_at = goal.started_at or _utcnow_iso()
        goal.reason = None
        self.save()
        return goal

    def complete_goal(self, goal_id: str, outcome: str = "") -> Goal:
        goal = self.get_goal(goal_id)
        goal.status = GoalStatus.COMPLETED
        goal.completed_at = _utcnow_iso()
        goal.outcome = outcome
        goal.reason = None
        self.save()
        return goal

    def stall_goal(self, goal_id: str, reason: str = "") -> Goal:
        goal = self.get_goal(goal_id)
        goal.status = GoalStatus.STALLED
        goal.reason = reason
        self.save()
        return goal

    def abort_goal(self, goal_id: str, reason: str = "") -> Goal:
        goal = self.get_goal(goal_id)
        goal.status = GoalStatus.ABORTED
        goal.completed_at = _utcnow_iso()
        goal.reason = reason
        self.save()
        return goal

    def resumable_goals(self) -> list[Goal]:
        return [
            goal
            for goal in self._goals.values()
            if goal.status in {GoalStatus.PENDING, GoalStatus.STALLED}
        ]


__all__ = ["Goal", "GoalManager", "GoalStatus", "GOALS_PATH"]
