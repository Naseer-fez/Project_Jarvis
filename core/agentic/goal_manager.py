"""
core/agentic/goal_manager.py

Owns the lifecycle of long-lived agent goals.
A Goal is a high-level desired outcome that may span multiple Missions.

Responsibilities:
- Create / update / complete / cancel goals
- Prioritise active goals
- Query which goal the agent should work on next
- Persist goal state (via snapshot / restore)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class GoalStatus(str, Enum):
    PENDING   = "pending"    # created, not yet started
    ACTIVE    = "active"     # currently being pursued
    PAUSED    = "paused"     # temporarily on hold
    COMPLETED = "completed"  # achieved
    FAILED    = "failed"     # could not be achieved
    CANCELLED = "cancelled"  # explicitly abandoned


@dataclass
class Goal:
    """A single long-lived agent objective."""

    goal_id: str
    description: str
    priority: int = 5              # 1 (highest) – 10 (lowest)
    status: GoalStatus = GoalStatus.PENDING
    parent_goal_id: Optional[str] = None   # for sub-goals
    metadata: dict = field(default_factory=dict)

    created_at: datetime = field(default_factory=_utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    deadline: Optional[datetime] = None

    outcome: Optional[str] = None   # human-readable result

    def start(self) -> None:
        if self.status != GoalStatus.PENDING:
            raise ValueError(f"Cannot start goal in status '{self.status}'")
        self.status = GoalStatus.ACTIVE
        self.started_at = _utcnow()

    def complete(self, outcome: str = "") -> None:
        self.status = GoalStatus.COMPLETED
        self.completed_at = _utcnow()
        self.outcome = outcome

    def fail(self, reason: str = "") -> None:
        self.status = GoalStatus.FAILED
        self.completed_at = _utcnow()
        self.outcome = reason

    def cancel(self, reason: str = "") -> None:
        self.status = GoalStatus.CANCELLED
        self.completed_at = _utcnow()
        self.outcome = reason

    def pause(self) -> None:
        if self.status == GoalStatus.ACTIVE:
            self.status = GoalStatus.PAUSED

    def resume(self) -> None:
        if self.status == GoalStatus.PAUSED:
            self.status = GoalStatus.ACTIVE

    @property
    def is_terminal(self) -> bool:
        return self.status in (GoalStatus.COMPLETED, GoalStatus.FAILED, GoalStatus.CANCELLED)

    def to_dict(self) -> dict:
        return {
            "goal_id": self.goal_id,
            "description": self.description,
            "priority": self.priority,
            "status": self.status.value,
            "parent_goal_id": self.parent_goal_id,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "outcome": self.outcome,
        }


class GoalManager:
    """
    Registry and lifecycle manager for all agent goals.

    Usage:
        gm = GoalManager()
        gid = gm.create_goal("Summarise all emails from today", priority=2)
        gm.start_goal(gid)
        ...
        gm.complete_goal(gid, outcome="12 emails summarised")
    """

    def __init__(self) -> None:
        self._goals: dict[str, Goal] = {}

    # ── CRUD ─────────────────────────────────────────────────────────────

    def create_goal(
        self,
        description: str,
        priority: int = 5,
        parent_goal_id: Optional[str] = None,
        deadline: Optional[datetime] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        goal_id = str(uuid.uuid4())
        self._goals[goal_id] = Goal(
            goal_id=goal_id,
            description=description,
            priority=priority,
            parent_goal_id=parent_goal_id,
            deadline=deadline,
            metadata=metadata or {},
        )
        return goal_id

    def get_goal(self, goal_id: str) -> Goal:
        if goal_id not in self._goals:
            raise KeyError(f"Unknown goal: {goal_id}")
        return self._goals[goal_id]

    def start_goal(self, goal_id: str) -> None:
        self.get_goal(goal_id).start()

    def complete_goal(self, goal_id: str, outcome: str = "") -> None:
        self.get_goal(goal_id).complete(outcome)

    def fail_goal(self, goal_id: str, reason: str = "") -> None:
        self.get_goal(goal_id).fail(reason)

    def cancel_goal(self, goal_id: str, reason: str = "") -> None:
        self.get_goal(goal_id).cancel(reason)

    # ── Queries ──────────────────────────────────────────────────────────

    def next_goal(self) -> Optional[Goal]:
        """Return the highest-priority pending or paused goal."""
        candidates = [
            g for g in self._goals.values()
            if g.status in (GoalStatus.PENDING, GoalStatus.PAUSED)
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda g: (g.priority, g.created_at))

    def active_goals(self) -> list[Goal]:
        return [g for g in self._goals.values() if g.status == GoalStatus.ACTIVE]

    def all_goals(self) -> list[Goal]:
        return list(self._goals.values())

    # ── Persistence ──────────────────────────────────────────────────────

    def snapshot(self) -> list[dict]:
        return [g.to_dict() for g in self._goals.values()]

    def restore(self, data: list[dict]) -> None:
        """Reload goals from a persisted snapshot (e.g. after restart)."""
        for d in data:
            goal = Goal(
                goal_id=d["goal_id"],
                description=d["description"],
                priority=d["priority"],
                status=GoalStatus(d["status"]),
                parent_goal_id=d.get("parent_goal_id"),
                metadata=d.get("metadata", {}),
                created_at=datetime.fromisoformat(d["created_at"]),
                outcome=d.get("outcome"),
            )
            if d.get("started_at"):
                goal.started_at = datetime.fromisoformat(d["started_at"])
            if d.get("completed_at"):
                goal.completed_at = datetime.fromisoformat(d["completed_at"])
            if d.get("deadline"):
                goal.deadline = datetime.fromisoformat(d["deadline"])
            self._goals[goal.goal_id] = goal
