"""
goal_manager.py — Persistent Goal Ownership for Jarvis Agentic Layer

Owns and persists goals across sessions.
Re-injects active/stalled goals into the planner after restarts.
Does NOT plan or execute — delegates to Mission and the existing planner.
"""

import json
import logging
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

GOALS_FILE = Path("data/agentic/goals.json")


class GoalStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    STALLED = "stalled"
    COMPLETED = "completed"
    ABORTED = "aborted"


class Goal:
    """A single persistent goal owned by the agent."""

    def __init__(
        self,
        description: str,
        priority: int = 5,
        goal_id: Optional[str] = None,
        status: GoalStatus = GoalStatus.PENDING,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
        context: Optional[Dict] = None,
        stall_reason: Optional[str] = None,
        abort_reason: Optional[str] = None,
        mission_id: Optional[str] = None,
    ):
        self.goal_id = goal_id or str(uuid.uuid4())
        self.description = description
        self.priority = priority  # 1 (low) – 10 (critical)
        self.status = GoalStatus(status)
        self.created_at = created_at or datetime.utcnow().isoformat()
        self.updated_at = updated_at or self.created_at
        self.context = context or {}
        self.stall_reason = stall_reason
        self.abort_reason = abort_reason
        self.mission_id = mission_id  # linked Mission, if any

    def to_dict(self) -> Dict:
        return {
            "goal_id": self.goal_id,
            "description": self.description,
            "priority": self.priority,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "context": self.context,
            "stall_reason": self.stall_reason,
            "abort_reason": self.abort_reason,
            "mission_id": self.mission_id,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Goal":
        return cls(**data)

    def _touch(self):
        self.updated_at = datetime.utcnow().isoformat()


class GoalManager:
    """
    Owns all agent goals.
    Persists goals to disk so they survive restarts.
    Re-injects pending/stalled goals into the planner on startup.

    Usage:
        gm = GoalManager()
        gm.load()
        goal = gm.add_goal("Book flight to Tokyo by Friday", priority=8)
        gm.transition(goal.goal_id, GoalStatus.ACTIVE)
        ...
        gm.save()
    """

    def __init__(self, storage_path: Path = GOALS_FILE):
        self.storage_path = storage_path
        self._goals: Dict[str, Goal] = {}

    # ------------------------------------------------------------------ I/O

    def load(self) -> None:
        """Load persisted goals from disk. Safe to call on first run."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.storage_path.exists():
            logger.info("No existing goals file — starting fresh.")
            return
        try:
            with self.storage_path.open() as f:
                raw = json.load(f)
            self._goals = {g["goal_id"]: Goal.from_dict(g) for g in raw}
            logger.info("Loaded %d goals from disk.", len(self._goals))
        except Exception as exc:
            logger.error("Failed to load goals: %s", exc)

    def save(self) -> None:
        """Persist all goals to disk atomically."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.storage_path.with_suffix(".tmp")
        try:
            with tmp.open("w") as f:
                json.dump([g.to_dict() for g in self._goals.values()], f, indent=2)
            tmp.replace(self.storage_path)
            logger.debug("Goals saved to %s", self.storage_path)
        except Exception as exc:
            logger.error("Failed to save goals: %s", exc)

    # ------------------------------------------------------------ CRUD

    def add_goal(
        self,
        description: str,
        priority: int = 5,
        context: Optional[Dict] = None,
    ) -> Goal:
        """Create and persist a new goal in PENDING state."""
        goal = Goal(description=description, priority=priority, context=context or {})
        self._goals[goal.goal_id] = goal
        self.save()
        logger.info("New goal [%s] '%s'", goal.goal_id[:8], description)
        return goal

    def get_goal(self, goal_id: str) -> Optional[Goal]:
        return self._goals.get(goal_id)

    def list_goals(self, status: Optional[GoalStatus] = None) -> List[Goal]:
        goals = list(self._goals.values())
        if status:
            goals = [g for g in goals if g.status == status]
        return sorted(goals, key=lambda g: (-g.priority, g.created_at))

    def transition(
        self,
        goal_id: str,
        new_status: GoalStatus,
        reason: Optional[str] = None,
    ) -> bool:
        """
        Move a goal to a new status.
        Logs the transition. Returns False if goal not found.
        """
        goal = self._goals.get(goal_id)
        if not goal:
            logger.warning("Transition failed — goal %s not found.", goal_id)
            return False

        old = goal.status
        goal.status = new_status
        goal._touch()

        if new_status == GoalStatus.STALLED:
            goal.stall_reason = reason
        elif new_status == GoalStatus.ABORTED:
            goal.abort_reason = reason

        logger.info(
            "Goal [%s] %s → %s%s",
            goal_id[:8],
            old.value,
            new_status.value,
            f" ({reason})" if reason else "",
        )
        self.save()
        return True

    def link_mission(self, goal_id: str, mission_id: str) -> None:
        """Associate a Mission with a Goal."""
        goal = self._goals.get(goal_id)
        if goal:
            goal.mission_id = mission_id
            goal._touch()
            self.save()

    # -------------------------------------------------- Post-restart logic

    def resumable_goals(self) -> List[Goal]:
        """
        Returns goals that should be re-injected into the planner after restart.
        These are PENDING or STALLED goals, ordered by priority.
        """
        resumable = [
            g for g in self._goals.values()
            if g.status in (GoalStatus.PENDING, GoalStatus.STALLED)
        ]
        resumable.sort(key=lambda g: (-g.priority, g.created_at))
        logger.info("%d goal(s) eligible for resumption.", len(resumable))
        return resumable

    def summary(self) -> Dict:
        """Return a high-level status count for logging/dashboards."""
        counts: Dict[str, int] = {s.value: 0 for s in GoalStatus}
        for g in self._goals.values():
            counts[g.status.value] += 1
        return counts
