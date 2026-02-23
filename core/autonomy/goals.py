"""
mission.py — Multi-step Mission Representation for Jarvis Agentic Layer

A Mission connects a Goal → Planner → Execution → Reflection.
It tracks checkpoints (steps) and supports pause / resume / abort.
Does NOT plan or execute directly — it coordinates existing systems.
"""

import json
import logging
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MISSIONS_DIR = Path("data/agentic/missions")


class MissionStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ABORTED = "aborted"
    FAILED = "failed"


class CheckpointStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class Checkpoint:
    """One discrete step inside a Mission."""

    def __init__(
        self,
        name: str,
        description: str,
        index: int,
        checkpoint_id: Optional[str] = None,
        status: CheckpointStatus = CheckpointStatus.PENDING,
        result: Optional[Any] = None,
        error: Optional[str] = None,
        started_at: Optional[str] = None,
        finished_at: Optional[str] = None,
    ):
        self.checkpoint_id = checkpoint_id or str(uuid.uuid4())
        self.name = name
        self.description = description
        self.index = index
        self.status = CheckpointStatus(status)
        self.result = result
        self.error = error
        self.started_at = started_at
        self.finished_at = finished_at

    def to_dict(self) -> Dict:
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
    def from_dict(cls, data: Dict) -> "Checkpoint":
        return cls(**data)


class Mission:
    """
    Represents a multi-step objective the agent is working toward.

    Lifecycle:
        created → running → (paused ↔ running) → completed | aborted | failed

    Checkpoints are executed in order by the dispatcher.
    The reflection engine reads the mission after completion.
    """

    def __init__(
        self,
        goal_id: str,
        title: str,
        checkpoints: Optional[List[Dict]] = None,
        mission_id: Optional[str] = None,
        status: MissionStatus = MissionStatus.CREATED,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
        abort_reason: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ):
        self.mission_id = mission_id or str(uuid.uuid4())
        self.goal_id = goal_id
        self.title = title
        self.status = MissionStatus(status)
        self.created_at = created_at or datetime.utcnow().isoformat()
        self.updated_at = updated_at or self.created_at
        self.abort_reason = abort_reason
        self.metadata = metadata or {}

        raw_checkpoints = checkpoints or []
        self.checkpoints: List[Checkpoint] = [
            Checkpoint.from_dict(c) if isinstance(c, dict) else c
            for c in raw_checkpoints
        ]

    # ---------------------------------------------------------------- I/O

    @classmethod
    def load(cls, mission_id: str) -> Optional["Mission"]:
        path = MISSIONS_DIR / f"{mission_id}.json"
        if not path.exists():
            return None
        try:
            with path.open() as f:
                return cls.from_dict(json.load(f))
        except Exception as exc:
            logger.error("Failed to load mission %s: %s", mission_id, exc)
            return None

    def save(self) -> None:
        MISSIONS_DIR.mkdir(parents=True, exist_ok=True)
        path = MISSIONS_DIR / f"{self.mission_id}.json"
        tmp = path.with_suffix(".tmp")
        try:
            with tmp.open("w") as f:
                json.dump(self.to_dict(), f, indent=2)
            tmp.replace(path)
        except Exception as exc:
            logger.error("Failed to save mission %s: %s", self.mission_id, exc)

    def to_dict(self) -> Dict:
        return {
            "mission_id": self.mission_id,
            "goal_id": self.goal_id,
            "title": self.title,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "abort_reason": self.abort_reason,
            "metadata": self.metadata,
            "checkpoints": [c.to_dict() for c in self.checkpoints],
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Mission":
        return cls(**data)

    # ---------------------------------------------------------- Checkpoints

    def add_checkpoint(self, name: str, description: str) -> Checkpoint:
        cp = Checkpoint(name=name, description=description, index=len(self.checkpoints))
        self.checkpoints.append(cp)
        return cp

    def current_checkpoint(self) -> Optional[Checkpoint]:
        """Return the first non-completed, non-skipped checkpoint."""
        for cp in self.checkpoints:
            if cp.status in (CheckpointStatus.PENDING, CheckpointStatus.RUNNING):
                return cp
        return None

    def mark_checkpoint(
        self,
        checkpoint_id: str,
        status: CheckpointStatus,
        result: Any = None,
        error: Optional[str] = None,
    ) -> bool:
        for cp in self.checkpoints:
            if cp.checkpoint_id == checkpoint_id:
                cp.status = status
                cp.result = result
                cp.error = error
                if status == CheckpointStatus.RUNNING:
                    cp.started_at = datetime.utcnow().isoformat()
                else:
                    cp.finished_at = datetime.utcnow().isoformat()
                self._touch()
                self.save()
                logger.info(
                    "Mission [%s] checkpoint '%s' → %s",
                    self.mission_id[:8],
                    cp.name,
                    status.value,
                )
                return True
        return False

    def progress(self) -> Dict:
        total = len(self.checkpoints)
        done = sum(1 for c in self.checkpoints if c.status == CheckpointStatus.DONE)
        failed = sum(1 for c in self.checkpoints if c.status == CheckpointStatus.FAILED)
        return {"total": total, "done": done, "failed": failed, "remaining": total - done - failed}

    # -------------------------------------------------- Lifecycle controls

    def start(self) -> None:
        self._require_status(MissionStatus.CREATED, MissionStatus.PAUSED)
        self.status = MissionStatus.RUNNING
        self._touch()
        self.save()
        logger.info("Mission [%s] '%s' started.", self.mission_id[:8], self.title)

    def pause(self, reason: str = "") -> None:
        self._require_status(MissionStatus.RUNNING)
        self.status = MissionStatus.PAUSED
        self.metadata["pause_reason"] = reason
        self._touch()
        self.save()
        logger.info("Mission [%s] paused. %s", self.mission_id[:8], reason)

    def resume(self) -> None:
        self._require_status(MissionStatus.PAUSED)
        self.status = MissionStatus.RUNNING
        self._touch()
        self.save()
        logger.info("Mission [%s] resumed.", self.mission_id[:8])

    def complete(self) -> None:
        self.status = MissionStatus.COMPLETED
        self._touch()
        self.save()
        logger.info("Mission [%s] '%s' completed.", self.mission_id[:8], self.title)

    def abort(self, reason: str) -> None:
        self.status = MissionStatus.ABORTED
        self.abort_reason = reason
        self._touch()
        self.save()
        logger.warning("Mission [%s] ABORTED: %s", self.mission_id[:8], reason)

    def fail(self, reason: str) -> None:
        self.status = MissionStatus.FAILED
        self.abort_reason = reason
        self._touch()
        self.save()
        logger.error("Mission [%s] FAILED: %s", self.mission_id[:8], reason)

    # -------------------------------------------------------------- Helpers

    def _touch(self) -> None:
        self.updated_at = datetime.utcnow().isoformat()

    def _require_status(self, *allowed: MissionStatus) -> None:
        if self.status not in allowed:
            raise RuntimeError(
                f"Mission {self.mission_id[:8]} is '{self.status.value}', "
                f"expected one of {[s.value for s in allowed]}"
            )

    def human_summary(self) -> str:
        p = self.progress()
        lines = [
            f"Mission: {self.title}",
            f"Status : {self.status.value}",
            f"Progress: {p['done']}/{p['total']} checkpoints done, {p['failed']} failed",
        ]
        for cp in self.checkpoints:
            marker = {"done": "✓", "failed": "✗", "running": "→", "pending": "·", "skipped": "–"}.get(
                cp.status.value, "?"
            )
            lines.append(f"  {marker} [{cp.index}] {cp.name}")
        return "\n".join(lines)

