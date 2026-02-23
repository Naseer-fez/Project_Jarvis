"""
core/agentic/scheduler.py

Manages delayed and retry execution of Missions.

Responsibilities:
- Queue missions for execution at a future time
- Implement exponential back-off for retries
- Expose the next due mission (pull model — no background threads)
- Persist schedule across restarts

Design note:
  This is a *pull-based* scheduler.  The caller (main loop / dispatcher)
  asks `scheduler.due()` on each tick.  There are no background threads,
  no asyncio tasks, no hidden loops — exactly as the spec requires.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ScheduleStatus(str, Enum):
    WAITING   = "waiting"    # not yet due
    DUE       = "due"        # ready to run
    RUNNING   = "running"    # currently executing
    COMPLETED = "completed"  # finished (success or final failure)
    CANCELLED = "cancelled"


@dataclass
class ScheduledMission:
    """Entry in the scheduler queue."""

    entry_id: str
    mission_id: str                     # refers to a Mission object (owned externally)
    goal_id: str

    run_at: datetime                    # when to execute
    status: ScheduleStatus = ScheduleStatus.WAITING

    # Retry book-keeping
    attempt_number: int = 1
    max_attempts: int = 3
    base_delay_seconds: float = 30.0   # first retry delay
    backoff_factor: float = 2.0        # multiply delay each attempt

    # Optional label for humans / LLMs
    description: str = ""

    created_at: datetime = field(default_factory=_utcnow)
    last_run_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @property
    def is_due(self) -> bool:
        return self.status == ScheduleStatus.WAITING and _utcnow() >= self.run_at

    @property
    def next_retry_delay(self) -> float:
        """Seconds to wait before the next retry attempt."""
        return self.base_delay_seconds * (self.backoff_factor ** (self.attempt_number - 1))

    def mark_running(self) -> None:
        self.status = ScheduleStatus.RUNNING
        self.last_run_at = _utcnow()

    def mark_completed(self) -> None:
        self.status = ScheduleStatus.COMPLETED
        self.completed_at = _utcnow()

    def mark_cancelled(self) -> None:
        self.status = ScheduleStatus.CANCELLED
        self.completed_at = _utcnow()

    def schedule_retry(self) -> bool:
        """
        Advance the attempt counter and set a new run_at.
        Returns False if max_attempts is exhausted (caller should cancel).
        """
        if self.attempt_number >= self.max_attempts:
            return False
        delay = self.next_retry_delay
        self.attempt_number += 1
        self.run_at = _utcnow() + timedelta(seconds=delay)
        self.status = ScheduleStatus.WAITING
        return True

    def to_dict(self) -> dict:
        return {
            "entry_id": self.entry_id,
            "mission_id": self.mission_id,
            "goal_id": self.goal_id,
            "run_at": self.run_at.isoformat(),
            "status": self.status.value,
            "attempt_number": self.attempt_number,
            "max_attempts": self.max_attempts,
            "base_delay_seconds": self.base_delay_seconds,
            "backoff_factor": self.backoff_factor,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "last_run_at": self.last_run_at.isoformat() if self.last_run_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class Scheduler:
    """
    Pull-based mission scheduler with exponential back-off.

    Usage:
        scheduler = Scheduler()
        scheduler.enqueue(mission_id="abc", goal_id="xyz", delay_seconds=0)

        # In your main loop:
        for entry in scheduler.due():
            entry.mark_running()
            run_mission(entry.mission_id)
            entry.mark_completed()
    """

    def __init__(self) -> None:
        self._queue: dict[str, ScheduledMission] = {}

    # ── Enqueue ──────────────────────────────────────────────────────────

    def enqueue(
        self,
        mission_id: str,
        goal_id: str,
        delay_seconds: float = 0.0,
        max_attempts: int = 3,
        base_delay_seconds: float = 30.0,
        backoff_factor: float = 2.0,
        description: str = "",
    ) -> ScheduledMission:
        entry = ScheduledMission(
            entry_id=str(uuid.uuid4()),
            mission_id=mission_id,
            goal_id=goal_id,
            run_at=_utcnow() + timedelta(seconds=delay_seconds),
            max_attempts=max_attempts,
            base_delay_seconds=base_delay_seconds,
            backoff_factor=backoff_factor,
            description=description,
        )
        self._queue[entry.entry_id] = entry
        return entry

    def enqueue_retry(self, entry: ScheduledMission) -> bool:
        """
        Re-schedule an existing entry for retry.
        Returns True if retry was scheduled; False if attempts exhausted.
        """
        success = entry.schedule_retry()
        if not success:
            entry.mark_cancelled()
        return success

    # ── Query ────────────────────────────────────────────────────────────

    def due(self) -> list[ScheduledMission]:
        """Return all entries that are currently due, sorted by run_at."""
        entries = [e for e in self._queue.values() if e.is_due]
        return sorted(entries, key=lambda e: e.run_at)

    def get(self, entry_id: str) -> ScheduledMission:
        if entry_id not in self._queue:
            raise KeyError(f"No scheduled entry: {entry_id}")
        return self._queue[entry_id]

    def cancel(self, entry_id: str) -> None:
        if entry_id in self._queue:
            self._queue[entry_id].mark_cancelled()

    def all_entries(self) -> list[ScheduledMission]:
        return list(self._queue.values())

    def pending(self) -> list[ScheduledMission]:
        return [e for e in self._queue.values() if e.status == ScheduleStatus.WAITING]

    # ── Serialisation ────────────────────────────────────────────────────

    def snapshot(self) -> list[dict]:
        return [e.to_dict() for e in self._queue.values()]

    def restore(self, data: list[dict]) -> None:
        for d in data:
            entry = ScheduledMission(
                entry_id=d["entry_id"],
                mission_id=d["mission_id"],
                goal_id=d["goal_id"],
                run_at=datetime.fromisoformat(d["run_at"]),
                status=ScheduleStatus(d["status"]),
                attempt_number=d["attempt_number"],
                max_attempts=d["max_attempts"],
                base_delay_seconds=d["base_delay_seconds"],
                backoff_factor=d["backoff_factor"],
                description=d.get("description", ""),
                created_at=datetime.fromisoformat(d["created_at"]),
                last_run_at=datetime.fromisoformat(d["last_run_at"]) if d.get("last_run_at") else None,
                completed_at=datetime.fromisoformat(d["completed_at"]) if d.get("completed_at") else None,
            )
            self._queue[entry.entry_id] = entry
