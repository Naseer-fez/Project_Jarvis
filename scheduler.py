"""
scheduler.py — Deferred Goal Scheduling for Jarvis Agentic Layer

Handles delayed goals, retry scheduling, "check back later" behavior,
and time-based triggers.

Does NOT execute goals — it signals GoalManager when a goal is ready
to be re-injected into the planner.
NO background threads or infinite loops are created here.
The caller is responsible for periodically calling `tick()`.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

SCHEDULE_FILE = Path("data/agentic/schedule.json")


class TriggerType(str, Enum):
    ONCE = "once"          # Fire at a specific datetime
    RETRY = "retry"        # Retry a stalled goal after a delay
    RECURRING = "recurring"  # Repeat on an interval (seconds)
    CHECK_BACK = "check_back"  # "Try again later" — user-defined delay


@dataclass
class ScheduledItem:
    """One scheduled event."""

    item_id: str
    goal_id: str
    trigger_type: TriggerType
    fire_at: str             # ISO datetime
    interval_seconds: Optional[int] = None  # For RECURRING
    retry_count: int = 0
    max_retries: int = 3
    description: str = ""
    fired: bool = False
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def is_due(self) -> bool:
        if self.fired and self.trigger_type != TriggerType.RECURRING:
            return False
        return datetime.utcnow() >= datetime.fromisoformat(self.fire_at)

    def reschedule(self) -> None:
        """Advance fire_at by interval_seconds for RECURRING items."""
        if self.trigger_type == TriggerType.RECURRING and self.interval_seconds:
            next_fire = datetime.utcnow() + timedelta(seconds=self.interval_seconds)
            self.fire_at = next_fire.isoformat()
            self.fired = False
            logger.debug("Rescheduled [%s] next fire at %s", self.item_id[:8], self.fire_at)

    def to_dict(self) -> Dict:
        return {
            "item_id": self.item_id,
            "goal_id": self.goal_id,
            "trigger_type": self.trigger_type.value,
            "fire_at": self.fire_at,
            "interval_seconds": self.interval_seconds,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "description": self.description,
            "fired": self.fired,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ScheduledItem":
        data = dict(data)
        data["trigger_type"] = TriggerType(data["trigger_type"])
        return cls(**data)


class AgentScheduler:
    """
    Time-based trigger system for deferred goals.

    Design:
        - Purely data-driven (no threads, no loops).
        - Caller calls `tick()` at regular intervals (e.g., every 30 s).
        - `tick()` returns a list of goal_ids that are now due.
        - Caller passes those goal_ids to GoalManager for re-injection.

    Usage:
        scheduler = AgentScheduler()
        scheduler.load()
        scheduler.schedule_retry("goal-xyz", delay_seconds=300, description="Retry after network error")

        # In main agent loop:
        due = scheduler.tick()
        for goal_id in due:
            goal_manager.transition(goal_id, GoalStatus.PENDING)
    """

    def __init__(self, storage_path: Path = SCHEDULE_FILE):
        self.storage_path = storage_path
        self._items: Dict[str, ScheduledItem] = {}

    # ────────────────────────────────────────────────────── I/O

    def load(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.storage_path.exists():
            logger.info("No schedule file — starting empty.")
            return
        try:
            with self.storage_path.open() as f:
                raw = json.load(f)
            self._items = {r["item_id"]: ScheduledItem.from_dict(r) for r in raw}
            logger.info("Loaded %d scheduled items.", len(self._items))
        except Exception as exc:
            logger.error("Failed to load schedule: %s", exc)

    def save(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.storage_path.with_suffix(".tmp")
        try:
            with tmp.open("w") as f:
                json.dump([i.to_dict() for i in self._items.values()], f, indent=2)
            tmp.replace(self.storage_path)
        except Exception as exc:
            logger.error("Failed to save schedule: %s", exc)

    # ────────────────────────────────────────────────────── Scheduling

    def schedule_once(
        self,
        goal_id: str,
        fire_at: datetime,
        description: str = "",
    ) -> ScheduledItem:
        """Schedule a goal to activate at an exact time."""
        return self._add(goal_id, TriggerType.ONCE, fire_at, description=description)

    def schedule_retry(
        self,
        goal_id: str,
        delay_seconds: int,
        max_retries: int = 3,
        description: str = "",
    ) -> Optional[ScheduledItem]:
        """
        Schedule a retry for a stalled goal.
        Returns None if max_retries already exhausted.
        """
        existing = self._find_retry(goal_id)
        if existing:
            if existing.retry_count >= existing.max_retries:
                logger.warning(
                    "Goal [%s] has exhausted %d retries — not scheduling more.",
                    goal_id[:8],
                    existing.max_retries,
                )
                return None
            existing.retry_count += 1
            existing.fired = False
            existing.fire_at = (datetime.utcnow() + timedelta(seconds=delay_seconds)).isoformat()
            self.save()
            logger.info(
                "Retry %d/%d scheduled for goal [%s] in %ds.",
                existing.retry_count,
                existing.max_retries,
                goal_id[:8],
                delay_seconds,
            )
            return existing

        return self._add(
            goal_id,
            TriggerType.RETRY,
            datetime.utcnow() + timedelta(seconds=delay_seconds),
            description=description,
            max_retries=max_retries,
        )

    def schedule_check_back(
        self,
        goal_id: str,
        delay_seconds: int,
        description: str = "Check back later",
    ) -> ScheduledItem:
        """Schedule a 'check back later' trigger."""
        return self._add(
            goal_id,
            TriggerType.CHECK_BACK,
            datetime.utcnow() + timedelta(seconds=delay_seconds),
            description=description,
        )

    def schedule_recurring(
        self,
        goal_id: str,
        interval_seconds: int,
        description: str = "",
    ) -> ScheduledItem:
        """Schedule a recurring trigger (e.g., every hour)."""
        return self._add(
            goal_id,
            TriggerType.RECURRING,
            datetime.utcnow() + timedelta(seconds=interval_seconds),
            interval_seconds=interval_seconds,
            description=description,
        )

    def cancel(self, goal_id: str) -> int:
        """Cancel all scheduled items for a goal. Returns count removed."""
        keys = [k for k, v in self._items.items() if v.goal_id == goal_id]
        for k in keys:
            del self._items[k]
        if keys:
            self.save()
        logger.info("Cancelled %d scheduled item(s) for goal [%s].", len(keys), goal_id[:8])
        return len(keys)

    # ────────────────────────────────────────────────────── Tick

    def tick(self) -> List[str]:
        """
        Called periodically by the agent main loop.
        Returns goal_ids that are now due and should be re-injected.
        Mutates internal state (marks items as fired / reschedules).
        """
        due_goal_ids: List[str] = []
        for item in list(self._items.values()):
            if not item.is_due():
                continue

            logger.info(
                "Scheduled item [%s] due: goal=%s type=%s — %s",
                item.item_id[:8],
                item.goal_id[:8],
                item.trigger_type.value,
                item.description,
            )
            due_goal_ids.append(item.goal_id)
            item.fired = True

            if item.trigger_type == TriggerType.RECURRING:
                item.reschedule()
            elif item.trigger_type not in (TriggerType.RECURRING,):
                # One-shot items: remove after firing
                del self._items[item.item_id]

        if due_goal_ids:
            self.save()

        return due_goal_ids

    # ────────────────────────────────────────────────────── Helpers

    def _add(
        self,
        goal_id: str,
        trigger_type: TriggerType,
        fire_at: datetime,
        description: str = "",
        interval_seconds: Optional[int] = None,
        max_retries: int = 3,
    ) -> ScheduledItem:
        import uuid
        item = ScheduledItem(
            item_id=str(uuid.uuid4()),
            goal_id=goal_id,
            trigger_type=trigger_type,
            fire_at=fire_at.isoformat(),
            interval_seconds=interval_seconds,
            max_retries=max_retries,
            description=description,
        )
        self._items[item.item_id] = item
        self.save()
        logger.info(
            "Scheduled [%s] %s for goal [%s] at %s. %s",
            item.item_id[:8],
            trigger_type.value,
            goal_id[:8],
            fire_at.strftime("%Y-%m-%d %H:%M:%S"),
            description,
        )
        return item

    def _find_retry(self, goal_id: str) -> Optional[ScheduledItem]:
        for item in self._items.values():
            if item.goal_id == goal_id and item.trigger_type == TriggerType.RETRY:
                return item
        return None

    def pending_items(self) -> List[ScheduledItem]:
        return [i for i in self._items.values() if not i.fired or i.trigger_type == TriggerType.RECURRING]

    def summary(self) -> str:
        items = list(self._items.values())
        if not items:
            return "Schedule is empty."
        lines = [f"Scheduled items ({len(items)}):"]
        for i in sorted(items, key=lambda x: x.fire_at):
            lines.append(
                f"  [{i.trigger_type.value}] goal={i.goal_id[:8]} at={i.fire_at[:19]} — {i.description}"
            )
        return "\n".join(lines)
