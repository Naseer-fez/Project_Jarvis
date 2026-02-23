"""
core/agentic/resume.py

Crash / Resume Contract — defines what state is persisted, what is safe
to resume automatically after a restart, and what requires human confirmation.

Responsibilities:
- Collect full agent state into a ResumableSnapshot
- Persist snapshot to disk (JSON)
- Load and validate snapshot on startup
- Classify each item as AUTO_RESUME, REQUIRES_CONFIRMATION, or DISCARD
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ResumePolicy(str, Enum):
    AUTO_RESUME           = "auto_resume"           # safe to restart without asking
    REQUIRES_CONFIRMATION = "requires_confirmation"  # show human before continuing
    DISCARD               = "discard"               # do not restore; start fresh


@dataclass
class ResumeItem:
    """
    Metadata for one piece of restorable state.

    Attach a ResumeItem to each subsystem snapshot so the loader knows
    what to do with it after a crash.
    """

    key: str                    # e.g. "scheduler", "goal_manager", "belief_state"
    policy: ResumePolicy
    data: Any                   # the actual serialisable state
    version: int = 1            # schema version for forward compat
    note: str = ""              # human-readable note shown at confirmation prompt

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "policy": self.policy.value,
            "data": self.data,
            "version": self.version,
            "note": self.note,
        }

    @staticmethod
    def from_dict(d: dict) -> "ResumeItem":
        return ResumeItem(
            key=d["key"],
            policy=ResumePolicy(d["policy"]),
            data=d["data"],
            version=d.get("version", 1),
            note=d.get("note", ""),
        )


@dataclass
class ResumableSnapshot:
    """Full serialisable snapshot of agent state at a point in time."""

    snapshot_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=_utcnow)
    items: list[ResumeItem] = field(default_factory=list)

    # ── RESUME POLICIES (defined here, one place) ────────────────────────
    #
    # Key           | Policy                | Reason
    # --------------|---------------------- |--------------------------------
    # context       | AUTO_RESUME           | Read-only runtime data; safe
    # goal_manager  | AUTO_RESUME           | Goals are persistent by design
    # scheduler     | REQUIRES_CONFIRMATION | Pending actions may be stale
    # belief_state  | AUTO_RESUME           | Beliefs are observations; safe
    # decision_trace| AUTO_RESUME           | Append-only audit log; safe
    # active_mission| REQUIRES_CONFIRMATION | Was mid-flight; may be partial
    # override_log  | AUTO_RESUME           | Human instructions; always keep
    # ────────────────────────────────────────────────────────────────────

    POLICIES: dict[str, ResumePolicy] = field(default_factory=lambda: {
        "context":        ResumePolicy.AUTO_RESUME,
        "goal_manager":   ResumePolicy.AUTO_RESUME,
        "scheduler":      ResumePolicy.REQUIRES_CONFIRMATION,
        "belief_state":   ResumePolicy.AUTO_RESUME,
        "decision_trace": ResumePolicy.AUTO_RESUME,
        "active_mission": ResumePolicy.REQUIRES_CONFIRMATION,
        "override_log":   ResumePolicy.AUTO_RESUME,
    })

    def add(self, key: str, data: Any, note: str = "") -> None:
        policy = self.POLICIES.get(key, ResumePolicy.REQUIRES_CONFIRMATION)
        self.items.append(ResumeItem(key=key, policy=policy, data=data, note=note))

    def to_dict(self) -> dict:
        return {
            "snapshot_id": self.snapshot_id,
            "created_at": self.created_at.isoformat(),
            "items": [i.to_dict() for i in self.items],
        }

    @staticmethod
    def from_dict(d: dict) -> "ResumableSnapshot":
        snap = ResumableSnapshot(
            snapshot_id=d["snapshot_id"],
            created_at=datetime.fromisoformat(d["created_at"]),
        )
        snap.items = [ResumeItem.from_dict(i) for i in d.get("items", [])]
        return snap

    def auto_items(self) -> list[ResumeItem]:
        return [i for i in self.items if i.policy == ResumePolicy.AUTO_RESUME]

    def confirmation_items(self) -> list[ResumeItem]:
        return [i for i in self.items if i.policy == ResumePolicy.REQUIRES_CONFIRMATION]

    def get(self, key: str) -> Optional[ResumeItem]:
        for item in self.items:
            if item.key == key:
                return item
        return None


class ResumeManager:
    """
    Writes and reads ResumableSnapshots.

    Usage:
        rm = ResumeManager(path=Path("data/resume.json"))
        rm.save(snapshot)         # called after each mission completes
        snapshot = rm.load()      # called on startup
        auto   = snapshot.auto_items()
        confirm = snapshot.confirmation_items()
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def save(self, snapshot: ResumableSnapshot) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(snapshot.to_dict(), fh, indent=2)

    def load(self) -> Optional[ResumableSnapshot]:
        if not self.path.exists():
            return None
        with open(self.path, encoding="utf-8") as fh:
            try:
                data = json.load(fh)
                return ResumableSnapshot.from_dict(data)
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(f"Corrupt resume file at {self.path}: {exc}") from exc

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()

    def exists(self) -> bool:
        return self.path.exists()

    def confirmation_summary(self, snapshot: ResumableSnapshot) -> str:
        """Return a human-readable list of items that need approval."""
        items = snapshot.confirmation_items()
        if not items:
            return "Nothing requires confirmation."
        lines = ["The following items require your confirmation before resuming:"]
        for item in items:
            note = f" — {item.note}" if item.note else ""
            lines.append(f"  [{item.key}]{note}")
        return "\n".join(lines)

