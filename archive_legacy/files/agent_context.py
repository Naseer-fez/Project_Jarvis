"""
core/agentic/agent_context.py

Single shared read-only runtime context passed to planner, dispatcher,
reflection, and autonomy policy. Eliminates global variables and hidden
dependencies. All mutations go through explicit update methods.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class AgentContext:
    """
    Immutable-by-convention runtime context for a single agent session.

    Pass this object (or a snapshot of it) to every subsystem that needs
    to know *where* the agent is in its lifecycle. Never mutate fields
    directly — call the provided update helpers so changes are logged.
    """

    # ── Identity ────────────────────────────────────────────────────────
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    current_goal_id: Optional[str] = None
    mission_id: Optional[str] = None

    # ── Scoring ─────────────────────────────────────────────────────────
    confidence_score: float = 1.0   # 0.0 – 1.0 (how sure the agent is)
    risk_score: float = 0.0         # 0.0 – 1.0 (how dangerous the action is)

    # ── Control flags ───────────────────────────────────────────────────
    interrupt_flag: bool = False     # set True to halt execution cleanly
    paused: bool = False             # set True while awaiting human input

    # ── Timestamps ──────────────────────────────────────────────────────
    created_at: datetime = field(default_factory=_utcnow)
    last_updated_at: datetime = field(default_factory=_utcnow)
    goal_started_at: Optional[datetime] = None
    mission_started_at: Optional[datetime] = None

    # ── Audit trail (internal) ──────────────────────────────────────────
    _change_log: list[dict] = field(default_factory=list, repr=False)

    # ────────────────────────────────────────────────────────────────────
    # Update helpers
    # ────────────────────────────────────────────────────────────────────

    def set_goal(self, goal_id: str) -> None:
        self._record("set_goal", goal_id=goal_id)
        self.current_goal_id = goal_id
        self.goal_started_at = _utcnow()
        self._touch()

    def set_mission(self, mission_id: str) -> None:
        self._record("set_mission", mission_id=mission_id)
        self.mission_id = mission_id
        self.mission_started_at = _utcnow()
        self._touch()

    def update_scores(self, confidence: float, risk: float) -> None:
        self._record("update_scores", confidence=confidence, risk=risk)
        self.confidence_score = max(0.0, min(1.0, confidence))
        self.risk_score = max(0.0, min(1.0, risk))
        self._touch()

    def raise_interrupt(self, reason: str = "") -> None:
        self._record("raise_interrupt", reason=reason)
        self.interrupt_flag = True
        self._touch()

    def clear_interrupt(self) -> None:
        self._record("clear_interrupt")
        self.interrupt_flag = False
        self._touch()

    def pause(self, reason: str = "") -> None:
        self._record("pause", reason=reason)
        self.paused = True
        self._touch()

    def resume(self) -> None:
        self._record("resume")
        self.paused = False
        self._touch()

    # ────────────────────────────────────────────────────────────────────
    # Derived properties
    # ────────────────────────────────────────────────────────────────────

    @property
    def is_safe_to_proceed(self) -> bool:
        """True when the agent may take the next action autonomously."""
        return (
            not self.interrupt_flag
            and not self.paused
            and self.confidence_score >= 0.5
            and self.risk_score <= 0.7
        )

    @property
    def change_log(self) -> list[dict]:
        return list(self._change_log)

    def snapshot(self) -> dict:
        """Return a JSON-serialisable dict snapshot for persistence."""
        return {
            "session_id": self.session_id,
            "current_goal_id": self.current_goal_id,
            "mission_id": self.mission_id,
            "confidence_score": self.confidence_score,
            "risk_score": self.risk_score,
            "interrupt_flag": self.interrupt_flag,
            "paused": self.paused,
            "created_at": self.created_at.isoformat(),
            "last_updated_at": self.last_updated_at.isoformat(),
            "goal_started_at": self.goal_started_at.isoformat() if self.goal_started_at else None,
            "mission_started_at": self.mission_started_at.isoformat() if self.mission_started_at else None,
        }

    # ────────────────────────────────────────────────────────────────────
    # Internal
    # ────────────────────────────────────────────────────────────────────

    def _touch(self) -> None:
        self.last_updated_at = _utcnow()

    def _record(self, event: str, **kwargs) -> None:
        self._change_log.append({"event": event, "at": _utcnow().isoformat(), **kwargs})

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"AgentContext(session={self.session_id[:8]}…, "
            f"goal={self.current_goal_id}, "
            f"conf={self.confidence_score:.2f}, "
            f"risk={self.risk_score:.2f}, "
            f"interrupt={self.interrupt_flag})"
        )

