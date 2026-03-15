"""
core/agentic/decision_trace.py

Records *why* the agent made each decision — or why it did NOT act.
This is the primary explainability surface for both humans and other LLMs.

Each TraceEntry answers: "At time T, given context C, the agent chose A because R."
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class DecisionType(str, Enum):
    ACTION_TAKEN      = "action_taken"      # agent executed a tool/step
    ACTION_SKIPPED    = "action_skipped"    # agent decided not to act
    GOAL_SELECTED     = "goal_selected"     # agent picked the next goal
    MISSION_CREATED   = "mission_created"   # planner produced a mission
    ESCALATED         = "escalated"         # agent deferred to human
    OVERRIDDEN        = "overridden"        # human overrode agent decision
    RETRY_SCHEDULED   = "retry_scheduled"   # agent scheduled a retry
    BELIEF_UPDATED    = "belief_updated"    # a key belief changed
    POLICY_BLOCKED    = "policy_blocked"    # autonomy policy denied action


@dataclass(frozen=True)
class TraceEntry:
    """
    An immutable record of one agent decision moment.

    Never modify entries after creation.  Append-only log.
    """

    trace_id: str
    decision_type: DecisionType
    action: str                       # what was (or was not) done
    reason: str                       # human-readable explanation
    outcome: Optional[str]            # result, if known at trace time
    context_snapshot: dict            # snapshot of AgentContext at decision time
    metadata: dict                    # arbitrary extra data (step_id, rule_name, …)
    created_at: datetime

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "decision_type": self.decision_type.value,
            "action": self.action,
            "reason": self.reason,
            "outcome": self.outcome,
            "context_snapshot": self.context_snapshot,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }

    def explain(self) -> str:
        """One-sentence human-readable explanation."""
        ts = self.created_at.strftime("%H:%M:%S")
        outcome_str = f" → {self.outcome}" if self.outcome else ""
        return f"[{ts}] {self.decision_type.value}: {self.action}{outcome_str}. Reason: {self.reason}"


class DecisionTrace:
    """
    Append-only log of agent decision trace entries.

    Usage:
        trace = DecisionTrace()
        trace.record(
            decision_type=DecisionType.ACTION_TAKEN,
            action="send_email",
            reason="User requested email summary",
            context=ctx,
        )
        for entry in trace.since(mission_id="abc"):
            print(entry.explain())
    """

    def __init__(self) -> None:
        self._entries: list[TraceEntry] = []

    # ── Write ────────────────────────────────────────────────────────────

    def record(
        self,
        decision_type: DecisionType,
        action: str,
        reason: str,
        context: Any,                      # AgentContext (Any to avoid circular)
        outcome: Optional[str] = None,
        **metadata,
    ) -> TraceEntry:
        entry = TraceEntry(
            trace_id=str(uuid.uuid4()),
            decision_type=decision_type,
            action=action,
            reason=reason,
            outcome=outcome,
            context_snapshot=context.snapshot() if hasattr(context, "snapshot") else {},
            metadata=metadata,
            created_at=_utcnow(),
        )
        self._entries.append(entry)
        return entry

    # ── Query ────────────────────────────────────────────────────────────

    def all(self) -> list[TraceEntry]:
        return list(self._entries)

    def for_mission(self, mission_id: str) -> list[TraceEntry]:
        return [e for e in self._entries if e.metadata.get("mission_id") == mission_id]

    def for_goal(self, goal_id: str) -> list[TraceEntry]:
        return [e for e in self._entries if e.metadata.get("goal_id") == goal_id]

    def of_type(self, decision_type: DecisionType) -> list[TraceEntry]:
        return [e for e in self._entries if e.decision_type == decision_type]

    def since(self, dt: datetime) -> list[TraceEntry]:
        return [e for e in self._entries if e.created_at >= dt]

    def last_n(self, n: int) -> list[TraceEntry]:
        return self._entries[-n:]

    def explain_last(self, n: int = 5) -> str:
        """Return a plain-text summary of the last N decisions."""
        entries = self.last_n(n)
        if not entries:
            return "No decisions recorded yet."
        lines = [f"Last {len(entries)} decisions:"]
        lines += [f"  {e.explain()}" for e in entries]
        return "\n".join(lines)

    # ── Serialisation ────────────────────────────────────────────────────

    def snapshot(self) -> list[dict]:
        return [e.to_dict() for e in self._entries]

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"DecisionTrace({len(self._entries)} entries)"

