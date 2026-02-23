"""
core/agentic/belief_state.py

Mutable, versioned store of agent beliefs — what the agent currently
"thinks is true" about the world, its environment, and its own state.

Beliefs differ from memory (episodic facts) in that they are:
- Actively reasoned about and updated
- Associated with a confidence level
- Retractable when contradicting evidence arrives
- Exposed to the autonomy policy for go/no-go decisions
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Belief:
    """A single agent belief with a provenance trail."""

    belief_id: str
    key: str                    # dot-path key, e.g. "env.network.available"
    value: Any
    confidence: float           # 0.0 – 1.0
    source: str                 # what produced this belief
    retracted: bool = False

    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)
    retracted_at: Optional[datetime] = None
    retraction_reason: Optional[str] = None

    def retract(self, reason: str = "") -> None:
        self.retracted = True
        self.retracted_at = _utcnow()
        self.retraction_reason = reason

    def update(self, value: Any, confidence: float, source: str) -> None:
        self.value = value
        self.confidence = max(0.0, min(1.0, confidence))
        self.source = source
        self.updated_at = _utcnow()
        self.retracted = False               # un-retract if re-asserted

    def to_dict(self) -> dict:
        return {
            "belief_id": self.belief_id,
            "key": self.key,
            "value": self.value,
            "confidence": self.confidence,
            "source": self.source,
            "retracted": self.retracted,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "retracted_at": self.retracted_at.isoformat() if self.retracted_at else None,
            "retraction_reason": self.retraction_reason,
        }


class BeliefState:
    """
    Mutable key-value store of agent beliefs, keyed by dot-path strings.

    Example keys:
        "env.network.available"
        "user.intent.confirmed"
        "task.email_sent"

    Usage:
        bs = BeliefState()
        bs.assert_belief("env.network.available", True, confidence=0.95, source="ping_tool")
        b = bs.get("env.network.available")
        bs.retract("env.network.available", reason="DNS failure observed")
    """

    def __init__(self) -> None:
        # key → list[Belief] (history; latest is last)
        self._store: dict[str, list[Belief]] = {}

    # ── Core ops ─────────────────────────────────────────────────────────

    def assert_belief(
        self,
        key: str,
        value: Any,
        confidence: float = 1.0,
        source: str = "agent",
    ) -> Belief:
        """Assert or update a belief."""
        if key in self._store and not self._store[key][-1].retracted:
            belief = self._store[key][-1]
            belief.update(value, confidence, source)
        else:
            belief = Belief(
                belief_id=str(uuid.uuid4()),
                key=key,
                value=value,
                confidence=max(0.0, min(1.0, confidence)),
                source=source,
            )
            self._store.setdefault(key, []).append(belief)
        return belief

    def retract(self, key: str, reason: str = "") -> bool:
        """Retract the current belief for a key. Returns True if found."""
        if key in self._store and not self._store[key][-1].retracted:
            self._store[key][-1].retract(reason)
            return True
        return False

    def get(self, key: str, default: Any = None) -> Optional[Belief]:
        """Return the current (non-retracted) belief, or None."""
        if key in self._store:
            b = self._store[key][-1]
            if not b.retracted:
                return b
        return default

    def get_value(self, key: str, default: Any = None) -> Any:
        b = self.get(key)
        return b.value if b is not None else default

    def get_confidence(self, key: str, default: float = 0.0) -> float:
        b = self.get(key)
        return b.confidence if b is not None else default

    # ── Bulk queries ─────────────────────────────────────────────────────

    def all_active(self) -> list[Belief]:
        """Return all non-retracted beliefs."""
        return [
            history[-1]
            for history in self._store.values()
            if not history[-1].retracted
        ]

    def low_confidence(self, threshold: float = 0.5) -> list[Belief]:
        return [b for b in self.all_active() if b.confidence < threshold]

    def history(self, key: str) -> list[Belief]:
        return list(self._store.get(key, []))

    # ── Serialisation ────────────────────────────────────────────────────

    def snapshot(self) -> list[dict]:
        result = []
        for history in self._store.values():
            result.extend(b.to_dict() for b in history)
        return result

    def restore(self, data: list[dict]) -> None:
        for d in data:
            belief = Belief(
                belief_id=d["belief_id"],
                key=d["key"],
                value=d["value"],
                confidence=d["confidence"],
                source=d["source"],
                retracted=d["retracted"],
                created_at=datetime.fromisoformat(d["created_at"]),
                updated_at=datetime.fromisoformat(d["updated_at"]),
                retracted_at=datetime.fromisoformat(d["retracted_at"]) if d.get("retracted_at") else None,
                retraction_reason=d.get("retraction_reason"),
            )
            self._store.setdefault(d["key"], []).append(belief)

    def __repr__(self) -> str:
        active = len(self.all_active())
        total  = sum(len(v) for v in self._store.values())
        return f"BeliefState(active={active}, total_history={total})"

