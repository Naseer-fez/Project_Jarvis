"""
decision_trace.py — Human-readable Decision Audit Trail for Jarvis Agentic Layer

Records why a plan was chosen, why alternatives were rejected,
confidence scores, and risk scores.
Must be inspectable by humans at any time.
"""

import json
import importlib as _importlib
import sys as _sys
_stdlib_logging = _importlib.import_module("logging")

logging = _stdlib_logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

TRACES_DIR = Path("data/agentic/traces")


@dataclass
class Alternative:
    """A rejected option with reason."""

    name: str
    description: str
    rejection_reason: str
    risk_score: float = 0.0


@dataclass
class DecisionEntry:
    """A single recorded decision."""

    trace_id: str
    goal_id: str
    mission_id: Optional[str]
    decision: str            # What was decided
    rationale: str           # Why this decision
    confidence: float        # 0.0 – 1.0
    risk_score: float        # 0.0 – 1.0
    alternatives_rejected: List[Alternative] = field(default_factory=list)
    context: Dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict:
        return {
            "trace_id": self.trace_id,
            "goal_id": self.goal_id,
            "mission_id": self.mission_id,
            "decision": self.decision,
            "rationale": self.rationale,
            "confidence": self.confidence,
            "risk_score": self.risk_score,
            "alternatives_rejected": [
                {
                    "name": a.name,
                    "description": a.description,
                    "rejection_reason": a.rejection_reason,
                    "risk_score": a.risk_score,
                }
                for a in self.alternatives_rejected
            ],
            "context": self.context,
            "timestamp": self.timestamp,
        }

    def human_readable(self) -> str:
        lines = [
            f"╔══ Decision Trace [{self.trace_id[:8]}] ══",
            f"║  Timestamp  : {self.timestamp}",
            f"║  Goal       : {self.goal_id[:8]}",
            f"║  Mission    : {self.mission_id[:8] if self.mission_id else 'N/A'}",
            f"║  Decision   : {self.decision}",
            f"║  Rationale  : {self.rationale}",
            f"║  Confidence : {self.confidence:.0%}",
            f"║  Risk Score : {self.risk_score:.0%}",
        ]
        if self.alternatives_rejected:
            lines.append("║  Rejected Alternatives:")
            for alt in self.alternatives_rejected:
                lines.append(f"║    ✗ {alt.name}: {alt.rejection_reason} (risk={alt.risk_score:.0%})")
        if self.context:
            lines.append(f"║  Context    : {json.dumps(self.context, default=str)}")
        lines.append("╚" + "═" * 44)
        return "\n".join(lines)


class DecisionTrace:
    """
    Records all significant agent decisions in an auditable, human-readable log.

    One trace file per goal (JSON lines format for easy streaming).

    Usage:
        trace = DecisionTrace(goal_id="abc123")
        trace.record(
            decision="Use planner strategy A",
            rationale="Strategy A has 80% historical success rate",
            confidence=0.80,
            risk_score=0.20,
            alternatives=[
                Alternative("Strategy B", "parallel execution", "higher risk of race conditions", 0.55)
            ]
        )
        trace.save()
    """

    def __init__(self, goal_id: str, mission_id: Optional[str] = None):
        self.goal_id = goal_id
        self.mission_id = mission_id
        self._entries: List[DecisionEntry] = []
        self._trace_file: Path = TRACES_DIR / f"{goal_id}.jsonl"

    # ────────────────────────────────────────────────────── Recording

    def record(
        self,
        decision: str,
        rationale: str,
        confidence: float,
        risk_score: float,
        alternatives: Optional[List[Alternative]] = None,
        context: Optional[Dict] = None,
    ) -> DecisionEntry:
        """
        Record a decision.  Appends to in-memory log and persists immediately.
        """
        import uuid

        entry = DecisionEntry(
            trace_id=str(uuid.uuid4()),
            goal_id=self.goal_id,
            mission_id=self.mission_id,
            decision=decision,
            rationale=rationale,
            confidence=max(0.0, min(1.0, confidence)),
            risk_score=max(0.0, min(1.0, risk_score)),
            alternatives_rejected=alternatives or [],
            context=context or {},
        )
        self._entries.append(entry)
        self._append_to_file(entry)

        logger.info(
            "Decision recorded [%s]: '%s' (conf=%.0f%%, risk=%.0f%%)",
            entry.trace_id[:8],
            decision,
            confidence * 100,
            risk_score * 100,
        )
        return entry

    # ────────────────────────────────────────────────────── Persistence

    def _append_to_file(self, entry: DecisionEntry) -> None:
        """Append a single JSON line to the trace file (fast, append-only)."""
        TRACES_DIR.mkdir(parents=True, exist_ok=True)
        try:
            with self._trace_file.open("a") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")
        except Exception as exc:
            logger.error("Failed to write decision trace: %s", exc)

    def save(self) -> None:
        """Alias kept for API consistency — append-only file is written per record."""
        pass  # Already persisted on each record()

    @classmethod
    def load_for_goal(cls, goal_id: str) -> "DecisionTrace":
        """Load all recorded decisions for a goal from disk."""
        instance = cls(goal_id=goal_id)
        path = TRACES_DIR / f"{goal_id}.jsonl"
        if not path.exists():
            return instance
        try:
            with path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    raw = json.loads(line)
                    alts = [
                        Alternative(**a) for a in raw.pop("alternatives_rejected", [])
                    ]
                    raw["alternatives_rejected"] = alts
                    instance._entries.append(DecisionEntry(**raw))
        except Exception as exc:
            logger.error("Failed to load trace for goal %s: %s", goal_id, exc)
        return instance

    # ────────────────────────────────────────────────────── Inspection

    def entries(self) -> List[DecisionEntry]:
        return list(self._entries)

    def print_full_trace(self) -> str:
        """Return complete human-readable audit log."""
        if not self._entries:
            return f"No decisions recorded for goal {self.goal_id[:8]}."
        return "\n\n".join(e.human_readable() for e in self._entries)

    def summary(self) -> str:
        total = len(self._entries)
        if total == 0:
            return "No decisions recorded."
        avg_conf = sum(e.confidence for e in self._entries) / total
        avg_risk = sum(e.risk_score for e in self._entries) / total
        return (
            f"{total} decision(s) | avg confidence={avg_conf:.0%} | avg risk={avg_risk:.0%}"
        )

