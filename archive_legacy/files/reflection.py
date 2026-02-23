"""
core/agentic/reflection.py

Post-action evaluation: the agent looks back at what it just did, decides
whether it succeeded, and updates confidence/risk scores accordingly.

Responsibilities:
- Evaluate the outcome of a completed Mission
- Produce a ReflectionRecord (stored, never mutated)
- Update the AgentContext with new scores
- Decide whether to retry, continue, or escalate
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from core.agentic.agent_context import AgentContext
    from core.agentic.mission import core.autonomy.goals


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ReflectionVerdict(str, Enum):
    CONTINUE   = "continue"    # mission succeeded, proceed to next
    RETRY      = "retry"       # mission failed but is retryable
    ESCALATE   = "escalate"    # human decision required
    ABANDON    = "abandon"     # goal cannot be achieved; stop
    PARTIAL    = "partial"     # partial success; decide based on policy


@dataclass(frozen=True)
class ReflectionRecord:
    """
    Immutable record of a single reflection event.
    Stored for audit, debugging, and learning.
    """

    reflection_id: str
    mission_id: str
    goal_id: str
    verdict: ReflectionVerdict

    confidence_before: float
    risk_before: float
    confidence_after: float
    risk_after: float

    summary: str                        # natural-language explanation
    lessons: list[str] = field(default_factory=list)  # bullet-points for future reference

    created_at: datetime = field(default_factory=_utcnow)

    def to_dict(self) -> dict:
        return {
            "reflection_id": self.reflection_id,
            "mission_id": self.mission_id,
            "goal_id": self.goal_id,
            "verdict": self.verdict.value,
            "confidence_before": self.confidence_before,
            "risk_before": self.risk_before,
            "confidence_after": self.confidence_after,
            "risk_after": self.risk_after,
            "summary": self.summary,
            "lessons": self.lessons,
            "created_at": self.created_at.isoformat(),
        }


class ReflectionEngine:
    """
    Evaluates a completed Mission and emits a ReflectionRecord.

    Inject custom scorers to override the default heuristics.
    """

    def __init__(
        self,
        confidence_decay_on_failure: float = 0.2,
        risk_escalation_on_failure: float = 0.15,
        confidence_boost_on_success: float = 0.05,
        risk_decay_on_success: float = 0.05,
    ) -> None:
        self._conf_decay   = confidence_decay_on_failure
        self._risk_esc     = risk_escalation_on_failure
        self._conf_boost   = confidence_boost_on_success
        self._risk_decay   = risk_decay_on_success
        self._history: list[ReflectionRecord] = []

    # ── Public API ───────────────────────────────────────────────────────

    def reflect(
        self,
        mission: "Mission",
        context: "AgentContext",
        max_retries: int = 2,
    ) -> ReflectionRecord:
        """
        Evaluate the mission outcome, update the context, and return a record.

        Call this immediately after a mission reaches a terminal state.
        """
        import uuid
        from core.agentic.mission import core.autonomy.goalsStatus

        conf_before = context.confidence_score
        risk_before = context.risk_score

        verdict, summary, lessons, conf_after, risk_after = self._evaluate(
            mission, conf_before, risk_before, max_retries
        )

        context.update_scores(conf_after, risk_after)

        record = ReflectionRecord(
            reflection_id=str(uuid.uuid4()),
            mission_id=mission.mission_id,
            goal_id=mission.goal_id,
            verdict=verdict,
            confidence_before=conf_before,
            risk_before=risk_before,
            confidence_after=conf_after,
            risk_after=risk_after,
            summary=summary,
            lessons=lessons,
        )
        self._history.append(record)
        return record

    def history(self, goal_id: Optional[str] = None) -> list[ReflectionRecord]:
        if goal_id:
            return [r for r in self._history if r.goal_id == goal_id]
        return list(self._history)

    def last(self) -> Optional[ReflectionRecord]:
        return self._history[-1] if self._history else None

    # ── Internal logic ───────────────────────────────────────────────────

    def _evaluate(
        self,
        mission: "Mission",
        conf: float,
        risk: float,
        max_retries: int,
    ) -> tuple[ReflectionVerdict, str, list[str], float, float]:
        from core.agentic.mission import core.autonomy.goalsStatus

        lessons: list[str] = []

        if mission.status == MissionStatus.SUCCEEDED:
            verdict  = ReflectionVerdict.CONTINUE
            summary  = f"Mission '{mission.description}' completed successfully."
            conf_out = min(1.0, conf + self._conf_boost)
            risk_out = max(0.0, risk - self._risk_decay)
            lessons.append("Action succeeded — strategy is valid.")

        elif mission.status == MissionStatus.ABORTED:
            verdict  = ReflectionVerdict.ESCALATE
            summary  = f"Mission '{mission.description}' was aborted by interrupt."
            conf_out = max(0.0, conf - self._conf_decay)
            risk_out = min(1.0, risk + self._risk_esc)
            lessons.append("Abort indicates unresolved conflict — require human review.")

        else:  # FAILED
            failed_steps = [s for s in mission.steps if s.status.value == "failed"]
            errors = [s.error or "unknown" for s in failed_steps]

            if mission.attempt_number >= max_retries:
                verdict = ReflectionVerdict.ABANDON
                summary = (
                    f"Mission '{mission.description}' failed after "
                    f"{mission.attempt_number} attempt(s). Giving up."
                )
                lessons.append("Repeated failure — reassess goal feasibility.")
            else:
                verdict = ReflectionVerdict.RETRY
                summary = (
                    f"Mission '{mission.description}' failed (attempt "
                    f"{mission.attempt_number}). Scheduling retry."
                )
                lessons.append("First failure — retry with same or adjusted params.")

            conf_out = max(0.0, conf - self._conf_decay)
            risk_out = min(1.0, risk + self._risk_esc)
            if errors:
                lessons.append(f"Step errors observed: {'; '.join(errors[:3])}")

        # Low confidence after multiple reflections → escalate regardless
        if conf_out < 0.3 and verdict not in (ReflectionVerdict.ABANDON, ReflectionVerdict.ESCALATE):
            verdict = ReflectionVerdict.ESCALATE
            summary += " [auto-escalated: confidence too low]"
            lessons.append("Confidence dropped below safe threshold — human needed.")

        return verdict, summary, lessons, round(conf_out, 4), round(risk_out, 4)

