"""
reflection.py — Post-execution Reflection Engine for Jarvis Agentic Layer

Runs after a Mission completes (success or failure).
Evaluates outcome vs intent, detects failure patterns,
produces lessons learned, and updates the belief state + memory.
Does NOT re-execute or re-plan — it only reads and writes state.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .mission import Mission
    from .belief_state import BeliefState

logger = logging.getLogger(__name__)


@dataclass
class ReflectionReport:
    """Structured output from one reflection cycle."""

    mission_id: str
    goal_id: str
    outcome: str  # "success" | "partial" | "failure"
    summary: str
    lessons: List[str] = field(default_factory=list)
    failure_patterns: List[str] = field(default_factory=list)
    belief_updates: Dict[str, float] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict:
        return self.__dict__.copy()

    def human_readable(self) -> str:
        lines = [
            f"── Reflection Report ──────────────────────",
            f"Mission : {self.mission_id[:8]}",
            f"Goal    : {self.goal_id[:8]}",
            f"Outcome : {self.outcome.upper()}",
            f"Summary : {self.summary}",
        ]
        if self.lessons:
            lines.append("Lessons :")
            for l in self.lessons:
                lines.append(f"  • {l}")
        if self.failure_patterns:
            lines.append("Patterns:")
            for p in self.failure_patterns:
                lines.append(f"  ⚠ {p}")
        if self.belief_updates:
            lines.append("Belief Δ:")
            for k, v in self.belief_updates.items():
                lines.append(f"  {k}: {v:+.2f}")
        lines.append(f"At      : {self.timestamp}")
        return "\n".join(lines)


class ReflectionEngine:
    """
    Evaluates a completed Mission and produces a ReflectionReport.
    Writes the report to memory and updates the agent's belief state.

    Usage:
        engine = ReflectionEngine(belief_state, memory)
        report = engine.reflect(mission)
    """

    # Patterns that indicate systemic failure (matched against error strings)
    KNOWN_FAILURE_PATTERNS = {
        "network": "Repeated network failures — consider retry with backoff or offline fallback.",
        "timeout": "Timeout pattern detected — tasks may be too large; consider decomposition.",
        "permission": "Permission/auth failures — credentials or scopes may need updating.",
        "not found": "Resource-not-found pattern — upstream data may have changed.",
        "rate limit": "Rate-limiting detected — reduce request frequency or batch calls.",
    }

    def __init__(self, belief_state: "BeliefState", memory=None):
        """
        Args:
            belief_state: The agent's BeliefState instance to update.
            memory: The existing hybrid_memory instance (optional).
                    If provided, reflection summaries are written to it.
        """
        self.belief_state = belief_state
        self.memory = memory

    def reflect(self, mission: "Mission") -> ReflectionReport:
        """
        Main entry point. Call after a mission finishes (any terminal state).
        Returns a ReflectionReport and side-effects belief state + memory.
        """
        from .mission import CheckpointStatus, MissionStatus

        logger.info("Reflecting on mission [%s] '%s'", mission.mission_id[:8], mission.title)

        # ── 1. Determine outcome ──────────────────────────────────────────
        total = len(mission.checkpoints)
        done = sum(1 for c in mission.checkpoints if c.status == CheckpointStatus.DONE)
        failed = sum(1 for c in mission.checkpoints if c.status == CheckpointStatus.FAILED)

        if mission.status == MissionStatus.COMPLETED and failed == 0:
            outcome = "success"
        elif done > 0 and done >= failed:
            outcome = "partial"
        else:
            outcome = "failure"

        # ── 2. Build summary ──────────────────────────────────────────────
        summary = self._build_summary(mission, outcome, done, total, failed)

        # ── 3. Extract lessons ────────────────────────────────────────────
        lessons = self._extract_lessons(mission, outcome)

        # ── 4. Detect failure patterns ────────────────────────────────────
        patterns = self._detect_patterns(mission)

        # ── 5. Compute belief updates ─────────────────────────────────────
        belief_updates = self._compute_belief_updates(mission, outcome, patterns)
        for key, delta in belief_updates.items():
            self.belief_state.update(key, delta, source="reflection")

        # ── 6. Assemble report ────────────────────────────────────────────
        report = ReflectionReport(
            mission_id=mission.mission_id,
            goal_id=mission.goal_id,
            outcome=outcome,
            summary=summary,
            lessons=lessons,
            failure_patterns=patterns,
            belief_updates=belief_updates,
        )

        logger.info("Reflection complete: %s\n%s", outcome, report.human_readable())

        # ── 7. Persist to memory ──────────────────────────────────────────
        self._write_to_memory(report)

        return report

    # ---------------------------------------------------------------- helpers

    def _build_summary(
        self,
        mission: "Mission",
        outcome: str,
        done: int,
        total: int,
        failed: int,
    ) -> str:
        title = mission.title
        if outcome == "success":
            return f"Mission '{title}' completed successfully. All {total} checkpoints passed."
        elif outcome == "partial":
            return (
                f"Mission '{title}' partially completed. "
                f"{done}/{total} checkpoints succeeded; {failed} failed."
            )
        else:
            abort = f" Reason: {mission.abort_reason}" if mission.abort_reason else ""
            return (
                f"Mission '{title}' failed. "
                f"{done}/{total} checkpoints completed.{abort}"
            )

    def _extract_lessons(self, mission: "Mission", outcome: str) -> List[str]:
        from .mission import CheckpointStatus

        lessons: List[str] = []

        failed_cps = [c for c in mission.checkpoints if c.status == CheckpointStatus.FAILED]
        done_cps = [c for c in mission.checkpoints if c.status == CheckpointStatus.DONE]

        if outcome == "success":
            lessons.append(f"Approach used in '{mission.title}' is effective — consider reuse.")
        if failed_cps:
            for cp in failed_cps:
                lessons.append(f"Step '{cp.name}' failed: {cp.error or 'unknown error'}.")
        if done_cps and failed_cps:
            lessons.append(
                "Some checkpoints succeeded before failure — partial results may be reusable."
            )
        if mission.abort_reason:
            lessons.append(f"Abort triggered: {mission.abort_reason}")

        return lessons

    def _detect_patterns(self, mission: "Mission") -> List[str]:
        from .mission import CheckpointStatus

        errors: List[str] = []
        for cp in mission.checkpoints:
            if cp.status == CheckpointStatus.FAILED and cp.error:
                errors.append(cp.error.lower())

        matched: List[str] = []
        for keyword, advice in self.KNOWN_FAILURE_PATTERNS.items():
            if any(keyword in e for e in errors):
                if advice not in matched:
                    matched.append(advice)
        return matched

    def _compute_belief_updates(
        self,
        mission: "Mission",
        outcome: str,
        patterns: List[str],
    ) -> Dict[str, float]:
        """
        Returns {belief_key: delta} — small signed floats to nudge beliefs.
        Actual clamping is done by BeliefState.update().
        """
        updates: Dict[str, float] = {}

        if outcome == "success":
            updates["system_reliability"] = +0.05
            updates["agent_confidence"] = +0.03
        elif outcome == "partial":
            updates["system_reliability"] = -0.02
        else:
            updates["system_reliability"] = -0.05
            updates["agent_confidence"] = -0.05

        if any("network" in p.lower() for p in patterns):
            updates["network_reliability"] = -0.10
        if any("rate limit" in p.lower() for p in patterns):
            updates["api_rate_limit_risk"] = +0.10

        return updates

    def _write_to_memory(self, report: ReflectionReport) -> None:
        """
        Write a reflection summary to the existing hybrid_memory.
        Safe to call even if memory is None (logs a warning instead).
        """
        if self.memory is None:
            logger.debug("No memory system attached — skipping memory write.")
            return
        try:
            # hybrid_memory is expected to have a store(key, value) or similar interface.
            key = f"reflection:{report.mission_id}"
            self.memory.store(key, report.to_dict())
            logger.debug("Reflection written to memory under key '%s'.", key)
        except Exception as exc:
            logger.warning("Could not write reflection to memory: %s", exc)
