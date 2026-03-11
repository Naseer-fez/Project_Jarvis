"""Post-mission reflection utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass

from .belief_state import BeliefState
from .mission import CheckpointStatus, Mission, MissionStatus, StepStatus


@dataclass
class ReflectionEngine:
    belief_state: BeliefState
    hybrid_memory: object | None = None
    last_report: dict[str, object] | None = None

    def reflect(self, mission: Mission) -> str:
        outcome = self._classify_outcome(mission)
        self._apply_belief_updates(mission, outcome)
        self.belief_state.save()

        report = {
            "mission_id": mission.mission_id,
            "goal_id": mission.goal_id,
            "title": mission.title,
            "outcome": outcome,
            "status": mission.status.value,
            "abort_reason": mission.abort_reason,
            "belief_after": self.belief_state.scores(),
        }
        self.last_report = report

        if self.hybrid_memory is not None and hasattr(self.hybrid_memory, "store_fact"):
            self.hybrid_memory.store_fact(
                f"reflection:{mission.mission_id}",
                json.dumps(report),
                source="reflection",
            )

        return outcome

    def _classify_outcome(self, mission: Mission) -> str:
        if mission.status in {MissionStatus.COMPLETED, MissionStatus.SUCCEEDED}:
            return "success"

        if any(checkpoint.status == CheckpointStatus.FAILED for checkpoint in mission.checkpoints):
            return "failure"

        if any(step.status == StepStatus.FAILED for step in mission.steps):
            return "failure"

        if mission.status in {MissionStatus.ABORTED, MissionStatus.FAILED}:
            return "failure"

        return "partial"

    def _apply_belief_updates(self, mission: Mission, outcome: str) -> None:
        error_text = " ".join(
            filter(
                None,
                [mission.abort_reason] + [checkpoint.error for checkpoint in mission.checkpoints],
            )
        ).lower()

        if outcome == "success":
            self.belief_state.update("agent_confidence", 0.05)
            self.belief_state.update("system_reliability", 0.02)
            return

        self.belief_state.update("agent_confidence", -0.1)

        if "network" in error_text or "timeout" in error_text:
            self.belief_state.update("network_reliability", -0.15)
        if "rate limit" in error_text:
            self.belief_state.update("api_rate_limit_risk", 0.2)


__all__ = ["ReflectionEngine"]
