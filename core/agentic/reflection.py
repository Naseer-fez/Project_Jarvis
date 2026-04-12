"""Reflection engine for updating belief state from mission outcomes."""

from __future__ import annotations

import json

from core.agentic.mission import CheckpointStatus, Mission, MissionStatus


class ReflectionEngine:
    def __init__(self, belief_state, hybrid_memory) -> None:
        self.belief_state = belief_state
        self.hybrid_memory = hybrid_memory
        self.last_report: dict | None = None

    def reflect(self, mission: Mission) -> str:
        if mission.status in {MissionStatus.ABORTED, MissionStatus.FAILED} or any(
            checkpoint.status == CheckpointStatus.FAILED
            for checkpoint in mission.checkpoints
        ):
            outcome = "failure"
            self.belief_state.update("agent_confidence", -0.1)
        elif mission.status == MissionStatus.SUCCEEDED:
            outcome = "success"
            self.belief_state.update("agent_confidence", 0.05)
        else:
            outcome = "partial"
            self.belief_state.update("agent_confidence", -0.02)

        self.belief_state.save()
        report = {
            "mission_id": mission.mission_id,
            "goal_id": mission.goal_id,
            "status": mission.status.value,
            "outcome": outcome,
            "belief_after": self.belief_state.scores(),
        }
        self.last_report = report
        self.hybrid_memory.store_fact(
            f"reflection:{mission.mission_id}",
            json.dumps(report),
            source="reflection",
        )
        return outcome


__all__ = ["ReflectionEngine"]
