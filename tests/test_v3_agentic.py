from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import core.agentic.belief_state as belief_state_module
import core.agentic.goal_manager as goal_manager_module
import core.agentic.mission as mission_module
from core.agentic.autonomy_policy import AutonomyPolicy, PolicyVerdict
from core.agentic.belief_state import BeliefState
from core.agentic.goal_manager import GoalManager, GoalStatus
from core.agentic.mission import CheckpointStatus, Mission, MissionBuilder, MissionStatus, StepStatus
from core.agentic.reflection import ReflectionEngine


@pytest.fixture()
def isolated_agentic_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    agentic_dir = tmp_path / "data" / "agentic"
    belief_path = agentic_dir / "belief_state.json"
    goals_path = agentic_dir / "goals.json"
    missions_dir = agentic_dir / "missions"

    monkeypatch.setattr(belief_state_module, "DATA_DIR", agentic_dir)
    monkeypatch.setattr(belief_state_module, "BELIEF_STATE_PATH", belief_path)
    monkeypatch.setattr(goal_manager_module, "GOALS_PATH", goals_path)
    monkeypatch.setattr(mission_module, "MISSIONS_DIR", missions_dir)

    return {
        "agentic_dir": agentic_dir,
        "belief_path": belief_path,
        "goals_path": goals_path,
        "missions_dir": missions_dir,
    }


def test_belief_state(isolated_agentic_storage: dict[str, Path]) -> None:
    state = BeliefState(agent_confidence=0.8)

    assert state.scores()["agent_confidence"] == 0.8
    assert state.update("agent_confidence", 0.5) == 1.0
    assert state.agent_confidence == 1.0
    assert state.update("agent_confidence", -1.5) == 0.0
    assert state.agent_confidence == 0.0

    saved_path = state.save()
    persisted_payload = json.loads(saved_path.read_text(encoding="utf-8"))
    loaded_state = BeliefState().load()

    assert saved_path == isolated_agentic_storage["belief_path"]
    assert persisted_payload == state.to_dict()
    assert loaded_state.to_dict() == state.to_dict()


def test_goal_manager(isolated_agentic_storage: dict[str, Path]) -> None:
    manager = GoalManager(storage_path=isolated_agentic_storage["goals_path"])

    goal_id = manager.create_goal("Ship agentic layer tests", priority=2)
    goal = manager.get_goal(goal_id)
    assert goal.status == GoalStatus.PENDING

    goal = manager.start_goal(goal_id)
    assert goal.status == GoalStatus.ACTIVE
    assert goal.started_at is not None

    goal = manager.complete_goal(goal_id, outcome="tests passed")
    assert goal.status == GoalStatus.COMPLETED
    assert goal.completed_at is not None

    reloaded = GoalManager(storage_path=isolated_agentic_storage["goals_path"]).load()
    assert reloaded.get_goal(goal_id).status == GoalStatus.COMPLETED

    pending_id = reloaded.create_goal("Pending follow-up")
    stalled_id = reloaded.create_goal("Stalled follow-up")
    completed_id = reloaded.create_goal("Already completed")
    aborted_id = reloaded.create_goal("No longer needed")

    reloaded.stall_goal(stalled_id, reason="waiting on input")
    reloaded.complete_goal(completed_id, outcome="done")
    reloaded.abort_goal(aborted_id, reason="cancelled")

    resumable_ids = {goal.goal_id for goal in reloaded.resumable_goals()}

    assert pending_id in resumable_ids
    assert stalled_id in resumable_ids
    assert goal_id not in resumable_ids
    assert completed_id not in resumable_ids
    assert aborted_id not in resumable_ids


def test_autonomy_policy() -> None:
    belief_state = MagicMock()
    belief_state.agent_confidence = 0.9
    belief_state.risk_tolerance = 0.9
    belief_state.should_ask_user = MagicMock(return_value=False)

    policy = AutonomyPolicy(belief_state)

    hard_deny = policy.evaluate("disable_logging", risk_score=0.01)
    always_confirm = policy.evaluate("send_email", risk_score=0.1)
    risk_deny = policy.evaluate("unknown_action", risk_score=0.98)
    risk_confirm = policy.evaluate("unknown_action", risk_score=0.65)

    assert hard_deny.verdict == PolicyVerdict.DENY
    assert hard_deny.rule_name == "hard_deny_list"
    assert always_confirm.verdict == PolicyVerdict.REQUIRE_CONFIRM
    assert always_confirm.rule_name == "always_confirm_list"
    assert risk_deny.verdict == PolicyVerdict.DENY
    assert risk_deny.rule_name == "risk_threshold_deny"
    assert risk_confirm.verdict == PolicyVerdict.REQUIRE_CONFIRM
    assert risk_confirm.rule_name == "risk_threshold_confirm"


def test_reflection_engine(isolated_agentic_storage: dict[str, Path]) -> None:
    hybrid_memory = MagicMock(spec=["store_fact"])
    belief_state = BeliefState()
    engine = ReflectionEngine(belief_state=belief_state, hybrid_memory=hybrid_memory)

    mission = Mission(goal_id="goal-123", title="Recover mailbox sync")
    checkpoint = mission.add_checkpoint("connect", "Connect to the upstream service")
    mission.start()
    mission.mark_checkpoint(
        checkpoint.checkpoint_id,
        CheckpointStatus.FAILED,
        error="network timeout while connecting",
    )
    mission.abort("network timeout while connecting")

    outcome = engine.reflect(mission)

    assert outcome == "failure"
    assert outcome in {"success", "partial", "failure"}
    assert isolated_agentic_storage["belief_path"].exists()
    assert engine.last_report is not None

    # ReflectionEngine currently persists via store_fact(...) on hybrid memory.
    hybrid_memory.store_fact.assert_called_once()
    args, kwargs = hybrid_memory.store_fact.call_args
    stored_report = json.loads(args[1])

    assert args[0] == f"reflection:{mission.mission_id}"
    assert kwargs == {"source": "reflection"}
    assert stored_report == engine.last_report
    assert stored_report["mission_id"] == mission.mission_id
    assert stored_report["outcome"] == outcome
    assert stored_report["belief_after"] == belief_state.scores()


def test_mission_loads_legacy_step_payload(
    isolated_agentic_storage: dict[str, Path],
) -> None:
    mission_id = "legacy-mission"
    legacy_payload = {
        "mission_id": mission_id,
        "goal_id": "goal-legacy",
        "description": "Legacy step mission",
        "status": "succeeded",
        "steps": [
            {
                "step_id": "step-1",
                "name": "search",
                "tool": "search_tool",
                "status": "succeeded",
            }
        ],
    }

    mission_path = isolated_agentic_storage["missions_dir"] / f"{mission_id}.json"
    mission_path.parent.mkdir(parents=True, exist_ok=True)
    mission_path.write_text(json.dumps(legacy_payload), encoding="utf-8")

    mission = Mission.load(mission_id)

    assert mission is not None
    assert mission.title == "Legacy step mission"
    assert mission.description == "Legacy step mission"
    assert mission.status == MissionStatus.SUCCEEDED
    assert mission.steps[0].status == StepStatus.SUCCEEDED
    assert mission.checkpoints[0].status == CheckpointStatus.DONE


def test_legacy_mission_builder_preserved() -> None:
    mission = (
        MissionBuilder("goal-123", "Legacy mission")
        .step("collect", "collector")
        .step("summarize", "summarizer", depends_on=["missing-id"])
        .build()
    )

    assert mission.status == MissionStatus.QUEUED
    assert mission.description == "Legacy mission"
    assert mission.steps[0].status == StepStatus.PENDING
    assert mission.next_ready_step() == mission.steps[0]
