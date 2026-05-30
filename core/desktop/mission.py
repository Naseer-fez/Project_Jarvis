"""Planner-executor-recovery loop for bounded desktop missions."""

from __future__ import annotations

import inspect
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable

from core.desktop.actions import DesktopActionExecutor
from core.desktop.contracts import (
    ApprovalDecision,
    DesktopAction,
    DesktopActionResult,
    DesktopActionStatus,
    DesktopChange,
    DesktopObservation,
)
from core.desktop.observation import DesktopObserver


class DesktopMissionStatus(str, Enum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    NEEDS_USER = "needs_user"
    STOPPED = "stopped"


class RecoveryDecision(str, Enum):
    NONE = "none"
    RETRY = "retry"
    REOBSERVE = "reobserve"
    ASK_USER = "ask_user"
    STOP = "stop"


@dataclass
class MissionStepRecord:
    step_id: str
    action: dict[str, Any]
    observation_before: dict[str, Any] | None = None
    approval: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    observation_after: dict[str, Any] | None = None
    change: dict[str, Any] | None = None
    recovery_decision: str = RecoveryDecision.NONE.value
    attempts: int = 0
    status: str = "pending"
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "action": dict(self.action),
            "observation_before": self.observation_before,
            "approval": self.approval,
            "result": self.result,
            "observation_after": self.observation_after,
            "change": self.change,
            "recovery_decision": self.recovery_decision,
            "attempts": self.attempts,
            "status": self.status,
            "error": self.error,
        }


@dataclass
class MissionExecutionRecord:
    goal: str
    plan: list[dict[str, Any]]
    mission_id: str = field(default_factory=lambda: f"mission_{uuid.uuid4().hex[:12]}")
    status: DesktopMissionStatus = DesktopMissionStatus.RUNNING
    steps: list[MissionStepRecord] = field(default_factory=list)
    final_summary: str = ""
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def close(self, status: DesktopMissionStatus, summary: str) -> None:
        self.status = status
        self.final_summary = summary
        self.ended_at = time.time()

    @property
    def duration_seconds(self) -> float:
        return max(0.0, (self.ended_at or time.time()) - self.started_at)

    def explain(self) -> str:
        if self.final_summary:
            return self.final_summary
        succeeded = sum(1 for step in self.steps if step.status == "succeeded")
        failed = [step for step in self.steps if step.status not in {"succeeded", "pending"}]
        if self.status == DesktopMissionStatus.SUCCEEDED:
            return f"Completed '{self.goal}' with {succeeded} step(s)."
        if failed:
            first = failed[0]
            return f"Stopped '{self.goal}' at step {first.step_id}: {first.error or first.status}."
        return f"Mission '{self.goal}' stopped before completion."

    def to_dict(self) -> dict[str, Any]:
        return {
            "mission_id": self.mission_id,
            "goal": self.goal,
            "plan": list(self.plan),
            "status": self.status.value,
            "steps": [step.to_dict() for step in self.steps],
            "final_summary": self.final_summary,
            "duration_seconds": round(self.duration_seconds, 3),
            "metadata": dict(self.metadata),
        }


class DesktopMissionExecutor:
    """Observe, act, verify, recover, and explain desktop missions."""

    def __init__(
        self,
        *,
        action_executor: DesktopActionExecutor | None = None,
        observer: DesktopObserver | None = None,
        approval_callback: Callable[[DesktopAction, str], Any] | None = None,
        audit_writer: Callable[[str, dict[str, Any]], str] | None = None,
        max_retries: int = 1,
        min_confidence: float = 0.35,
    ) -> None:
        self.action_executor = action_executor or DesktopActionExecutor()
        self.observer = observer or DesktopObserver()
        self.approval_callback = approval_callback
        if audit_writer is None:
            from core.logging.logger import audit

            audit_writer = audit
        self.audit_writer = audit_writer
        self.max_retries = max(0, int(max_retries))
        self.min_confidence = max(0.0, min(1.0, float(min_confidence)))

    async def run(
        self,
        *,
        goal: str,
        actions: Iterable[DesktopAction],
        plan_summary: str = "",
    ) -> MissionExecutionRecord:
        action_list = list(actions)
        record = MissionExecutionRecord(
            goal=goal,
            plan=[action.to_dict() for action in action_list],
            metadata={"plan_summary": plan_summary} if plan_summary else {},
        )
        self._audit("desktop_mission_started", record.to_dict())

        for action in action_list:
            step = MissionStepRecord(step_id=action.action_id, action=action.to_dict())
            record.steps.append(step)

            before = await self._observe_with_recovery(step, "before")
            if before.confidence < self.min_confidence and self.action_executor.requires_approval(action):
                decision = await self._approval(action, "Low-confidence desktop observation before action.")
                step.approval = decision.to_dict()
                if not decision.approved:
                    step.status = "needs_user"
                    step.recovery_decision = RecoveryDecision.ASK_USER.value
                    step.error = decision.reason or "User approval is required before low-confidence desktop action."
                    record.close(DesktopMissionStatus.NEEDS_USER, self._summary_for(record))
                    self._audit("desktop_mission_finished", record.to_dict())
                    return record

            approval = await self._approval_if_required(action)
            step.approval = approval.to_dict()
            if approval.required and not approval.approved:
                step.status = "needs_user"
                step.recovery_decision = RecoveryDecision.ASK_USER.value
                step.error = approval.reason or "User approval is required."
                record.close(DesktopMissionStatus.NEEDS_USER, self._summary_for(record))
                self._audit("desktop_mission_finished", record.to_dict())
                return record

            success = await self._execute_with_verification(step, action, before, approval)
            self._audit("desktop_mission_step", step.to_dict())
            if not success:
                record.close(DesktopMissionStatus.FAILED, self._summary_for(record))
                self._audit("desktop_mission_finished", record.to_dict())
                return record

        record.close(DesktopMissionStatus.SUCCEEDED, self._summary_for(record))
        self._audit("desktop_mission_finished", record.to_dict())
        return record

    async def _observe_with_recovery(self, step: MissionStepRecord, label: str) -> DesktopObservation:
        observation = await self.observer.observe(label)
        if label == "before":
            step.observation_before = observation.to_dict()
        else:
            step.observation_after = observation.to_dict()

        if observation.confidence >= self.min_confidence:
            return observation

        step.recovery_decision = RecoveryDecision.REOBSERVE.value
        second = await self.observer.observe(f"{label}_retry")
        if label == "before":
            step.observation_before = second.to_dict()
        else:
            step.observation_after = second.to_dict()
        return second

    async def _execute_with_verification(
        self,
        step: MissionStepRecord,
        action: DesktopAction,
        before: DesktopObservation,
        approval: ApprovalDecision,
    ) -> bool:
        current_before = before
        last_result: DesktopActionResult | None = None
        last_change: DesktopChange | None = None

        for attempt in range(1, self.max_retries + 2):
            step.attempts = attempt
            result = await self.action_executor.execute(action, approved=approval.approved if approval.required else None)
            after = await self._observe_with_recovery(step, "after")
            change = self.observer.compare(current_before, after)
            step.result = result.to_dict()
            step.change = change.to_dict()
            last_result = result
            last_change = change

            if result.status in {DesktopActionStatus.BLOCKED, DesktopActionStatus.NEEDS_APPROVAL}:
                step.status = result.status.value
                step.recovery_decision = RecoveryDecision.STOP.value
                step.error = result.error
                return False

            expected_change_missing = bool(action.expected_change) and not change.changed
            if result.success and not expected_change_missing:
                step.status = "succeeded"
                if step.recovery_decision == RecoveryDecision.REOBSERVE.value:
                    step.recovery_decision = RecoveryDecision.NONE.value
                return True

            if attempt <= self.max_retries:
                step.recovery_decision = RecoveryDecision.RETRY.value
                current_before = after
                continue

            step.status = "failed"
            if result.success and expected_change_missing:
                step.error = f"No-op detected: expected change '{action.expected_change}' was not observed."
            else:
                step.error = result.error or "Desktop action failed."
            return False

        step.status = "failed"
        if last_result is not None:
            step.result = last_result.to_dict()
        if last_change is not None:
            step.change = last_change.to_dict()
        step.error = step.error or "Desktop action did not complete."
        return False

    async def _approval_if_required(self, action: DesktopAction) -> ApprovalDecision:
        if not self.action_executor.requires_approval(action):
            return ApprovalDecision(required=False, approved=True, reason="Approval not required.")
        return await self._approval(action, "Desktop action requires approval.")

    async def _approval(self, action: DesktopAction, reason: str) -> ApprovalDecision:
        if self.approval_callback is None:
            return ApprovalDecision(
                required=True,
                approved=False,
                reason=reason,
                mode="user_required",
            )
        try:
            result = self.approval_callback(action, reason)
            if inspect.isawaitable(result):
                result = await result
            return ApprovalDecision(
                required=True,
                approved=bool(result),
                reason=reason,
                mode="callback",
            )
        except Exception as exc:  # noqa: BLE001
            return ApprovalDecision(
                required=True,
                approved=False,
                reason=f"Approval callback failed: {exc}",
                mode="callback",
            )

    def _summary_for(self, record: MissionExecutionRecord) -> str:
        succeeded = sum(1 for step in record.steps if step.status == "succeeded")
        total = len(record.steps)
        failed = [step for step in record.steps if step.status not in {"succeeded", "pending"}]

        if not failed and total:
            return f"Completed '{record.goal}' with {succeeded}/{total} desktop step(s) verified."
        if failed:
            first = failed[0]
            return (
                f"Paused '{record.goal}' after {succeeded}/{total} desktop step(s). "
                f"Step {first.step_id} status: {first.status}. {first.error}".strip()
            )
        return f"No desktop steps were executed for '{record.goal}'."

    def _audit(self, event_type: str, payload: dict[str, Any]) -> None:
        try:
            self.audit_writer(event_type, payload)
        except Exception:
            return


__all__ = [
    "DesktopMissionExecutor",
    "DesktopMissionStatus",
    "MissionExecutionRecord",
    "MissionStepRecord",
    "RecoveryDecision",
]
