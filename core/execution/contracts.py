"""
core/execution/contracts.py

Execution Contracts — the shared type language between planner, dispatcher,
and tools.  Every crossing of a module boundary uses one of these types.

Rules:
- Planner  MUST return a PlannerOutput.
- Dispatcher MUST accept a PlannerOutput and return a DispatchResult.
- Tools    MUST return a ToolResult.
- Failures MUST be expressed as ToolResult(success=False, ...).

No silent None returns.  No magic dict shapes.  No "trust me" conventions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── Enumerations ──────────────────────────────────────────────────────────────

class FailureReason(str, Enum):
    UNKNOWN            = "unknown"
    TOOL_NOT_FOUND     = "tool_not_found"
    INVALID_PARAMS     = "invalid_params"
    PERMISSION_DENIED  = "permission_denied"
    TIMEOUT            = "timeout"
    RATE_LIMITED       = "rate_limited"
    NETWORK_ERROR      = "network_error"
    TOOL_ERROR         = "tool_error"
    POLICY_BLOCKED     = "policy_blocked"
    DEPENDENCY_FAILED  = "dependency_failed"
    CANCELLED          = "cancelled"


# ── Tool contracts ────────────────────────────────────────────────────────────

@dataclass
class ToolResult:
    """
    Every tool MUST return this type.

    success=True  → result contains the tool output.
    success=False → error_reason and error_message describe what went wrong.
    """

    tool_name: str
    success: bool
    result: Any = None
    error_reason: FailureReason = FailureReason.UNKNOWN
    error_message: str = ""

    # Metadata
    duration_ms: Optional[float] = None
    retryable: bool = False            # hint to dispatcher/scheduler
    produced_at: datetime = field(default_factory=_utcnow)

    def to_dict(self) -> dict:
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "result": self.result,
            "error_reason": self.error_reason.value if not self.success else None,
            "error_message": self.error_message if not self.success else None,
            "duration_ms": self.duration_ms,
            "retryable": self.retryable,
            "produced_at": self.produced_at.isoformat(),
        }

    @staticmethod
    def ok(tool_name: str, result: Any, duration_ms: Optional[float] = None) -> "ToolResult":
        return ToolResult(tool_name=tool_name, success=True, result=result, duration_ms=duration_ms)

    @staticmethod
    def fail(
        tool_name: str,
        reason: FailureReason,
        message: str,
        retryable: bool = False,
        duration_ms: Optional[float] = None,
    ) -> "ToolResult":
        return ToolResult(
            tool_name=tool_name,
            success=False,
            error_reason=reason,
            error_message=message,
            retryable=retryable,
            duration_ms=duration_ms,
        )


# ── Planner contracts ─────────────────────────────────────────────────────────

@dataclass
class PlannedStep:
    """A single step as emitted by the planner."""

    step_name: str
    tool: str
    params: dict = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)  # other step_names
    description: str = ""
    estimated_risk: float = 0.0           # 0.0 – 1.0 planner hint


@dataclass
class PlannerOutput:
    """
    What the planner MUST return.

    The dispatcher converts each PlannedStep into a Mission Step.
    """

    goal_id: str
    plan_id: str
    description: str
    steps: list[PlannedStep]
    estimated_confidence: float = 1.0    # planner's confidence in its plan
    estimated_duration_seconds: Optional[float] = None
    notes: str = ""
    created_at: datetime = field(default_factory=_utcnow)

    def to_dict(self) -> dict:
        return {
            "goal_id": self.goal_id,
            "plan_id": self.plan_id,
            "description": self.description,
            "steps": [
                {
                    "step_name": s.step_name,
                    "tool": s.tool,
                    "params": s.params,
                    "depends_on": s.depends_on,
                    "description": s.description,
                    "estimated_risk": s.estimated_risk,
                }
                for s in self.steps
            ],
            "estimated_confidence": self.estimated_confidence,
            "estimated_duration_seconds": self.estimated_duration_seconds,
            "notes": self.notes,
            "created_at": self.created_at.isoformat(),
        }

    def validate(self) -> list[str]:
        """
        Return a list of validation error strings.
        Empty list = valid.
        """
        errors: list[str] = []
        if not self.goal_id:
            errors.append("goal_id is required")
        if not self.plan_id:
            errors.append("plan_id is required")
        if not self.steps:
            errors.append("at least one step is required")
        known_names = {s.step_name for s in self.steps}
        for step in self.steps:
            for dep in step.depends_on:
                if dep not in known_names:
                    errors.append(f"step '{step.step_name}' depends on unknown step '{dep}'")
            if not step.tool:
                errors.append(f"step '{step.step_name}' has no tool specified")
        return errors


# ── Dispatcher contracts ──────────────────────────────────────────────────────

class DispatchStatus(str, Enum):
    COMPLETED = "completed"
    PARTIAL   = "partial"     # some steps succeeded
    FAILED    = "failed"
    ABORTED   = "aborted"


@dataclass
class DispatchResult:
    """
    What the dispatcher MUST return after executing a PlannerOutput.
    """

    mission_id: str
    goal_id: str
    status: DispatchStatus
    step_results: list[ToolResult] = field(default_factory=list)

    started_at: datetime = field(default_factory=_utcnow)
    finished_at: Optional[datetime] = None

    error_summary: str = ""   # human-readable if status != COMPLETED

    def to_dict(self) -> dict:
        return {
            "mission_id": self.mission_id,
            "goal_id": self.goal_id,
            "status": self.status.value,
            "step_results": [r.to_dict() for r in self.step_results],
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "error_summary": self.error_summary,
        }

    @property
    def succeeded(self) -> bool:
        return self.status == DispatchStatus.COMPLETED

    @property
    def failed_steps(self) -> list[ToolResult]:
        return [r for r in self.step_results if not r.success]

    @property
    def all_retryable(self) -> bool:
        """True if every failure is marked retryable."""
        failures = self.failed_steps
        return bool(failures) and all(r.retryable for r in failures)
