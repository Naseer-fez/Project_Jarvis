"""Workflow Chain Engine — DAG-based sequential tool executor.

Architecture:
  LLM → Planner → WorkflowEngine → RiskEvaluator → Tool (via IntegrationRegistry)

The engine accepts a list of WorkflowStep objects, evaluates risk per step,
executes via the IntegrationRegistry, logs outcomes to JSONL, and supports:
  - Sequential DAG execution (parallel-ready in future)
  - Per-step retry with configurable count
  - Per-step timeout
  - Failure rollback hooks
  - Partial completion tracking
  - Structured JSONL logging

Usage:
    engine = WorkflowEngine()
    result = await engine.execute(steps, registry=reg, risk_evaluator=evaluator)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

from core.logging.tool_execution_logger import ToolExecutionLogger

_logger = logging.getLogger(__name__)

RollbackHook = Callable[[str, dict[str, Any]], Awaitable[None]]


@dataclass
class WorkflowStep:
    """A single step in a workflow plan."""

    tool_name: str
    args: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)  # step IDs this step waits on
    retry_count: int = 1
    timeout: float = 30.0  # seconds
    step_id: str = ""  # unique within the workflow; auto-set to tool_name if empty

    def __post_init__(self) -> None:
        if not self.step_id:
            self.step_id = self.tool_name


@dataclass
class WorkflowResult:
    """Aggregated outcome of a workflow execution."""

    success: bool
    steps_completed: list[str] = field(default_factory=list)
    steps_failed: list[str] = field(default_factory=list)
    steps_skipped: list[str] = field(default_factory=list)
    outputs: dict[str, Any] = field(default_factory=dict)  # step_id → result data
    error: str = ""
    total_duration_ms: float = 0.0

    @property
    def partial(self) -> bool:
        return bool(self.steps_completed) and bool(self.steps_failed)


class WorkflowEngine:
    """
    Sequential DAG-based workflow executor.

    Future extension: swap _execute_sequential for _execute_parallel
    once dependency tracking is implemented.
    """

    def __init__(self, log_file: Any = None) -> None:
        self._exec_logger = ToolExecutionLogger(log_file) if log_file else ToolExecutionLogger()
        self._rollback_hooks: dict[str, RollbackHook] = {}

    def register_rollback(self, tool_name: str, hook: RollbackHook) -> None:
        """Register an async rollback callback for a specific tool."""
        self._rollback_hooks[tool_name] = hook

    async def execute(
        self,
        plan: list[WorkflowStep],
        *,
        registry: Any,
        risk_evaluator: Any,
    ) -> WorkflowResult:
        """
        Execute a workflow plan.

        Args:
            plan:           Ordered list of WorkflowStep objects.
            registry:       IntegrationRegistry instance for tool dispatch.
            risk_evaluator: RiskEvaluator instance for per-step risk gate.

        Returns:
            WorkflowResult with per-step outcomes.
        """
        overall_start = time.perf_counter()

        if not plan:
            return WorkflowResult(success=True, error="Empty plan — nothing to execute.")

        result = WorkflowResult(success=True)
        completed_ids: set[str] = set()

        for step in plan:
            step_id = step.step_id or step.tool_name

            # ── Dependency check ──────────────────────────────────────────────
            unmet = [dep for dep in step.depends_on if dep not in completed_ids]
            if unmet:
                msg = f"Step '{step_id}' skipped — unmet dependencies: {unmet}"
                _logger.warning(msg)
                result.steps_skipped.append(step_id)
                result.outputs[step_id] = {"skipped": True, "reason": msg}
                continue

            # ── Risk gate ─────────────────────────────────────────────────────
            risk_result = risk_evaluator.evaluate([step.tool_name])
            if risk_result.is_blocked:
                msg = f"Step '{step_id}' blocked by risk evaluator: {risk_result.summary()}"
                _logger.error(msg)
                result.steps_failed.append(step_id)
                result.outputs[step_id] = {"blocked": True, "reason": msg}
                result.success = False
                # Blocked steps abort the rest of the workflow
                break

            # ── Execute with retry ────────────────────────────────────────────
            step_success, step_output, step_error = await self._execute_step_with_retry(
                step=step,
                registry=registry,
            )

            if step_success:
                completed_ids.add(step_id)
                result.steps_completed.append(step_id)
                result.outputs[step_id] = step_output
                _logger.info("Workflow step '%s' completed successfully.", step_id)
            else:
                result.steps_failed.append(step_id)
                result.outputs[step_id] = {"error": step_error}
                result.success = False
                _logger.error("Workflow step '%s' failed: %s", step_id, step_error)

                # Run rollback hook if registered
                await self._run_rollback(step.tool_name, step.args)

                # Abort remaining steps on first failure (sequential behavior)
                # Mark remaining as skipped
                remaining_steps = plan[plan.index(step) + 1 :]
                for remaining in remaining_steps:
                    result.steps_skipped.append(remaining.step_id or remaining.tool_name)
                break

        result.total_duration_ms = (time.perf_counter() - overall_start) * 1000
        self._log_workflow_summary(result)
        return result

    async def _execute_step_with_retry(
        self,
        step: WorkflowStep,
        registry: Any,
    ) -> tuple[bool, Any, str]:
        """Attempt a step up to step.retry_count times. Returns (success, data, error)."""
        last_error = ""
        for attempt in range(max(1, step.retry_count)):
            step_start = time.perf_counter()
            try:
                coro = registry.execute(step.tool_name, step.args)
                task_result = await asyncio.wait_for(coro, timeout=step.timeout)
                duration_ms = (time.perf_counter() - step_start) * 1000

                if task_result.get("success"):
                    self._exec_logger.log(
                        tool_name=step.tool_name,
                        args=step.args,
                        success=True,
                        duration_ms=duration_ms,
                    )
                    return True, task_result.get("data"), ""

                last_error = task_result.get("error") or "Tool returned success=False"
                self._exec_logger.log(
                    tool_name=step.tool_name,
                    args=step.args,
                    success=False,
                    duration_ms=duration_ms,
                    error=last_error,
                )

            except asyncio.TimeoutError:
                last_error = f"Step timed out after {step.timeout}s"
                self._exec_logger.log(
                    tool_name=step.tool_name,
                    args=step.args,
                    success=False,
                    duration_ms=(time.perf_counter() - step_start) * 1000,
                    error=last_error,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                self._exec_logger.log(
                    tool_name=step.tool_name,
                    args=step.args,
                    success=False,
                    duration_ms=(time.perf_counter() - step_start) * 1000,
                    error=last_error,
                )

            if attempt + 1 < step.retry_count:
                _logger.debug(
                    "Retrying step '%s' (attempt %d/%d): %s",
                    step.step_id,
                    attempt + 1,
                    step.retry_count,
                    last_error,
                )
                await asyncio.sleep(0.5 * (attempt + 1))  # simple back-off

        return False, None, last_error

    async def _run_rollback(self, tool_name: str, args: dict[str, Any]) -> None:
        hook = self._rollback_hooks.get(tool_name)
        if hook:
            try:
                await hook(tool_name, args)
                _logger.info("Rollback hook executed for '%s'", tool_name)
            except Exception as exc:  # noqa: BLE001
                _logger.warning("Rollback hook failed for '%s': %s", tool_name, exc)

    def _log_workflow_summary(self, result: WorkflowResult) -> None:
        _logger.info(
            "Workflow finished | success=%s | completed=%d | failed=%d | skipped=%d | %.1fms",
            result.success,
            len(result.steps_completed),
            len(result.steps_failed),
            len(result.steps_skipped),
            result.total_duration_ms,
        )


def build_steps_from_plan(plan: list[dict[str, Any]]) -> list[WorkflowStep]:
    """Convert a planner-output list of dicts into WorkflowStep objects.

    Expected dict format::

        {
            "tool": "send_telegram",
            "args": {"message": "Hello"},
            "depends_on": [],           # optional
            "retry_count": 1,           # optional
            "timeout": 30.0             # optional
        }
    """
    steps: list[WorkflowStep] = []
    for item in plan:
        tool_name = str(item.get("tool") or item.get("tool_name") or "").strip()
        if not tool_name:
            continue
        steps.append(
            WorkflowStep(
                tool_name=tool_name,
                args=dict(item.get("args") or {}),
                depends_on=list(item.get("depends_on") or []),
                retry_count=int(item.get("retry_count") or 1),
                timeout=float(item.get("timeout") or 30.0),
            )
        )
    return steps


__all__ = ["WorkflowEngine", "WorkflowStep", "WorkflowResult", "build_steps_from_plan"]
