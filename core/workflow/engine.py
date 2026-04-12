"""Simple sequential workflow engine used by legacy tests."""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from core.autonomy.risk_evaluator import RiskLevel


@dataclass
class WorkflowStep:
    tool_name: str
    args: dict[str, Any] = field(default_factory=dict)
    step_id: str = ""
    retry_count: int = 1
    timeout: float = 30.0
    depends_on: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.step_id:
            self.step_id = self.tool_name


@dataclass
class WorkflowResult:
    success: bool
    partial: bool = False
    steps_completed: list[str] = field(default_factory=list)
    steps_failed: list[str] = field(default_factory=list)
    steps_skipped: list[str] = field(default_factory=list)


class WorkflowEngine:
    def __init__(self, log_file: str | Path | None = None) -> None:
        self.log_file = Path(log_file) if log_file else None
        self._rollback_handlers: dict[str, Callable[[str, dict[str, Any]], Awaitable[None] | None]] = {}

    def register_rollback(
        self,
        tool_name: str,
        handler: Callable[[str, dict[str, Any]], Awaitable[None] | None],
    ) -> None:
        self._rollback_handlers[tool_name] = handler

    async def execute(self, steps, *, registry, risk_evaluator) -> WorkflowResult:
        steps = list(steps or [])
        result = WorkflowResult(success=True)

        if not steps:
            return result

        completed_ids: set[str] = set()

        for index, step in enumerate(steps):
            if any(dep not in completed_ids for dep in step.depends_on):
                result.steps_skipped.append(step.step_id)
                await self._log_step(step, "skipped", "unmet_dependency")
                continue

            risk = risk_evaluator.evaluate([step.tool_name])
            if risk.level >= RiskLevel.CRITICAL:
                result.success = False
                result.steps_failed.append(step.tool_name)
                result.steps_skipped.extend(
                    later.step_id for later in steps[index + 1 :] if later.step_id not in result.steps_skipped
                )
                await self._log_step(step, "blocked", "risk")
                break

            payload = await self._call_registry(registry, step.tool_name, step.args)
            success = bool(payload.get("success", False))
            error = str(payload.get("error", "") or "")

            if success:
                result.steps_completed.append(step.tool_name)
                completed_ids.add(step.step_id)
                await self._log_step(step, "success", payload.get("data"))
                continue

            result.success = False
            result.steps_failed.append(step.tool_name)
            result.partial = bool(result.steps_completed)
            result.steps_skipped.extend(
                later.step_id for later in steps[index + 1 :] if later.step_id not in result.steps_skipped
            )
            await self._log_step(step, "failure", error)
            await self._run_rollback(step.tool_name, step.args)
            break

        if result.success:
            result.partial = False
        return result

    async def _call_registry(self, registry, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        payload = registry.execute(tool_name, args)
        if inspect.isawaitable(payload):
            payload = await payload
        return dict(payload or {})

    async def _run_rollback(self, tool_name: str, args: dict[str, Any]) -> None:
        handler = self._rollback_handlers.get(tool_name)
        if handler is None:
            return
        outcome = handler(tool_name, args)
        if inspect.isawaitable(outcome):
            await outcome

    async def _log_step(self, step: WorkflowStep, status: str, detail: Any) -> None:
        if self.log_file is None:
            return
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "step_id": step.step_id,
            "tool_name": step.tool_name,
            "status": status,
            "detail": detail,
        }
        with self.log_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")


def build_steps_from_plan(plan: list[dict[str, Any]]) -> list[WorkflowStep]:
    steps: list[WorkflowStep] = []
    for item in plan or []:
        if not isinstance(item, dict):
            continue
        tool_name = str(item.get("tool", "")).strip()
        if not tool_name:
            continue
        args = item.get("args", {})
        steps.append(
            WorkflowStep(
                tool_name=tool_name,
                args=args if isinstance(args, dict) else {},
                step_id=str(item.get("step_id") or tool_name),
                retry_count=int(item.get("retry_count", 1)),
                timeout=float(item.get("timeout", 30.0)),
                depends_on=list(item.get("depends_on", []) or []),
            )
        )
    return steps


__all__ = ["WorkflowEngine", "WorkflowResult", "WorkflowStep", "build_steps_from_plan"]
