"""
core/introspection/health.py

System and tool health monitoring.

Provides:
- HealthCheck: run a single check and capture its result
- HealthReport: aggregate of many checks
- Helpers for common checks (context integrity, scheduler backlog, etc.)

Usage:
    report = run_health_check(context=ctx, scheduler=sched, goal_manager=gm)
    print(report.summary())
    if not report.is_healthy:
        alert_operator(report)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class HealthStatus(str, Enum):
    OK      = "ok"
    WARN    = "warn"     # degraded but operational
    FAIL    = "fail"     # subsystem is broken or dangerous
    UNKNOWN = "unknown"  # check could not run


@dataclass
class HealthCheckResult:
    name: str
    status: HealthStatus
    message: str
    details: dict = field(default_factory=dict)
    checked_at: datetime = field(default_factory=_utcnow)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "checked_at": self.checked_at.isoformat(),
        }


@dataclass
class HealthReport:
    checks: list[HealthCheckResult] = field(default_factory=list)
    generated_at: datetime = field(default_factory=_utcnow)

    def add(self, result: HealthCheckResult) -> None:
        self.checks.append(result)

    @property
    def is_healthy(self) -> bool:
        return all(c.status == HealthStatus.OK for c in self.checks)

    @property
    def has_failures(self) -> bool:
        return any(c.status == HealthStatus.FAIL for c in self.checks)

    @property
    def has_warnings(self) -> bool:
        return any(c.status == HealthStatus.WARN for c in self.checks)

    def summary(self) -> str:
        total = len(self.checks)
        ok    = sum(1 for c in self.checks if c.status == HealthStatus.OK)
        warn  = sum(1 for c in self.checks if c.status == HealthStatus.WARN)
        fail  = sum(1 for c in self.checks if c.status == HealthStatus.FAIL)
        overall = "✅ HEALTHY" if self.is_healthy else ("⚠️ DEGRADED" if not self.has_failures else "❌ FAILED")
        lines = [f"Health Report — {overall} ({ok}/{total} OK, {warn} warn, {fail} fail)"]
        for c in self.checks:
            icon = {"ok": "✅", "warn": "⚠️", "fail": "❌", "unknown": "❓"}.get(c.status.value, "?")
            lines.append(f"  {icon} {c.name}: {c.message}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at.isoformat(),
            "is_healthy": self.is_healthy,
            "checks": [c.to_dict() for c in self.checks],
        }


# ── Built-in check functions ──────────────────────────────────────────────────

def check_context(context: Any) -> HealthCheckResult:
    if context is None:
        return HealthCheckResult("context", HealthStatus.FAIL, "AgentContext is None")
    if context.interrupt_flag:
        return HealthCheckResult("context", HealthStatus.WARN, "Interrupt flag is raised",
                                  details=context.snapshot())
    if context.risk_score > 0.85:
        return HealthCheckResult("context", HealthStatus.FAIL,
                                  f"Risk score {context.risk_score:.2f} is critical",
                                  details=context.snapshot())
    if context.confidence_score < 0.3:
        return HealthCheckResult("context", HealthStatus.WARN,
                                  f"Confidence {context.confidence_score:.2f} is low",
                                  details=context.snapshot())
    return HealthCheckResult("context", HealthStatus.OK,
                              f"conf={context.confidence_score:.2f} risk={context.risk_score:.2f}")


def check_scheduler(scheduler: Any, warn_backlog: int = 10) -> HealthCheckResult:
    if scheduler is None:
        return HealthCheckResult("scheduler", HealthStatus.UNKNOWN, "Scheduler not provided")
    pending = scheduler.pending()
    due     = scheduler.due()
    if len(pending) > warn_backlog:
        return HealthCheckResult("scheduler", HealthStatus.WARN,
                                  f"{len(pending)} pending entries (>{warn_backlog} threshold)",
                                  details={"pending": len(pending), "due": len(due)})
    return HealthCheckResult("scheduler", HealthStatus.OK,
                              f"{len(pending)} pending, {len(due)} due",
                              details={"pending": len(pending), "due": len(due)})


def check_goal_manager(goal_manager: Any) -> HealthCheckResult:
    if goal_manager is None:
        return HealthCheckResult("goal_manager", HealthStatus.UNKNOWN, "GoalManager not provided")
    active = goal_manager.active_goals()
    next_g = goal_manager.next_goal()
    return HealthCheckResult(
        "goal_manager",
        HealthStatus.OK,
        f"{len(active)} active goal(s), next pending: {next_g.description[:40] if next_g else 'none'}",
        details={"active_count": len(active)},
    )


def check_override_protocol(override_protocol: Any) -> HealthCheckResult:
    if override_protocol is None:
        return HealthCheckResult("override_protocol", HealthStatus.UNKNOWN, "Not provided")
    if override_protocol.is_stopped():
        return HealthCheckResult("override_protocol", HealthStatus.FAIL,
                                  "Emergency stop is ACTIVE — agent is halted")
    return HealthCheckResult("override_protocol", HealthStatus.OK, "No active emergency stop")


# ── Aggregate runner ──────────────────────────────────────────────────────────

def run_health_check(
    context: Optional[Any] = None,
    scheduler: Optional[Any] = None,
    goal_manager: Optional[Any] = None,
    override_protocol: Optional[Any] = None,
    custom_checks: Optional[list[Callable[[], HealthCheckResult]]] = None,
) -> HealthReport:
    """Run all available health checks and return a HealthReport."""
    report = HealthReport()

    if context is not None:
        report.add(check_context(context))
    if scheduler is not None:
        report.add(check_scheduler(scheduler))
    if goal_manager is not None:
        report.add(check_goal_manager(goal_manager))
    if override_protocol is not None:
        report.add(check_override_protocol(override_protocol))

    if custom_checks:
        for fn in custom_checks:
            try:
                report.add(fn())
            except Exception as exc:
                report.add(HealthCheckResult("custom_check", HealthStatus.FAIL, str(exc)))

    return report
