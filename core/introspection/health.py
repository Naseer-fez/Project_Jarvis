"""
core/introspection/health.py

System and tool health monitoring.

Provides:
- HealthCheck: run a single check and capture its result
- HealthReport: aggregate of many checks
- Helpers for common checks (context integrity, scheduler backlog, etc.)
- run_lightweight_health_check: fast pre-controller check (no model loading)

Usage:
    report = run_health_check(context=ctx, scheduler=sched, goal_manager=gm)
    print(report.summary())
    if not report.is_healthy:
        alert_operator(report)
"""

from __future__ import annotations

import configparser
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
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
    duration_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "checked_at": self.checked_at.isoformat(),
            "duration_ms": self.duration_ms,
        }


@dataclass
class HealthReport:
    checks: list[HealthCheckResult] = field(default_factory=list)
    generated_at: datetime = field(default_factory=_utcnow)

    def add(self, result: HealthCheckResult) -> None:
        self.checks.append(result)

    @property
    def is_healthy(self) -> bool:
        return not self.has_failures

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


def run_startup_health_check(controller=None) -> "HealthReport":
    """Run all startup checks, print status, and return a HealthReport."""
    checks: dict[str, Any] = {}

    import logging
    import time
    logger = logging.getLogger(__name__)
    durations: dict[str, float] = {}

    # 1) Ollama reachable
    t0 = time.perf_counter()
    try:
        import urllib.request
        import urllib.error
        import socket
        from core.config.defaults import OLLAMA_BASE_URL

        urllib.request.urlopen(OLLAMA_BASE_URL, timeout=3)
        checks["ollama_reachable"] = True
    except urllib.error.URLError as e:
        checks["ollama_reachable"] = False
        checks["ollama_error"] = str(e.reason)
    except socket.timeout:
        checks["ollama_reachable"] = False
        checks["ollama_error"] = "timeout"
    except Exception as e:
        checks["ollama_reachable"] = False
        checks["ollama_error"] = str(e)
    durations["ollama_reachable"] = (time.perf_counter() - t0) * 1000

    # 2) ChromaDB installed (optional)
    try:
        import chromadb  # noqa: F401

        checks["chromadb_ready"] = True
    except ImportError:
        checks["chromadb_ready"] = None
    except Exception:
        checks["chromadb_ready"] = False

    # 3) Memory SQLite accessible — use controller's own db_path when available
    try:
        if controller is not None and hasattr(controller, "memory") and hasattr(controller.memory, "db_path"):
            db = Path(controller.memory.db_path)
        else:
            raw = os.environ.get("JARVIS_MEMORY_DB", "data/jarvis_memory.db")
            db = Path(raw)
        checks["memory_sqlite"] = db.exists()
    except Exception:
        checks["memory_sqlite"] = False

    # 4) Voice dependencies installed (optional)
    try:
        import pvporcupine  # noqa: F401
        import sounddevice  # noqa: F401

        checks["voice_deps"] = True
    except ImportError:
        checks["voice_deps"] = None
    except Exception:
        checks["voice_deps"] = False

    # 5) Config file exists
    try:
        checks["config_loaded"] = Path("config/jarvis.ini").exists()
    except Exception:
        checks["config_loaded"] = False

    # 6) Integrations loaded (if controller available)
    if controller and hasattr(controller, "integration_loader"):
        try:
            result = getattr(controller, "_integration_result", {})
            loaded = result.get("loaded", []) if isinstance(result, dict) else []
            checks["integrations"] = f"{len(loaded)} loaded"
        except Exception:
            checks["integrations"] = "unknown"
    else:
        checks["integrations"] = "not wired"

    logger.info("═══ JARVIS STARTUP HEALTH ═══")
    for key, val in checks.items():
        if val is True or (isinstance(val, str) and val and key != "ollama_error"):
            logger.info("  [OK] %s: %s", key, val)
        elif val is None:
            logger.warning("  [WARN] %s: %s", key, val)
        elif key == "ollama_error":
            logger.error("  [ERROR] %s: %s", key, val)
        else:
            logger.error("  [FAIL] %s: %s", key, val)
    logger.info("══════════════════════════════")

    # Build HealthReport using existing schema and attach direct attrs for convenience.
    report = HealthReport()
    for key, val in checks.items():
        if val is True or (isinstance(val, str) and val):
            status = HealthStatus.OK
        elif val is None:
            status = HealthStatus.WARN
        else:
            status = HealthStatus.FAIL
        report.add(
            HealthCheckResult(
                name=key,
                status=status,
                message=str(val),
                details={"value": val},
                duration_ms=durations.get(key, 0.0),
            )
        )
        try:
            setattr(report, key, val)
        except Exception:
            pass

    return report


def run_lightweight_health_check(
    config: Optional[configparser.ConfigParser] = None,
) -> "HealthReport":
    """
    Fast, pre-controller health probe used by ``--health-check``.

    Deliberately does NOT import chromadb, sentence_transformers, or any
    controller code.  Safe to run before the controller is constructed.
    """
    checks: dict[str, Any] = {}

    # 1) Config file existence
    try:
        cfg_path_str = (
            config.get("general", "config_file", fallback="config/jarvis.ini")
            if config is not None
            else "config/jarvis.ini"
        )
        checks["config_loaded"] = Path(cfg_path_str).exists() or Path("config/jarvis.ini").exists()
    except Exception:
        checks["config_loaded"] = False

    # 2) Ollama reachable
    try:
        import urllib.request
        base = (
            config.get("ollama", "base_url", fallback="http://localhost:11434")
            if config is not None
            else "http://localhost:11434"
        )
        urllib.request.urlopen(base, timeout=3)
        checks["ollama_reachable"] = True
    except Exception:
        checks["ollama_reachable"] = False

    # 3) SQLite path exists (config-aware, no DB connection attempted)
    try:
        raw_db = (
            config.get(
                "memory",
                "sqlite_file",
                fallback=config.get("memory", "db_path", fallback="data/jarvis_memory.db"),
            )
            if config is not None
            else os.environ.get("JARVIS_MEMORY_DB", "data/jarvis_memory.db")
        )
        checks["memory_sqlite"] = Path(raw_db).exists()
    except Exception:
        checks["memory_sqlite"] = False

    report = HealthReport()
    for key, val in checks.items():
        if val is True or (isinstance(val, str) and val):
            status = HealthStatus.OK
        elif val is None:
            status = HealthStatus.WARN
        else:
            status = HealthStatus.FAIL
        report.add(
            HealthCheckResult(
                name=key,
                status=status,
                message=str(val),
                details={"value": val},
            )
        )
        try:
            setattr(report, key, val)
        except Exception:
            pass

    return report
