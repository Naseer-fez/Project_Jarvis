"""Startup and lightweight runtime health checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from urllib.request import urlopen


class HealthStatus(str, Enum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True)
class HealthCheck:
    name: str
    status: HealthStatus
    message: str


@dataclass
class HealthReport:
    checks: list[HealthCheck] = field(default_factory=list)

    @property
    def has_failures(self) -> bool:
        return any(check.status == HealthStatus.FAIL for check in self.checks)

    @property
    def is_healthy(self) -> bool:
        return not self.has_failures

    @property
    def ollama_reachable(self) -> bool:
        check = next((item for item in self.checks if item.name == "ollama"), None)
        return bool(check and check.status == HealthStatus.OK)

    def summary(self) -> str:
        return "\n".join(
            f"{check.name}: {check.status.value.upper()} - {check.message}"
            for check in self.checks
        )


def run_startup_health_check(controller, verbose: bool = False) -> HealthReport:
    del verbose
    checks: list[HealthCheck] = []

    db_path = Path(getattr(getattr(controller, "memory", None), "db_path", ""))
    exists = db_path.exists()
    checks.append(
        HealthCheck(
            name="memory_sqlite",
            status=HealthStatus.OK if exists else HealthStatus.FAIL,
            message=str(exists),
        )
    )

    base_url = getattr(getattr(controller, "llm", None), "base_url", "http://localhost:11434")
    ollama_ok = False
    try:
        urlopen(f"{base_url}/api/tags")
        ollama_ok = True
    except Exception:
        ollama_ok = False
    checks.append(
        HealthCheck(
            name="ollama",
            status=HealthStatus.OK if ollama_ok else HealthStatus.WARN,
            message=str(ollama_ok),
        )
    )

    return HealthReport(checks=checks)


def run_lightweight_health_check(config) -> HealthReport:
    checks: list[HealthCheck] = []
    sqlite_path = ""
    try:
        sqlite_path = config.get("memory", "sqlite_file", fallback="")
    except Exception:
        sqlite_path = ""

    checks.append(
        HealthCheck(
            name="config_sqlite_path",
            status=HealthStatus.OK if bool(sqlite_path) else HealthStatus.WARN,
            message=str(bool(sqlite_path)),
        )
    )

    base_url = "http://localhost:11434"
    try:
        base_url = config.get("ollama", "base_url", fallback=base_url)
    except Exception:
        pass

    reachable = False
    try:
        urlopen(f"{base_url}/api/tags")
        reachable = True
    except Exception:
        reachable = False
    checks.append(
        HealthCheck(
            name="ollama",
            status=HealthStatus.OK if reachable else HealthStatus.WARN,
            message=str(reachable),
        )
    )

    return HealthReport(checks=checks)
