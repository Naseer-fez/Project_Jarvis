"""Startup and lightweight runtime health checks."""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from urllib.request import urlopen

from core.ops.production import is_production, validate_production_config
from core.runtime.paths import _resolve_path


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


def _config_get(config, section: str, option: str, fallback: str = "") -> str:
    try:
        return str(config.get(section, option, fallback=fallback))
    except Exception:
        return fallback


def _config_get_bool(
    config,
    section: str,
    option: str,
    fallback: bool = False,
) -> bool:
    try:
        return bool(config.getboolean(section, option, fallback=fallback))
    except Exception:
        return fallback


def _module_available(import_name: str) -> bool:
    return importlib.util.find_spec(import_name) is not None


def _path_ready(path: Path, *, expect_file: bool) -> tuple[HealthStatus, str]:
    if not str(path):
        return HealthStatus.FAIL, "not configured"
    resolved_path = _resolve_path(path)
    target = resolved_path.parent if expect_file else resolved_path
    try:
        exists = target.exists()
        writable = os.access(target if exists else target.parent, os.W_OK)
    except OSError as exc:
        return HealthStatus.FAIL, f"{resolved_path} ({exc})"

    if exists and writable:
        return HealthStatus.OK, str(resolved_path)
    if exists:
        return HealthStatus.WARN, f"{resolved_path} (exists but may not be writable)"
    return HealthStatus.WARN, f"{resolved_path} (path will be created at runtime)"


def _collect_config_checks(config) -> list[HealthCheck]:
    checks: list[HealthCheck] = []

    environment = _config_get(
        config,
        "general",
        "environment",
        os.environ.get("JARVIS_ENV", "development"),
    )
    checks.append(
        HealthCheck(
            name="runtime_environment",
            status=HealthStatus.OK,
            message=environment or "development",
        )
    )

    sqlite_path = _config_get(config, "memory", "sqlite_file", "")
    if sqlite_path:
        sqlite_status, sqlite_message = _path_ready(Path(sqlite_path), expect_file=True)
    else:
        sqlite_status, sqlite_message = HealthStatus.FAIL, "memory.sqlite_file is not configured"
    checks.append(
        HealthCheck(
            name="memory_sqlite_config",
            status=sqlite_status,
            message=sqlite_message,
        )
    )

    app_log_path = _config_get(config, "logging", "app_file", "")
    if app_log_path:
        app_status, app_message = _path_ready(Path(app_log_path), expect_file=True)
    else:
        app_status, app_message = HealthStatus.WARN, "logging.app_file is not configured"
    checks.append(
        HealthCheck(
            name="logging_app_file",
            status=app_status,
            message=app_message,
        )
    )

    audit_path = _config_get(config, "logging", "audit_file", "")
    if audit_path:
        audit_status, audit_message = _path_ready(Path(audit_path), expect_file=True)
    else:
        audit_status, audit_message = HealthStatus.WARN, "logging.audit_file is not configured"
    checks.append(
        HealthCheck(
            name="logging_audit_file",
            status=audit_status,
            message=audit_message,
        )
    )

    raw_safe_dirs = _config_get(config, "execution", "safe_directories", "")
    safe_dirs = [Path(item.strip()) for item in raw_safe_dirs.split(",") if item.strip()]
    if not safe_dirs:
        checks.append(
            HealthCheck(
                name="execution_safe_directories",
                status=HealthStatus.WARN,
                message="No execution.safe_directories configured",
            )
        )
    else:
        ready = []
        pending = []
        for path in safe_dirs:
            status, _ = _path_ready(path, expect_file=False)
            if status == HealthStatus.OK:
                ready.append(str(path))
            else:
                pending.append(str(path))
        if pending:
            message = f"ready={len(ready)} pending={len(pending)}"
            status = HealthStatus.WARN
        else:
            message = ", ".join(str(path) for path in safe_dirs)
            status = HealthStatus.OK
        checks.append(
            HealthCheck(
                name="execution_safe_directories",
                status=status,
                message=message,
            )
        )

    if _config_get_bool(config, "automation", "enabled", True):
        drop_root = _config_get(config, "automation", "drop_root", "workspace/jarvis_dropbox")
        drop_status, drop_message = _path_ready(Path(drop_root), expect_file=False)
        checks.append(
            HealthCheck(
                name="automation_drop_root",
                status=drop_status,
                message=drop_message,
            )
        )

        watch_recordings = _config_get_bool(config, "automation", "watch_recordings", True)
        if watch_recordings:
            has_cv2 = _module_available("cv2")
            checks.append(
                HealthCheck(
                    name="automation_video_ocr_dependency",
                    status=HealthStatus.OK if has_cv2 else HealthStatus.WARN,
                    message="opencv available" if has_cv2 else "opencv (cv2) not installed; video OCR will be limited",
                )
            )

        uses_ocr = _config_get_bool(config, "automation", "watch_screenshots", True) or _config_get_bool(
            config,
            "automation",
            "live_screen_enabled",
            True,
        )
        if uses_ocr:
            def _check_tesseract() -> tuple[HealthStatus, str]:
                if not _module_available("pytesseract"):
                    return HealthStatus.WARN, "pytesseract not installed; screenshot OCR will be limited"

                import sys
                import subprocess

                # Configure Tesseract path dynamically
                if getattr(sys, "frozen", False):
                    base_dir = getattr(sys, "_MEIPASS", "")
                    tesseract_dir = os.path.join(base_dir, "bin", "tesseract")
                    cmd = os.path.join(tesseract_dir, "tesseract.exe")
                else:
                    cmd = os.environ.get("TESSERACT_CMD")
                    if not cmd:
                        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                        local_bundled = os.path.join(project_root, "bin", "tesseract", "tesseract.exe")
                        if os.path.exists(local_bundled):
                            cmd = local_bundled
                        else:
                            cmd = "tesseract"

                try:
                    res = subprocess.run([cmd, "--version"], capture_output=True, text=True, timeout=5)
                    if res.returncode == 0:
                        version_line = res.stdout.splitlines()[0] if res.stdout else "unknown version"
                        return HealthStatus.OK, f"pytesseract available, tesseract executable responsive: {version_line}"
                    else:
                        return HealthStatus.FAIL, f"tesseract --version failed with return code {res.returncode}"
                except Exception as exc:
                    return HealthStatus.FAIL, f"tesseract executable not found or not executable at '{cmd}': {exc}"

            tess_status, tess_msg = _check_tesseract()
            checks.append(
                HealthCheck(
                    name="automation_ocr_dependency",
                    status=tess_status,
                    message=tess_msg,
                )
            )

    control_file = _config_get(config, "dashboard", "control_file", "")
    if control_file:
        control_status, control_message = _path_ready(Path(control_file), expect_file=True)
        checks.append(
            HealthCheck(
                name="dashboard_control_file",
                status=control_status,
                message=control_message,
            )
        )

    if _config_get_bool(config, "voice", "enabled", False):
        required_voice_modules = [
            "sounddevice",
            "speech_recognition",
            "pvporcupine",
            "pvrecorder",
        ]
        missing = [
            module_name for module_name in required_voice_modules if not _module_available(module_name)
        ]
        checks.append(
            HealthCheck(
                name="voice_dependencies",
                status=HealthStatus.FAIL if missing else HealthStatus.OK,
                message="missing: " + ", ".join(missing) if missing else "voice dependencies available",
            )
        )

    if _config_get_bool(config, "execution", "allow_gui_automation", False):
        required_desktop_modules = [
            "pyautogui",
            "pygetwindow",
            "pytesseract",
            "PIL",
        ]
        missing = [
            module_name for module_name in required_desktop_modules if not _module_available(module_name)
        ]
        checks.append(
            HealthCheck(
                name="desktop_dependencies",
                status=HealthStatus.FAIL if missing else HealthStatus.OK,
                message="missing: " + ", ".join(missing) if missing else "desktop dependencies available",
            )
        )

    if _config_get_bool(config, "execution", "allow_web_search", True) and _config_get_bool(
        config,
        "web_search",
        "enabled",
        True,
    ):
        ready_providers: list[str] = []
        issues: list[str] = []
        tavily_api_key = (
            os.environ.get("TAVILY_API_KEY")
            or _config_get(config, "web_search", "tavily_api_key", "")
        ).strip()

        if _module_available("ddgs"):
            ready_providers.append("ddgs")
        else:
            issues.append("ddgs package missing")

        if tavily_api_key:
            if _module_available("requests"):
                ready_providers.append("tavily")
            else:
                issues.append("requests package missing for Tavily")

        if ready_providers:
            checks.append(
                HealthCheck(
                    name="web_search_dependencies",
                    status=HealthStatus.OK,
                    message="ready providers: " + ", ".join(ready_providers),
                )
            )
        else:
            checks.append(
                HealthCheck(
                    name="web_search_dependencies",
                    status=HealthStatus.WARN,
                    message="no web-search provider ready" + (
                        f" ({'; '.join(issues)})" if issues else ""
                    ),
                )
            )

    production_check = validate_production_config(config)
    if production_check.errors:
        checks.append(
            HealthCheck(
                name="production_guardrails",
                status=HealthStatus.FAIL,
                message="; ".join(production_check.errors),
            )
        )
    elif production_check.warnings:
        checks.append(
            HealthCheck(
                name="production_guardrails",
                status=HealthStatus.WARN,
                message="; ".join(production_check.warnings),
            )
        )
    else:
        message = (
            "production guardrails passed"
            if is_production(config)
            else "not running in production mode"
        )
        checks.append(
            HealthCheck(
                name="production_guardrails",
                status=HealthStatus.OK,
                message=message,
            )
        )

    return checks


def _ollama_check(base_url: str) -> HealthCheck:
    reachable = False
    try:
        with urlopen(f"{base_url}/api/tags", timeout=5):
            reachable = True
    except Exception:
        reachable = False
    return HealthCheck(
        name="ollama",
        status=HealthStatus.OK if reachable else HealthStatus.WARN,
        message=f"{base_url} reachable={reachable}",
    )


def run_startup_health_check(controller, verbose: bool = False) -> HealthReport:
    del verbose
    checks: list[HealthCheck] = []

    config = getattr(controller, "config", None)
    if config is not None:
        checks.extend(_collect_config_checks(config))

    raw_db_path = getattr(getattr(controller, "memory", None), "db_path", "")
    db_path = _resolve_path(raw_db_path) if str(raw_db_path) else None
    exists = bool(db_path and db_path.exists())
    checks.append(
        HealthCheck(
            name="memory_sqlite",
            status=HealthStatus.OK if exists else HealthStatus.FAIL,
            message=str(db_path) if db_path is not None else "controller memory path missing",
        )
    )

    # Preflight Import dependency validation
    try:
        from core.runtime.import_validator import StartupValidator
        proj_root = Path(__file__).resolve().parents[2]
        validator = StartupValidator(proj_root)
        preflight = validator.run_preflight_checks()
        if preflight["status"] == "GREEN":
            checks.append(
                HealthCheck(
                    name="import_dependency_health",
                    status=HealthStatus.OK,
                    message="all critical submodules loaded successfully",
                )
            )
        else:
            failed_names = [f["module"] for f in preflight["failed"]]
            checks.append(
                HealthCheck(
                    name="import_dependency_health",
                    status=HealthStatus.FAIL,
                    message=f"failed modules: {', '.join(failed_names)}",
                )
            )
    except Exception as exc:
        checks.append(
            HealthCheck(
                name="import_dependency_health",
                status=HealthStatus.WARN,
                message=f"failed to run import preflight validation: {exc}",
            )
        )

    base_url = getattr(getattr(controller, "llm", None), "base_url", "http://localhost:11434")
    checks.append(_ollama_check(str(base_url)))
    return HealthReport(checks=checks)


def run_lightweight_health_check(config) -> HealthReport:
    checks = _collect_config_checks(config)

    # Lightweight import dependency validation
    try:
        from core.runtime.import_validator import StartupValidator
        proj_root = Path(__file__).resolve().parents[2]
        validator = StartupValidator(proj_root)
        preflight = validator.run_preflight_checks()
        if preflight["status"] == "GREEN":
            checks.append(
                HealthCheck(
                    name="import_dependency_health",
                    status=HealthStatus.OK,
                    message="all critical submodules loaded successfully",
                )
            )
        else:
            failed_names = [f["module"] for f in preflight["failed"]]
            checks.append(
                HealthCheck(
                    name="import_dependency_health",
                    status=HealthStatus.FAIL,
                    message=f"failed modules: {', '.join(failed_names)}",
                )
            )
    except Exception as exc:
        checks.append(
            HealthCheck(
                name="import_dependency_health",
                status=HealthStatus.WARN,
                message=f"failed to run import preflight validation: {exc}",
            )
        )

    base_url = _config_get(config, "ollama", "base_url", "http://localhost:11434")
    checks.append(_ollama_check(base_url))
    return HealthReport(checks=checks)
