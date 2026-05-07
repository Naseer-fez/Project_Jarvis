from __future__ import annotations

from configparser import ConfigParser
from pathlib import Path
import core.runtime.bootstrap as bootstrap_mod
from unittest.mock import patch

from core.controller.services import build_controller_services
from core.introspection.health import HealthStatus, run_lightweight_health_check
from core.logging import logger as logger_mod
from core.runtime.bootstrap import (
    _prepare_runtime_paths,
    _resolve_runtime_mode,
    _validate_startup_settings,
)


def test_prepare_runtime_paths_creates_safe_directories(tmp_path):
    cfg = ConfigParser()
    cfg["logging"] = {
        "app_file": str(tmp_path / "logs" / "app.log"),
        "audit_file": str(tmp_path / "logs" / "audit.jsonl"),
    }
    cfg["memory"] = {
        "sqlite_file": str(tmp_path / "data" / "jarvis_memory.db"),
        "chroma_dir": str(tmp_path / "data" / "chroma"),
    }
    cfg["execution"] = {
        "safe_directories": f"{tmp_path / 'workspace'},{tmp_path / 'outputs'}",
    }

    _prepare_runtime_paths(cfg)

    assert (tmp_path / "logs").exists()
    assert (tmp_path / "data").exists()
    assert (tmp_path / "data" / "chroma").exists()
    assert (tmp_path / "workspace").exists()
    assert (tmp_path / "outputs").exists()


def test_logger_setup_uses_configured_app_file(tmp_path):
    cfg = ConfigParser()
    app_file = tmp_path / "logs" / "app.log"
    cfg["logging"] = {
        "level": "INFO",
        "app_file": str(app_file),
        "audit_file": str(tmp_path / "logs" / "audit.jsonl"),
    }

    logger_mod.setup(cfg)
    log = logger_mod.get()
    log.info("phase1 app file logging works")
    for handler in log.handlers:
        flush = getattr(handler, "flush", None)
        if callable(flush):
            flush()

    assert app_file.exists()
    assert "phase1 app file logging works" in app_file.read_text(encoding="utf-8")


def test_logger_setup_resolves_relative_paths_from_project_root(tmp_path, monkeypatch):
    project_root = tmp_path / "project"
    project_root.mkdir()
    monkeypatch.setattr(bootstrap_mod, "PROJECT_ROOT", project_root)
    monkeypatch.chdir(tmp_path)

    cfg = ConfigParser()
    cfg["logging"] = {
        "level": "INFO",
        "app_file": "logs/app.log",
        "audit_file": "logs/audit.jsonl",
    }

    logger_mod.setup(cfg)
    log = logger_mod.get()
    log.info("relative app file logging works")
    logger_mod.audit("relative-path-check", {"ok": True})
    for handler in log.handlers:
        flush = getattr(handler, "flush", None)
        if callable(flush):
            flush()

    app_file = project_root / "logs" / "app.log"
    audit_file = project_root / "logs" / "audit.jsonl"
    assert app_file.exists()
    assert audit_file.exists()
    assert "relative app file logging works" in app_file.read_text(encoding="utf-8")
    assert not (tmp_path / "logs" / "app.log").exists()


def test_lightweight_health_fails_when_voice_enabled_but_deps_missing(tmp_path):
    cfg = ConfigParser()
    cfg["general"] = {"environment": "development"}
    cfg["memory"] = {"sqlite_file": str(tmp_path / "data" / "jarvis_memory.db")}
    cfg["logging"] = {
        "app_file": str(tmp_path / "logs" / "app.log"),
        "audit_file": str(tmp_path / "logs" / "audit.jsonl"),
    }
    cfg["voice"] = {"enabled": "true"}

    with patch("importlib.util.find_spec", return_value=None), patch(
        "urllib.request.urlopen",
        side_effect=OSError("offline"),
    ):
        report = run_lightweight_health_check(cfg)

    voice_check = next(check for check in report.checks if check.name == "voice_dependencies")
    assert voice_check.status == HealthStatus.FAIL
    assert "sounddevice" in voice_check.message


def test_lightweight_health_resolves_relative_paths_from_project_root(tmp_path, monkeypatch):
    project_root = tmp_path / "project"
    (project_root / "data").mkdir(parents=True)
    (project_root / "logs").mkdir(parents=True)
    monkeypatch.setattr(bootstrap_mod, "PROJECT_ROOT", project_root)
    monkeypatch.chdir(tmp_path)

    cfg = ConfigParser()
    cfg["general"] = {"environment": "development"}
    cfg["memory"] = {"sqlite_file": "data/jarvis_memory.db"}
    cfg["logging"] = {
        "app_file": "logs/app.log",
        "audit_file": "logs/audit.jsonl",
    }

    with patch("urllib.request.urlopen", side_effect=OSError("offline")):
        report = run_lightweight_health_check(cfg)

    checks = {check.name: check for check in report.checks}
    assert checks["memory_sqlite_config"].status == HealthStatus.OK
    assert checks["logging_app_file"].status == HealthStatus.OK
    assert checks["logging_audit_file"].status == HealthStatus.OK
    assert checks["memory_sqlite_config"].message == str(project_root / "data" / "jarvis_memory.db")


def test_build_controller_services_resolves_project_relative_paths(tmp_path, monkeypatch):
    project_root = tmp_path / "project"
    project_root.mkdir()
    monkeypatch.setattr(bootstrap_mod, "PROJECT_ROOT", project_root)

    cfg = ConfigParser()
    cfg["memory"] = {
        "sqlite_file": "data/jarvis_memory.db",
        "chroma_dir": "data/chroma",
        "goals_file": "data/goals.json",
    }
    cfg["models"] = {"chat_model": "mistral:7b"}
    cfg["ollama"] = {"base_url": "http://localhost:11434"}
    cfg["proactive"] = {"goal_check_interval_minutes": "5"}

    class DummyMemory:
        def __init__(self, db_path: str, chroma_path: str, model_name: str) -> None:
            self.db_path = db_path
            self.chroma_path = chroma_path
            self.model_name = model_name

        def set_llm(self, llm, *, enable_context_titles: bool = True) -> None:
            self.llm = llm
            self.enable_context_titles = enable_context_titles

    class DummyProfile:
        pass

    class DummyLLM:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.model_router = None

        def set_router(self, router) -> None:
            self.model_router = router

    class DummySynthesizer:
        def __init__(self, llm) -> None:
            self.llm = llm

    class DummyStateMachine:
        pass

    class DummyTaskPlanner:
        def __init__(self, config) -> None:
            self.config = config

    class DummyToolRouter:
        pass

    class DummyRiskEvaluator:
        def __init__(self, config) -> None:
            self.config = config

    class DummyGovernor:
        def __init__(self, level: int) -> None:
            self.level = level

    class DummyAgentLoop:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class DummyGoalManager:
        pass

    class DummyScheduler:
        pass

    class DummyNotifier:
        pass

    class DummyMonitor:
        def __init__(self, notifier, config) -> None:
            self.notifier = notifier
            self.config = config

    settings, _ = build_controller_services(
        cfg,
        memory_cls=DummyMemory,
        profile_cls=DummyProfile,
        llm_cls=DummyLLM,
        synthesizer_cls=DummySynthesizer,
        state_machine_cls=DummyStateMachine,
        task_planner_cls=DummyTaskPlanner,
        tool_router_cls=DummyToolRouter,
        risk_evaluator_cls=DummyRiskEvaluator,
        autonomy_governor_cls=DummyGovernor,
        agent_loop_cls=DummyAgentLoop,
        goal_manager_cls=DummyGoalManager,
        scheduler_cls=DummyScheduler,
        notifier_cls=DummyNotifier,
        monitor_cls=DummyMonitor,
        register_tools=lambda *args, **kwargs: None,
    )

    assert settings.db_path == str(project_root / "data" / "jarvis_memory.db")
    assert settings.chroma_path == str(project_root / "data" / "chroma")
    assert settings.goals_file == project_root / "data" / "goals.json"


def test_resolve_runtime_mode_prefers_headless_and_dashboard_combinations():
    assert _resolve_runtime_mode(
        voice_enabled=True,
        dashboard_enabled=True,
        headless=True,
    ) == "headless+dashboard"
    assert _resolve_runtime_mode(
        voice_enabled=True,
        dashboard_enabled=False,
        headless=False,
    ) == "voice"


def test_validate_startup_settings_rejects_invalid_dashboard_port():
    cfg = ConfigParser()
    cfg["dashboard"] = {"host": "127.0.0.1", "port": "70000"}
    cfg["execution"] = {"safe_directories": "workspace"}

    class Args:
        verify = False
        health_check = False
        dashboard_host = None
        dashboard_port = None

    result = _validate_startup_settings(
        cfg,
        Args(),
        voice_enabled=False,
        dashboard_enabled=True,
        headless=False,
        shutdown_timeout=15.0,
    )

    assert result.is_valid is False
    assert any("Dashboard port" in error for error in result.errors)


def test_validate_startup_settings_warns_when_headless_overrides_voice():
    cfg = ConfigParser()
    cfg["execution"] = {"safe_directories": "workspace"}

    class Args:
        verify = False
        health_check = False
        dashboard_host = None
        dashboard_port = None

    result = _validate_startup_settings(
        cfg,
        Args(),
        voice_enabled=True,
        dashboard_enabled=False,
        headless=True,
        shutdown_timeout=15.0,
    )

    assert result.is_valid is True
    assert any("Headless mode" in warning for warning in result.warnings)


def test_validate_startup_settings_rejects_missing_production_secrets(monkeypatch):
    monkeypatch.setenv("JARVIS_ENV", "production")
    monkeypatch.delenv("JARVIS_SECRET_KEY", raising=False)
    monkeypatch.delenv("JARVIS_ADMIN_USER", raising=False)
    monkeypatch.delenv("JARVIS_ADMIN_PASSWORD", raising=False)

    cfg = ConfigParser()
    cfg["general"] = {"environment": "production"}
    cfg["execution"] = {"safe_directories": "workspace"}

    class Args:
        verify = False
        health_check = False
        dashboard_host = None
        dashboard_port = None

    result = _validate_startup_settings(
        cfg,
        Args(),
        voice_enabled=False,
        dashboard_enabled=False,
        headless=False,
        shutdown_timeout=15.0,
    )

    assert result.is_valid is False
    assert any("JARVIS_SECRET_KEY" in error for error in result.errors)
    assert any("JARVIS_ADMIN_USER" in error for error in result.errors)


def test_lightweight_health_reports_production_guardrail_failures(monkeypatch, tmp_path):
    monkeypatch.setenv("JARVIS_ENV", "production")
    monkeypatch.delenv("JARVIS_SECRET_KEY", raising=False)
    monkeypatch.delenv("JARVIS_ADMIN_USER", raising=False)
    monkeypatch.delenv("JARVIS_ADMIN_PASSWORD", raising=False)

    cfg = ConfigParser()
    cfg["general"] = {"environment": "production"}
    cfg["memory"] = {"sqlite_file": str(tmp_path / "data" / "jarvis_memory.db")}
    cfg["logging"] = {
        "app_file": str(tmp_path / "logs" / "app.log"),
        "audit_file": str(tmp_path / "logs" / "audit.jsonl"),
    }

    with patch("urllib.request.urlopen", side_effect=OSError("offline")):
        report = run_lightweight_health_check(cfg)

    guardrail_check = next(
        check for check in report.checks if check.name == "production_guardrails"
    )
    assert guardrail_check.status == HealthStatus.FAIL
    assert "JARVIS_SECRET_KEY" in guardrail_check.message
