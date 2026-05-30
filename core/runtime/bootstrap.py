from __future__ import annotations

import argparse
import asyncio
import configparser
import contextlib
import dataclasses
import faulthandler
import io
import json
import logging
import os
import signal
import sys
import threading
from pathlib import Path
from typing import Any

from core.ops.production import validate_production_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = "config/jarvis.ini"
DEFAULT_DASHBOARD_HOST = "127.0.0.1"
DEFAULT_DASHBOARD_PORT = 7070
DEFAULT_SHUTDOWN_TIMEOUT_S = 15.0


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=PROJECT_ROOT / ".env")
    except Exception:
        return


def _enable_fault_diagnostics() -> None:
    try:
        faulthandler.enable(all_threads=True)
    except Exception:
        return


def _configure_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                continue


_load_dotenv()
_enable_fault_diagnostics()
_configure_stdio()


def _uprint(msg: str, *, file=None) -> None:
    """Print safely even on Windows consoles with non-UTF encodings."""
    target = file or sys.stdout
    try:
        print(msg, file=target)
    except UnicodeEncodeError:
        raw = getattr(target, "buffer", None)
        if raw is not None:
            raw.write((msg + "\n").encode("utf-8", errors="replace"))
        else:
            fallback = msg.encode("ascii", errors="replace").decode("ascii")
            print(fallback, file=target)


_bootstrap = logging.getLogger("jarvis.bootstrap")
if not _bootstrap.handlers:
    _bootstrap.addHandler(logging.StreamHandler(sys.stderr))
_bootstrap_level = (
    logging.DEBUG
    if os.environ.get("JARVIS_LOG_LEVEL", "").upper() == "DEBUG"
    else logging.INFO
)
_bootstrap.setLevel(_bootstrap_level)
_bootstrap.propagate = False


class ExitCode:
    OK = 0
    GENERIC_ERROR = 1
    CONFIG_ERROR = 2
    AUDIT_FAILED = 3
    STARTUP_ERROR = 4


@dataclasses.dataclass
class StartupValidation:
    errors: list[str] = dataclasses.field(default_factory=list)
    warnings: list[str] = dataclasses.field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors


def _resolve_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _ensure_section(config: configparser.ConfigParser, section: str) -> None:
    if not config.has_section(section):
        config.add_section(section)


def load_config(config_path: str) -> configparser.ConfigParser:
    """
    Load INI config from an absolute path or relative to PROJECT_ROOT.
    Raises SystemExit(CONFIG_ERROR) if the file is missing in production.
    """
    from core.config import load_config as _load_config
    config = _load_config(config_path)
    _bootstrap.debug("Config loaded from %s", config_path)
    return config


def apply_cli_overrides(
    config: configparser.ConfigParser,
    args: argparse.Namespace,
) -> None:
    """Merge CLI arguments into config without clobbering unrelated keys."""
    if getattr(args, "log_level", None):
        _ensure_section(config, "logging")
        config["logging"]["level"] = str(args.log_level)

    if getattr(args, "session_name", None):
        _ensure_section(config, "general")
        config["general"]["session_name"] = str(args.session_name)

    if getattr(args, "voice", False):
        _ensure_section(config, "voice")
        config["voice"]["enabled"] = "true"

    dashboard_host = getattr(args, "dashboard_host", None)
    if dashboard_host:
        _ensure_section(config, "dashboard")
        config["dashboard"]["host"] = str(dashboard_host)

    dashboard_port = getattr(args, "dashboard_port", None)
    if dashboard_port is not None:
        _ensure_section(config, "dashboard")
        config["dashboard"]["port"] = str(int(dashboard_port))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    dashboard_port_default = os.environ.get("JARVIS_DASHBOARD_PORT")

    parser = argparse.ArgumentParser(
        description="Jarvis local runtime",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--voice", action="store_true", help="Start Jarvis in voice mode")
    parser.add_argument("--gui", action="store_true", help="Start the dashboard server")
    parser.add_argument("--dashboard", action="store_true", help="Alias for --gui")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Do not start CLI or voice loop; keep services alive until shutdown",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify audit log integrity and exit",
    )
    parser.add_argument(
        "--health-check",
        action="store_true",
        help="Run startup health diagnostics and exit",
    )
    parser.add_argument(
        "--strict-health",
        action="store_true",
        help="Fail startup if health diagnostics report failures",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print configured model routing and discovered availability",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the effective config after CLI overrides",
    )
    parser.add_argument(
        "--config",
        default=os.environ.get("JARVIS_CONFIG", DEFAULT_CONFIG_PATH),
        help="Config file path (also reads JARVIS_CONFIG)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=os.environ.get("JARVIS_LOG_LEVEL"),
        help="Override log level (also reads JARVIS_LOG_LEVEL)",
    )
    parser.add_argument("--session-name", help="Optional session name for this run")
    parser.add_argument(
        "--dashboard-host",
        default=os.environ.get("JARVIS_DASHBOARD_HOST"),
        help="Dashboard bind host",
    )
    parser.add_argument(
        "--dashboard-port",
        type=int,
        default=int(dashboard_port_default) if dashboard_port_default else None,
        help="Dashboard bind port",
    )
    parser.add_argument(
        "--shutdown-timeout",
        type=float,
        default=float(
            os.environ.get(
                "JARVIS_SHUTDOWN_TIMEOUT_S",
                str(DEFAULT_SHUTDOWN_TIMEOUT_S),
            )
        ),
        help="Graceful shutdown timeout in seconds",
    )
    parser.add_argument(
        "--replay",
        help="Path to an execution trace snapshot file (.json) to replay",
    )
    return parser.parse_args(argv)


class _ShutdownCoordinator:
    """Signal-aware shutdown gate."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._event = asyncio.Event()

    def request_shutdown(self, signame: str = "manual") -> None:
        _bootstrap.info("Shutdown requested via %s", signame)
        self._loop.call_soon_threadsafe(self._event.set)

    def install_signal_handlers(self) -> None:
        if sys.platform == "win32":
            for sig in (signal.SIGINT, signal.SIGBREAK):  # type: ignore[attr-defined]
                signal.signal(
                    sig,
                    lambda *_, _signame=sig.name: self.request_shutdown(_signame),
                )
            return

        for sig in (signal.SIGINT, signal.SIGTERM):
            self._loop.add_signal_handler(sig, self.request_shutdown, sig.name)

    async def wait(self) -> None:
        await self._event.wait()


def _install_process_exception_hooks(log: logging.Logger) -> None:
    def _sys_hook(exc_type, exc_value, exc_tb) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        log.critical(
            "Unhandled process exception",
            exc_info=(exc_type, exc_value, exc_tb),
        )

    sys.excepthook = _sys_hook

    if hasattr(threading, "excepthook"):
        def _thread_hook(args) -> None:
            if issubclass(args.exc_type, KeyboardInterrupt):
                return
            log.critical(
                "Unhandled thread exception in %s",
                getattr(args.thread, "name", "unknown"),
                exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
            )

        threading.excepthook = _thread_hook


def _install_loop_exception_handler(
    loop: asyncio.AbstractEventLoop,
    log: logging.Logger,
) -> None:
    def _handler(_loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
        exc = context.get("exception")
        message = context.get("message", "Unhandled event loop exception")
        if exc is None:
            log.error("%s | context=%s", message, context)
            return
        log.error("%s", message, exc_info=exc)

    loop.set_exception_handler(_handler)


def _prepare_runtime_environment(config: configparser.ConfigParser) -> None:
    environment = config.get(
        "general",
        "environment",
        fallback=os.environ.get("JARVIS_ENV", "development"),
    )
    os.environ.setdefault("JARVIS_ENV", str(environment))


def _prepare_runtime_paths(config: configparser.ConfigParser) -> None:
    entries = [
        ("logging", "log_dir", False),
        ("logging", "app_file", True),
        ("logging", "audit_file", True),
        ("logging", "trace_dir", False),
        ("memory", "data_dir", False),
        ("memory", "sqlite_file", True),
        ("memory", "db_path", True),
        ("memory", "chroma_dir", False),
        ("memory", "chroma_path", False),
        ("dashboard", "control_file", True),
        ("plugins", "manifest_directory", False),
        ("plugins", "marketplace_directory", False),
        ("ai_os", "workflow_catalog_dir", False),
        ("automation", "drop_root", False),
        ("automation", "commands_folder", False),
        ("automation", "rag_folder", False),
        ("automation", "processed_folder", False),
        ("automation", "failed_folder", False),
        ("automation", "screenshots_folder", False),
        ("automation", "recordings_folder", False),
        ("automation", "ingest_log_file", True),
        ("automation", "state_file", True),
    ]

    for section, key, is_file in entries:
        if not config.has_option(section, key):
            continue
        raw_value = config.get(section, key, fallback="").strip()
        if not raw_value:
            continue
        path = _resolve_path(raw_value)
        target = path.parent if is_file else path
        target.mkdir(parents=True, exist_ok=True)

    raw_safe_dirs = config.get("execution", "safe_directories", fallback="")
    for raw_dir in raw_safe_dirs.split(","):
        value = raw_dir.strip()
        if not value:
            continue
        _resolve_path(value).mkdir(parents=True, exist_ok=True)


def _resolve_voice_enabled(
    config: configparser.ConfigParser,
    args: argparse.Namespace,
) -> bool:
    cli_value = bool(getattr(args, "voice", False))
    try:
        config_value = config.getboolean("voice", "enabled", fallback=False)
    except (configparser.Error, ValueError):
        config_value = False
    return cli_value or config_value


def _resolve_dashboard_binding(
    config: configparser.ConfigParser,
    args: argparse.Namespace,
) -> tuple[str, int]:
    host = str(
        getattr(args, "dashboard_host", None)
        or config.get("dashboard", "host", fallback=DEFAULT_DASHBOARD_HOST)
        or DEFAULT_DASHBOARD_HOST
    )
    port_arg = getattr(args, "dashboard_port", None)
    if port_arg is not None:
        return host, int(port_arg)
    return host, config.getint("dashboard", "port", fallback=DEFAULT_DASHBOARD_PORT)


def _resolve_runtime_mode(
    *,
    voice_enabled: bool,
    dashboard_enabled: bool,
    headless: bool,
) -> str:
    if headless and dashboard_enabled:
        return "headless+dashboard"
    if headless:
        return "headless"
    if voice_enabled and dashboard_enabled:
        return "voice+dashboard"
    if voice_enabled:
        return "voice"
    if dashboard_enabled:
        return "cli+dashboard"
    return "cli"


def _validate_startup_settings(
    config: configparser.ConfigParser,
    args: argparse.Namespace,
    *,
    voice_enabled: bool,
    dashboard_enabled: bool,
    headless: bool,
    shutdown_timeout: float,
) -> StartupValidation:
    result = StartupValidation()

    if getattr(args, "verify", False) and getattr(args, "health_check", False):
        result.errors.append("Choose either --verify or --health-check, not both.")

    if shutdown_timeout <= 0:
        result.errors.append("--shutdown-timeout must be greater than 0 seconds.")

    if dashboard_enabled:
        host = str(
            getattr(args, "dashboard_host", None)
            or config.get("dashboard", "host", fallback=DEFAULT_DASHBOARD_HOST)
            or ""
        ).strip()
        if not host:
            result.errors.append("Dashboard host cannot be empty when dashboard mode is enabled.")

        port = getattr(args, "dashboard_port", None)
        if port is None:
            port = config.getint("dashboard", "port", fallback=DEFAULT_DASHBOARD_PORT)
        if not 1 <= int(port) <= 65535:
            result.errors.append("Dashboard port must be between 1 and 65535.")

    raw_safe_dirs = config.get("execution", "safe_directories", fallback="")
    safe_dirs = [item.strip() for item in raw_safe_dirs.split(",") if item.strip()]
    if not safe_dirs:
        result.warnings.append(
            "No execution.safe_directories are configured; file operations may be blocked more often."
        )

    if headless and voice_enabled:
        result.warnings.append(
            "Headless mode disables the interactive voice loop even if voice mode is enabled."
        )

    production_check = validate_production_config(
        config,
        dashboard_enabled=dashboard_enabled,
    )
    result.errors.extend(production_check.errors)
    result.warnings.extend(production_check.warnings)

    return result


def _redact_key(key: str, value: str) -> str:
    lowered = key.lower()
    if any(
        token in lowered
        for token in ("secret", "token", "password", "api_key", "access_key")
    ):
        return "***REDACTED***"
    return value


def _config_snapshot(config: configparser.ConfigParser) -> dict[str, dict[str, str]]:
    snapshot: dict[str, dict[str, str]] = {}
    for section in config.sections():
        snapshot[section] = {
            key: _redact_key(key, value)
            for key, value in config.items(section)
        }
    return snapshot


def _print_config_snapshot(
    config: configparser.ConfigParser,
    config_path: Path,
) -> None:
    payload = {
        "config_path": str(config_path),
        "project_root": str(PROJECT_ROOT),
        "sections": _config_snapshot(config),
    }
    _uprint(json.dumps(payload, indent=2, sort_keys=True))


def _build_model_inventory(
    config: configparser.ConfigParser,
) -> dict[str, dict[str, Any]]:
    from core.llm.model_router import ModelRouter

    router = ModelRouter(config=config)
    inventory: dict[str, dict[str, Any]] = {}

    for task_type in (
        "intent_classification",
        "memory_summarization",
        "tool_selection",
        "planning",
        "chat",
        "vision",
        "synthesis",
        "fallback",
    ):
        primary = router.route(task_type)
        entry: dict[str, Any] = {
            "primary": primary,
            "primary_available": router.is_available(primary),
        }
        try:
            entry["best_available"] = router.get_best_available(task_type)
        except Exception as exc:
            entry["best_available"] = None
            entry["error"] = str(exc)
        inventory[task_type] = entry

    inventory["discovered"] = router.list_available()
    return inventory


def _print_model_inventory(config: configparser.ConfigParser) -> None:
    inventory = _build_model_inventory(config)
    _uprint(json.dumps(inventory, indent=2, sort_keys=True))


def _should_exit_after_info(args: argparse.Namespace) -> bool:
    info_requested = bool(
        getattr(args, "print_config", False)
        or getattr(args, "list_models", False)
    )
    runtime_requested = bool(
        getattr(args, "voice", False)
        or getattr(args, "gui", False)
        or getattr(args, "dashboard", False)
        or getattr(args, "headless", False)
        or getattr(args, "verify", False)
        or getattr(args, "health_check", False)
    )
    return info_requested and not runtime_requested


def _safe_audit(
    logger_mod: Any,
    event_type: str,
    payload: dict[str, Any],
    log: logging.Logger,
) -> None:
    audit_fn = getattr(logger_mod, "audit", None)
    if not callable(audit_fn):
        return
    try:
        audit_fn(event_type, payload)
    except Exception:
        log.debug("Audit event '%s' failed", event_type, exc_info=True)


def _load_logger_module():
    logger_mod = sys.modules.get("core.logging.logger")
    if logger_mod is not None:
        return logger_mod

    from core.logging import logger as logger_mod

    return logger_mod


def _load_controller_class():
    controller_mod = sys.modules.get("core.controller_v2")
    if controller_mod is not None:
        return controller_mod.JarvisControllerV2

    from core.controller_v2 import JarvisControllerV2

    return JarvisControllerV2


def _load_integrations(
    controller: Any,
    config: configparser.ConfigParser,
    log: logging.Logger,
) -> dict[str, list[str]]:
    try:
        from integrations.loader import IntegrationLoader
        from integrations.registry import integration_registry

        loader = IntegrationLoader()
        result = loader.load_all(config=config, registry=integration_registry)

        # Dynamically register loaded integration safety rules & risk profiles
        gov = getattr(controller, "autonomy_governor", None)
        risk_eval = getattr(controller, "risk_evaluator", None)
        disp = getattr(controller, "dispatcher", None)
        integration_registry.register_safety_rules(gov, risk_eval, disp)

        setattr(controller, "integration_loader", loader)
        setattr(controller, "integration_registry", integration_registry)
        setattr(controller, "_integration_result", result)
        log.info(
            "Integrations loaded=%d skipped=%d",
            len(result.get("loaded", [])),
            len(result.get("skipped", [])),
        )
        return result
    except Exception:
        log.exception("Integration bootstrap failed")
        result = {"loaded": [], "skipped": ["bootstrap failed"]}
        setattr(controller, "_integration_result", result)
        return result


def _run_startup_health_check(controller: Any | None, *, verbose: bool) -> Any:
    from core.introspection.health import HealthReport, run_startup_health_check

    if verbose:
        return run_startup_health_check(controller)

    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        report = run_startup_health_check(controller)
    if report is None:
        return HealthReport()
    return report


async def _cancel_task(task: asyncio.Task[Any]) -> None:
    if task.done():
        with contextlib.suppress(Exception):
            task.exception()
        return

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await task


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_DASHBOARD_HOST",
    "DEFAULT_DASHBOARD_PORT",
    "DEFAULT_SHUTDOWN_TIMEOUT_S",
    "ExitCode",
    "PROJECT_ROOT",
    "_ShutdownCoordinator",
    "_bootstrap",
    "_build_model_inventory",
    "_cancel_task",
    "_config_snapshot",
    "_install_loop_exception_handler",
    "_install_process_exception_hooks",
    "_load_controller_class",
    "_load_integrations",
    "_load_logger_module",
    "_prepare_runtime_environment",
    "_prepare_runtime_paths",
    "_print_config_snapshot",
    "_print_model_inventory",
    "_resolve_dashboard_binding",
    "_resolve_runtime_mode",
    "_resolve_path",
    "_resolve_voice_enabled",
    "_run_startup_health_check",
    "_safe_audit",
    "_should_exit_after_info",
    "_uprint",
    "_validate_startup_settings",
    "apply_cli_overrides",
    "load_config",
    "parse_args",
    "StartupValidation",
]
