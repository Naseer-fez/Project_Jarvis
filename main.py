"""
Production-ready Jarvis entry point.

Usage:
  python main.py
  python main.py --voice
  python main.py --gui
  python main.py --headless --gui
  python main.py --health-check
  python main.py --verify
"""

from __future__ import annotations

import argparse
import asyncio
import configparser
import contextlib
import faulthandler
import importlib
import io
import json
import logging
import os
import signal
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
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


if TYPE_CHECKING:
    from core.controller_v2 import Controller


_bootstrap = logging.getLogger("jarvis.bootstrap")
if not _bootstrap.handlers:
    _bootstrap.addHandler(logging.StreamHandler(sys.stderr))
_bootstrap_level = logging.DEBUG if os.environ.get("JARVIS_LOG_LEVEL", "").upper() == "DEBUG" else logging.INFO
_bootstrap.setLevel(_bootstrap_level)
_bootstrap.propagate = False


class ExitCode:
    OK = 0
    GENERIC_ERROR = 1
    CONFIG_ERROR = 2
    AUDIT_FAILED = 3
    STARTUP_ERROR = 4


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
    config = configparser.ConfigParser()
    path = _resolve_path(config_path)

    if not path.exists():
        env = os.environ.get("JARVIS_ENV", "development").lower()
        msg = f"Config not found: {path}"
        if env == "production":
            _bootstrap.critical(msg)
            raise SystemExit(ExitCode.CONFIG_ERROR)
        _bootstrap.warning("%s - using defaults", msg)
        return config

    try:
        with path.open("r", encoding="utf-8") as handle:
            config.read_file(handle)
    except configparser.Error as exc:
        _bootstrap.critical("Config parse error: %s", exc)
        raise SystemExit(ExitCode.CONFIG_ERROR) from exc
    except OSError as exc:
        _bootstrap.critical("Config read error: %s", exc)
        raise SystemExit(ExitCode.CONFIG_ERROR) from exc

    _bootstrap.debug("Config loaded from %s", path)
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
    parser.add_argument(
        "--voice",
        action="store_true",
        help="Start Jarvis in voice mode",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Start the dashboard server",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Alias for --gui",
    )
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
    parser.add_argument(
        "--session-name",
        help="Optional session name for this run",
    )
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
        default=float(os.environ.get("JARVIS_SHUTDOWN_TIMEOUT_S", str(DEFAULT_SHUTDOWN_TIMEOUT_S))),
        help="Graceful shutdown timeout in seconds",
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


class DashboardRuntime:
    def __init__(
        self,
        host: str,
        port: int,
        log: logging.Logger,
    ) -> None:
        self.host = host
        self.port = port
        self.log = log
        self._server = None
        self._thread: threading.Thread | None = None
        self._thread_error: BaseException | None = None

    async def start(self, controller: Any, health_report: Any | None = None) -> None:
        if self._thread and self._thread.is_alive():
            return

        import uvicorn
        from dashboard.server import app as dashboard_app
        from dashboard.server import set_controller, update_state

        set_controller(controller)
        config = uvicorn.Config(
            dashboard_app,
            host=self.host,
            port=self.port,
            log_level="warning",
            access_log=False,
        )
        self._server = uvicorn.Server(config)

        def _serve() -> None:
            try:
                self._server.run()
            except BaseException as exc:  # noqa: BLE001
                self._thread_error = exc

        self._thread_error = None
        self._thread = threading.Thread(
            target=_serve,
            name="jarvis-dashboard",
            daemon=True,
        )
        self._thread.start()

        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if self._thread_error is not None:
                raise RuntimeError("Dashboard thread crashed during startup") from self._thread_error
            if getattr(self._server, "started", False):
                break
            if self._thread and not self._thread.is_alive():
                raise RuntimeError("Dashboard server exited before reporting ready")
            await asyncio.sleep(0.1)

        if not getattr(self._server, "started", False):
            self.log.warning("Dashboard startup was not confirmed within 10 seconds")

        llm_obj = getattr(controller, "llm", None)
        model_name = getattr(llm_obj, "model", getattr(llm_obj, "model_name", "unknown"))
        active_goals = 0
        goal_manager = getattr(controller, "goal_manager", None)
        if hasattr(goal_manager, "active_goals"):
            with contextlib.suppress(Exception):
                active_goals = len(goal_manager.active_goals())

        update_state(
            session_id=str(getattr(controller, "session_id", "jarvis")),
            model=str(model_name),
            state="IDLE",
            active_goals=active_goals,
            ollama_online=bool(getattr(health_report, "ollama_reachable", False)),
        )
        self.log.info("Dashboard listening on http://%s:%s", self.host, self.port)

    def stop(self, timeout: float = 5.0) -> None:
        if self._server is not None:
            try:
                self._server.should_exit = True
            except Exception:
                pass

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                self.log.warning("Dashboard thread did not stop within %.1f seconds", timeout)

        with contextlib.suppress(Exception):
            from dashboard.server import update_state

            update_state(state="OFFLINE")


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


def _redact_key(key: str, value: str) -> str:
    lowered = key.lower()
    if any(token in lowered for token in ("secret", "token", "password", "api_key", "access_key")):
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


def _print_config_snapshot(config: configparser.ConfigParser, config_path: Path) -> None:
    payload = {
        "config_path": str(config_path),
        "project_root": str(PROJECT_ROOT),
        "sections": _config_snapshot(config),
    }
    _uprint(json.dumps(payload, indent=2, sort_keys=True))


def _build_model_inventory(config: configparser.ConfigParser) -> dict[str, dict[str, Any]]:
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
        except Exception as exc:  # noqa: BLE001
            entry["best_available"] = None
            entry["error"] = str(exc)
        inventory[task_type] = entry

    inventory["discovered"] = router.list_available()
    return inventory


def _print_model_inventory(config: configparser.ConfigParser) -> None:
    inventory = _build_model_inventory(config)
    _uprint(json.dumps(inventory, indent=2, sort_keys=True))


def _should_exit_after_info(args: argparse.Namespace) -> bool:
    info_requested = bool(getattr(args, "print_config", False) or getattr(args, "list_models", False))
    runtime_requested = bool(
        getattr(args, "voice", False)
        or getattr(args, "gui", False)
        or getattr(args, "dashboard", False)
        or getattr(args, "headless", False)
        or getattr(args, "verify", False)
        or getattr(args, "health_check", False)
    )
    return info_requested and not runtime_requested


def _safe_audit(logger_mod: Any, event_type: str, payload: dict[str, Any], log: logging.Logger) -> None:
    audit_fn = getattr(logger_mod, "audit", None)
    if not callable(audit_fn):
        return
    try:
        audit_fn(event_type, payload)
    except Exception:
        log.debug("Audit event '%s' failed", event_type, exc_info=True)


def _load_logger_module():
    from core.logging import logger as logger_mod
    return logger_mod


def _load_controller_class():
    from core.controller import Controller
    return Controller


def _load_integrations(controller: Any, config: configparser.ConfigParser, log: logging.Logger) -> dict[str, list[str]]:
    try:
        from integrations.loader import IntegrationLoader
        from integrations.registry import integration_registry

        loader = IntegrationLoader()
        result = loader.load_all(config=config, registry=integration_registry)
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


async def _run_runtime_loop(
    controller: Any,
    shutdown: _ShutdownCoordinator,
    *,
    headless: bool,
    log: logging.Logger,
) -> int:
    run_cli = getattr(controller, "run_cli", None)

    if headless:
        log.info("Running in headless mode; waiting for shutdown signal")
        await shutdown.wait()
        return ExitCode.OK

    if not callable(run_cli):
        log.warning("Controller has no run_cli(); waiting for shutdown signal")
        await shutdown.wait()
        return ExitCode.OK

    cli_task = asyncio.create_task(run_cli(), name="jarvis-cli")
    shutdown_task = asyncio.create_task(shutdown.wait(), name="jarvis-shutdown")

    done, pending = await asyncio.wait(
        {cli_task, shutdown_task},
        return_when=asyncio.FIRST_COMPLETED,
    )

    for task in pending:
        await _cancel_task(task)

    if cli_task in done and not cli_task.cancelled():
        exc = cli_task.exception()
        if exc is not None:
            raise exc

    return ExitCode.OK


async def async_main(args: argparse.Namespace) -> int:
    """
    Core coroutine. Returns an integer exit code.
    Never calls sys.exit() directly.
    """
    config_path = _resolve_path(getattr(args, "config", DEFAULT_CONFIG_PATH))
    config = load_config(str(config_path))
    apply_cli_overrides(config, args)
    _prepare_runtime_environment(config)
    _prepare_runtime_paths(config)

    try:
        logger_mod = _load_logger_module()

        logger_mod.setup(config)
        log = logger_mod.get()
    except Exception as exc:
        _bootstrap.critical("Failed to initialize logging subsystem: %s", exc)
        return ExitCode.STARTUP_ERROR

    _install_process_exception_hooks(log)

    if getattr(args, "print_config", False):
        _print_config_snapshot(config, config_path)

    if getattr(args, "list_models", False):
        try:
            _print_model_inventory(config)
        except Exception:
            log.exception("Failed to inspect model inventory")
            if _should_exit_after_info(args):
                return ExitCode.STARTUP_ERROR

    if _should_exit_after_info(args):
        return ExitCode.OK

    if getattr(args, "verify", False):
        try:
            ok, count, err = logger_mod.verify_audit()
            if ok:
                _uprint(f"[OK] Audit OK - {count} entries verified")
                log.info("Audit verification passed (%d entries)", count)
                return ExitCode.OK
            _uprint(f"[FAIL] Audit FAILED - {err}", file=sys.stderr)
            log.error("Audit verification failed: %s", err)
            return ExitCode.AUDIT_FAILED
        except Exception:
            log.exception("Unexpected error during audit verification")
            _uprint("[ERROR] Audit verification crashed", file=sys.stderr)
            return ExitCode.GENERIC_ERROR

    voice_enabled = _resolve_voice_enabled(config, args)
    dashboard_enabled = bool(getattr(args, "gui", False) or getattr(args, "dashboard", False))
    headless = bool(getattr(args, "headless", False))
    shutdown_timeout = float(getattr(args, "shutdown_timeout", DEFAULT_SHUTDOWN_TIMEOUT_S))
    version = config.get("general", "version", fallback="unknown")
    environment = config.get("general", "environment", fallback=os.environ.get("JARVIS_ENV", "development"))

    loop = asyncio.get_running_loop()
    _install_loop_exception_handler(loop, log)
    shutdown = _ShutdownCoordinator(loop)
    shutdown.install_signal_handlers()

    controller: Optional[Controller] = None
    dashboard: DashboardRuntime | None = None
    health_report: Any | None = None
    exit_code = ExitCode.OK
    phase = "startup"

    if headless and voice_enabled:
        log.warning("Voice mode requested together with headless mode; headless mode wins")

    log.info(
        "Starting Jarvis version=%s env=%s voice=%s headless=%s dashboard=%s config=%s",
        version,
        environment,
        voice_enabled,
        headless,
        dashboard_enabled,
        config_path,
    )

    # ------------------------------------------------------------------
    # --health-check: run a LIGHTWEIGHT pre-controller probe and exit.
    # The controller and SentenceTransformer are never loaded on this path.
    # ------------------------------------------------------------------
    if getattr(args, "health_check", False):
        from core.introspection.health import run_lightweight_health_check

        light_report = run_lightweight_health_check(config)
        log.info("Lightweight health check complete: is_healthy=%s", light_report.is_healthy)
        # Print human-readable summary to stdout so operators see it
        _uprint(light_report.summary())
        has_failures = bool(getattr(light_report, "has_failures", False))
        if bool(getattr(args, "strict_health", False)) and has_failures:
            log.error("Health check failed in strict mode")
            return ExitCode.STARTUP_ERROR
        # Ollama being offline is WARN not FAIL for a local-first runtime;
        # exit 0 unless something structural (config missing) fails.
        return ExitCode.STARTUP_ERROR if has_failures else ExitCode.OK

    try:
        controller_cls = _load_controller_class()
        controller = controller_cls(config=config, voice=voice_enabled)
        _load_integrations(controller, config, log)

        _safe_audit(
            logger_mod,
            "startup",
            {
                "config": str(config_path),
                "environment": environment,
                "voice": voice_enabled,
                "headless": headless,
                "dashboard": dashboard_enabled,
            },
            log,
        )

        await controller.start()

        verbose_health = not headless
        health_report = _run_startup_health_check(controller, verbose=verbose_health)
        if getattr(args, "strict_health", False) and bool(getattr(health_report, "has_failures", False)):
            log.error("Startup health check reported failures and strict mode is enabled")
            return ExitCode.STARTUP_ERROR

        if dashboard_enabled:
            host, port = _resolve_dashboard_binding(config, args)
            dashboard = DashboardRuntime(
                host=host,
                port=port,
                log=log,
            )
            await dashboard.start(controller, health_report=health_report)
            _uprint(f"Dashboard: http://{host}:{port}")

        phase = "runtime"
        exit_code = await _run_runtime_loop(
            controller,
            shutdown,
            headless=headless,
            log=log,
        )

    except asyncio.CancelledError:
        log.info("Main task cancelled during %s", phase)
        exit_code = ExitCode.OK
    except Exception:
        if phase == "startup":
            log.critical("Startup failure", exc_info=True)
            exit_code = ExitCode.STARTUP_ERROR
        else:
            log.critical("Unhandled runtime failure", exc_info=True)
            exit_code = ExitCode.GENERIC_ERROR
    finally:
        if dashboard is not None:
            dashboard.stop(timeout=min(5.0, shutdown_timeout))

        if controller is not None:
            try:
                await asyncio.wait_for(controller.shutdown(), timeout=shutdown_timeout)
                log.info("Controller shut down cleanly")
            except asyncio.TimeoutError:
                log.error("Controller shutdown timed out after %.1f seconds", shutdown_timeout)
                exit_code = ExitCode.GENERIC_ERROR
            except Exception:
                log.exception("Error during controller shutdown")
                exit_code = ExitCode.GENERIC_ERROR

        if controller is not None:
            summary_fn = getattr(controller, "session_summary", None)
            session_summary = summary_fn() if callable(summary_fn) else {}
            payload = {
                "exit_code": exit_code,
                "phase": phase,
                "session_id": getattr(controller, "session_id", None),
                "summary": session_summary,
            }
        else:
            payload = {
                "exit_code": exit_code,
                "phase": phase,
            }
        _safe_audit(logger_mod, "shutdown", payload, log)

    return exit_code


def main() -> None:
    args = parse_args()

    try:
        exit_code = asyncio.run(async_main(args))
    except KeyboardInterrupt:
        _uprint("Interrupted - goodbye.", file=sys.stderr)
        exit_code = ExitCode.OK
    except Exception:
        _bootstrap.critical(
            "Unhandled top-level exception:\n%s",
            "".join(traceback.format_exception(*sys.exc_info())),
        )
        exit_code = ExitCode.GENERIC_ERROR

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
