"""
Main_No_Voice.py — Jarvis V2 headless / text-only entry point.

Identical to main.py but with voice permanently disabled.
All integrations (Telegram, Gmail, Google Calendar, Notion, Spotify),
the workflow engine, risk evaluator, and LLMClientV2 routing are all active.

Usage:
  python Main_No_Voice.py               # plain CLI
  python Main_No_Voice.py --verify      # verify audit log integrity and exit
  python Main_No_Voice.py --gui         # CLI + web dashboard at http://localhost:7070
  python Main_No_Voice.py --log-level DEBUG
"""

from __future__ import annotations

import argparse
import asyncio
import configparser
import logging
import os
import signal
import sys
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Optional

try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=_env_path)
except ImportError:
    pass


def _uprint(msg: str, *, file=None) -> None:
    """Print msg safely regardless of terminal encoding (e.g. cp1252 on Windows)."""
    import io
    target = file or sys.stdout
    try:
        print(msg, file=target)
    except UnicodeEncodeError:
        raw = getattr(target, "buffer", None)
        if raw:
            raw.write((msg + "\n").encode("utf-8", errors="replace"))
        else:
            print(msg.encode("ascii", errors="replace").decode("ascii"), file=target)


if TYPE_CHECKING:
    from core.controller_v2 import Controller

PROJECT_ROOT = Path(__file__).resolve().parent

_bootstrap = logging.getLogger("jarvis.bootstrap")
_bootstrap.setLevel(logging.DEBUG)
_bootstrap.addHandler(logging.StreamHandler(sys.stderr))


# ─────────────────────────────────────────────────────────────
# Exit codes (same as main.py)
# ─────────────────────────────────────────────────────────────
class ExitCode:
    OK = 0
    GENERIC_ERROR = 1
    CONFIG_ERROR = 2
    AUDIT_FAILED = 3
    STARTUP_ERROR = 4


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
def load_config(config_path: str) -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / config_path

    if not path.exists():
        env = os.environ.get("JARVIS_ENV", "development").lower()
        msg = f"Config not found: {path}"
        if env == "production":
            _bootstrap.critical(msg)
            sys.exit(ExitCode.CONFIG_ERROR)
        else:
            _bootstrap.warning(f"{msg} — using defaults")
            return config

    try:
        config.read(path, encoding="utf-8")
        _bootstrap.debug(f"Config loaded from {path}")
    except configparser.Error as exc:
        _bootstrap.critical(f"Config parse error: {exc}")
        sys.exit(ExitCode.CONFIG_ERROR)

    return config


def apply_cli_overrides(
    config: configparser.ConfigParser, args: argparse.Namespace
) -> None:
    if args.log_level:
        config.setdefault("logging", {})
        config["logging"]["level"] = args.log_level

    if args.session_name:
        config.setdefault("general", {})
        config["general"]["session_name"] = args.session_name

    # ── Force voice off ───────────────────────────────────────
    config.setdefault("voice", {})
    config["voice"]["enabled"] = "false"


# ─────────────────────────────────────────────────────────────
# CLI args  (--voice removed)
# ─────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Jarvis — Text-only (no voice) entry point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Start web dashboard at http://localhost:7070",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Alias for --gui (backward compatibility)",
    )
    parser.add_argument(
        "--verify", action="store_true", help="Verify audit log and exit"
    )
    parser.add_argument(
        "--config",
        default=os.environ.get("JARVIS_CONFIG", "config/jarvis.ini"),
        help="Config file path (also reads JARVIS_CONFIG env var)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=os.environ.get("JARVIS_LOG_LEVEL"),
        help="Override log level (also reads JARVIS_LOG_LEVEL env var)",
    )
    parser.add_argument("--session-name", help="Optional session name for this run")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────
# Shutdown coordinator (identical to main.py)
# ─────────────────────────────────────────────────────────────
class _ShutdownCoordinator:
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._event = asyncio.Event()

    def request_shutdown(self, signame: str = "manual") -> None:
        _bootstrap.info(f"Shutdown requested via {signame}")
        self._loop.call_soon_threadsafe(self._event.set)

    def install_signal_handlers(self) -> None:
        if sys.platform == "win32":
            for sig in (signal.SIGINT, signal.SIGBREAK):  # type: ignore[attr-defined]
                signal.signal(sig, lambda *_: self.request_shutdown(sig.name))
        else:
            for sig in (signal.SIGINT, signal.SIGTERM):
                self._loop.add_signal_handler(sig, self.request_shutdown, sig.name)

    async def wait(self) -> None:
        await self._event.wait()


# ─────────────────────────────────────────────────────────────
# Async main — voice=False always
# ─────────────────────────────────────────────────────────────
async def async_main(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    apply_cli_overrides(config, args)  # also sets voice.enabled=false

    # ── Logging ──────────────────────────────────────────────
    try:
        from core import logger as logger_mod
        logger_mod.setup(config)
        log = logger_mod.get()
    except Exception as exc:
        _bootstrap.critical(f"Failed to initialise logging subsystem: {exc}")
        return ExitCode.STARTUP_ERROR

    # ── Audit verification ───────────────────────────────────
    if args.verify:
        try:
            from core import logger as logger_mod  # noqa: F811
            ok, count, err = logger_mod.verify_audit()
            msg = f"{count} entries verified"
            if ok:
                log.info("Audit OK — %s", msg)
                _uprint(f"[OK] Audit OK — {msg}")
                return ExitCode.OK
            else:
                log.error("Audit FAILED — %s", err)
                _uprint(f"[FAIL] Audit FAILED — {err}", file=sys.stderr)
                return ExitCode.AUDIT_FAILED
        except Exception as exc:
            log.exception("Unexpected error during audit verification")
            _uprint(f"[ERROR] Audit error: {exc}", file=sys.stderr)
            return ExitCode.GENERIC_ERROR

    # ── Signal handlers ──────────────────────────────────────
    loop = asyncio.get_running_loop()
    shutdown = _ShutdownCoordinator(loop)
    shutdown.install_signal_handlers()

    # ── Controller — voice=False always ──────────────────────
    controller: Optional[Controller] = None

    try:
        from core.controller_v2 import Controller

        log.info(
            "Starting Jarvis V2 (no-voice) | session=%s",
            config.get("general", "session_name", fallback="default"),
        )

        controller = Controller(config=config, voice=False)  # ← voice permanently off

        from core.introspection.health import HealthReport, run_startup_health_check
        try:
            health_report = run_startup_health_check(controller)
        except Exception as exc:  # noqa: BLE001
            log.warning("Startup health check failed: %s", exc)
            health_report = HealthReport()

        if args.gui or getattr(args, "dashboard", False):
            try:
                import threading
                import uvicorn
                from dashboard.server import (
                    app as dashboard_app,
                    set_controller,
                    update_state,
                )
                set_controller(controller)

                def _run_dashboard() -> None:
                    uvicorn.run(
                        dashboard_app,
                        host="127.0.0.1",
                        port=7070,
                        log_level="warning",
                    )

                t = threading.Thread(target=_run_dashboard, daemon=True)
                t.start()
                try:
                    llm_obj = getattr(controller, "llm", None)
                    model_name = getattr(llm_obj, "model", getattr(llm_obj, "model_name", "unknown"))
                    update_state(
                        session_id=getattr(controller, "session_id", "jarvis-1"),
                        model=model_name,
                        state="IDLE",
                        ollama_online=getattr(health_report, "ollama_reachable", False),
                    )
                except Exception:
                    pass
                print("Dashboard: http://localhost:7070")
            except ImportError as exc:
                log.warning("Dashboard unavailable: %s", exc)
            except Exception as exc:  # noqa: BLE001
                log.warning("Dashboard failed to start: %s", exc)

        await controller.start()

    except Exception:
        log.critical("Controller failed to start:\n%s", traceback.format_exc())
        return ExitCode.STARTUP_ERROR

    # ── Run loop ─────────────────────────────────────────────
    exit_code = ExitCode.OK

    try:
        if hasattr(controller, "run_cli"):
            cli_task = asyncio.create_task(controller.run_cli(), name="cli_loop")
            shutdown_task = asyncio.create_task(shutdown.wait(), name="shutdown_gate")

            done, pending = await asyncio.wait(
                [cli_task, shutdown_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in pending:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

            for task in done:
                if task.get_name() == "cli_loop" and not task.cancelled():
                    exc = task.exception()
                    if exc:
                        log.error("CLI loop raised: %s", exc, exc_info=exc)
                        exit_code = ExitCode.GENERIC_ERROR
        else:
            log.warning("Controller has no run_cli() — headless mode. Send SIGTERM to stop.")
            await shutdown.wait()

    except asyncio.CancelledError:
        log.info("Main task cancelled — shutting down")
    except Exception:
        log.critical("Unhandled error in run loop:\n%s", traceback.format_exc())
        exit_code = ExitCode.GENERIC_ERROR
    finally:
        if controller is not None:
            log.info("Shutting down controller…")
            try:
                await asyncio.wait_for(controller.shutdown(), timeout=15)
                log.info("Controller shut down cleanly")
            except asyncio.TimeoutError:
                log.error("Controller.shutdown() timed out after 15 s — forcing exit")
                exit_code = ExitCode.GENERIC_ERROR
            except Exception:
                log.exception("Error during controller shutdown")
                exit_code = ExitCode.GENERIC_ERROR

    return exit_code


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    try:
        exit_code = asyncio.run(async_main(args))
    except KeyboardInterrupt:
        print("\nInterrupted — goodbye.", file=sys.stderr)
        exit_code = ExitCode.OK
    except Exception:
        _bootstrap.critical("Unhandled top-level exception:\n%s", traceback.format_exc())
        exit_code = ExitCode.GENERIC_ERROR

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
