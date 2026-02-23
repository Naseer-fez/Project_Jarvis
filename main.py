"""
main.py — Jarvis V2 entry point.

Usage:
  python main.py               # CLI only (safe default)
  python main.py --voice       # Voice + CLI hybrid
  python main.py --verify      # Verify audit log integrity and exit
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

if TYPE_CHECKING:
    from core.controller import Controller

# ─────────────────────────────────────────────────────────────
# Project root — never mutate cwd; use absolute paths instead
# ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent

# Bootstrap logger used before full logging is configured
_bootstrap = logging.getLogger("jarvis.bootstrap")
_bootstrap.setLevel(logging.DEBUG)
_bootstrap.addHandler(logging.StreamHandler(sys.stderr))


# ─────────────────────────────────────────────────────────────
# Exit codes
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
    """
    Load INI config from an absolute path or relative to PROJECT_ROOT.
    Raises SystemExit(CONFIG_ERROR) if the file is missing in production
    (i.e. when JARVIS_ENV=production).
    """
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
    """Merge CLI arguments into config without clobbering unrelated sections."""
    if args.log_level:
        config.setdefault("logging", {})
        config["logging"]["level"] = args.log_level

    if args.session_name:
        config.setdefault("general", {})
        config["general"]["session_name"] = args.session_name


# ─────────────────────────────────────────────────────────────
# CLI args
# ─────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Jarvis — Local Agentic AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--voice", action="store_true", help="Enable voice mode")
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
# Shutdown coordinator
# ─────────────────────────────────────────────────────────────
class _ShutdownCoordinator:
    """
    Thread-safe, signal-aware shutdown gate.

    Usage:
        coordinator = _ShutdownCoordinator(loop)
        coordinator.install_signal_handlers()
        await coordinator.wait()          # blocks until signal arrives
    """

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._event = asyncio.Event()

    def request_shutdown(self, signame: str = "manual") -> None:
        _bootstrap.info(f"Shutdown requested via {signame}")
        self._loop.call_soon_threadsafe(self._event.set)

    def install_signal_handlers(self) -> None:
        # Windows does not support add_signal_handler
        if sys.platform == "win32":
            for sig in (signal.SIGINT, signal.SIGBREAK):  # type: ignore[attr-defined]
                signal.signal(sig, lambda *_: self.request_shutdown(sig.name))
        else:
            for sig in (signal.SIGINT, signal.SIGTERM):
                self._loop.add_signal_handler(
                    sig, self.request_shutdown, sig.name
                )

    async def wait(self) -> None:
        await self._event.wait()


# ─────────────────────────────────────────────────────────────
# Async main
# ─────────────────────────────────────────────────────────────
async def async_main(args: argparse.Namespace) -> int:
    """
    Core coroutine. Returns an integer exit code.
    Never calls sys.exit() directly — that's the caller's job.
    """
    config = load_config(args.config)
    apply_cli_overrides(config, args)

    # ── Set up structured logging ────────────────────────────
    try:
        from core.logging import logger as logger_mod

        logger_mod.setup(config)
        log = logger_mod.get()
    except Exception as exc:
        _bootstrap.critical(f"Failed to initialise logging subsystem: {exc}")
        return ExitCode.STARTUP_ERROR

    # ── Audit verification mode ──────────────────────────────
    if args.verify:
        try:
            from core.logging import logger as logger_mod  # noqa: F811

            ok, count, err = logger_mod.verify_audit()
            msg = f"{count} entries verified"

            if ok:
                log.info(f"Audit OK — {msg}")
                print(f"✅ Audit OK — {msg}")
                return ExitCode.OK
            else:
                log.error(f"Audit FAILED — {err}")
                print(f"❌ Audit FAILED — {err}", file=sys.stderr)
                return ExitCode.AUDIT_FAILED

        except Exception as exc:
            log.exception("Unexpected error during audit verification")
            print(f"❌ Audit error: {exc}", file=sys.stderr)
            return ExitCode.GENERIC_ERROR

    # ── Resolve voice flag ───────────────────────────────────
    try:
        voice_enabled = args.voice or config.getboolean(
            "voice", "enabled", fallback=False
        )
    except (configparser.Error, ValueError) as exc:
        log.warning(f"Could not parse voice.enabled from config ({exc}) — defaulting to CLI arg")
        voice_enabled = args.voice

    # ── Install signal handlers ──────────────────────────────
    loop = asyncio.get_running_loop()
    shutdown = _ShutdownCoordinator(loop)
    shutdown.install_signal_handlers()

    # ── Initialise controller ────────────────────────────────
    controller: Optional[Controller] = None

    try:
        from core.controller import Controller

        log.info(
            "Starting Jarvis V2 | voice=%s | session=%s",
            voice_enabled,
            config.get("general", "session_name", fallback="default"),
        )

        controller = Controller(config, voice=voice_enabled)
        await controller.start()

    except Exception:
        log.critical("Controller failed to start:\n%s", traceback.format_exc())
        return ExitCode.STARTUP_ERROR

    # ── Run until done or signalled ──────────────────────────
    exit_code = ExitCode.OK

    try:
        if hasattr(controller, "run_cli"):
            # run_cli and the shutdown event race; whichever finishes first wins
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

            # Propagate exceptions from the CLI task
            for task in done:
                if task.get_name() == "cli_loop" and not task.cancelled():
                    exc = task.exception()
                    if exc:
                        log.error("CLI loop raised: %s", exc, exc_info=exc)
                        exit_code = ExitCode.GENERIC_ERROR

        else:
            log.warning(
                "Controller has no run_cli() — running in headless mode. "
                "Send SIGTERM or SIGINT to stop."
            )
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
        # Ctrl-C before the event loop installs its own handler
        print("\nInterrupted — goodbye.", file=sys.stderr)
        exit_code = ExitCode.OK
    except Exception:
        _bootstrap.critical("Unhandled top-level exception:\n%s", traceback.format_exc())
        exit_code = ExitCode.GENERIC_ERROR

    sys.exit(exit_code)


if __name__ == "__main__":
    main()