"""Backward-compatible text-only entry point."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import traceback

import main as jarvis_main

ExitCode = jarvis_main.ExitCode
PROJECT_ROOT = jarvis_main.PROJECT_ROOT
_ShutdownCoordinator = jarvis_main._ShutdownCoordinator
_bootstrap = jarvis_main._bootstrap
load_config = jarvis_main.load_config


def apply_cli_overrides(
    config,
    args: argparse.Namespace,
) -> None:
    jarvis_main.apply_cli_overrides(config, args)
    config.setdefault("voice", {})
    config["voice"]["enabled"] = "false"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Jarvis text-only entry point",
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
        "--verify",
        action="store_true",
        help="Verify audit log and exit",
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


async def async_main(args: argparse.Namespace) -> int:
    if not hasattr(args, "voice"):
        args.voice = False
    else:
        args.voice = False
    return await jarvis_main.async_main(args)


def main() -> None:
    args = parse_args()
    try:
        exit_code = asyncio.run(async_main(args))
    except KeyboardInterrupt:
        print("\nInterrupted - goodbye.", file=sys.stderr)
        exit_code = ExitCode.OK
    except Exception:
        _bootstrap.critical("Unhandled top-level exception:\n%s", traceback.format_exc())
        exit_code = ExitCode.GENERIC_ERROR

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
