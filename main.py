r"""
Production-ready Jarvis entry point.

Usage (cross-platform):
  python main.py
  python main.py --voice
  python main.py --gui
  python main.py --headless --gui
  python main.py --health-check
  python main.py --verify

Windows PowerShell convenience:
  .\Start.ps1
  .\Start.ps1 --voice
  .\Start.ps1 --gui
  .\Start.ps1 --headless --gui
  .\Start.ps1 --health-check
  .\Start.ps1 --verify
"""

from __future__ import annotations

import asyncio
import signal
import sys
import traceback
import argparse
from collections.abc import Callable, Awaitable

from core.runtime.bootstrap import (
    ExitCode,
    PROJECT_ROOT,
    _ShutdownCoordinator,
    _bootstrap,
    _uprint,
    apply_cli_overrides,
    load_config,
    parse_args,
)
from core.runtime.entrypoint import async_run

ArgsParser = Callable[[list[str] | None], argparse.Namespace]
AsyncEntry = Callable[[argparse.Namespace], Awaitable[int]]


async def async_main(args: argparse.Namespace) -> int:
    return await async_run(args, shutdown_cls=_ShutdownCoordinator)


def run_entrypoint(
    *,
    parse_args_fn: ArgsParser = parse_args,
    async_entry_fn: AsyncEntry = async_main,
    argv: list[str] | None = None,
    interrupted_message: str = "Interrupted - goodbye.",
) -> int:
    args = parse_args_fn(argv)

    try:
        return asyncio.run(async_entry_fn(args))  # type: ignore[arg-type]
    except KeyboardInterrupt:
        _uprint(interrupted_message, file=sys.stderr)
        return ExitCode.OK
    except Exception:
        _bootstrap.critical(
            "Unhandled top-level exception:\n%s",
            "".join(traceback.format_exception(*sys.exc_info())),
        )
        return ExitCode.GENERIC_ERROR


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(
        run_entrypoint(
            parse_args_fn=parse_args,
            async_entry_fn=async_main,
            argv=argv,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "ExitCode",
    "PROJECT_ROOT",
    "_ShutdownCoordinator",
    "_bootstrap",
    "_uprint",
    "apply_cli_overrides",
    "async_main",
    "load_config",
    "main",
    "parse_args",
    "run_entrypoint",
    "signal",
    "sys",
]
