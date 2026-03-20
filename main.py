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

import signal
import sys

from main_connector import (
    ExitCode,
    PROJECT_ROOT,
    _ShutdownCoordinator,
    _bootstrap,
    _uprint,
    apply_cli_overrides,
    load_config,
    parse_args,
    run_entrypoint,
)
from core.runtime.entrypoint import async_run


async def async_main(args) -> int:
    return await async_run(args, shutdown_cls=_ShutdownCoordinator)


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
