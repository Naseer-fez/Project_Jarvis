"""
Shared runtime connector for Jarvis entrypoints.

This keeps the public launcher modules thin while preserving a stable
import surface for compatibility layers and tests.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import traceback
from collections.abc import Awaitable, Callable

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
        return asyncio.run(async_entry_fn(args))
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
    raise SystemExit(run_entrypoint(argv=argv))


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
]
