# API Analyst Report: runtime\bootstrap.py

## Dependencies
- `from __future__ import annotations`
- `import argparse`
- `import asyncio`
- `import configparser`
- `import contextlib`
- `import dataclasses`
- `import faulthandler`
- `import io`
- `import json`
- `import logging`
- `import math`
- `import os`
- `import signal`
- `import sys`
- `import threading`
- `from pathlib import Path`
- `from typing import Any`
- `from core.ops.production import validate_production_config`
- `from core.runtime.paths import PROJECT_ROOT`
- `from core.runtime.paths import _resolve_path`

## Configuration Variables
- `DEFAULT_CONFIG_PATH` = `'config/jarvis.ini'`
- `DEFAULT_DASHBOARD_HOST` = `'127.0.0.1'`
- `DEFAULT_DASHBOARD_PORT` = `7070`
- `DEFAULT_SHUTDOWN_TIMEOUT_S` = `15.0`
- `OK` = `0`
- `GENERIC_ERROR` = `1`
- `CONFIG_ERROR` = `2`
- `AUDIT_FAILED` = `3`
- `STARTUP_ERROR` = `4`

## Schemas & API Contracts (Classes)

### Class `ExitCode`


### Class `StartupValidation`
**Fields/Schema:**
  - `errors: list[str]`
  - `warnings: list[str]`

**Methods:**
- @property
- `def is_valid(self) -> bool`


### Class `_ShutdownCoordinator`
> Signal-aware shutdown gate.

**Methods:**
- `def __init__(self, loop: asyncio.AbstractEventLoop) -> None`
- `def request_shutdown(self, signame: str='manual') -> None`
- `def install_signal_handlers(self) -> None`
- `async def wait(self) -> None`


## Functions & Endpoints

### `_load_dotenv`
`def _load_dotenv() -> None`
### `_enable_fault_diagnostics`
`def _enable_fault_diagnostics() -> None`
### `_configure_stdio`
`def _configure_stdio() -> None`
### `_uprint`
`def _uprint(msg: str, *, file=None) -> None`
> Print safely even on Windows consoles with non-UTF encodings.

### `_ensure_section`
`def _ensure_section(config: configparser.ConfigParser, section: str) -> None`
### `load_config`
`def load_config(config_path: str) -> configparser.ConfigParser`
> Load INI config from an absolute path or relative to PROJECT_ROOT.
Raises SystemExit(CONFIG_ERROR) if the file is missing in production.

### `apply_cli_overrides`
`def apply_cli_overrides(config: configparser.ConfigParser, args: argparse.Namespace) -> None`
> Merge CLI arguments into config without clobbering unrelated keys.

### `parse_args`
`def parse_args(argv: list[str] | None=None) -> argparse.Namespace`
### `_install_process_exception_hooks`
`def _install_process_exception_hooks(log: logging.Logger) -> None`
### `_install_loop_exception_handler`
`def _install_loop_exception_handler(loop: asyncio.AbstractEventLoop, log: logging.Logger) -> None`
### `_prepare_runtime_environment`
`def _prepare_runtime_environment(config: configparser.ConfigParser) -> None`
### `_prepare_runtime_paths`
`def _prepare_runtime_paths(config: configparser.ConfigParser) -> None`
### `_resolve_voice_enabled`
`def _resolve_voice_enabled(config: configparser.ConfigParser, args: argparse.Namespace) -> bool`
### `_resolve_dashboard_binding`
`def _resolve_dashboard_binding(config: configparser.ConfigParser, args: argparse.Namespace) -> tuple[str, int]`
### `_resolve_runtime_mode`
`def _resolve_runtime_mode(*, voice_enabled: bool, dashboard_enabled: bool, headless: bool) -> str`
### `_validate_startup_settings`
`def _validate_startup_settings(config: configparser.ConfigParser, args: argparse.Namespace, *, voice_enabled: bool, dashboard_enabled: bool, headless: bool, shutdown_timeout: float) -> StartupValidation`
### `_redact_key`
`def _redact_key(key: str, value: str) -> str`
### `_config_snapshot`
`def _config_snapshot(config: configparser.ConfigParser) -> dict[str, dict[str, str]]`
### `_print_config_snapshot`
`def _print_config_snapshot(config: configparser.ConfigParser, config_path: Path) -> None`
### `_build_model_inventory`
`def _build_model_inventory(config: configparser.ConfigParser) -> dict[str, Any]`
### `_print_model_inventory`
`def _print_model_inventory(config: configparser.ConfigParser) -> None`
### `_should_exit_after_info`
`def _should_exit_after_info(args: argparse.Namespace) -> bool`
### `_safe_audit`
`def _safe_audit(logger_mod: Any, event_type: str, payload: dict[str, Any], log: logging.Logger) -> None`
### `_load_logger_module`
`def _load_logger_module()`
### `_load_controller_class`
`def _load_controller_class()`
### `_load_integrations`
`def _load_integrations(controller: Any, config: configparser.ConfigParser, log: logging.Logger) -> dict[str, list[str]]`
### `_run_startup_health_check`
`def _run_startup_health_check(controller: Any | None, *, verbose: bool) -> Any`
### `_cancel_task`
`async def _cancel_task(task: asyncio.Task[Any]) -> None`