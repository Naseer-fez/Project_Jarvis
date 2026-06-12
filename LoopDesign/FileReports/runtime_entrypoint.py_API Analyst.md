# API Analyst Report: runtime\entrypoint.py

## Dependencies
- `from __future__ import annotations`
- `import asyncio`
- `import contextlib`
- `import logging`
- `import os`
- `import sys`
- `from typing import Any`
- `from core.introspection.health import HealthStatus`
- `from core.introspection.health import run_lightweight_health_check`
- `from core.runtime.paths import _resolve_path`
- `from core.runtime.bootstrap import DEFAULT_CONFIG_PATH`
- `from core.runtime.bootstrap import DEFAULT_SHUTDOWN_TIMEOUT_S`
- `from core.runtime.bootstrap import ExitCode`
- `from core.runtime.bootstrap import _cancel_task`
- `from core.runtime.bootstrap import _install_loop_exception_handler`
- `from core.runtime.bootstrap import _install_process_exception_hooks`
- `from core.runtime.bootstrap import _load_controller_class`
- `from core.runtime.bootstrap import _load_integrations`
- `from core.runtime.bootstrap import _load_logger_module`
- `from core.runtime.bootstrap import _prepare_runtime_environment`
- `from core.runtime.bootstrap import _prepare_runtime_paths`
- `from core.runtime.bootstrap import _print_config_snapshot`
- `from core.runtime.bootstrap import _print_model_inventory`
- `from core.runtime.bootstrap import _resolve_dashboard_binding`
- `from core.runtime.bootstrap import _resolve_runtime_mode`
- `from core.runtime.bootstrap import _resolve_voice_enabled`
- `from core.runtime.bootstrap import _run_startup_health_check`
- `from core.runtime.bootstrap import _safe_audit`
- `from core.runtime.bootstrap import _should_exit_after_info`
- `from core.runtime.bootstrap import _uprint`
- `from core.runtime.bootstrap import _validate_startup_settings`
- `from core.runtime.bootstrap import apply_cli_overrides`
- `from core.runtime.bootstrap import load_config`
- `from core.runtime.bootstrap import _ShutdownCoordinator as DefaultShutdownCoordinator`
- `from core.runtime.dashboard_runtime import DashboardRuntime`

## Functions & Endpoints

### `_log_health_report`
`def _log_health_report(log: logging.Logger, report: Any, *, prefix: str) -> None`
### `_run_runtime_loop`
`async def _run_runtime_loop(controller: Any, shutdown: Any, *, headless: bool, log: logging.Logger) -> int`
### `async_run`
`async def async_run(args, *, shutdown_cls: type=DefaultShutdownCoordinator) -> int`
> Core coroutine. Returns an integer exit code.
Never calls sys.exit() directly.
