from __future__ import annotations

import asyncio
import logging
import os
import sys
from typing import Any

from core.introspection.health import HealthStatus, run_lightweight_health_check
from core.runtime.bootstrap import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_SHUTDOWN_TIMEOUT_S,
    ExitCode,
    _cancel_task,
    _install_loop_exception_handler,
    _install_process_exception_hooks,
    _load_controller_class,
    _load_integrations,
    _load_logger_module,
    _prepare_runtime_environment,
    _prepare_runtime_paths,
    _print_config_snapshot,
    _print_model_inventory,
    _resolve_dashboard_binding,
    _resolve_runtime_mode,
    _resolve_path,
    _resolve_voice_enabled,
    _run_startup_health_check,
    _safe_audit,
    _should_exit_after_info,
    _uprint,
    _validate_startup_settings,
    apply_cli_overrides,
    load_config,
)
from core.runtime.bootstrap import _ShutdownCoordinator as DefaultShutdownCoordinator
from core.runtime.dashboard_runtime import DashboardRuntime


def _log_health_report(log: logging.Logger, report: Any, *, prefix: str) -> None:
    for check in getattr(report, "checks", []):
        status = getattr(check, "status", HealthStatus.OK)
        message = f"{prefix} health {check.name}: {status.value.upper()} - {check.message}"
        if status == HealthStatus.FAIL:
            log.error(message)
        elif status == HealthStatus.WARN:
            log.warning(message)
        else:
            log.info(message)


async def _run_runtime_loop(
    controller: Any,
    shutdown: Any,
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


async def async_run(
    args,
    *,
    shutdown_cls: type = DefaultShutdownCoordinator,
) -> int:
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
        from core.runtime.bootstrap import _bootstrap

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
    dashboard_enabled = bool(
        getattr(args, "gui", False) or getattr(args, "dashboard", False)
    )
    headless = bool(getattr(args, "headless", False))
    runtime_mode = _resolve_runtime_mode(
        voice_enabled=voice_enabled,
        dashboard_enabled=dashboard_enabled,
        headless=headless,
    )
    shutdown_timeout = float(
        getattr(args, "shutdown_timeout", DEFAULT_SHUTDOWN_TIMEOUT_S)
    )
    version = config.get("general", "version", fallback="unknown")
    environment = config.get(
        "general",
        "environment",
        fallback=os.environ.get("JARVIS_ENV", "development"),
    )
    validation = _validate_startup_settings(
        config,
        args,
        voice_enabled=voice_enabled,
        dashboard_enabled=dashboard_enabled,
        headless=headless,
        shutdown_timeout=shutdown_timeout,
    )
    for warning in validation.warnings:
        log.warning("Startup validation: %s", warning)
    if not validation.is_valid:
        for error in validation.errors:
            log.error("Startup validation failed: %s", error)
            _uprint(f"[ERROR] {error}", file=sys.stderr)
        return ExitCode.CONFIG_ERROR

    loop = asyncio.get_running_loop()
    _install_loop_exception_handler(loop, log)
    shutdown = shutdown_cls(loop)
    shutdown.install_signal_handlers()

    controller = None
    dashboard: DashboardRuntime | None = None
    health_report: Any | None = None
    exit_code = ExitCode.OK
    phase = "startup"

    if headless and voice_enabled:
        log.warning(
            "Voice mode requested together with headless mode; headless mode wins"
        )

    log.info(
        "Starting Jarvis version=%s env=%s mode=%s voice=%s headless=%s dashboard=%s config=%s",
        version,
        environment,
        runtime_mode,
        voice_enabled,
        headless,
        dashboard_enabled,
        config_path,
    )

    if getattr(args, "health_check", False):
        light_report = run_lightweight_health_check(config)
        log.info(
            "Lightweight health check complete: is_healthy=%s",
            light_report.is_healthy,
        )
        _log_health_report(log, light_report, prefix="Lightweight")
        _uprint(light_report.summary())
        has_failures = bool(getattr(light_report, "has_failures", False))
        if bool(getattr(args, "strict_health", False)) and has_failures:
            log.error("Health check failed in strict mode")
            return ExitCode.STARTUP_ERROR
        return ExitCode.STARTUP_ERROR if has_failures else ExitCode.OK

    preflight_report = run_lightweight_health_check(config)
    _log_health_report(log, preflight_report, prefix="Preflight")
    if bool(getattr(args, "strict_health", False)) and bool(
        getattr(preflight_report, "has_failures", False)
    ):
        log.error("Preflight health check failed in strict mode")
        return ExitCode.STARTUP_ERROR

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
                "mode": runtime_mode,
            },
            log,
        )

        await controller.start()

        verbose_health = not headless
        health_report = _run_startup_health_check(controller, verbose=verbose_health)
        if getattr(args, "strict_health", False) and bool(
            getattr(health_report, "has_failures", False)
        ):
            log.error(
                "Startup health check reported failures and strict mode is enabled"
            )
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
                await asyncio.wait_for(
                    controller.shutdown(),
                    timeout=shutdown_timeout,
                )
                log.info("Controller shut down cleanly")
            except asyncio.TimeoutError:
                log.error(
                    "Controller shutdown timed out after %.1f seconds",
                    shutdown_timeout,
                )
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


__all__ = ["async_run"]
