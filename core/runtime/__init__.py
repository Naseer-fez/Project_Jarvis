from core.runtime.bootstrap import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_DASHBOARD_HOST,
    DEFAULT_DASHBOARD_PORT,
    DEFAULT_SHUTDOWN_TIMEOUT_S,
    ExitCode,
    PROJECT_ROOT,
    _ShutdownCoordinator,
    _bootstrap,
    apply_cli_overrides,
    load_config,
    parse_args,
)


def __getattr__(name: str):
    if name == "async_run":
        from core.runtime.entrypoint import async_run

        return async_run
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_DASHBOARD_HOST",
    "DEFAULT_DASHBOARD_PORT",
    "DEFAULT_SHUTDOWN_TIMEOUT_S",
    "ExitCode",
    "PROJECT_ROOT",
    "_ShutdownCoordinator",
    "_bootstrap",
    "apply_cli_overrides",
    "async_run",
    "load_config",
    "parse_args",
]
