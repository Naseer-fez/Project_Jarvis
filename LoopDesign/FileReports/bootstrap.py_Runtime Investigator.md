# Runtime Investigator Report: bootstrap.py

## Role Relevancy
Handles the very bottom execution frame initializing configurations, environments, dependencies, subsystem components, and process hooks.

## Assumptions
- Uses Python's native `argparse` and `configparser`.
- Loads `.env` via `dotenv` (first `config/settings.env`, then root `.env` as override).
- Sets up exception handlers globally across threads (`threading.excepthook`) and asyncio (`loop.set_exception_handler`).
- Bootstraps missing runtime paths directly via `pathlib.mkdir`.

## Schema & API Contracts
- Configuration keys include heavily referenced structures (e.g. `[logging]`, `[memory]`, `[execution]`, `[dashboard]`, `[voice]`, `[autonomy]`, `[plugins]`, `[automation]`).
- Exposes `async_run` logic parameters and CLI flags to core controllers.

## Dependencies
- `core.ops.production`
- `core.runtime.paths`
- Optional dynamic load of `integrations.loader`, `core.introspection.health`.

## Configuration Variables
- `DEFAULT_CONFIG_PATH = "config/jarvis.ini"`
- `DEFAULT_DASHBOARD_HOST = "127.0.0.1"`
- `DEFAULT_DASHBOARD_PORT = 7070`
- `DEFAULT_SHUTDOWN_TIMEOUT_S = 15.0`

## Prompts
- None.
