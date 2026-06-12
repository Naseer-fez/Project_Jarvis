# Runtime Flow Map

## Initialization Phase
1. Process start (`sys.argv`)
2. Path resolution (`PROJECT_ROOT`, `__file__`)
3. Configuration tree merged (INI + ENV + CLI)
4. Logging singleton instantiation
5. Preflight check

## Subsystem Boot Phase
1. Dynamic loading of `Controller` via `_load_controller_class()`.
2. Plugins and integrations bound to controller instance.
3. `await controller.start()` -> controller mounts state machine, connects to databases, warms up LLM/memory.
4. Health check validation.
5. If GUI enabled, `DashboardRuntime.start(controller)` starts HTTP listener.

## Execution Phase
1. Headless: Wait on `_ShutdownCoordinator.wait()`.
2. Interactive/CLI: `asyncio.wait([cli_task, shutdown_task])`.

## Teardown Phase
1. Dashboard graceful stop.
2. `asyncio.wait_for(controller.shutdown(), timeout=shutdown_timeout)`.
3. Process cleanup, `_safe_audit` writes session summary.
4. `ExitCode` returned to `sys.exit`.
