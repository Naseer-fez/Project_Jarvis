# Runtime Investigator Report: entrypoint.py

## Role Relevancy
Top-level entrypoint script establishing the primary runtime async event loop orchestration.

## Assumptions
- Uses `asyncio.wait(return_when=asyncio.FIRST_COMPLETED)` for parallel `cli_task` and `shutdown_task` completion monitoring.
- Triggers strict pre-flight and startup health checks, gracefully degrading/shutting down on `ExitCode` definitions.
- Sets up the `DashboardRuntime` background server task before engaging the controller block.

## Schema & API Contracts
- Resolves CLI execution modes: `headless+dashboard`, `voice+dashboard`, `cli`, etc.
- Emits health states via `HealthReport`.
- Resolves `JarvisControllerV2` as the core module handler via dynamic module loading to break cyclic dependency paths.

## Dependencies
- `core.introspection.health`
- `core.runtime.paths`, `core.runtime.bootstrap`
- `core.runtime.dashboard_runtime`

## Configuration Variables
- Driven by parsed bootstrap flags.

## Prompts
- None.
