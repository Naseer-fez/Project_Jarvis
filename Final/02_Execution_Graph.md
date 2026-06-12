# Execution Graph

## Global Entrypoints
1. `main.py` -> Direct CLI invocation. System Entry calls `run_entrypoint` with `async_main()`.
2. `Start.ps1` -> Windows PowerShell convenience wrapper.

## Application Boot Sequence (`core.runtime.bootstrap` -> `core.runtime.entrypoint.async_run`)
1. **Bootstrapping (`core.runtime.bootstrap`)**:
   - `_load_dotenv()`: Loads `.env` and `settings.env`.
   - `_enable_fault_diagnostics()`: Configures `faulthandler`.
   - **Configuration Resolution**: Reads `jarvis.ini`, applies CLI overrides (`apply_cli_overrides()`).
2. **Environment Preparation**: Sets up paths, `JARVIS_ENV` checks.
3. **Logging Initialization**: Dynamically loads logger module (`logger.setup()`), configures process exception hooks.
4. **Argument Interception**: Handles `--print_config`, `--list_models`, `--verify`, `--health_check`.
5. **Pre-flight Checks**: Validates startup settings (`_validate_startup_settings`). Runs `run_lightweight_health_check()`. Can abort if `--strict_health` is set.
6. **Controller Instantiation**: Dynamically loads `ControllerClass` (e.g., `controller_v2.py`), initializes `JarvisControllerV2`.
7. **Integration & Plugin Loading**: `_load_integrations` bounds integrations to controller via `IntegrationLoader`.
8. **Startup Audit**: Records "startup" audit event.
9. **Subsystem Boot**: Await `controller.start()`.
10. **Startup Health Check**: Validates subsystem states post-startup.
11. **Main Loop Execution (`_run_runtime_loop`)**:
    - **Dashboard Boot**: If GUI enabled, starts `DashboardRuntime`.
    - Triggers `JarvisControllerV2.run_cli()` as an asyncio task alongside the `ShutdownCoordinator.wait()` task.
12. **Shutdown**:
    - Triggered by `SIGINT`, `SIGTERM`, or internal cancellation.
    - Cancels the CLI task, stops Dashboard.
    - Awaits `JarvisControllerV2.shutdown()`. Coordinated teardown of Dashboard, Controller, and event loop tasks.
    - Records "shutdown" audit event.

## Threat Analysis
- Hard dependencies on dynamic loading (`_load_controller_class`, `_load_logger_module`) can fail silently or mask exceptions.
- `asyncio.CancelledError` swallowing during shutdown.
