# Analysis Report for entrypoint.py

## Dependencies
- __future__.annotations
- asyncio
- contextlib
- logging
- os
- sys
- typing.Any
- core.introspection.health.HealthStatus
- core.introspection.health.run_lightweight_health_check
- core.runtime.paths._resolve_path
- core.runtime.bootstrap.DEFAULT_CONFIG_PATH
- core.runtime.bootstrap.DEFAULT_SHUTDOWN_TIMEOUT_S
- core.runtime.bootstrap.ExitCode
- core.runtime.bootstrap._cancel_task
- core.runtime.bootstrap._install_loop_exception_handler
- core.runtime.bootstrap._install_process_exception_hooks
- core.runtime.bootstrap._load_controller_class
- core.runtime.bootstrap._load_integrations
- core.runtime.bootstrap._load_logger_module
- core.runtime.bootstrap._prepare_runtime_environment
- core.runtime.bootstrap._prepare_runtime_paths
- core.runtime.bootstrap._print_config_snapshot
- core.runtime.bootstrap._print_model_inventory
- core.runtime.bootstrap._resolve_dashboard_binding
- core.runtime.bootstrap._resolve_runtime_mode
- core.runtime.bootstrap._resolve_voice_enabled
- core.runtime.bootstrap._run_startup_health_check
- core.runtime.bootstrap._safe_audit
- core.runtime.bootstrap._should_exit_after_info
- core.runtime.bootstrap._uprint
- core.runtime.bootstrap._validate_startup_settings
- core.runtime.bootstrap.apply_cli_overrides
- core.runtime.bootstrap.load_config
- core.runtime.bootstrap._ShutdownCoordinator
- core.runtime.dashboard_runtime.DashboardRuntime

## Schemas
None

## API Contracts
- _log_health_report(log, report)

## Configuration Variables
None

## Assumptions & Notes
None

