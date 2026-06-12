# Analysis Report for bootstrap.py

## Dependencies
- __future__.annotations
- argparse
- asyncio
- configparser
- contextlib
- dataclasses
- faulthandler
- io
- json
- logging
- math
- os
- signal
- sys
- threading
- pathlib.Path
- typing.Any
- core.ops.production.validate_production_config
- core.runtime.paths.PROJECT_ROOT
- core.runtime.paths._resolve_path

## Schemas
- ExitCode
- StartupValidation
- StartupValidation attribute: errors
- StartupValidation attribute: warnings
- _ShutdownCoordinator

## API Contracts
- _load_dotenv()
- _enable_fault_diagnostics()
- _configure_stdio()
- _uprint(msg)
- StartupValidation.is_valid(self)
- _ensure_section(config, section)
- load_config(config_path)
- apply_cli_overrides(config, args)
- parse_args(argv)
- _ShutdownCoordinator.__init__(self, loop)
- _ShutdownCoordinator.request_shutdown(self, signame)
- _ShutdownCoordinator.install_signal_handlers(self)
- _install_process_exception_hooks(log)
- _install_loop_exception_handler(loop, log)
- _prepare_runtime_environment(config)
- _prepare_runtime_paths(config)
- _resolve_voice_enabled(config, args)
- _resolve_dashboard_binding(config, args)
- _resolve_runtime_mode()
- _validate_startup_settings(config, args)
- _redact_key(key, value)
- _config_snapshot(config)
- _print_config_snapshot(config, config_path)
- _build_model_inventory(config)
- _print_model_inventory(config)
- _should_exit_after_info(args)
- _safe_audit(logger_mod, event_type, payload, log)
- _load_logger_module()
- _load_controller_class()
- _load_integrations(controller, config, log)
- _run_startup_health_check(controller)

## Configuration Variables
- DEFAULT_CONFIG_PATH
- DEFAULT_DASHBOARD_HOST
- DEFAULT_DASHBOARD_PORT
- DEFAULT_SHUTDOWN_TIMEOUT_S

## Assumptions & Notes
None

