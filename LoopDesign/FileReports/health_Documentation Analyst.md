# Analysis Report for health.py

## Dependencies
- __future__.annotations
- importlib.util
- os
- dataclasses.dataclass
- dataclasses.field
- enum.Enum
- pathlib.Path
- urllib.request.urlopen
- core.ops.production.is_production
- core.ops.production.validate_production_config
- core.runtime.paths._resolve_path

## Schemas
- HealthStatus
- HealthCheck
- HealthCheck attribute: name
- HealthCheck attribute: status
- HealthCheck attribute: message
- HealthReport
- HealthReport attribute: checks

## API Contracts
- HealthReport.has_failures(self)
- HealthReport.is_healthy(self)
- HealthReport.ollama_reachable(self)
- HealthReport.summary(self)
- _config_get(config, section, option, fallback)
- _config_get_bool(config, section, option, fallback)
- _module_available(import_name)
- _path_ready(path)
- _collect_config_checks(config)
- _ollama_check(base_url)
- run_startup_health_check(controller, verbose)
- run_lightweight_health_check(config)

## Configuration Variables
None

## Assumptions & Notes
- Module Docstring: Startup and lightweight runtime health checks.

