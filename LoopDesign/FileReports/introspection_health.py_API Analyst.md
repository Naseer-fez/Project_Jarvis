# API Analyst Report: introspection\health.py

## Dependencies
- `from __future__ import annotations`
- `import importlib.util`
- `import os`
- `from dataclasses import dataclass`
- `from dataclasses import field`
- `from enum import Enum`
- `from pathlib import Path`
- `from urllib.request import urlopen`
- `from core.ops.production import is_production`
- `from core.ops.production import validate_production_config`
- `from core.runtime.paths import _resolve_path`

## Configuration Variables
- `OK` = `'ok'`
- `WARN` = `'warn'`
- `FAIL` = `'fail'`

## Schemas & API Contracts (Classes)

### Class `HealthStatus(str, Enum)`


### Class `HealthCheck`
**Fields/Schema:**
  - `name: str`
  - `status: HealthStatus`
  - `message: str`



### Class `HealthReport`
**Fields/Schema:**
  - `checks: list[HealthCheck]`

**Methods:**
- @property
- `def has_failures(self) -> bool`
- @property
- `def is_healthy(self) -> bool`
- @property
- `def ollama_reachable(self) -> bool`
- `def summary(self) -> str`


## Functions & Endpoints

### `_config_get`
`def _config_get(config, section: str, option: str, fallback: str='') -> str`
### `_config_get_bool`
`def _config_get_bool(config, section: str, option: str, fallback: bool=False) -> bool`
### `_module_available`
`def _module_available(import_name: str) -> bool`
### `_path_ready`
`def _path_ready(path: Path, *, expect_file: bool) -> tuple[HealthStatus, str]`
### `_collect_config_checks`
`def _collect_config_checks(config) -> list[HealthCheck]`
### `_ollama_check`
`def _ollama_check(base_url: str) -> HealthCheck`
### `run_startup_health_check`
`def run_startup_health_check(controller, verbose: bool=False) -> HealthReport`
### `run_lightweight_health_check`
`def run_lightweight_health_check(config) -> HealthReport`