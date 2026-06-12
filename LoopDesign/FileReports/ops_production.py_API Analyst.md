# API Analyst Report: ops\production.py

## Dependencies
- `from __future__ import annotations`
- `import os`
- `from dataclasses import dataclass`
- `from dataclasses import field`
- `from typing import Any`

## Configuration Variables
- `PUBLIC_HOSTS` = `{'0.0.0.0', '::', '[::]'}`
- `DANGEROUS_ENV_FLAGS` = `{'allow_gui_automation': 'JARVIS_ENABLE_GUI_AUTOMATION', 'allow_shell_execution': 'JARVIS_ENABLE_SHELL', 'hardware_enabled': 'JARVIS_ENABLE_HARDWARE'}`

## Schemas & API Contracts (Classes)

### Class `ProductionCheck`
**Fields/Schema:**
  - `errors: list[str]`
  - `warnings: list[str]`

**Methods:**
- @property
- `def ok(self) -> bool`


## Functions & Endpoints

### `_get`
`def _get(config: Any, section: str, key: str, fallback: str='') -> str`
### `_get_bool`
`def _get_bool(config: Any, section: str, key: str, fallback: bool=False) -> bool`
### `is_production`
`def is_production(config: Any) -> bool`
### `validate_production_config`
`def validate_production_config(config: Any, *, dashboard_enabled: bool=False) -> ProductionCheck`