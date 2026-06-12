# API Analyst Report: runtime\import_validator.py

## Dependencies
- `from __future__ import annotations`
- `import ast`
- `import importlib`
- `import importlib.util`
- `import logging`
- `import os`
- `import sys`
- `from dataclasses import dataclass`
- `from pathlib import Path`
- `from typing import Any`
- `from typing import Callable`
- `from typing import TypeVar`
- `from typing import cast`

## Configuration Variables
- `F` = `TypeVar('F', bound=Callable[..., Any])`
- `CRITICAL_MODULES` = `['core.controller_v2', 'core.controller.intents', 'core.controller.intent_router', 'core.controller.services', 'core.controller.request_rules', 'core.controller.web_search', 'core.tools.builtin_tools', 'core.tools.web_tools', 'core.memory.hybrid_memory', 'core.memory.semantic_memory', 'core.runtime.paths', 'core.runtime.bootstrap']`

## Schemas & API Contracts (Classes)

### Class `FallbackMock`
> Mock object that acts as a fallback for missing modules, logging warnings when accessed.

**Methods:**
- `def __init__(self, name: str, reason: str='') -> None`
- `def __getattr__(self, item: str) -> Any`
- `def __call__(self, *args: Any, **kwargs: Any) -> Any`


### Class `ImportDiagnostic`
**Fields/Schema:**
  - `file_path: Path`
  - `line_number: int`
  - `raw_import_string: str`
  - `target_module: str`
  - `is_relative: bool`
  - `status: str`
  - `error_message: str`



### Class `DependencyScanner`
> Scans codebase python files, extracts imports via AST, and validates they resolve.

**Methods:**
- `def __init__(self, root_dir: Path) -> None`
- `def scan_project(self) -> list[ImportDiagnostic]`
- `def _scan_file(self, file_path: Path) -> list[ImportDiagnostic]`
- `def _validate_import(self, file_path: Path, lineno: int, module_name: str, raw_str: str, is_relative: bool, level: int=0) -> ImportDiagnostic`


### Class `StartupValidator`
> Verifies that all core controllers, tools, and memory submodules can resolve imports.

**Methods:**
- `def __init__(self, root_dir: Path) -> None`
- `def run_preflight_checks(self) -> dict[str, Any]`
  - *Perform preflight checks on critical imports and returns a summary health report.*
- `def generate_dependency_graph(self) -> dict[str, list[str]]`
  - *Static AST scan to map modules to their dependencies.*


## Functions & Endpoints

### `safe_import`
`def safe_import(module_name: str, fallback_obj: Any=None) -> Any`
> Attempt to import a module. Returns the imported module,
or a fallback mock object if the module is missing or fails to load.

### `protect_runtime`
`def protect_runtime(fallback_value: Any=None) -> Callable[[F], F]`
> Decorator to wrap sync or async functions in a runtime safety boundary.
Catches all exceptions, logs them, and returns a fallback value instead of crashing.

### `run_diagnostics`
`def run_diagnostics(root_dir: Path) -> None`
> Runs a complete diagnostics scan and prints it out.
