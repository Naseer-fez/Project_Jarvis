# API Analyst Report: logging\logger.py

## Dependencies
- `from __future__ import annotations`
- `import atexit`
- `import hashlib`
- `import json`
- `import logging`
- `import logging.handlers`
- `import queue`
- `import re`
- `import threading`
- `from pathlib import Path`
- `from typing import Any`
- `import contextvars`

## Configuration Variables
- `_MANAGED_STREAM_HANDLER_NAME` = `'jarvis_stream'`
- `_MANAGED_FILE_HANDLER_NAME` = `'jarvis_app_file'`

## Schemas & API Contracts (Classes)

### Class `AuditLog`
**Methods:**
- `def __init__(self, file_path: str) -> None`
- `def _start_worker(self) -> None`
- `def _write_worker(self) -> None`
- `def write(self, event_type: str, payload: dict[str, Any]) -> str`
- `def stop(self) -> None`
- `def verify(self) -> tuple[bool, int, str]`


### Class `JSONFormatter(logging.Formatter)`
**Methods:**
- `def format(self, record: logging.LogRecord) -> str`


### Class `FlushingQueueListener(logging.handlers.QueueListener)`
**Methods:**
- `def handle(self, record: logging.LogRecord) -> None`
- `def flush(self) -> None`


### Class `JarvisQueueHandler(logging.handlers.QueueHandler)`
**Methods:**
- `def prepare(self, record: logging.LogRecord) -> logging.LogRecord`


## Functions & Endpoints

### `set_trace_ids`
`def set_trace_ids(trace_id: str | None, task_id: str | None) -> tuple[contextvars.Token[str | None], contextvars.Token[str | None]]`
> Set correlation IDs for the current async context.

### `reset_trace_ids`
`def reset_trace_ids(trace_token: contextvars.Token[str | None], task_token: contextvars.Token[str | None]) -> None`
> Restore correlation IDs for the current async context.

### `redact_sensitive_data`
`def redact_sensitive_data(val: Any) -> Any`
> Recursively redact sensitive patterns (passwords, secrets, tokens) from metadata and strings.

### `_build_formatter`
`def _build_formatter() -> logging.Formatter`
### `_find_managed_handler`
`def _find_managed_handler(name: str) -> logging.Handler | None`
### `_resolve_config_path`
`def _resolve_config_path(path_value: str) -> Path`
### `setup`
`def setup(config=None) -> None`
### @atexit.register
`def cleanup_logging() -> None`
### `get`
`def get() -> logging.Logger`
### `get_logger`
`def get_logger(name: str | None=None) -> logging.Logger`
### `audit`
`def audit(event_type: str, payload: dict[str, Any]) -> str`
### `verify_audit`
`def verify_audit() -> tuple[bool, int, str]`
### `flush`
`def flush() -> None`
## Assumptions & Notes
- Line 364: # Reduce spam from third-party libraries by defaulting root logger to WARNING.
- Line 372: # avoiding the root logger's WARNING suppression.
