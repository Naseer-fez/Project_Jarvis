# API Analyst Report: desktop\observation.py

## Dependencies
- `from __future__ import annotations`
- `import hashlib`
- `import inspect`
- `from pathlib import Path`
- `from typing import Any`
- `from typing import Callable`
- `from core.desktop.contracts import DesktopChange`
- `from core.desktop.contracts import DesktopObservation`
- `from core.desktop.contracts import ScreenTarget`

## Schemas & API Contracts (Classes)

### Class `DesktopObserver`
> Capture normalized evidence about the current desktop state.

**Methods:**
- `def __init__(self, *, capture_screen: ObservationHandler | None=None, active_window: ObservationHandler | None=None, ocr: ObservationHandler | None=None) -> None`
- `async def observe(self, label: str='') -> DesktopObservation`
- `def compare(self, before: DesktopObservation | None, after: DesktopObservation | None) -> DesktopChange`
- @staticmethod
- `def _default_capture_screen() -> Any`
- @staticmethod
- `def _default_active_window() -> Any`
- @staticmethod
- `def _default_ocr() -> Any`


## Functions & Endpoints

### `_maybe_await`
`async def _maybe_await(value: Any) -> Any`
### `_result_success`
`def _result_success(result: Any) -> bool`
### `_result_error`
`def _result_error(result: Any) -> str`
### `_result_payload`
`def _result_payload(result: Any) -> dict[str, Any]`
### `_hash_path`
`def _hash_path(path_value: str) -> str`
### `_target_from_payload`
`def _target_from_payload(payload: dict[str, Any]) -> ScreenTarget | None`