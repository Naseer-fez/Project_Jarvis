# API Analyst Report: tools\gui_control.py

## Dependencies
- `from __future__ import annotations`
- `from core.types.common import ToolResult`
- `import asyncio`
- `import configparser`
- `import json`
- `import logging`
- `import re`
- `import time`
- `from pathlib import Path`
- `from typing import Any`

## Configuration Variables
- `GUI_AUDIT_DIR` = `Path('outputs/gui_audit')`
- `_CONFIG_PATH` = `Path('config/jarvis.ini')`
- `_LAST_CLICK_TIME` = `time.time()`

## Functions & Endpoints

### `_save_audit_screenshot`
`def _save_audit_screenshot(label: str) -> str`
### `_validate_coords`
`def _validate_coords(x: int, y: int) -> bool`
### `_require_pyautogui`
`def _require_pyautogui()`
### `_contains_sensitive_text`
`def _contains_sensitive_text(text: str) -> str | None`
### `_tool_result_payload`
`def _tool_result_payload(result: Any) -> dict[str, Any]`
### `_extract_json_object`
`def _extract_json_object(raw: str) -> dict[str, Any]`
### `_match_center`
`def _match_center(match: dict[str, Any]) -> tuple[int, int]`
### `_runtime_config`
`def _runtime_config() -> configparser.ConfigParser`
### `_vision_locate_target`
`def _vision_locate_target(target: str)`
### `click`
`async def click(x: int, y: int, button: str='left')`
> Click at screen coordinates with bounds checks and audit capture.

### `double_click`
`async def double_click(x: int, y: int)`
> Double-click at screen coordinates.

### `right_click`
`async def right_click(x: int, y: int)`
> Right-click at screen coordinates.

### `click_text_on_screen`
`async def click_text_on_screen(text: str, *, occurrence: int=1, button: str='left', match_mode: str='contains')`
> Locate visible text on screen and click its center.

### `_resolve_target_coordinates`
`async def _resolve_target_coordinates(target: str, *, occurrence: int=1, match_mode: str='contains', min_confidence: float=0.2)`
> Resolve coordinates for a described screen target, trying OCR first, then Vision.
Returns:
    (x, y, resolved_metadata, error_result)
    If resolved successfully: (x, y, metadata, None)
    If failed: (None, None, None, ToolResult)

### `click_screen_target`
`async def click_screen_target(target: str, *, occurrence: int=1, button: str='left', match_mode: str='contains', min_confidence: float=0.2)`
> Click a described screen target without hard-coded coordinates.

### `double_click_screen_target`
`async def double_click_screen_target(target: str, *, occurrence: int=1, match_mode: str='contains', min_confidence: float=0.2)`
> Double-click a described screen target without hard-coded coordinates.

### `right_click_screen_target`
`async def right_click_screen_target(target: str, *, occurrence: int=1, match_mode: str='contains', min_confidence: float=0.2)`
> Right-click a described screen target without hard-coded coordinates.

### `move_mouse`
`async def move_mouse(x: int, y: int, duration: float=0.0)`
### `scroll`
`async def scroll(clicks: int, x: int | None=None, y: int | None=None)`
### `drag`
`async def drag(start_x: int, start_y: int, end_x: int, end_y: int, *, duration: float=0.2, button: str='left')`
### `type_text`
`async def type_text(text: str, interval: float=0.05)`
### `press_key`
`async def press_key(key: str, presses: int=1, interval: float=0.05)`
### `hotkey`
`async def hotkey(*keys: str)`
### `focus_window`
`def focus_window(title: str)`
### `get_active_window`
`def get_active_window()`
### `clipboard_get`
`def clipboard_get()`
### `clipboard_set`
`def clipboard_set(text: str)`
### `clipboard_paste`
`async def clipboard_paste()`
### `get_mouse_position`
`async def get_mouse_position()`
### `mouse_down`
`async def mouse_down(button: str='left')`
### `mouse_up`
`async def mouse_up(button: str='left')`
### `key_down`
`async def key_down(key: str)`
### `key_up`
`async def key_up(key: str)`
### `middle_click`
`async def middle_click(x: int, y: int)`
> Middle-click at screen coordinates.
