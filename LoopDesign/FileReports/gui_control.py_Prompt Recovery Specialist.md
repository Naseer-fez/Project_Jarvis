# File Report: gui_control.py
**Role**: Prompt Recovery Specialist

## Dependencies
- pygetwindow
- pyperclip
- pyautogui
- time
- re
- typing
- core.types.common
- json
- asyncio
- logging
- core.tools.screen
- __future__
- core.tools.vision
- configparser
- pathlib

## Configuration Variables & Constants
- `prompt`: (Too long, 279 chars. Extracted to Prompts if applicable)
- `text_error`: ``
- `error`: `f-string: {...} Vision fallback: {...}`

## Schemas & API Contracts
### Function `_save_audit_screenshot`
**Args**: label

### Function `_validate_coords`
**Args**: x, y

### Function `_require_pyautogui`
**Args**: 

### Function `_contains_sensitive_text`
**Args**: text

### Function `_tool_result_payload`
**Args**: result

### Function `_extract_json_object`
**Args**: raw

### Function `_match_center`
**Args**: match

### Function `_runtime_config`
**Args**: 

### Function `_vision_locate_target`
**Args**: target

### Function `click`
**Args**: x, y, button
**Assumptions/Doc**: Click at screen coordinates with bounds checks and audit capture.

### Function `double_click`
**Args**: x, y
**Assumptions/Doc**: Double-click at screen coordinates.

### Function `right_click`
**Args**: x, y
**Assumptions/Doc**: Right-click at screen coordinates.

### Function `click_text_on_screen`
**Args**: text
**Assumptions/Doc**: Locate visible text on screen and click its center.

### Function `_resolve_target_coordinates`
**Args**: target
**Assumptions/Doc**: Resolve coordinates for a described screen target, trying OCR first, then Vision.
Returns:
    (x, y, resolved_metadata, error_result)
    If resolved successfully: (x, y, metadata, None)
    If failed: (None, None, None, ToolResult)

### Function `click_screen_target`
**Args**: target
**Assumptions/Doc**: Click a described screen target without hard-coded coordinates.

### Function `double_click_screen_target`
**Args**: target
**Assumptions/Doc**: Double-click a described screen target without hard-coded coordinates.

### Function `right_click_screen_target`
**Args**: target
**Assumptions/Doc**: Right-click a described screen target without hard-coded coordinates.

### Function `move_mouse`
**Args**: x, y, duration

### Function `scroll`
**Args**: clicks, x, y

### Function `drag`
**Args**: start_x, start_y, end_x, end_y

### Function `type_text`
**Args**: text, interval

### Function `press_key`
**Args**: key, presses, interval

### Function `hotkey`
**Args**: 

### Function `focus_window`
**Args**: title

### Function `get_active_window`
**Args**: 

### Function `clipboard_get`
**Args**: 

### Function `clipboard_set`
**Args**: text

### Function `clipboard_paste`
**Args**: 

### Function `get_mouse_position`
**Args**: 

### Function `mouse_down`
**Args**: button

### Function `mouse_up`
**Args**: button

### Function `key_down`
**Args**: key

### Function `key_up`
**Args**: key

### Function `middle_click`
**Args**: x, y
**Assumptions/Doc**: Middle-click at screen coordinates.

## Prompts and LLM Directives
- Extracted `prompt` to Prompts directory.
- Extracted `error` to Prompts directory.
