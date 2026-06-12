# Analysis Report for gui_control.py

## Dependencies
- __future__.annotations
- core.types.common.ToolResult
- asyncio
- configparser
- json
- logging
- re
- time
- pathlib.Path
- typing.Any

## Schemas
None

## API Contracts
- _save_audit_screenshot(label)
- _validate_coords(x, y)
- _require_pyautogui()
- _contains_sensitive_text(text)
- _tool_result_payload(result)
- _extract_json_object(raw)
- _match_center(match)
- _runtime_config()
- _vision_locate_target(target)
- focus_window(title)
- get_active_window()
- clipboard_get()
- clipboard_set(text)

## Configuration Variables
- GUI_AUDIT_DIR
- _CONFIG_PATH
- _FORBIDDEN_KEYWORDS (typed)
- _LAST_CLICK_TIME (typed)
- _CLICK_SAFETY_DELAY (typed)

## Assumptions & Notes
- Module Docstring: Desktop GUI automation tools for Jarvis.

