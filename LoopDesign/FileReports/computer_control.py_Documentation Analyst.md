# Documentation Report: clients/computer_control.py

## Assumptions
- Uses `pyautogui` for all interaction (mouse, keyboard, screenshot).
- Explicitly enables `pyautogui.FAILSAFE` which aborts if mouse is thrown to the corner of the screen.
- Screenshots are forcibly constrained to be inside an `outputs` directory.
- `keyboard_type` defaults to `interval=0.02` per key.

## Schema / API Contract
- Tool: `move_mouse(x: int, y: int)`
- Tool: `mouse_click(x?: int, y?: int, button?: str, double?: bool)`
- Tool: `keyboard_type(text: str, press_enter?: bool, interval?: float)`
- Tool: `take_screenshot(path?: str)` returns absolute path to saved file

## Dependencies
- `pyautogui` (external)
- `asyncio`, `os`, `logging` (stdlib)

## Configuration Variables
None natively, relies on host display server constraints.

## Prompts
None.
