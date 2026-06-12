# `computer_control.py` - API Analyst Report

## Overview
Provides control over the host's mouse, keyboard, and screen for UI automation using `pyautogui`.

## Endpoints / Tools
1. `move_mouse`
   - Description: Move mouse to absolute screen coordinates.
   - Risk: confirm (write)
   - Arguments: `x` (integer, required), `y` (integer, required).
2. `mouse_click`
   - Description: Click the mouse at the current or optional absolute coordinates.
   - Risk: confirm (write)
   - Arguments: `x` (integer), `y` (integer), `button` (string, default "left"), `double` (boolean, default False).
3. `keyboard_type`
   - Description: Type text rapidly using the keyboard.
   - Risk: confirm (write)
   - Arguments: `text` (string, required), `press_enter` (boolean, default False), `interval` (number, default 0.02).
4. `take_screenshot`
   - Description: Take a screenshot of the main display.
   - Risk: medium
   - Arguments: `path` (string, default "outputs/screenshot.png").

## External Contracts / Dependencies
- Relies heavily on the `pyautogui` library for all OS interactions.

## Assumptions
- Uses `pyautogui.FAILSAFE = True` which aborts the script if the user moves the mouse to a corner of the screen.
- Screen coordinate space assumes an origin (0, 0) at the top-left corner.
- For `take_screenshot`, the `path` must be within the absolute `outputs/` directory to avoid directory traversal vulnerabilities.
- `pyautogui` operations block the thread, so they are executed within `asyncio.get_running_loop().run_in_executor` to prevent blocking the async loop.
- Default path for screenshot assumes `outputs/` directory relative to current working directory.
