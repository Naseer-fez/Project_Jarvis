# clients/computer_control.py API Analyst Report

## Overview
Integration for GUI automation, enabling mouse movement, clicks, typing, and screenshots.

## API Contracts & Methods
- `ComputerControlIntegration(BaseIntegration)`
  - `is_available()`: Checks for `pyautogui` library.

## Tools Exposed
- `move_mouse`
  - **Risk:** `confirm`
  - **Args:** `x` (int), `y` (int)
  - **Behavior:** Moves mouse in 0.5s via `pyautogui.moveTo`.
- `mouse_click`
  - **Risk:** `confirm`
  - **Args:** `x` (int, optional), `y` (int, optional), `button` (str, default "left"), `double` (bool, default False)
- `keyboard_type`
  - **Risk:** `confirm`
  - **Args:** `text` (str), `press_enter` (bool, default False), `interval` (float, default 0.02)
- `take_screenshot`
  - **Risk:** `medium`
  - **Args:** `path` (str, default "outputs/screenshot.png")
  - **Behavior:** Strictly validates that the path resides within the `outputs` directory to prevent arbitrary file writes.

## Assumptions & Constants
- `pyautogui.FAILSAFE` is strictly enabled.
- Assumes an active display/windowing environment.

## Dependencies
- `pyautogui`

## Prompts
- None.
