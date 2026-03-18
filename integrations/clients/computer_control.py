from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from integrations.base import BaseIntegration

logger = logging.getLogger(__name__)


class ComputerControlIntegration(BaseIntegration):
    name = "computer_control"
    description = "Control the mouse, keyboard, and screen for UI automation."
    required_config = []

    def is_available(self) -> bool:
        try:
            import pyautogui
            _ = pyautogui.FAILSAFE  # Use it to avoid F401
            return True
        except ImportError:
            return False

    def get_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "move_mouse",
                "description": "Move the mouse to absolute screen coordinates (x, y).",
                "risk": "confirm",
                "args": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                },
                "required_args": ["x", "y"],
            },
            {
                "name": "mouse_click",
                "description": "Click the mouse at the current position or optional absolute screen coordinates.",
                "risk": "confirm",
                "args": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                    "button": {"type": "string", "default": "left"},
                    "double": {"type": "boolean", "default": False},
                },
                "required_args": [],
            },
            {
                "name": "keyboard_type",
                "description": "Type text rapidly using the keyboard.",
                "risk": "confirm",
                "args": {
                    "text": {"type": "string"},
                    "press_enter": {"type": "boolean", "default": False},
                    "interval": {"type": "number", "default": 0.02},
                },
                "required_args": ["text"],
            },
            {
                "name": "take_screenshot",
                "description": "Take a screenshot of the main display.",
                "risk": "medium",
                "args": {
                    "path": {"type": "string", "description": "Optional output path.", "default": "outputs/screenshot.png"},
                },
                "required_args": [],
            },
        ]

    async def execute(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        import pyautogui

        # PyAutoGUI has a built-in failsafe (moving mouse to corner of screen aborts)
        pyautogui.FAILSAFE = True

        args = args or {}
        loop = asyncio.get_running_loop()
        try:
            if tool_name == "move_mouse":
                await loop.run_in_executor(None, pyautogui.moveTo, args["x"], args["y"], 0.5)
                return {"success": True, "data": f"Moved to {args['x']}, {args['y']}", "error": None}

            if tool_name == "mouse_click":
                x = args.get("x")
                y = args.get("y")
                clicks = 2 if args.get("double") else 1
                button = str(args.get("button", "left") or "left")

                def _click() -> None:
                    if x is not None and y is not None:
                        pyautogui.click(int(x), int(y), button=button, clicks=clicks)
                    else:
                        pyautogui.click(button=button, clicks=clicks)

                await loop.run_in_executor(None, _click)
                location = f" at {int(x)}, {int(y)}" if x is not None and y is not None else ""
                return {"success": True, "data": f"Clicked{location}", "error": None}

            if tool_name == "keyboard_type":
                interval = float(args.get("interval", 0.02) or 0.02)
                await loop.run_in_executor(None, lambda: pyautogui.write(args["text"], interval=interval))
                if args.get("press_enter"):
                    await loop.run_in_executor(None, lambda: pyautogui.press("enter"))
                return {"success": True, "data": "Typed text", "error": None}

            if tool_name == "take_screenshot":
                path = os.path.abspath(str(args.get("path", "outputs/screenshot.png") or "outputs/screenshot.png"))
                os.makedirs(os.path.dirname(path), exist_ok=True)
                await loop.run_in_executor(None, lambda: pyautogui.screenshot(path))
                return {"success": True, "data": f"Screenshot saved to {path}", "error": None}

            return {"success": False, "data": None, "error": f"Unknown tool: {tool_name}"}
        except Exception as exc:
            return {"success": False, "data": None, "error": str(exc)}
