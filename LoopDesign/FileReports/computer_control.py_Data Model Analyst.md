# File Report: computer_control.py
**Path**: `d:\AI\Jarvis\integrations\clients\computer_control.py`
**Role**: Data Model Analyst

## Analysis Summary
This file has been analyzed for schemas, DTOs, state objects, config variables, and dependencies.

## Dependencies
- __future__.annotations
- asyncio
- logging
- os
- typing.Any
- integrations.base.BaseIntegration
- pyautogui
- pyautogui

## Classes and State Objects
### `ComputerControlIntegration`
**Variables**: name, description, required_config
**Methods**: is_available, get_tools, execute

## Tool Schemas / DTOs
```python
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

```

## Assumptions & API Contracts
1. Config vars are expected in environment variables.
2. Schema validation is typically deferred to the registry or client implementation.