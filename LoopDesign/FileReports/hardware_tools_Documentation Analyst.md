# Analysis Report for hardware_tools.py

## Dependencies
- __future__.annotations
- core.types.common.ToolResult
- logging

## Schemas
None

## API Contracts
- _get_registry()

## Configuration Variables
None

## Assumptions & Notes
- Module Docstring: core/tools/hardware_tools.py
------------------------------
Async tool functions for interacting with registered hardware devices via
the DeviceRegistry / SerialController layer.

All functions return a ToolResult so they integrate cleanly with the ToolRouter.

