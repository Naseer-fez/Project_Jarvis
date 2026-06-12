# API Analyst Report: autonomy\autonomy_governor.py

## Dependencies
- `import logging`
- `import threading`
- `from enum import IntEnum`
- `from typing import Any`

## Configuration Variables
- `CHAT_ONLY` = `0`
- `SUGGEST_ONLY` = `1`
- `READ_ONLY` = `2`
- `WRITE_WITH_CONFIRM` = `3`
- `AUTONOMOUS` = `4`

## Schemas & API Contracts (Classes)

### Class `AutonomyLevel(IntEnum)`


### Class `AutonomyGovernor`
**Methods:**
- `def __init__(self, level: int=1, registry: Any=None)`
- `def register_read_only_tool(self, tool_name: str) -> None`
  - *Dynamically register a tool as read-only.*
- `def register_write_tool(self, tool_name: str) -> None`
  - *Dynamically register a tool as a write tool.*
- `def _is_known_tool(self, tool_name: str) -> bool`
- `def _is_write_tool(self, tool_name: str) -> bool`
- `def can_execute(self, tool_name: str) -> tuple[bool, str]`
  - *Returns (allowed: bool, reason: str).*
- `def requires_confirmation(self, tool_name: str) -> bool`
  - *Write tools at LEVEL_3 always need explicit user confirmation.*
- `def escalate(self, new_level: int) -> bool`
  - *Temporarily escalate autonomy (user must consent upstream).*
- `def describe(self) -> str`

