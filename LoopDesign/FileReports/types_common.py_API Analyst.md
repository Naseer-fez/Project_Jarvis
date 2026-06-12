# API Analyst Report: types\common.py

## Dependencies
- `from __future__ import annotations`
- `from dataclasses import dataclass`
- `from dataclasses import field`
- `from enum import Enum`
- `from typing import Any`

## Configuration Variables
- `READ_ONLY` = `'READ_ONLY_TOOLS'`
- `CONFIRM` = `'CONFIRM_TOOLS'`
- `HIGH_RISK` = `'HIGH_RISK_TOOLS'`

## Schemas & API Contracts (Classes)

### Class `ToolResult`
> Standardised return type for all Jarvis tool functions.

Attributes:
    success: True if the tool call succeeded.
    data:    Payload on success (arbitrary dict).
    error:   Human-readable error message on failure.
    tool_name: (Optional) Name of the tool.

**Fields/Schema:**
  - `success: bool`
  - `data: dict[str, Any]`
  - `error: str`
  - `tool_name: str`

**Methods:**
- `def to_llm_string(self) -> str`
- `def __repr__(self) -> str`


### Class `IntegrationRiskLevel(str, Enum)`

