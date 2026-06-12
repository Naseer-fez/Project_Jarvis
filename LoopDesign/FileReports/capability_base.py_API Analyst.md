# API Analyst Report: capability\base.py

## Dependencies
- `from __future__ import annotations`
- `import logging`
- `from dataclasses import dataclass`
- `from typing import Any`
- `from typing import Optional`
- `from core.autonomy.risk_evaluator import RiskLevel`
- `from core.context.context import TaskExecutionContext`

## Schemas & API Contracts (Classes)

### Class `Capability`
> Base class for all tools and capabilities in Jarvis.

**Fields/Schema:**
  - `name: str`
  - `description: str`
  - `risk_level: RiskLevel`
  - `is_write: bool`

**Methods:**
- @property
- `def is_write_operation(self) -> bool`
  - *Alias for is_write, conforming to the abstract Capability base interface.*
- `async def run(self, args: dict[str, Any], context: TaskExecutionContext) -> ToolObservation`
  - *Execute the capability logic in the provided task context.*


### Class `ToolObservation`
**Fields/Schema:**
  - `tool_name: str`
  - `arguments: dict`
  - `execution_status: str`
  - `output_summary: str`
  - `error_message: Optional[str]`
  - `duration_seconds: float`
  - `metadata: dict[str, Any] | None`

**Methods:**
- `def to_dict(self) -> dict`


## Functions & Endpoints

### `_normalize_tool_result`
`def _normalize_tool_result(result: Any) -> tuple[bool, str, str]`
### `_first_non_empty`
`def _first_non_empty(*values: Any) -> str`
### `_stringify_payload`
`def _stringify_payload(value: Any) -> str`