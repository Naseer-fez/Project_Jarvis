# API Analyst Report: registry\base.py

## Dependencies
- `from __future__ import annotations`
- `from abc import ABC`
- `from abc import abstractmethod`
- `from enum import Enum`
- `from typing import Any`
- `from core.capability.base import ToolObservation`
- `from core.context.context import TaskExecutionContext`

## Configuration Variables
- `LOW` = `'low'`
- `MEDIUM` = `'medium'`
- `CONFIRM` = `'confirm'`
- `HIGH` = `'high'`
- `CRITICAL` = `'critical'`

## Schemas & API Contracts (Classes)

### Class `RiskLevel(Enum)`


### Class `Capability(ABC)`
> Abstract base class for all tools and integrations.

**Methods:**
- @property @abstractmethod
- `def name(self) -> str`
  - *Unique identifier of the capability.*
- @property @abstractmethod
- `def is_write_operation(self) -> bool`
  - *True if the tool mutates state, files, or sends outbound payloads.*
- @property @abstractmethod
- `def risk_level(self) -> RiskLevel`
  - *The tool risk profile (e.g. LOW, CRITICAL).*
- @property @abstractmethod
- `def schema(self) -> dict[str, Any]`
  - *JSON schema defining the expected arguments.*
- @abstractmethod
- `async def run(self, args: dict[str, Any], context: TaskExecutionContext) -> ToolObservation`
  - *Asynchronous, non-blocking execution callback.*

