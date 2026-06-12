# API Analyst Report: permission_matrix.py

## Dependencies
- `from __future__ import annotations`
- `from dataclasses import dataclass`
- `from dataclasses import field`

## Schemas & API Contracts (Classes)

### Class `PermissionResult`
**Fields/Schema:**
  - `blocked_actions: list[str]`
  - `confirmation_actions: list[str]`

**Methods:**
- @property
- `def has_blocked(self) -> bool`
- @property
- `def needs_confirmation(self) -> bool`


### Class `PermissionMatrix`
**Methods:**
- `def __init__(self, config=None) -> None`
- `def evaluate(self, actions: list[str]) -> PermissionResult`
- `def _parse_csv(self, section: str, key: str) -> set[str]`

