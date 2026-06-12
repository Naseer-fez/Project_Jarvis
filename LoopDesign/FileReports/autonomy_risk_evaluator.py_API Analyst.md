# API Analyst Report: autonomy\risk_evaluator.py

## Dependencies
- `from __future__ import annotations`
- `import threading`
- `from dataclasses import dataclass`
- `from dataclasses import field`
- `from enum import IntEnum`
- `from typing import Sequence`
- `from typing import Any`

## Configuration Variables
- `LOW` = `0`
- `MEDIUM` = `1`
- `CONFIRM` = `2`
- `HIGH` = `3`
- `CRITICAL` = `4`
- `FORBIDDEN` = `4`

## Schemas & API Contracts (Classes)

### Class `RiskLevel(IntEnum)`
**Methods:**
- `def label(self) -> str`


### Class `RiskResult`
**Fields/Schema:**
  - `level: RiskLevel`
  - `blocking_actions: list[str]`
  - `confirm_actions: list[str]`
  - `high_risk_actions: list[str]`
  - `reasons: list[str]`

**Methods:**
- @property
- `def is_blocked(self) -> bool`
- @property
- `def requires_confirmation(self) -> bool`
- `def summary(self) -> str`


### Class `RiskEvaluator`
> Evaluates a list of action names into LOW/MEDIUM/CONFIRM/HIGH/CRITICAL.

**Methods:**
- `def __init__(self, config=None, registry: Any=None) -> None`
- `def register_critical_action(self, action: str) -> None`
  - *Dynamically register an action as CRITICAL risk level.*
- `def register_confirm_action(self, action: str) -> None`
  - *Dynamically register an action as CONFIRM risk level.*
- `def register_high_action(self, action: str) -> None`
  - *Dynamically register an action as HIGH risk level.*
- `def register_medium_action(self, action: str) -> None`
  - *Dynamically register an action as MEDIUM risk level.*
- `def register_low_action(self, action: str) -> None`
  - *Dynamically register an action as LOW risk level.*
- `def _load_config(self, config) -> None`
- `def evaluate(self, actions: Sequence[str]) -> RiskResult`
- `def evaluate_plan(self, plan: dict) -> RiskResult`

