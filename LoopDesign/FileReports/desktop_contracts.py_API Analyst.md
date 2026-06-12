# API Analyst Report: desktop\contracts.py

## Dependencies
- `from __future__ import annotations`
- `import time`
- `import uuid`
- `from dataclasses import dataclass`
- `from dataclasses import field`
- `from enum import Enum`
- `from typing import Any`

## Configuration Variables
- `LAUNCH_APP` = `'launch_application'`
- `FOCUS_WINDOW` = `'focus_window'`
- `MOVE_MOUSE` = `'move_mouse'`
- `CLICK` = `'click'`
- `DOUBLE_CLICK` = `'double_click'`
- `RIGHT_CLICK` = `'right_click'`
- `CLICK_TEXT_ON_SCREEN` = `'click_text_on_screen'`
- `CLICK_SCREEN_TARGET` = `'click_screen_target'`
- `DOUBLE_CLICK_SCREEN_TARGET` = `'double_click_screen_target'`
- `RIGHT_CLICK_SCREEN_TARGET` = `'right_click_screen_target'`
- `SCROLL` = `'scroll'`
- `DRAG` = `'drag'`
- `TYPE_TEXT` = `'type_text'`
- `PRESS_KEY` = `'press_key'`
- `HOTKEY` = `'hotkey'`
- `CLIPBOARD_GET` = `'clipboard_get'`
- `CLIPBOARD_SET` = `'clipboard_set'`
- `CLIPBOARD_PASTE` = `'clipboard_paste'`
- `LOW` = `'low'`
- `MEDIUM` = `'medium'`
- `CONFIRM` = `'confirm'`
- `HIGH` = `'high'`
- `BLOCKED` = `'blocked'`
- `SUCCESS` = `'success'`
- `FAILURE` = `'failure'`
- `BLOCKED` = `'blocked'`
- `NEEDS_APPROVAL` = `'needs_approval'`

## Schemas & API Contracts (Classes)

### Class `DesktopActionType(str, Enum)`


### Class `DesktopRiskTier(str, Enum)`


### Class `DesktopActionStatus(str, Enum)`


### Class `DesktopAction`
**Fields/Schema:**
  - `action_type: DesktopActionType | str`
  - `params: dict[str, Any]`
  - `description: str`
  - `expected_change: str`
  - `risk_tier: DesktopRiskTier | str | None`
  - `requires_approval: bool | None`
  - `action_id: str`
  - `metadata: dict[str, Any]`

**Methods:**
- @property
- `def action_name(self) -> str`
- `def to_dict(self) -> dict[str, Any]`


### Class `DesktopActionResult`
**Fields/Schema:**
  - `action_id: str`
  - `action_type: str`
  - `success: bool`
  - `status: DesktopActionStatus`
  - `output: str`
  - `error: str`
  - `risk_tier: DesktopRiskTier`
  - `audit_hash: str`
  - `started_at: float`
  - `ended_at: float`
  - `metadata: dict[str, Any]`

**Methods:**
- @property
- `def duration_seconds(self) -> float`
- `def to_dict(self) -> dict[str, Any]`


### Class `ScreenTarget`
**Fields/Schema:**
  - `label: str`
  - `x: int`
  - `y: int`
  - `width: int`
  - `height: int`
  - `confidence: float`
  - `metadata: dict[str, Any]`

**Methods:**
- `def to_dict(self) -> dict[str, Any]`


### Class `DesktopObservation`
**Fields/Schema:**
  - `observation_id: str`
  - `screenshot_path: str`
  - `screenshot_fingerprint: str`
  - `active_window: dict[str, Any]`
  - `ocr_text: str`
  - `targets: list[ScreenTarget]`
  - `confidence: float`
  - `low_confidence_reason: str`
  - `metadata: dict[str, Any]`
  - `captured_at: float`

**Methods:**
- `def to_dict(self) -> dict[str, Any]`


### Class `DesktopChange`
**Fields/Schema:**
  - `changed: bool`
  - `confidence: float`
  - `summary: str`
  - `before_observation_id: str`
  - `after_observation_id: str`
  - `metadata: dict[str, Any]`

**Methods:**
- `def to_dict(self) -> dict[str, Any]`


### Class `ApprovalDecision`
**Fields/Schema:**
  - `required: bool`
  - `approved: bool`
  - `reason: str`
  - `mode: str`

**Methods:**
- `def to_dict(self) -> dict[str, Any]`


## Functions & Endpoints

### `_new_id`
`def _new_id(prefix: str) -> str`