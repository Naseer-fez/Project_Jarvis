# API Analyst Report: planner\planner.py

## Dependencies
- `from __future__ import annotations`
- `import json`
- `import logging`
- `import re`
- `import inspect`
- `from typing import Any`
- `from core.autonomy.risk_evaluator import RiskLevel`
- `from core.autonomy.risk_evaluator import RiskEvaluator`

## Configuration Variables
- `_GUI_TOOL_NAMES` = `{'click', 'double_click', 'right_click', 'click_text_on_screen', 'click_screen_target', 'double_click_screen_target', 'right_click_screen_target', 'move_mouse', 'scroll', 'drag', 'type_text', 'press_key', 'hotkey', 'focus_window', 'clipboard_get', 'clipboard_set', 'clipboard_paste'}`

## Schemas & API Contracts (Classes)

### Class `TaskPlanner`
**Methods:**
- `def __init__(self, config=None, llm=None, registry=None) -> None`
- `def _tool_schema(self) -> dict[str, list[dict[str, Any]]]`
- `async def _call_ollama(self, prompt: str) -> str`
- `async def plan(self, user_input: str, context: str='') -> dict[str, Any]`
- `def _build_prompt(self, user_input: str, context: str) -> str`
- `def _parse_llm_plan(self, raw: str) -> dict[str, Any] | None`
- `def _fallback_plan(self, text: str) -> dict[str, Any]`
- `def _clarification_plan(self, text: str) -> dict[str, Any]`
- `def _enrich_plan(self, text: str, plan: dict[str, Any]) -> dict[str, Any]`
- `def _normalize_steps(self, steps: Any) -> list[dict[str, Any]]`


## Functions & Endpoints

### `_strip_planner_artifacts`
`def _strip_planner_artifacts(raw: str) -> str`