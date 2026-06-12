# API Analyst Report: desktop\actions.py

## Dependencies
- `from __future__ import annotations`
- `import inspect`
- `import time`
- `from typing import Any`
- `from typing import Callable`
- `from core.autonomy.risk_evaluator import RiskEvaluator`
- `from core.autonomy.risk_evaluator import RiskLevel`
- `from core.autonomy.risk_evaluator import RiskResult`
- `from core.desktop.contracts import DesktopAction`
- `from core.desktop.contracts import DesktopActionResult`
- `from core.desktop.contracts import DesktopActionStatus`
- `from core.desktop.contracts import DesktopActionType`
- `from core.desktop.contracts import DesktopRiskTier`

## Configuration Variables
- `_SENSITIVE_TEXT_MARKERS` = `('password', 'passwd', 'secret', 'token', 'api_key', 'apikey', 'private key')`

## Schemas & API Contracts (Classes)

### Class `DesktopActionExecutor`
> Execute every desktop operation through one action contract.

**Methods:**
- `def __init__(self, *, risk_evaluator: RiskEvaluator | None=None, audit_writer: Callable[[str, dict[str, Any]], str] | None=None, action_handlers: dict[str | DesktopActionType, ActionHandler] | None=None) -> None`
- `def evaluate_risk(self, action: DesktopAction) -> tuple[DesktopRiskTier, RiskResult]`
- `def requires_approval(self, action: DesktopAction) -> bool`
- `async def execute(self, action: DesktopAction, *, approved: bool | None=None) -> DesktopActionResult`
- `def _audit(self, action: DesktopAction, result: DesktopActionResult) -> DesktopActionResult`
- @staticmethod
- `def _contains_sensitive_text(action: DesktopAction) -> bool`
- @staticmethod
- `def _result(action: DesktopAction, *, started_at: float, risk_tier: DesktopRiskTier, status: DesktopActionStatus, success: bool, output: str='', error: str='', metadata: dict[str, Any] | None=None) -> DesktopActionResult`
- @staticmethod
- `def _default_handlers() -> dict[str, ActionHandler]`


## Functions & Endpoints

### `_maybe_await`
`async def _maybe_await(value: Any) -> Any`
### `_stringify`
`def _stringify(value: Any) -> str`
### `_normalize_tool_result`
`def _normalize_tool_result(result: Any) -> tuple[bool, str, str, dict[str, Any]]`
### `_launch_application`
`async def _launch_application(target: str | None=None, args: list[str] | None=None, application: str | None=None) -> Any`
### `_click`
`async def _click(x: int, y: int, button: str='left') -> Any`
### `_double_click`
`async def _double_click(x: int, y: int) -> Any`
### `_right_click`
`async def _right_click(x: int, y: int) -> Any`
### `_click_text_on_screen`
`async def _click_text_on_screen(text: str, occurrence: int=1, button: str='left', match_mode: str='contains') -> Any`
### `_click_screen_target`
`async def _click_screen_target(target: str, occurrence: int=1, button: str='left', match_mode: str='contains', min_confidence: float=0.2) -> Any`
### `_double_click_screen_target`
`async def _double_click_screen_target(target: str, occurrence: int=1, match_mode: str='contains', min_confidence: float=0.2) -> Any`
### `_right_click_screen_target`
`async def _right_click_screen_target(target: str, occurrence: int=1, match_mode: str='contains', min_confidence: float=0.2) -> Any`
### `_type_text`
`async def _type_text(text: str, interval: float=0.05) -> Any`
### `_press_key`
`async def _press_key(key: str, presses: int=1, interval: float=0.05) -> Any`
### `_hotkey`
`async def _hotkey(keys: list[str] | tuple[str, ...] | str) -> Any`
### `_move_mouse`
`async def _move_mouse(x: int, y: int, duration: float=0.0) -> Any`
### `_scroll`
`async def _scroll(clicks: int, x: int | None=None, y: int | None=None) -> Any`
### `_drag`
`async def _drag(start_x: int, start_y: int, end_x: int, end_y: int, duration: float=0.2, button: str='left') -> Any`
### `_focus_window`
`async def _focus_window(title: str) -> Any`
### `_clipboard_get`
`async def _clipboard_get() -> Any`
### `_clipboard_set`
`async def _clipboard_set(text: str) -> Any`
### `_clipboard_paste`
`async def _clipboard_paste() -> Any`