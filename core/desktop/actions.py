"""Normalized desktop action execution with risk and audit metadata."""

from __future__ import annotations

import inspect
import time
from typing import Any, Callable

from core.autonomy.risk_evaluator import RiskEvaluator, RiskLevel, RiskResult
from core.desktop.contracts import (
    DesktopAction,
    DesktopActionResult,
    DesktopActionStatus,
    DesktopActionType,
    DesktopRiskTier,
)


ActionHandler = Callable[..., Any]

_SENSITIVE_TEXT_MARKERS = (
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "private key",
)


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _stringify(value: Any) -> str:
    if value in (None, "", {}, []):
        return ""
    return str(value)


def _normalize_tool_result(result: Any) -> tuple[bool, str, str, dict[str, Any]]:
    if isinstance(result, dict):
        success = bool(result.get("success", False))
        output = _stringify(
            result.get("output")
            or result.get("data")
            or result.get("metadata")
        )
        error = str(result.get("error", "") or "")
        metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
        data = result.get("data") if isinstance(result.get("data"), dict) else {}
        return success, output, error, {"data": data, **metadata}

    success_attr = getattr(result, "success", None)
    if success_attr is None:
        return True, _stringify(result) or "Action completed successfully.", "", {}

    success = bool(success_attr)
    output = _stringify(
        getattr(result, "output", None)
        or getattr(result, "data", None)
        or getattr(result, "metadata", None)
    )
    error = str(getattr(result, "error", "") or "")
    metadata = getattr(result, "metadata", None)
    data = getattr(result, "data", None)
    normalized_metadata: dict[str, Any] = {}
    if isinstance(data, dict):
        normalized_metadata["data"] = data
    if isinstance(metadata, dict):
        normalized_metadata.update(metadata)
    return success, output, error, normalized_metadata


class DesktopActionExecutor:
    """Execute every desktop operation through one action contract."""

    def __init__(
        self,
        *,
        risk_evaluator: RiskEvaluator | None = None,
        audit_writer: Callable[[str, dict[str, Any]], str] | None = None,
        action_handlers: dict[str | DesktopActionType, ActionHandler] | None = None,
    ) -> None:
        self.risk_evaluator = risk_evaluator or RiskEvaluator()
        if audit_writer is None:
            from core.logging.logger import audit

            audit_writer = audit
        self.audit_writer = audit_writer
        self.action_handlers = self._default_handlers()
        for key, handler in (action_handlers or {}).items():
            action_name = key.value if isinstance(key, DesktopActionType) else str(key)
            self.action_handlers[action_name] = handler

    def evaluate_risk(self, action: DesktopAction) -> tuple[DesktopRiskTier, RiskResult]:
        risk = self.risk_evaluator.evaluate([action.action_name])
        if risk.is_blocked:
            return DesktopRiskTier.BLOCKED, risk
        if risk.level >= RiskLevel.HIGH:
            return DesktopRiskTier.HIGH, risk
        if risk.requires_confirmation:
            return DesktopRiskTier.CONFIRM, risk
        if risk.level >= RiskLevel.MEDIUM:
            return DesktopRiskTier.MEDIUM, risk
        return DesktopRiskTier.LOW, risk

    def requires_approval(self, action: DesktopAction) -> bool:
        if action.requires_approval is not None:
            return action.requires_approval
        risk_tier, risk = self.evaluate_risk(action)
        return risk.requires_confirmation or risk_tier in {DesktopRiskTier.HIGH, DesktopRiskTier.BLOCKED}

    async def execute(
        self,
        action: DesktopAction,
        *,
        approved: bool | None = None,
    ) -> DesktopActionResult:
        started_at = time.time()
        risk_tier, risk = self.evaluate_risk(action)

        if self._contains_sensitive_text(action):
            result = self._result(
                action,
                started_at=started_at,
                risk_tier=DesktopRiskTier.BLOCKED,
                status=DesktopActionStatus.BLOCKED,
                success=False,
                error="Sensitive text entry is blocked by desktop policy.",
                metadata={"risk": risk.summary()},
            )
            return self._audit(action, result)

        if risk.is_blocked:
            result = self._result(
                action,
                started_at=started_at,
                risk_tier=DesktopRiskTier.BLOCKED,
                status=DesktopActionStatus.BLOCKED,
                success=False,
                error=risk.summary(),
                metadata={"blocking_actions": list(risk.blocking_actions)},
            )
            return self._audit(action, result)

        if self.requires_approval(action) and approved is not True:
            result = self._result(
                action,
                started_at=started_at,
                risk_tier=risk_tier,
                status=DesktopActionStatus.NEEDS_APPROVAL,
                success=False,
                error="Desktop action requires user approval.",
                metadata={"risk": risk.summary()},
            )
            return self._audit(action, result)

        handler = self.action_handlers.get(action.action_name)
        if handler is None:
            result = self._result(
                action,
                started_at=started_at,
                risk_tier=risk_tier,
                status=DesktopActionStatus.FAILURE,
                success=False,
                error=f"No desktop action handler registered for '{action.action_name}'.",
                metadata={"risk": risk.summary()},
            )
            return self._audit(action, result)

        try:
            raw_result = await _maybe_await(handler(**dict(action.params)))
            success, output, error, metadata = _normalize_tool_result(raw_result)
            result = self._result(
                action,
                started_at=started_at,
                risk_tier=risk_tier,
                status=DesktopActionStatus.SUCCESS if success else DesktopActionStatus.FAILURE,
                success=success,
                output=output,
                error=error,
                metadata={"risk": risk.summary(), **metadata},
            )
            return self._audit(action, result)
        except Exception as exc:  # noqa: BLE001
            result = self._result(
                action,
                started_at=started_at,
                risk_tier=risk_tier,
                status=DesktopActionStatus.FAILURE,
                success=False,
                error=str(exc),
                metadata={"risk": risk.summary()},
            )
            return self._audit(action, result)

    def _audit(self, action: DesktopAction, result: DesktopActionResult) -> DesktopActionResult:
        try:
            result.audit_hash = self.audit_writer(
                "desktop_action",
                {
                    "action": action.to_dict(),
                    "result": result.to_dict(),
                },
            )
        except Exception as exc:  # noqa: BLE001
            result.metadata["audit_error"] = str(exc)
        return result

    @staticmethod
    def _contains_sensitive_text(action: DesktopAction) -> bool:
        if action.action_name != DesktopActionType.TYPE_TEXT.value:
            return False
        text = str(action.params.get("text", "") or "").lower()
        return any(marker in text for marker in _SENSITIVE_TEXT_MARKERS)

    @staticmethod
    def _result(
        action: DesktopAction,
        *,
        started_at: float,
        risk_tier: DesktopRiskTier,
        status: DesktopActionStatus,
        success: bool,
        output: str = "",
        error: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> DesktopActionResult:
        return DesktopActionResult(
            action_id=action.action_id,
            action_type=action.action_name,
            success=success,
            status=status,
            output=output,
            error=error,
            risk_tier=risk_tier,
            started_at=started_at,
            ended_at=time.time(),
            metadata=dict(metadata or {}),
        )

    @staticmethod
    def _default_handlers() -> dict[str, ActionHandler]:
        return {
            DesktopActionType.LAUNCH_APP.value: _launch_application,
            DesktopActionType.FOCUS_WINDOW.value: _focus_window,
            DesktopActionType.MOVE_MOUSE.value: _move_mouse,
            DesktopActionType.CLICK.value: _click,
            DesktopActionType.DOUBLE_CLICK.value: _double_click,
            DesktopActionType.RIGHT_CLICK.value: _right_click,
            DesktopActionType.CLICK_TEXT_ON_SCREEN.value: _click_text_on_screen,
            DesktopActionType.CLICK_SCREEN_TARGET.value: _click_screen_target,
            DesktopActionType.DOUBLE_CLICK_SCREEN_TARGET.value: _double_click_screen_target,
            DesktopActionType.RIGHT_CLICK_SCREEN_TARGET.value: _right_click_screen_target,
            DesktopActionType.SCROLL.value: _scroll,
            DesktopActionType.DRAG.value: _drag,
            DesktopActionType.TYPE_TEXT.value: _type_text,
            DesktopActionType.PRESS_KEY.value: _press_key,
            DesktopActionType.HOTKEY.value: _hotkey,
            DesktopActionType.CLIPBOARD_GET.value: _clipboard_get,
            DesktopActionType.CLIPBOARD_SET.value: _clipboard_set,
            DesktopActionType.CLIPBOARD_PASTE.value: _clipboard_paste,
        }


async def _launch_application(target: str, args: list[str] | None = None) -> Any:
    from core.tools.system_automation import async_launch_application

    return await async_launch_application(target, args)


async def _click(x: int, y: int, button: str = "left") -> Any:
    from core.tools.gui_control import click

    return await click(x=x, y=y, button=button)


async def _double_click(x: int, y: int) -> Any:
    from core.tools.gui_control import double_click

    return await double_click(x=x, y=y)


async def _right_click(x: int, y: int) -> Any:
    from core.tools.gui_control import right_click

    return await right_click(x=x, y=y)


async def _click_text_on_screen(
    text: str,
    occurrence: int = 1,
    button: str = "left",
    match_mode: str = "contains",
) -> Any:
    from core.tools.gui_control import click_text_on_screen

    return await click_text_on_screen(
        text=text,
        occurrence=occurrence,
        button=button,
        match_mode=match_mode,
    )


async def _click_screen_target(
    target: str,
    occurrence: int = 1,
    button: str = "left",
    match_mode: str = "contains",
    min_confidence: float = 0.2,
) -> Any:
    from core.tools.gui_control import click_screen_target

    return await click_screen_target(
        target=target,
        occurrence=occurrence,
        button=button,
        match_mode=match_mode,
        min_confidence=min_confidence,
    )


async def _double_click_screen_target(
    target: str,
    occurrence: int = 1,
    match_mode: str = "contains",
    min_confidence: float = 0.2,
) -> Any:
    from core.tools.gui_control import double_click_screen_target

    return await double_click_screen_target(
        target=target,
        occurrence=occurrence,
        match_mode=match_mode,
        min_confidence=min_confidence,
    )


async def _right_click_screen_target(
    target: str,
    occurrence: int = 1,
    match_mode: str = "contains",
    min_confidence: float = 0.2,
) -> Any:
    from core.tools.gui_control import right_click_screen_target

    return await right_click_screen_target(
        target=target,
        occurrence=occurrence,
        match_mode=match_mode,
        min_confidence=min_confidence,
    )


async def _type_text(text: str, interval: float = 0.05) -> Any:
    from core.tools.gui_control import type_text

    return await type_text(text=text, interval=interval)


async def _press_key(key: str, presses: int = 1, interval: float = 0.05) -> Any:
    from core.tools.gui_control import press_key

    return await press_key(key=key, presses=presses, interval=interval)


async def _hotkey(keys: list[str] | tuple[str, ...] | str) -> Any:
    from core.tools.gui_control import hotkey

    if isinstance(keys, str):
        key_list = [part.strip() for part in keys.split("+") if part.strip()]
    else:
        key_list = [str(key) for key in keys]
    return await hotkey(*key_list)


def _pyautogui_result(success: bool, data: dict[str, Any] | None = None, error: str = "") -> dict[str, Any]:
    return {"success": success, "data": data or {}, "error": error}


def _require_pyautogui() -> Any:
    try:
        import pyautogui

        return pyautogui
    except ImportError as exc:
        raise ImportError("pyautogui not installed - run: pip install pyautogui") from exc


def _move_mouse(x: int, y: int, duration: float = 0.0) -> dict[str, Any]:
    pag = _require_pyautogui()
    pag.moveTo(x, y, duration=duration)
    return _pyautogui_result(True, {"action": "move_mouse", "x": x, "y": y})


def _scroll(clicks: int, x: int | None = None, y: int | None = None) -> dict[str, Any]:
    pag = _require_pyautogui()
    if x is not None and y is not None:
        pag.moveTo(x, y)
    pag.scroll(clicks)
    return _pyautogui_result(True, {"action": "scroll", "clicks": clicks, "x": x, "y": y})


def _drag(
    start_x: int,
    start_y: int,
    end_x: int,
    end_y: int,
    duration: float = 0.2,
    button: str = "left",
) -> dict[str, Any]:
    pag = _require_pyautogui()
    pag.moveTo(start_x, start_y)
    pag.dragTo(end_x, end_y, duration=duration, button=button)
    return _pyautogui_result(
        True,
        {
            "action": "drag",
            "start_x": start_x,
            "start_y": start_y,
            "end_x": end_x,
            "end_y": end_y,
            "button": button,
        },
    )


def _focus_window(title: str) -> dict[str, Any]:
    try:
        import pygetwindow as gw
    except ImportError:
        return _pyautogui_result(False, error="pygetwindow not installed - run: pip install pygetwindow")

    windows = gw.getWindowsWithTitle(title)
    if not windows:
        return _pyautogui_result(False, error=f"No window found matching '{title}'.")
    window = windows[0]
    window.activate()
    return _pyautogui_result(True, {"title": getattr(window, "title", title)})


def _clipboard_get() -> dict[str, Any]:
    try:
        import pyperclip
    except ImportError:
        return _pyautogui_result(False, error="pyperclip not installed - run: pip install pyperclip")

    text = pyperclip.paste()
    return _pyautogui_result(True, {"length": len(text), "text": text})


def _clipboard_set(text: str) -> dict[str, Any]:
    lower = str(text).lower()
    if any(marker in lower for marker in _SENSITIVE_TEXT_MARKERS):
        return _pyautogui_result(False, error="Sensitive clipboard text is blocked by desktop policy.")
    try:
        import pyperclip
    except ImportError:
        return _pyautogui_result(False, error="pyperclip not installed - run: pip install pyperclip")

    pyperclip.copy(text)
    return _pyautogui_result(True, {"length": len(text)})


def _clipboard_paste() -> dict[str, Any]:
    pag = _require_pyautogui()
    pag.hotkey("ctrl", "v")
    return _pyautogui_result(True, {"action": "clipboard_paste"})


__all__ = ["DesktopActionExecutor"]
