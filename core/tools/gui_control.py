"""Desktop GUI automation tools for Jarvis."""

from __future__ import annotations
from core.types.common import ToolResult

import asyncio
import configparser
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

GUI_AUDIT_DIR = Path("outputs/gui_audit")
_CONFIG_PATH = Path("config/jarvis.ini")

_FORBIDDEN_KEYWORDS: tuple[str, ...] = (
    "password",
    "passwd",
    "secret",
    "token",
    "apikey",
    "api_key",
)

_LAST_CLICK_TIME: float = 0.0
_CLICK_SAFETY_DELAY: float = 0.3


def _save_audit_screenshot(label: str) -> str:
    try:
        import pyautogui

        GUI_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time() * 1000)
        path = GUI_AUDIT_DIR / f"{timestamp}_{label}.png"
        pyautogui.screenshot().save(str(path))
        return str(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Audit screenshot failed (%s): %s", label, exc)
        return ""


def _validate_coords(x: int, y: int) -> bool:
    try:
        import pyautogui

        width, height = pyautogui.size()
        return bool(0 <= x < width and 0 <= y < height)
    except Exception:
        return True


def _require_pyautogui():
    try:
        import pyautogui

        return pyautogui
    except ImportError as exc:
        raise ImportError("pyautogui not installed - run: pip install pyautogui") from exc


def _contains_sensitive_text(text: str) -> str | None:
    lowered = str(text or "").lower()
    for keyword in _FORBIDDEN_KEYWORDS:
        if keyword in lowered:
            return keyword
    return None


def _tool_result_payload(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        data = result.get("data")
        return data if isinstance(data, dict) else {}
    data = getattr(result, "data", None)
    return data if isinstance(data, dict) else {}


def _extract_json_object(raw: str) -> dict[str, Any]:
    candidate = str(raw or "").strip()
    fenced = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
    if fenced is not None:
        candidate = fenced.group(0)
    parsed = json.loads(candidate)
    if not isinstance(parsed, dict):
        raise ValueError("Vision locator did not return a JSON object.")
    return parsed


def _match_center(match: dict[str, Any]) -> tuple[int, int]:
    x = int(match.get("x", 0))
    y = int(match.get("y", 0))
    width = int(match.get("w", match.get("width", 0)) or 0)
    height = int(match.get("h", match.get("height", 0)) or 0)
    return x + max(0, width // 2), y + max(0, height // 2)


def _runtime_config() -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    if _CONFIG_PATH.is_file():
        config.read(_CONFIG_PATH, encoding="utf-8")
    return config


def _vision_locate_target(target: str):
    from core.tools.screen import capture_screen
    from core.tools.vision import VisionTool

    screen_result = capture_screen()
    if not screen_result.success:
        return ToolResult(success=False, error=screen_result.error)

    screenshot_path = str(screen_result.data.get("path", "") or "")
    if not screenshot_path:
        return ToolResult(success=False, error="Vision locator could not capture a screenshot.")

    prompt = (
        "Locate the UI target described below in this screenshot.\n"
        f"Target: {target}\n"
        "Return strict JSON only with keys: found, x, y, width, height, confidence, reason.\n"
        "Use absolute image pixels for x and y as the center point of the target.\n"
        "If you cannot find it, return found=false."
    )

    try:
        raw = VisionTool(_runtime_config()).analyze(screenshot_path, prompt)
        parsed = _extract_json_object(raw)
    except Exception as exc:  # noqa: BLE001
        return ToolResult(success=False, error=f"Vision locator failed: {exc}")

    found = bool(parsed.get("found", True))
    x = int(parsed.get("x", 0) or 0)
    y = int(parsed.get("y", 0) or 0)
    confidence = float(parsed.get("confidence", 0.0) or 0.0)
    if not found:
        return ToolResult(
            success=False,
            data={"target": target, "confidence": confidence, "reason": str(parsed.get("reason", "") or "")},
            error=f"Target '{target}' was not found by the vision locator.",
        )
    if not _validate_coords(x, y):
        return ToolResult(success=False, error=f"Vision returned out-of-bounds coordinates ({x}, {y}).")
    return ToolResult(
        success=True,
        data={
            "target": target,
            "x": x,
            "y": y,
            "width": int(parsed.get("width", 0) or 0),
            "height": int(parsed.get("height", 0) or 0),
            "confidence": confidence,
            "reason": str(parsed.get("reason", "") or ""),
            "method": "vision",
            "screenshot_path": screenshot_path,
        },
    )


async def click(x: int, y: int, button: str = "left"):
    """Click at screen coordinates with bounds checks and audit capture."""

    global _LAST_CLICK_TIME
    try:
        pag = _require_pyautogui()
    except ImportError as exc:
        return ToolResult(success=False, error=str(exc))

    if not _validate_coords(x, y):
        return ToolResult(success=False, error=f"Coordinates ({x}, {y}) are outside screen bounds")

    await asyncio.to_thread(_save_audit_screenshot, "before_click")
    await asyncio.sleep(_CLICK_SAFETY_DELAY)
    _LAST_CLICK_TIME = time.time()
    await asyncio.to_thread(pag.click, x, y, button=button)
    await asyncio.to_thread(_save_audit_screenshot, "after_click")
    logger.info("click(%d, %d, button=%s)", x, y, button)
    return ToolResult(success=True, data={"action": "click", "x": x, "y": y, "button": button})


async def double_click(x: int, y: int):
    """Double-click at screen coordinates."""

    try:
        pag = _require_pyautogui()
    except ImportError as exc:
        return ToolResult(success=False, error=str(exc))

    if not _validate_coords(x, y):
        return ToolResult(success=False, error=f"Coordinates ({x}, {y}) are outside screen bounds")

    await asyncio.to_thread(_save_audit_screenshot, "before_dblclick")
    await asyncio.sleep(_CLICK_SAFETY_DELAY)
    await asyncio.to_thread(pag.doubleClick, x, y)
    await asyncio.to_thread(_save_audit_screenshot, "after_dblclick")
    logger.info("double_click(%d, %d)", x, y)
    return ToolResult(success=True, data={"action": "double_click", "x": x, "y": y})


async def right_click(x: int, y: int):
    """Right-click at screen coordinates."""

    try:
        pag = _require_pyautogui()
    except ImportError as exc:
        return ToolResult(success=False, error=str(exc))

    if not _validate_coords(x, y):
        return ToolResult(success=False, error=f"Coordinates ({x}, {y}) are outside screen bounds")

    await asyncio.to_thread(_save_audit_screenshot, "before_rightclick")
    await asyncio.sleep(_CLICK_SAFETY_DELAY)
    await asyncio.to_thread(pag.rightClick, x, y)
    await asyncio.to_thread(_save_audit_screenshot, "after_rightclick")
    logger.info("right_click(%d, %d)", x, y)
    return ToolResult(success=True, data={"action": "right_click", "x": x, "y": y})


async def click_text_on_screen(
    text: str,
    *,
    occurrence: int = 1,
    button: str = "left",
    match_mode: str = "contains",
):
    """Locate visible text on screen and click its center."""
    from core.tools.screen import find_text_on_screen

    result = find_text_on_screen(text, match_mode=match_mode)
    if not result.success:
        return result

    matches = list(result.data.get("matches", []))
    if not matches:
        return ToolResult(
            success=False,
            data={"query": text, "match_mode": match_mode},
            error=f"No visible text matched '{text}'.",
        )

    index = max(0, int(occurrence) - 1)
    if index >= len(matches):
        return ToolResult(
            success=False,
            data={"query": text, "matches_found": len(matches)},
            error=f"Requested match {occurrence}, but only {len(matches)} match(es) were found for '{text}'.",
        )

    match = matches[index]
    center_x, center_y = _match_center(match)
    click_result = await click(center_x, center_y, button=button)
    if not click_result.success:
        return click_result

    return ToolResult(
        success=True,
        data={
            **_tool_result_payload(click_result),
            "query": text,
            "match_mode": match_mode,
            "occurrence": occurrence,
            "matched_text": str(match.get("text", "") or ""),
            "match": match,
            "method": "ocr_text",
        },
    )


async def _resolve_target_coordinates(
    target: str,
    *,
    occurrence: int = 1,
    match_mode: str = "contains",
    min_confidence: float = 0.2,
):
    """Resolve coordinates for a described screen target, trying OCR first, then Vision.
    Returns:
        (x, y, resolved_metadata, error_result)
        If resolved successfully: (x, y, metadata, None)
        If failed: (None, None, None, ToolResult)
    """
    from core.tools.screen import find_text_on_screen

    # 1. Try OCR text locator first
    text_res = await asyncio.to_thread(find_text_on_screen, target, match_mode=match_mode)
    text_error = ""
    if text_res.success:
        matches = list(text_res.data.get("matches", []))
        index = max(0, int(occurrence) - 1)
        if matches and index < len(matches):
            match = matches[index]
            center_x, center_y = _match_center(match)
            return center_x, center_y, {
                "matched_text": str(match.get("text", "") or ""),
                "match": match,
                "target": target,
                "occurrence": occurrence,
                "match_mode": match_mode,
                "method": "ocr_text",
            }, None
        else:
            text_error = f"No visible text matched '{target}'." if not matches else f"Requested match {occurrence}, but only {len(matches)} match(es) were found."
    else:
        text_error = text_res.error or "OCR text lookup failed."

    # 2. Try Vision locator fallback
    vision_result = await asyncio.to_thread(_vision_locate_target, target)
    if not vision_result.success:
        error = text_error or vision_result.error
        if text_error and vision_result.error:
            error = f"{text_error} Vision fallback: {vision_result.error}"
        return None, None, None, ToolResult(
            success=False,
            data={"target": target, "text_locator_error": text_error, "vision_locator_error": vision_result.error},
            error=error,
        )

    confidence = float(vision_result.data.get("confidence", 0.0) or 0.0)
    if confidence < float(min_confidence):
        return None, None, None, ToolResult(
            success=False,
            data=dict(vision_result.data),
            error=(
                f"Vision confidence for '{target}' was {confidence:.2f}, below the minimum "
                f"threshold of {float(min_confidence):.2f}."
            ),
        )

    x = int(vision_result.data.get("x", 0) or 0)
    y = int(vision_result.data.get("y", 0) or 0)
    return x, y, {
        **dict(vision_result.data),
        "target": target,
        "method": "vision",
    }, None


async def click_screen_target(
    target: str,
    *,
    occurrence: int = 1,
    button: str = "left",
    match_mode: str = "contains",
    min_confidence: float = 0.2,
):
    """Click a described screen target without hard-coded coordinates."""

    x, y, resolved, err = await _resolve_target_coordinates(
        target,
        occurrence=occurrence,
        match_mode=match_mode,
        min_confidence=min_confidence,
    )
    if err is not None:
        return err

    click_result = await click(x, y, button=button)
    if not click_result.success:
        return click_result

    return ToolResult(
        success=True,
        data={
            **_tool_result_payload(click_result),
            **resolved,
        },
    )


async def double_click_screen_target(
    target: str,
    *,
    occurrence: int = 1,
    match_mode: str = "contains",
    min_confidence: float = 0.2,
):
    """Double-click a described screen target without hard-coded coordinates."""

    x, y, resolved, err = await _resolve_target_coordinates(
        target,
        occurrence=occurrence,
        match_mode=match_mode,
        min_confidence=min_confidence,
    )
    if err is not None:
        return err

    click_result = await double_click(x, y)
    if not click_result.success:
        return click_result

    return ToolResult(
        success=True,
        data={
            **_tool_result_payload(click_result),
            **resolved,
        },
    )


async def right_click_screen_target(
    target: str,
    *,
    occurrence: int = 1,
    match_mode: str = "contains",
    min_confidence: float = 0.2,
):
    """Right-click a described screen target without hard-coded coordinates."""

    x, y, resolved, err = await _resolve_target_coordinates(
        target,
        occurrence=occurrence,
        match_mode=match_mode,
        min_confidence=min_confidence,
    )
    if err is not None:
        return err

    click_result = await right_click(x, y)
    if not click_result.success:
        return click_result

    return ToolResult(
        success=True,
        data={
            **_tool_result_payload(click_result),
            **resolved,
        },
    )


async def move_mouse(x: int, y: int, duration: float = 0.0):

    try:
        pag = _require_pyautogui()
    except ImportError as exc:
        return ToolResult(success=False, error=str(exc))

    if not _validate_coords(x, y):
        return ToolResult(success=False, error=f"Coordinates ({x}, {y}) are outside screen bounds")

    await asyncio.to_thread(pag.moveTo, x, y, duration=max(0.0, float(duration)))
    return ToolResult(success=True, data={"action": "move_mouse", "x": x, "y": y, "duration": duration})


async def scroll(clicks: int, x: int | None = None, y: int | None = None):

    try:
        pag = _require_pyautogui()
    except ImportError as exc:
        return ToolResult(success=False, error=str(exc))

    if x is not None and y is not None:
        if not _validate_coords(int(x), int(y)):
            return ToolResult(success=False, error=f"Coordinates ({x}, {y}) are outside screen bounds")
        await asyncio.to_thread(pag.moveTo, int(x), int(y))
    await asyncio.to_thread(pag.scroll, int(clicks))
    return ToolResult(success=True, data={"action": "scroll", "clicks": int(clicks), "x": x, "y": y})


async def drag(
    start_x: int,
    start_y: int,
    end_x: int,
    end_y: int,
    *,
    duration: float = 0.2,
    button: str = "left",
):

    try:
        pag = _require_pyautogui()
    except ImportError as exc:
        return ToolResult(success=False, error=str(exc))

    if not _validate_coords(start_x, start_y) or not _validate_coords(end_x, end_y):
        return ToolResult(success=False, error="Drag coordinates are outside screen bounds")

    await asyncio.to_thread(pag.moveTo, start_x, start_y)
    await asyncio.to_thread(pag.dragTo, end_x, end_y, duration=max(0.0, float(duration)), button=button)
    return ToolResult(
        success=True,
        data={
            "action": "drag",
            "start_x": start_x,
            "start_y": start_y,
            "end_x": end_x,
            "end_y": end_y,
            "button": button,
            "duration": duration,
        },
    )


async def type_text(text: str, interval: float = 0.05):

    forbidden = _contains_sensitive_text(text)
    if forbidden is not None:
        logger.warning("type_text refused because it contained '%s'", forbidden)
        return ToolResult(success=False, error=f"Refused: text contains sensitive keyword '{forbidden}'")

    try:
        pag = _require_pyautogui()
    except ImportError as exc:
        return ToolResult(success=False, error=str(exc))

    await asyncio.sleep(_CLICK_SAFETY_DELAY)
    await asyncio.to_thread(pag.typewrite, text, interval=max(0.0, float(interval)))
    logger.info("type_text: typed %d character(s)", len(text))
    return ToolResult(success=True, data={"action": "type_text", "length": len(text)})


async def press_key(key: str, presses: int = 1, interval: float = 0.05):

    try:
        pag = _require_pyautogui()
    except ImportError as exc:
        return ToolResult(success=False, error=str(exc))

    await asyncio.sleep(0.1)
    await asyncio.to_thread(pag.press, str(key), presses=max(1, int(presses)), interval=max(0.0, float(interval)))
    return ToolResult(
        success=True,
        data={"action": "press_key", "key": str(key), "presses": max(1, int(presses))},
    )


async def hotkey(*keys: str):

    try:
        pag = _require_pyautogui()
    except ImportError as exc:
        return ToolResult(success=False, error=str(exc))

    await asyncio.sleep(0.1)
    await asyncio.to_thread(pag.hotkey, *keys)
    logger.info("hotkey: %s", "+".join(keys))
    return ToolResult(success=True, data={"action": "hotkey", "keys": list(keys)})


def focus_window(title: str):

    try:
        import pygetwindow as gw
    except ImportError:
        return ToolResult(success=False, error="pygetwindow not installed - run: pip install pygetwindow")

    windows = gw.getWindowsWithTitle(title)
    if not windows:
        return ToolResult(success=False, error=f"No window found matching '{title}'.")
    window = windows[0]
    window.activate()
    return ToolResult(success=True, data={"title": getattr(window, "title", title)})


def get_active_window():

    try:
        import pygetwindow as gw
    except ImportError:
        return ToolResult(success=False, error="pygetwindow not installed - run: pip install pygetwindow")

    window = gw.getActiveWindow()
    if window is None:
        return ToolResult(success=True, data={"title": None, "message": "No active window detected"})
    return ToolResult(
        success=True,
        data={
            "title": window.title,
            "x": window.left,
            "y": window.top,
            "width": window.width,
            "height": window.height,
        },
    )


def clipboard_get():

    try:
        import pyperclip
    except ImportError:
        return ToolResult(success=False, error="pyperclip not installed - run: pip install pyperclip")

    text = pyperclip.paste()
    return ToolResult(success=True, data={"length": len(text), "text": text})


def clipboard_set(text: str):

    forbidden = _contains_sensitive_text(text)
    if forbidden is not None:
        return ToolResult(success=False, error=f"Refused: text contains sensitive keyword '{forbidden}'")

    try:
        import pyperclip
    except ImportError:
        return ToolResult(success=False, error="pyperclip not installed - run: pip install pyperclip")

    pyperclip.copy(text)
    return ToolResult(success=True, data={"length": len(text)})


async def clipboard_paste():

    result = await hotkey("ctrl", "v")
    if not result.success:
        return result
    return ToolResult(success=True, data={"action": "clipboard_paste"})


async def get_mouse_position():

    try:
        pag = _require_pyautogui()
    except ImportError as exc:
        return ToolResult(success=False, error=str(exc))

    x, y = await asyncio.to_thread(pag.position)
    return ToolResult(success=True, data={"action": "get_mouse_position", "x": x, "y": y})


async def mouse_down(button: str = "left"):

    try:
        pag = _require_pyautogui()
    except ImportError as exc:
        return ToolResult(success=False, error=str(exc))

    await asyncio.to_thread(pag.mouseDown, button=button)
    logger.info("mouse_down(button=%s)", button)
    return ToolResult(success=True, data={"action": "mouse_down", "button": button})


async def mouse_up(button: str = "left"):

    try:
        pag = _require_pyautogui()
    except ImportError as exc:
        return ToolResult(success=False, error=str(exc))

    await asyncio.to_thread(pag.mouseUp, button=button)
    logger.info("mouse_up(button=%s)", button)
    return ToolResult(success=True, data={"action": "mouse_up", "button": button})


async def key_down(key: str):

    try:
        pag = _require_pyautogui()
    except ImportError as exc:
        return ToolResult(success=False, error=str(exc))

    await asyncio.to_thread(pag.keyDown, str(key))
    logger.info("key_down(key=%s)", key)
    return ToolResult(success=True, data={"action": "key_down", "key": str(key)})


async def key_up(key: str):

    try:
        pag = _require_pyautogui()
    except ImportError as exc:
        return ToolResult(success=False, error=str(exc))

    await asyncio.to_thread(pag.keyUp, str(key))
    logger.info("key_up(key=%s)", key)
    return ToolResult(success=True, data={"action": "key_up", "key": str(key)})


async def middle_click(x: int, y: int):
    """Middle-click at screen coordinates."""

    try:
        pag = _require_pyautogui()
    except ImportError as exc:
        return ToolResult(success=False, error=str(exc))

    if not _validate_coords(x, y):
        return ToolResult(success=False, error=f"Coordinates ({x}, {y}) are outside screen bounds")

    await asyncio.to_thread(_save_audit_screenshot, "before_middleclick")
    await asyncio.sleep(_CLICK_SAFETY_DELAY)
    await asyncio.to_thread(pag.middleClick, x, y)
    await asyncio.to_thread(_save_audit_screenshot, "after_middleclick")
    logger.info("middle_click(%d, %d)", x, y)
    return ToolResult(success=True, data={"action": "middle_click", "x": x, "y": y})


__all__ = [
    "click",
    "click_screen_target",
    "click_text_on_screen",
    "clipboard_get",
    "clipboard_paste",
    "clipboard_set",
    "double_click",
    "double_click_screen_target",
    "drag",
    "focus_window",
    "get_active_window",
    "get_mouse_position",
    "hotkey",
    "key_down",
    "key_up",
    "middle_click",
    "mouse_down",
    "mouse_up",
    "move_mouse",
    "press_key",
    "right_click",
    "right_click_screen_target",
    "scroll",
    "type_text",
]
