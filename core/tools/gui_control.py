"""
core/tools/gui_control.py
--------------------------
Desktop GUI automation tools for Jarvis.

SAFETY RULES (hard-coded, non-negotiable):
  • 300 ms sleep before every click action — never remove.
  • Screenshot saved before AND after every click for audit trail.
  • Coordinates validated against screen bounds before clicking.
  • type_text REFUSES input containing sensitive keywords.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

GUI_AUDIT_DIR = Path("outputs/gui_audit")

# Keywords that type_text will ALWAYS refuse — no exceptions.
_FORBIDDEN_KEYWORDS: tuple[str, ...] = (
    "password",
    "passwd",
    "secret",
    "token",
    "apikey",
    "api_key",
)

_LAST_CLICK_TIME: float = 0.0
_CLICK_SAFETY_DELAY: float = 0.3   # seconds — do NOT reduce


# ── Internal helpers ──────────────────────────────────────────────────────────

def _save_audit_screenshot(label: str) -> str:
    """Capture a screenshot to the GUI audit directory."""
    try:
        import pyautogui
        GUI_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
        ts = int(time.time() * 1000)
        path = GUI_AUDIT_DIR / f"{ts}_{label}.png"
        pyautogui.screenshot().save(str(path))
        return str(path)
    except Exception as e:
        logger.warning("Audit screenshot failed (%s): %s", label, e)
        return ""


def _validate_coords(x: int, y: int) -> bool:
    """Return True if (x, y) is within the current screen bounds."""
    try:
        import pyautogui
        w, h = pyautogui.size()
        return 0 <= x < w and 0 <= y < h
    except Exception:
        return True   # Cannot determine bounds — allow and log


def _require_pyautogui():
    """Import pyautogui or raise ImportError with a helpful message."""
    try:
        import pyautogui
        return pyautogui
    except ImportError as exc:
        raise ImportError("pyautogui not installed — run: pip install pyautogui") from exc


# ── Click tools ───────────────────────────────────────────────────────────────

async def click(x: int, y: int, button: str = "left"):
    """Click at screen coordinates.

    Safety: validates bounds, 300 ms delay, audit screenshots before/after.

    Args:
        x, y:   Screen coordinates (pixels).
        button: ``"left"`` (default), ``"right"``, or ``"middle"``.
    """
    from integrations.base import ToolResult
    global _LAST_CLICK_TIME
    try:
        pag = _require_pyautogui()
    except ImportError as e:
        return ToolResult(success=False, error=str(e))

    if not _validate_coords(x, y):
        return ToolResult(success=False, error=f"Coordinates ({x}, {y}) are outside screen bounds")

    _save_audit_screenshot("before_click")
    await asyncio.sleep(_CLICK_SAFETY_DELAY)   # mandatory safety pause
    _LAST_CLICK_TIME = time.time()
    pag.click(x, y, button=button)
    _save_audit_screenshot("after_click")
    logger.info("click(%d, %d, button=%s)", x, y, button)
    return ToolResult(success=True, data={"action": "click", "x": x, "y": y, "button": button})


async def double_click(x: int, y: int):
    """Double-click at screen coordinates.

    Args:
        x, y: Screen coordinates (pixels).
    """
    from integrations.base import ToolResult
    try:
        pag = _require_pyautogui()
    except ImportError as e:
        return ToolResult(success=False, error=str(e))

    if not _validate_coords(x, y):
        return ToolResult(success=False, error=f"Coordinates ({x}, {y}) are outside screen bounds")

    _save_audit_screenshot("before_dblclick")
    await asyncio.sleep(_CLICK_SAFETY_DELAY)
    pag.doubleClick(x, y)
    _save_audit_screenshot("after_dblclick")
    logger.info("double_click(%d, %d)", x, y)
    return ToolResult(success=True, data={"action": "double_click", "x": x, "y": y})


async def right_click(x: int, y: int):
    """Right-click at screen coordinates.

    Args:
        x, y: Screen coordinates (pixels).
    """
    from integrations.base import ToolResult
    try:
        pag = _require_pyautogui()
    except ImportError as e:
        return ToolResult(success=False, error=str(e))

    if not _validate_coords(x, y):
        return ToolResult(success=False, error=f"Coordinates ({x}, {y}) are outside screen bounds")

    _save_audit_screenshot("before_rightclick")
    await asyncio.sleep(_CLICK_SAFETY_DELAY)
    pag.rightClick(x, y)
    _save_audit_screenshot("after_rightclick")
    logger.info("right_click(%d, %d)", x, y)
    return ToolResult(success=True, data={"action": "right_click", "x": x, "y": y})


# ── Keyboard tools ────────────────────────────────────────────────────────────

async def type_text(text: str, interval: float = 0.05):
    """Type text at the current cursor position.

    HARD SAFETY: Refuses if *text* contains sensitive keywords
    (``password``, ``passwd``, ``secret``, ``token``, ``apikey``, ``api_key``).

    Args:
        text:     The string to type.
        interval: Delay between keystrokes in seconds (default 0.05).
    """
    from integrations.base import ToolResult
    lower = text.lower()
    for kw in _FORBIDDEN_KEYWORDS:
        if kw in lower:
            logger.warning("type_text REFUSED — text contains forbidden keyword '%s'", kw)
            return ToolResult(success=False, error=f"Refused: text contains sensitive keyword '{kw}'")

    try:
        pag = _require_pyautogui()
    except ImportError as e:
        return ToolResult(success=False, error=str(e))

    await asyncio.sleep(_CLICK_SAFETY_DELAY)
    pag.typewrite(text, interval=interval)
    logger.info("type_text: typed %d character(s)", len(text))
    return ToolResult(success=True, data={"action": "type_text", "length": len(text)})


async def hotkey(*keys: str):
    """Press a keyboard shortcut (e.g. ``hotkey('ctrl', 'c')``).

    Args:
        keys: Sequence of key names as recognised by pyautogui (e.g. ``'ctrl'``, ``'c'``).
    """
    from integrations.base import ToolResult
    try:
        pag = _require_pyautogui()
    except ImportError as e:
        return ToolResult(success=False, error=str(e))

    await asyncio.sleep(0.1)
    pag.hotkey(*keys)
    logger.info("hotkey: %s", "+".join(keys))
    return ToolResult(success=True, data={"action": "hotkey", "keys": list(keys)})


# ── Window tools ──────────────────────────────────────────────────────────────

def get_active_window():
    """Return title and geometry of the currently focused window.

    Returns:
        ToolResult with ``title``, ``x``, ``y``, ``width``, ``height``.
    """
    from integrations.base import ToolResult
    try:
        import pygetwindow as gw
    except ImportError:
        return ToolResult(success=False, error="pygetwindow not installed — run: pip install pygetwindow")

    win = gw.getActiveWindow()
    if win is None:
        return ToolResult(success=True, data={"title": None, "message": "No active window detected"})
    return ToolResult(
        success=True,
        data={
            "title": win.title,
            "x": win.left,
            "y": win.top,
            "width": win.width,
            "height": win.height,
        },
    )


__all__ = [
    "click",
    "double_click",
    "right_click",
    "type_text",
    "hotkey",
    "get_active_window",
]
