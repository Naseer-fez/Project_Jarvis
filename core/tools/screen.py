"""
core/tools/screen.py
---------------------
Screen capture and OCR tools for Jarvis.

Functions:
  capture_screen()            — full-screen PNG to outputs/screenshots/
  capture_region(x,y,w,h)    — region PNG
  find_text_on_screen(text)   — OCR + bounding-box search via pytesseract
  describe_screen(llm_client) — vision description or OCR fallback
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

SCREENSHOT_DIR = Path("outputs/screenshots")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_dir() -> None:
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)


# ── Tool functions ─────────────────────────────────────────────────────────────

def capture_screen():
    """Take a full-screen screenshot and save it to outputs/screenshots/.

    Returns:
        ToolResult with ``path``, ``width``, and ``height`` on success.
    """
    from integrations.base import ToolResult
    try:
        import pyautogui
    except ImportError:
        return ToolResult(success=False, error="pyautogui not installed")

    try:
        _ensure_dir()
        path = SCREENSHOT_DIR / f"{_ts()}.png"
        img = pyautogui.screenshot()
        img.save(str(path))
        logger.info("Screenshot saved: %s", path)
        return ToolResult(success=True, data={"path": str(path), "width": img.width, "height": img.height})
    except Exception as e:
        logger.error("capture_screen failed: %s", e)
        return ToolResult(success=False, error=str(e))


def capture_region(x: int, y: int, width: int, height: int):
    """Screenshot a specific screen region.

    Args:
        x, y:         Top-left corner (pixels).
        width, height: Region dimensions (pixels).

    Returns:
        ToolResult with ``path`` on success.
    """
    from integrations.base import ToolResult
    try:
        import pyautogui
    except ImportError:
        return ToolResult(success=False, error="pyautogui not installed")

    try:
        _ensure_dir()
        path = SCREENSHOT_DIR / f"{_ts()}_region.png"
        img = pyautogui.screenshot(region=(x, y, width, height))
        img.save(str(path))
        logger.info("Region screenshot saved: %s", path)
        return ToolResult(success=True, data={"path": str(path), "x": x, "y": y, "width": width, "height": height})
    except Exception as e:
        logger.error("capture_region failed: %s", e)
        return ToolResult(success=False, error=str(e))


def find_text_on_screen(text: str):
    """Search for a text string on the current screen using OCR.

    Args:
        text: The text to search for (case-insensitive substring match).

    Returns:
        ToolResult with ``matches`` list, each entry containing ``text``,
        ``x``, ``y``, ``w``, ``h``.
    """
    from integrations.base import ToolResult
    try:
        import pyautogui
        import pytesseract
        from PIL import Image  # noqa: F401 — ensure Pillow is present
    except ImportError as e:
        return ToolResult(success=False, error=f"Missing dependency: {e}")

    try:
        img = pyautogui.screenshot()
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        matches = []
        for i, word in enumerate(data["text"]):
            if word and text.lower() in word.lower():
                matches.append(
                    {
                        "text": word,
                        "x": data["left"][i],
                        "y": data["top"][i],
                        "w": data["width"][i],
                        "h": data["height"][i],
                    }
                )
        logger.info("find_text_on_screen('%s'): %d match(es)", text, len(matches))
        return ToolResult(success=True, data={"query": text, "matches": matches})
    except Exception as e:
        logger.error("find_text_on_screen failed: %s", e)
        return ToolResult(success=False, error=str(e))


def describe_screen(llm_client=None):
    """Describe the current screen contents.

    Tries LLaVA-style vision via *llm_client* first; falls back to OCR text dump.

    Args:
        llm_client: Optional LLM client with a ``complete()`` method that
                    accepts ``images=[b64_str]`` (e.g. LLaVA/Ollama vision).

    Returns:
        ToolResult with ``description`` or ``ocr_text``.
    """
    from integrations.base import ToolResult
    try:
        import pyautogui
    except ImportError:
        return ToolResult(success=False, error="pyautogui not installed")

    try:
        img = pyautogui.screenshot()
    except Exception as e:
        return ToolResult(success=False, error=f"Screenshot failed: {e}")

    # Try vision LLM if available
    if llm_client is not None:
        try:
            import base64
            import io
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            desc = llm_client.complete(
                prompt="Describe what you see on this screen briefly.",
                images=[b64],
                task_type="vision",
            )
            return ToolResult(success=True, data={"description": desc})
        except Exception:
            pass  # Fall through to OCR

    # OCR fallback
    try:
        import pytesseract
        text = pytesseract.image_to_string(img)
        return ToolResult(success=True, data={"ocr_text": text[:2000]})
    except ImportError:
        return ToolResult(success=True, data={"description": "Screen captured but no OCR available"})
    except Exception as e:
        logger.warning("describe_screen OCR failed: %s", e)
        return ToolResult(success=True, data={"description": "Screen captured but OCR failed"})


__all__ = [
    "capture_screen",
    "capture_region",
    "find_text_on_screen",
    "describe_screen",
]
