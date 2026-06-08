"""Screen capture, OCR, and screen-state tools for Jarvis."""

from __future__ import annotations
from core.types.common import ToolResult

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SCREENSHOT_DIR = Path("outputs/screenshots")
_OCR_MAX_TEXT_CHARS = 4000
_TESSERACT_WARN_LOGGED = False



def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_dir() -> None:
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)


def _require_pyautogui():
    try:
        import pyautogui

        return pyautogui
    except ImportError as exc:
        raise ImportError("pyautogui not installed") from exc


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _capture_image(*, region: tuple[int, int, int, int] | None = None):
    pyautogui = _require_pyautogui()
    return pyautogui.screenshot(region=region)


def _ocr_words_from_data(data: dict[str, Any]) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    count = len(data.get("text", []))
    for idx in range(count):
        text = _clean_text(data["text"][idx])
        if not text:
            continue
        width = _safe_int(data.get("width", [0])[idx])
        height = _safe_int(data.get("height", [0])[idx])
        words.append(
            {
                "text": text,
                "x": _safe_int(data.get("left", [0])[idx]),
                "y": _safe_int(data.get("top", [0])[idx]),
                "w": width,
                "h": height,
                "width": width,
                "height": height,
                "confidence": _safe_float(data.get("conf", [0])[idx]),
                "block_num": _safe_int(data.get("block_num", [0])[idx]),
                "par_num": _safe_int(data.get("par_num", [0])[idx]),
                "line_num": _safe_int(data.get("line_num", [0])[idx]),
                "source": "word",
            }
        )
    return words


def _ocr_lines_from_words(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for word in words:
        key = (
            _safe_int(word.get("block_num")),
            _safe_int(word.get("par_num")),
            _safe_int(word.get("line_num")),
        )
        grouped.setdefault(key, []).append(word)

    lines: list[dict[str, Any]] = []
    for line_words in grouped.values():
        ordered = sorted(line_words, key=lambda item: (item.get("y", 0), item.get("x", 0)))
        if not ordered:
            continue
        left = min(_safe_int(item.get("x")) for item in ordered)
        top = min(_safe_int(item.get("y")) for item in ordered)
        right = max(_safe_int(item.get("x")) + _safe_int(item.get("w")) for item in ordered)
        bottom = max(_safe_int(item.get("y")) + _safe_int(item.get("h")) for item in ordered)
        text = _clean_text(" ".join(str(item.get("text", "")) for item in ordered))
        if not text:
            continue
        lines.append(
            {
                "text": text,
                "x": left,
                "y": top,
                "w": max(0, right - left),
                "h": max(0, bottom - top),
                "width": max(0, right - left),
                "height": max(0, bottom - top),
                "confidence": round(
                    sum(_safe_float(item.get("confidence")) for item in ordered) / max(1, len(ordered)),
                    3,
                ),
                "word_count": len(ordered),
                "source": "line",
            }
        )
    return sorted(lines, key=lambda item: (item.get("y", 0), item.get("x", 0)))


def _match_entries(
    entries: list[dict[str, Any]],
    query: str,
    *,
    match_mode: str = "contains",
) -> list[dict[str, Any]]:
    normalized_query = _clean_text(query).casefold()
    if not normalized_query:
        return []

    matches: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int, int, int]] = set()
    for entry in entries:
        haystack = _clean_text(entry.get("text", "")).casefold()
        if not haystack:
            continue

        matched = False
        if match_mode == "exact":
            matched = haystack == normalized_query
        elif match_mode == "starts_with":
            matched = haystack.startswith(normalized_query)
        else:
            matched = normalized_query in haystack

        if not matched:
            continue

        key = (
            _clean_text(entry.get("text", "")),
            _safe_int(entry.get("x")),
            _safe_int(entry.get("y")),
            _safe_int(entry.get("w", entry.get("width"))),
            _safe_int(entry.get("h", entry.get("height"))),
        )
        if key in seen:
            continue
        seen.add(key)
        matches.append(
            {
                "text": _clean_text(entry.get("text", "")),
                "x": _safe_int(entry.get("x")),
                "y": _safe_int(entry.get("y")),
                "w": _safe_int(entry.get("w", entry.get("width"))),
                "h": _safe_int(entry.get("h", entry.get("height"))),
                "width": _safe_int(entry.get("width", entry.get("w"))),
                "height": _safe_int(entry.get("height", entry.get("h"))),
                "confidence": _safe_float(entry.get("confidence")),
                "source": str(entry.get("source", "word") or "word"),
            }
        )
    return matches


def capture_screen():
    """Take a full-screen screenshot and save it to outputs/screenshots/."""

    try:
        _ensure_dir()
        path = SCREENSHOT_DIR / f"{_ts()}.png"
        img = _capture_image()
        img.save(str(path))
        logger.info("Screenshot saved: %s", path)
        return ToolResult(success=True, data={"path": str(path), "width": img.width, "height": img.height})
    except Exception as exc:  # noqa: BLE001
        logger.error("capture_screen failed: %s", exc, exc_info=True)
        return ToolResult(success=False, error=str(exc))


def capture_region(x: int, y: int, width: int, height: int):
    """Screenshot a specific screen region."""

    try:
        _ensure_dir()
        path = SCREENSHOT_DIR / f"{_ts()}_region.png"
        img = _capture_image(region=(x, y, width, height))
        img.save(str(path))
        logger.info("Region screenshot saved: %s", path)
        return ToolResult(success=True, data={"path": str(path), "x": x, "y": y, "width": width, "height": height})
    except Exception as exc:  # noqa: BLE001
        logger.error("capture_region failed: %s", exc, exc_info=True)
        return ToolResult(success=False, error=str(exc))


def read_screen_text(
    query: str = "",
    *,
    match_mode: str = "contains",
    include_words: bool = False,
    include_lines: bool = False,
    max_text_chars: int = _OCR_MAX_TEXT_CHARS,
):
    """Read visible screen text with OCR and optional query matching."""

    try:
        import os
        import sys
        import pytesseract
        from PIL import Image  # noqa: F401

        # Configure Tesseract path dynamically
        if getattr(sys, "frozen", False):
            # Packaged desktop app mode: use bundled Tesseract binary from temporary directory
            base_dir = getattr(sys, '_MEIPASS')
            tesseract_dir = os.path.join(base_dir, "bin", "tesseract")
            os.environ["TESSDATA_PREFIX"] = os.path.join(tesseract_dir, "tessdata")
            pytesseract.pytesseract.tesseract_cmd = os.path.join(tesseract_dir, "tesseract.exe")
        else:
            # Development mode: check if custom TESSERACT_CMD env variable is set
            tesseract_cmd = os.environ.get("TESSERACT_CMD")
            if tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            else:
                # Check for the local bundled folder inside the workspace
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                local_bundled = os.path.join(project_root, "bin", "tesseract", "tesseract.exe")
                if os.path.exists(local_bundled):
                    pytesseract.pytesseract.tesseract_cmd = local_bundled
                    os.environ["TESSDATA_PREFIX"] = os.path.join(project_root, "bin", "tesseract", "tessdata")
    except ImportError as exc:
        return ToolResult(success=False, error=f"Missing dependency: {exc}")

    try:
        img = _capture_image()
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        words = _ocr_words_from_data(data)
        lines = _ocr_lines_from_words(words)
        full_text = "\n".join(line["text"] for line in lines)

        matches: list[dict[str, Any]] = []
        if _clean_text(query):
            matches.extend(_match_entries(lines, query, match_mode=match_mode))
            matches.extend(_match_entries(words, query, match_mode=match_mode))

        payload: dict[str, Any] = {
            "query": query,
            "text": full_text[:max_text_chars],
            "matches": matches,
            "match_count": len(matches),
            "line_count": len(lines),
            "word_count": len(words),
        }
        if include_lines:
            payload["lines"] = lines
        if include_words:
            payload["words"] = words

        logger.info("read_screen_text(query=%r): %d match(es)", query, len(matches))
        return ToolResult(success=True, data=payload)
    except Exception as exc:  # noqa: BLE001
        global _TESSERACT_WARN_LOGGED
        err_msg = str(exc)
        if "tesseract is not installed" in err_msg or "TesseractNotFoundError" in type(exc).__name__:
            if not _TESSERACT_WARN_LOGGED:
                logger.warning(
                    "read_screen_text: Tesseract OCR is not installed or not in your PATH. "
                    "Screen-aware features will be limited. See README.MD to set it up."
                )
                _TESSERACT_WARN_LOGGED = True
            else:
                logger.debug("read_screen_text failed (Tesseract not installed): %s", exc)
        else:
            logger.error("read_screen_text failed: %s", exc, exc_info=True)
        return ToolResult(success=False, error=err_msg)


def find_text_on_screen(text: str, match_mode: str = "contains"):
    """Search for visible screen text using phrase-aware OCR matching."""

    result = read_screen_text(
        query=text,
        match_mode=match_mode,
        include_lines=False,
        include_words=False,
    )
    if not result.success:
        return result

    return ToolResult(
        success=True,
        data={
            "query": text,
            "match_mode": match_mode,
            "matches": list(result.data.get("matches", [])),
            "text": str(result.data.get("text", "") or ""),
        },
    )


async def wait_for_text_on_screen(
    text: str,
    *,
    timeout_seconds: float = 10.0,
    poll_interval_seconds: float = 0.5,
    match_mode: str = "contains",
):
    """Poll OCR until the requested text appears on screen."""

    timeout_value = max(0.1, float(timeout_seconds))
    deadline = time.monotonic() + timeout_value
    started_at = time.monotonic()
    attempts = 0
    last_error = ""

    while time.monotonic() < deadline:
        attempts += 1
        result = find_text_on_screen(text, match_mode=match_mode)
        if result.success and result.data.get("matches"):
            payload = dict(result.data)
            payload["attempts"] = attempts
            payload["elapsed_seconds"] = round(max(0.0, time.monotonic() - started_at), 3)
            return ToolResult(success=True, data=payload)
        if not result.success:
            last_error = result.error
        await asyncio.sleep(max(0.1, float(poll_interval_seconds)))

    return ToolResult(
        success=False,
        data={
            "query": text,
            "attempts": attempts,
            "elapsed_seconds": round(max(0.0, time.monotonic() - started_at), 3),
        },
        error=last_error or f"Timed out waiting for '{text}' to appear on screen.",
    )


def describe_screen(llm_client=None):
    """Describe the current screen contents."""

    try:
        img = _capture_image()
    except Exception as exc:  # noqa: BLE001
        return ToolResult(success=False, error=f"Screenshot failed: {exc}")

    if llm_client is not None:
        try:
            import base64
            import io

            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            b64 = base64.b64encode(buffer.getvalue()).decode()
            description = llm_client.complete(
                prompt="Describe what you see on this screen briefly.",
                images=[b64],
                task_type="vision",
            )
            return ToolResult(success=True, data={"description": description})
        except Exception:
            pass

    ocr_result = read_screen_text()
    if ocr_result.success:
        return ToolResult(success=True, data={"ocr_text": str(ocr_result.data.get("text", "") or "")})
    if "Missing dependency" in ocr_result.error:
        return ToolResult(success=True, data={"description": "Screen captured but no OCR backend is available."})
    logger.warning("describe_screen OCR failed: %s", ocr_result.error)
    return ToolResult(success=True, data={"description": "Screen captured but OCR failed."})


__all__ = [
    "capture_screen",
    "capture_region",
    "describe_screen",
    "find_text_on_screen",
    "read_screen_text",
    "wait_for_text_on_screen",
]
