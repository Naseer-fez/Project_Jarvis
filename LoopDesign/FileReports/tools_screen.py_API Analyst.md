# API Analyst Report: tools\screen.py

## Dependencies
- `from __future__ import annotations`
- `from core.types.common import ToolResult`
- `import asyncio`
- `import logging`
- `import time`
- `from datetime import datetime`
- `from pathlib import Path`
- `from typing import Any`

## Configuration Variables
- `SCREENSHOT_DIR` = `Path('outputs/screenshots')`
- `_OCR_MAX_TEXT_CHARS` = `4000`
- `_TESSERACT_WARN_LOGGED` = `False`
- `_TESSERACT_WARN_LOGGED` = `True`

## Functions & Endpoints

### `_ts`
`def _ts() -> str`
### `_ensure_dir`
`def _ensure_dir() -> None`
### `_require_pyautogui`
`def _require_pyautogui()`
### `_clean_text`
`def _clean_text(value: Any) -> str`
### `_safe_int`
`def _safe_int(value: Any, default: int=0) -> int`
### `_safe_float`
`def _safe_float(value: Any, default: float=0.0) -> float`
### `_capture_image`
`def _capture_image(*, region: tuple[int, int, int, int] | None=None)`
### `_ocr_words_from_data`
`def _ocr_words_from_data(data: dict[str, Any]) -> list[dict[str, Any]]`
### `_ocr_lines_from_words`
`def _ocr_lines_from_words(words: list[dict[str, Any]]) -> list[dict[str, Any]]`
### `_match_entries`
`def _match_entries(entries: list[dict[str, Any]], query: str, *, match_mode: str='contains') -> list[dict[str, Any]]`
### `capture_screen`
`def capture_screen()`
> Take a full-screen screenshot and save it to outputs/screenshots/.

### `capture_region`
`def capture_region(x: int, y: int, width: int, height: int)`
> Screenshot a specific screen region.

### `read_screen_text`
`def read_screen_text(query: str='', *, match_mode: str='contains', include_words: bool=False, include_lines: bool=False, max_text_chars: int=_OCR_MAX_TEXT_CHARS)`
> Read visible screen text with OCR and optional query matching.

### `find_text_on_screen`
`def find_text_on_screen(text: str, match_mode: str='contains')`
> Search for visible screen text using phrase-aware OCR matching.

### `wait_for_text_on_screen`
`async def wait_for_text_on_screen(text: str, *, timeout_seconds: float=10.0, poll_interval_seconds: float=0.5, match_mode: str='contains')`
> Poll OCR until the requested text appears on screen.

### `describe_screen`
`def describe_screen(llm_client=None)`
> Describe the current screen contents.
