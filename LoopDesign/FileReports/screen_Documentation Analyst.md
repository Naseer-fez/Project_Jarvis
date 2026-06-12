# Analysis Report for screen.py

## Dependencies
- __future__.annotations
- core.types.common.ToolResult
- asyncio
- logging
- time
- datetime.datetime
- pathlib.Path
- typing.Any

## Schemas
None

## API Contracts
- _ts()
- _ensure_dir()
- _require_pyautogui()
- _clean_text(value)
- _safe_int(value, default)
- _safe_float(value, default)
- _capture_image()
- _ocr_words_from_data(data)
- _ocr_lines_from_words(words)
- _match_entries(entries, query)
- capture_screen()
- capture_region(x, y, width, height)
- read_screen_text(query)
- find_text_on_screen(text, match_mode)
- describe_screen(llm_client)

## Configuration Variables
- SCREENSHOT_DIR
- _OCR_MAX_TEXT_CHARS
- _TESSERACT_WARN_LOGGED

## Assumptions & Notes
- Module Docstring: Screen capture, OCR, and screen-state tools for Jarvis.

