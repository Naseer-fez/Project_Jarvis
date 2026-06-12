# File Report: screen.py
**Role**: Prompt Recovery Specialist

## Dependencies
- datetime
- pyautogui
- time
- typing
- core.types.common
- logging
- asyncio
- sys
- pytesseract
- __future__
- PIL
- base64
- io
- os
- pathlib

## Configuration Variables & Constants
- `last_error`: ``

## Schemas & API Contracts
### Function `_ts`
**Args**: 

### Function `_ensure_dir`
**Args**: 

### Function `_require_pyautogui`
**Args**: 

### Function `_clean_text`
**Args**: value

### Function `_safe_int`
**Args**: value, default

### Function `_safe_float`
**Args**: value, default

### Function `_capture_image`
**Args**: 

### Function `_ocr_words_from_data`
**Args**: data

### Function `_ocr_lines_from_words`
**Args**: words

### Function `_match_entries`
**Args**: entries, query

### Function `capture_screen`
**Args**: 
**Assumptions/Doc**: Take a full-screen screenshot and save it to outputs/screenshots/.

### Function `capture_region`
**Args**: x, y, width, height
**Assumptions/Doc**: Screenshot a specific screen region.

### Function `read_screen_text`
**Args**: query
**Assumptions/Doc**: Read visible screen text with OCR and optional query matching.

### Function `find_text_on_screen`
**Args**: text, match_mode
**Assumptions/Doc**: Search for visible screen text using phrase-aware OCR matching.

### Function `wait_for_text_on_screen`
**Args**: text
**Assumptions/Doc**: Poll OCR until the requested text appears on screen.

### Function `describe_screen`
**Args**: llm_client
**Assumptions/Doc**: Describe the current screen contents.

## Prompts and LLM Directives
No explicit prompts found in module scope.
