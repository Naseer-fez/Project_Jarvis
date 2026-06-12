# API Analyst Report: automation\live_automation.py

## Dependencies
- `from __future__ import annotations`
- `import asyncio`
- `import contextlib`
- `import hashlib`
- `import json`
- `import logging`
- `import re`
- `import shutil`
- `import time`
- `from dataclasses import asdict`
- `from dataclasses import dataclass`
- `from datetime import datetime`
- `from datetime import timezone`
- `from pathlib import Path`
- `from typing import Any`
- `from typing import Awaitable`
- `from typing import Callable`
- `from core.automation.scan_pipeline import ScanBatch`
- `from core.automation.scan_pipeline import ScanPipeline`
- `from core.automation.scan_rules import ScanRoute`
- `from core.automation.scan_rules import build_scan_routes`
- `from core.runtime.paths import _resolve_path`

## Configuration Variables
- `_TEXT_EXTENSIONS` = `{'.txt', '.md', '.rst', '.json', '.yaml', '.yml', '.csv', '.tsv', '.py', '.js', '.ts', '.html', '.css', '.ini', '.log'}`
- `_IMAGE_EXTENSIONS` = `{'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff'}`
- `_VIDEO_EXTENSIONS` = `{'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'}`
- `_COMMAND_EXTENSIONS` = `{'.txt', '.md', '.task', '.cmd'}`
- `_DEFAULT_DROP_ROOT` = `'workspace/jarvis_dropbox'`
- `_DEFAULT_SCREENSHOT_DIR` = `'outputs/screenshots'`
- `_DEFAULT_RECORDING_DIR` = `'outputs/screen_recordings'`

## Schemas & API Contracts (Classes)

### Class `AutomationStats`
**Fields/Schema:**
  - `started_at: str`
  - `last_scan_at: str`
  - `last_error: str`
  - `scanned_files: int`
  - `ingested_files: int`
  - `ingested_chunks: int`
  - `commands_executed: int`
  - `failed_files: int`
  - `skipped_files: int`
  - `live_screen_updates: int`



### Class `LiveAutomationEngine`
> Poll-based automation engine for command inbox and RAG ingestion.

**Methods:**
- `def __init__(self, *, config: Any, memory: Any, llm: Any | None=None, command_handler: CommandHandler | None=None, desktop_observer: Any | None=None, notifier: Any | None=None, dag_executor: Any | None=None) -> None`
- `async def start(self) -> None`
- `async def stop(self) -> None`
- `async def enable(self) -> dict[str, Any]`
- `async def disable(self) -> dict[str, Any]`
- `async def force_scan(self) -> dict[str, Any]`
- `async def scan_once(self) -> dict[str, Any]`
- `async def _build_scan_batches(self) -> list[ScanBatch]`
- `def _build_command_scan_batch(self, route: ScanRoute, candidates: tuple[Path, ...]) -> ScanBatch`
- `def _build_ingest_scan_batch(self, route: ScanRoute, candidates: tuple[Path, ...]) -> ScanBatch`
- `def _scan_readiness(self, path: Path, mark_seen: bool) -> tuple[bool, str]`
- `def _handle_scan_failure(self, route: ScanRoute, path: Path, exc: Exception) -> None`
- `def _apply_scan_summary(self, summary: dict[str, int]) -> None`
- `def status(self) -> dict[str, Any]`
- `def status_line(self) -> str`
- `async def search_rag(self, query: str, top_k: int=5) -> str`
- `async def _run_loop(self) -> None`
- `async def _process_command_file(self, path: Path) -> None`
- `async def _ingest_file(self, path: Path, *, source: str, move_after: bool) -> int`
- `def _extract_text_payload(self, path: Path) -> str`
- `async def _poll_live_screen(self) -> None`
- `async def _store_rag_text(self, *, source: str, path: Path, text: str) -> int`
- `def _file_ready(self, path: Path, *, mark_seen: bool) -> tuple[bool, str]`
- `async def _iter_files(self, folder: Path, allowed_extensions: set[str] | None) -> list[Path]`
- @staticmethod
- `def _extract_command(raw_text: str) -> str`
- @staticmethod
- `async def _read_text_file(path: Path, max_bytes: int=2000000) -> str`
- `def _move_to_failed(self, path: Path, *, error: str) -> None`
- `def _relocate(self, source: Path, destination_dir: Path) -> Path`
- @staticmethod
- `def _unique_path(path: Path) -> Path`
- `def _fingerprint(self, path: Path, stat: Any | None=None) -> str`
- `def _remember_file(self, path: Path) -> None`
- `def _remember_fingerprint(self, fingerprint: str) -> None`
- `def _ensure_directories(self) -> None`
- `async def _append_log(self, payload: dict[str, Any]) -> None`
- `def _load_state(self) -> None`
- `def _save_state(self) -> None`
- @staticmethod
- `def _extract_metadata_value(block: str, key: str) -> str`


## Functions & Endpoints

### `_cfg_bool`
`def _cfg_bool(config: Any, section: str, key: str, fallback: bool) -> bool`
### `_cfg_float`
`def _cfg_float(config: Any, section: str, key: str, fallback: float) -> float`
### `_cfg_int`
`def _cfg_int(config: Any, section: str, key: str, fallback: int) -> int`
### `_iso_now`
`def _iso_now() -> str`
### `_normalize_text`
`def _normalize_text(value: str) -> str`
### `_truncate`
`def _truncate(text: str, max_chars: int) -> str`