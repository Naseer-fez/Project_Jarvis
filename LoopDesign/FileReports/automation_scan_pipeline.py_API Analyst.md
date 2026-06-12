# API Analyst Report: automation\scan_pipeline.py

## Dependencies
- `from __future__ import annotations`
- `from dataclasses import dataclass`
- `from pathlib import Path`
- `from typing import Awaitable`
- `from typing import Callable`

## Configuration Variables
- `SUMMARY_KEYS` = `('commands_processed', 'files_ingested', 'chunks_ingested', 'failed_files', 'skipped_files', 'scanned_files')`

## Schemas & API Contracts (Classes)

### Class `ScanBatch`
**Fields/Schema:**
  - `name: str`
  - `candidates: tuple[Path, ...]`
  - `mark_seen: bool`
  - `process: Callable[[Path], Awaitable[dict[str, int]]]`
  - `on_preexisting: Callable[[Path], None] | None`
  - `on_error: Callable[[Path, Exception], None] | None`



### Class `ScanPipeline`
**Methods:**
- `def __init__(self, batches: list[ScanBatch] | tuple[ScanBatch, ...]) -> None`
- `async def run(self, readiness: ReadinessCheck) -> dict[str, int]`


## Functions & Endpoints

### `blank_scan_summary`
`def blank_scan_summary() -> dict[str, int]`