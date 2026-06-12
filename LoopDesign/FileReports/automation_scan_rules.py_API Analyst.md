# API Analyst Report: automation\scan_rules.py

## Dependencies
- `from __future__ import annotations`
- `from dataclasses import dataclass`
- `from pathlib import Path`
- `from typing import Literal`

## Schemas & API Contracts (Classes)

### Class `ScanRoute`
**Fields/Schema:**
  - `name: str`
  - `kind: ScanRouteKind`
  - `folder: Path`
  - `allowed_extensions: set[str] | None`
  - `mark_seen: bool`
  - `source: str`
  - `move_after: bool`
  - `move_to_failed: bool`
  - `failure_label: str`



## Functions & Endpoints

### `build_scan_routes`
`def build_scan_routes(*, commands_dir: Path, rag_dir: Path, screenshots_dir: Path, recordings_dir: Path, command_extensions: set[str], image_extensions: set[str], video_extensions: set[str], watch_screenshots: bool, watch_recordings: bool) -> tuple[ScanRoute, ...]`