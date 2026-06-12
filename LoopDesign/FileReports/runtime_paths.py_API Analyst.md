# API Analyst Report: runtime\paths.py

## Dependencies
- `from __future__ import annotations`
- `import os`
- `from pathlib import Path`

## Configuration Variables
- `PROJECT_ROOT` = `Path(__file__).resolve().parents[2]`

## Functions & Endpoints

### `_resolve_path`
`def _resolve_path(path_str: str | Path) -> Path`
> Resolve a path relative to the project root directory if it is relative,
otherwise return it as an absolute path.
