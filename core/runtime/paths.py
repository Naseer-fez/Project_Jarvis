from __future__ import annotations

import os
from pathlib import Path

# Path to the project root folder (two levels up from core/runtime/paths.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(path_str: str | Path) -> Path:
    """
    Resolve a path relative to the project root directory if it is relative,
    otherwise return it as an absolute path.
    """
    raw_path = Path(os.path.expandvars(str(path_str))).expanduser()
    if raw_path.is_absolute():
        return raw_path.resolve(strict=False)
    return (PROJECT_ROOT / raw_path).resolve(strict=False)
