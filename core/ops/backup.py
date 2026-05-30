from __future__ import annotations

import shutil
import time
import zipfile
from pathlib import Path
from typing import Any

from core.runtime.bootstrap import PROJECT_ROOT, _resolve_path


def _config_get(config: Any, section: str, key: str, fallback: str) -> str:
    try:
        return str(config.get(section, key, fallback=fallback))
    except Exception:
        return fallback


def create_backup(config: Any, output_dir: str | Path = "outputs/backups") -> Path:
    target_dir = _resolve_path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    archive = target_dir / f"jarvis-backup-{time.strftime('%Y%m%d-%H%M%S')}.zip"

    candidates = [
        _config_get(config, "memory", "sqlite_file", "data/jarvis_memory.db"),
        _config_get(config, "memory", "goals_file", "data/goals.json"),
        _config_get(config, "logging", "audit_file", "logs/audit.jsonl"),
        _config_get(config, "dashboard", "control_file", "runtime/control_flags.json"),
        "config/jarvis.production.ini",
    ]
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for raw in candidates:
            path = _resolve_path(raw)
            if path.exists() and path.is_file():
                zf.write(path, path.relative_to(PROJECT_ROOT))
    return archive


def restore_backup(archive_path: str | Path, *, overwrite: bool = False) -> list[Path]:
    archive = _resolve_path(archive_path)
    restored: list[Path] = []
    if not archive.exists():
        raise FileNotFoundError(str(archive))

    allowed_roots = [
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "logs",
        PROJECT_ROOT / "runtime",
        PROJECT_ROOT / "config",
    ]
    with zipfile.ZipFile(archive, "r") as zf:
        for member in zf.infolist():
            target = (PROJECT_ROOT / member.filename).resolve()
            if not any(target.is_relative_to(root.resolve()) for root in allowed_roots):
                raise ValueError(f"Backup member outside restore roots: {member.filename}")
            if target.exists() and not overwrite:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            restored.append(target)
    return restored


__all__ = ["create_backup", "restore_backup"]
