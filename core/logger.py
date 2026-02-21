"""
core/logger.py — Immutable audit + application logging.

Audit log: append-only JSONL. Each entry is SHA-256 chained to the previous
so any tampering is detectable.

App log: standard rotating file + coloured console output.
"""

from __future__ import annotations

import hashlib
import json
import logging
import logging.handlers
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from colorama import Fore, Style, init as colorama_init
except ImportError:
    class _DummyColor:
        BLACK = ""
        RED = ""
        GREEN = ""
        YELLOW = ""
        BLUE = ""
        MAGENTA = ""
        CYAN = ""
        WHITE = ""
        RESET_ALL = ""

    Fore = _DummyColor()
    Style = _DummyColor()

    def colorama_init(*args, **kwargs):
        del args, kwargs
        return None

colorama_init(autoreset=True)

# ── Colour map ────────────────────────────────────────────────────────────────
_LEVEL_COLOURS = {
    "DEBUG":    Fore.CYAN,
    "INFO":     Fore.GREEN,
    "WARNING":  Fore.YELLOW,
    "ERROR":    Fore.RED,
    "CRITICAL": Fore.MAGENTA,
}


class ColouredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        colour = _LEVEL_COLOURS.get(record.levelname, "")
        prefix = f"{colour}[{record.levelname[:4]}]{Style.RESET_ALL}"
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime("%H:%M:%S")
        return f"{Fore.WHITE}{ts}{Style.RESET_ALL} {prefix} {record.getMessage()}"


# ── Audit log ─────────────────────────────────────────────────────────────────
class AuditLog:
    """Append-only, hash-chained JSONL audit log."""

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._prev_hash = self._load_last_hash()

    def _load_last_hash(self) -> str:
        genesis = "0" * 64
        if not self._path.exists():
            return genesis
        last_line = ""
        with self._path.open("r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if stripped:
                    last_line = stripped
        if not last_line:
            return genesis
        try:
            entry = json.loads(last_line)
            return entry.get("hash", genesis)
        except (json.JSONDecodeError, KeyError):
            return genesis

    def write(self, event_type: str, payload: dict[str, Any]) -> str:
        """Append one entry. Returns the entry's hash."""
        with self._lock:
            entry = {
                "ts": datetime.now(tz=timezone.utc).isoformat(),
                "event": event_type,
                "payload": payload,
                "prev_hash": self._prev_hash,
            }
            serialised = json.dumps(entry, ensure_ascii=False, sort_keys=True)
            entry_hash = hashlib.sha256(serialised.encode()).hexdigest()
            entry["hash"] = entry_hash

            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

            self._prev_hash = entry_hash
            return entry_hash

    def verify(self) -> tuple[bool, int, str]:
        """Re-compute the hash chain. Returns (ok, entries_checked, error_msg)."""
        if not self._path.exists():
            return True, 0, ""
        prev = "0" * 64
        count = 0
        with self._path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as exc:
                    return False, count, f"Line {lineno}: invalid JSON — {exc}"

                stored_hash = entry.pop("hash", None)
                serialised = json.dumps(entry, ensure_ascii=False, sort_keys=True)
                computed = hashlib.sha256(serialised.encode()).hexdigest()

                if stored_hash != computed:
                    return False, count, f"Line {lineno}: hash mismatch"
                if entry.get("prev_hash") != prev:
                    return False, count, f"Line {lineno}: chain broken"
                prev = stored_hash
                count += 1
        return True, count, ""


# ── Module-level singletons (initialised by setup()) ─────────────────────────
_audit: AuditLog | None = None
_app_logger: logging.Logger | None = None


def setup(config: Any) -> None:
    """Call once at startup with the parsed ConfigParser object."""
    global _audit, _app_logger

    log_dir = Path(config.get("logging", "log_dir", fallback="logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    audit_path = config.get("logging", "audit_file", fallback="logs/audit.jsonl")
    _audit = AuditLog(audit_path)

    app_path = config.get("logging", "app_file", fallback="logs/app.log")
    level_str = config.get("logging", "level", fallback="INFO")
    level = getattr(logging, level_str.upper(), logging.INFO)

    logger = logging.getLogger("jarvis")
    logger.setLevel(level)
    logger.handlers.clear()

    # Console
    ch = logging.StreamHandler()
    ch.setFormatter(ColouredFormatter())
    ch.setLevel(level)
    logger.addHandler(ch)

    # File (rotating, 5 MB × 3)
    fh = logging.handlers.RotatingFileHandler(
        app_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    fh.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"
    ))
    fh.setLevel(level)
    logger.addHandler(fh)

    _app_logger = logger


def get() -> logging.Logger:
    if _app_logger is None:
        raise RuntimeError("Logger not initialised — call logger.setup() first")
    return _app_logger


def audit(event_type: str, payload: dict[str, Any]) -> str:
    if _audit is None:
        raise RuntimeError("Audit log not initialised — call logger.setup() first")
    return _audit.write(event_type, payload)


def verify_audit() -> tuple[bool, int, str]:
    if _audit is None:
        return False, 0, "Audit log not initialised"
    return _audit.verify()
