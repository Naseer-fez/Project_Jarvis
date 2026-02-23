from __future__ import annotations

import configparser
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

_GENESIS_HASH = "0" * 64

_logger: logging.Logger | None = None
_audit: "AuditLog | None" = None


def _parse_level(level: str | int) -> int:
    if isinstance(level, int):
        return level
    mapping = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    return mapping.get(str(level).upper(), logging.INFO)


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


def _entry_hash(ts: str, event: str, payload: Any, prev_hash: str) -> str:
    body = {
        "ts": ts,
        "event": event,
        "payload": payload,
        "prev_hash": prev_hash,
    }
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class AuditLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._last_hash = self._load_last_hash()

    def _load_last_hash(self) -> str:
        if not self.path.exists():
            return _GENESIS_HASH

        try:
            with self.path.open("r", encoding="utf-8") as handle:
                for line in reversed(handle.read().splitlines()):
                    raw = line.strip()
                    if not raw:
                        continue
                    try:
                        entry = json.loads(raw)
                    except json.JSONDecodeError:
                        return _GENESIS_HASH
                    digest = entry.get("hash")
                    if isinstance(digest, str) and len(digest) == 64:
                        return digest
                    return _GENESIS_HASH
        except OSError:
            return _GENESIS_HASH

        return _GENESIS_HASH

    def write(self, event: str, payload: dict[str, Any]) -> str:
        event_name = str(event)
        payload_safe = _json_safe(payload if isinstance(payload, dict) else {"value": payload})
        ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")

        with self._lock:
            prev_hash = self._last_hash
            digest = _entry_hash(ts=ts, event=event_name, payload=payload_safe, prev_hash=prev_hash)
            entry = {
                "ts": ts,
                "event": event_name,
                "payload": payload_safe,
                "prev_hash": prev_hash,
                "hash": digest,
            }
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self._last_hash = digest

        return digest

    def verify(self) -> tuple[bool, int, str | None]:
        if not self.path.exists():
            return True, 0, None

        count = 0
        prev_hash = _GENESIS_HASH

        try:
            with self.path.open("r", encoding="utf-8") as handle:
                for line_no, raw_line in enumerate(handle, start=1):
                    raw = raw_line.strip()
                    if not raw:
                        continue

                    try:
                        entry = json.loads(raw)
                    except json.JSONDecodeError as exc:
                        return False, count, f"line {line_no}: invalid JSON ({exc})"

                    for field in ("ts", "event", "payload", "prev_hash", "hash"):
                        if field not in entry:
                            return False, count, f"line {line_no}: missing field '{field}'"

                    if not isinstance(entry["prev_hash"], str) or not isinstance(entry["hash"], str):
                        return False, count, f"line {line_no}: invalid hash types"

                    if entry["prev_hash"] != prev_hash:
                        return (
                            False,
                            count,
                            f"line {line_no}: prev_hash mismatch (expected {prev_hash}, found {entry['prev_hash']})",
                        )

                    expected = _entry_hash(
                        ts=str(entry["ts"]),
                        event=str(entry["event"]),
                        payload=entry["payload"],
                        prev_hash=entry["prev_hash"],
                    )
                    if entry["hash"] != expected:
                        return False, count, f"line {line_no}: hash mismatch"

                    prev_hash = entry["hash"]
                    count += 1
        except OSError as exc:
            return False, count, f"I/O error: {exc}"

        with self._lock:
            self._last_hash = prev_hash

        return True, count, None


def _resolve_setup(
    config_or_name: configparser.ConfigParser | str = "Jarvis",
    level: str | int = "INFO",
    log_file: str | Path | None = None,
) -> tuple[str, str | int, str | None, str]:
    name = "Jarvis"
    resolved_level: str | int = level
    resolved_log_file: str | None = None
    audit_file = "logs/audit.jsonl"

    if isinstance(config_or_name, configparser.ConfigParser):
        cfg = config_or_name
        name = cfg.get("general", "name", fallback="Jarvis")
        resolved_level = cfg.get("logging", "level", fallback=str(level))
        resolved_log_file = cfg.get("logging", "app_file", fallback="")
        if not resolved_log_file:
            resolved_log_file = cfg.get("logging", "file", fallback="")
        if not resolved_log_file:
            resolved_log_file = None
        audit_file = cfg.get("logging", "audit_file", fallback="logs/audit.jsonl")
    else:
        if isinstance(config_or_name, str) and config_or_name.strip():
            name = config_or_name.strip()
        if isinstance(log_file, Path):
            resolved_log_file = str(log_file)
        elif isinstance(log_file, str) and log_file.strip():
            resolved_log_file = log_file

    return name, resolved_level, resolved_log_file, audit_file


def setup(
    config_or_name: configparser.ConfigParser | str = "Jarvis",
    level: str | int = "INFO",
    log_file: str | Path | None = None,
) -> logging.Logger:
    global _logger, _audit

    name, resolved_level, resolved_log_file, audit_file = _resolve_setup(
        config_or_name=config_or_name,
        level=level,
        log_file=log_file,
    )

    logger = logging.getLogger(name)
    logger.setLevel(_parse_level(resolved_level))
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if resolved_log_file:
        path = Path(resolved_log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _logger = logger
    _audit = AuditLog(audit_file)
    logger.debug("Logging subsystem initialized")
    return logger


def get(name: str | None = None) -> logging.Logger:
    if name:
        return logging.getLogger(name)
    if _logger is not None:
        return _logger
    return logging.getLogger("Jarvis")


def get_logger(name: str = "Jarvis") -> logging.Logger:
    return get(name=name)


def audit(event_type: str, payload: dict[str, Any]) -> str:
    global _audit
    if _audit is None:
        _audit = AuditLog("logs/audit.jsonl")
    return _audit.write(event_type, payload)


def verify_audit() -> tuple[bool, int, str | None]:
    global _audit
    if _audit is None:
        _audit = AuditLog("logs/audit.jsonl")
    return _audit.verify()


__all__ = [
    "AuditLog",
    "setup",
    "get",
    "get_logger",
    "audit",
    "verify_audit",
]
