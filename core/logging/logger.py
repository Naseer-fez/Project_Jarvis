"""Application logging and append-only audit log support."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any


class AuditLog:
    def __init__(self, file_path: str) -> None:
        self.path = Path(file_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._last_hash = "0" * 64

    def write(self, event_type: str, payload: dict[str, Any]) -> str:
        body = {
            "event_type": event_type,
            "payload": payload,
            "prev_hash": self._last_hash,
        }
        digest = hashlib.sha256(
            json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        record = body | {"hash": digest}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._last_hash = digest
        return digest

    def verify(self) -> tuple[bool, int, str]:
        if not self.path.exists():
            return True, 0, ""

        previous = "0" * 64
        count = 0
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            expected = hashlib.sha256(
                json.dumps(
                    {
                        "event_type": payload.get("event_type"),
                        "payload": payload.get("payload"),
                        "prev_hash": previous,
                    },
                    sort_keys=True,
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest()
            if payload.get("hash") != expected:
                return False, count, "Hash chain mismatch"
            previous = expected
            count += 1
        return True, count, ""


_logger = logging.getLogger("Jarvis")
_audit: AuditLog | None = None


def setup(config=None) -> None:
    global _audit

    level_name = "INFO"
    audit_file = "logs/audit.jsonl"
    if config is not None:
        try:
            level_name = config.get("logging", "level", fallback=level_name)
        except Exception:
            pass
        try:
            audit_file = config.get("logging", "audit_file", fallback=audit_file)
        except Exception:
            pass

    level = getattr(logging, str(level_name).upper(), logging.INFO)
    _logger.setLevel(level)
    if not _logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        _logger.addHandler(handler)
    _audit = AuditLog(audit_file)


def get():
    return _logger


def get_logger(name: str | None = None):
    if not name:
        return _logger
    return logging.getLogger(name)


def audit(event_type: str, payload: dict[str, Any]) -> str:
    global _audit
    if _audit is None:
        setup()
    assert _audit is not None
    return _audit.write(event_type, payload)


def verify_audit() -> tuple[bool, int, str]:
    if _audit is None:
        setup()
    assert _audit is not None
    return _audit.verify()


__all__ = ["AuditLog", "_audit", "audit", "get", "get_logger", "setup", "verify_audit"]
