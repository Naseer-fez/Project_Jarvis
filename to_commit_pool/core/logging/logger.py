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
_MANAGED_STREAM_HANDLER_NAME = "jarvis_stream"
_MANAGED_FILE_HANDLER_NAME = "jarvis_app_file"


def _build_formatter() -> logging.Formatter:
    return logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")


def _find_managed_handler(name: str) -> logging.Handler | None:
    for handler in _logger.handlers:
        if getattr(handler, "name", None) == name:
            return handler
    return None


def _resolve_config_path(path_value: str) -> Path:
    from core.runtime.bootstrap import _resolve_path

    return _resolve_path(path_value).resolve()


def setup(config=None) -> None:
    global _audit

    level_name = "INFO"
    audit_file = "logs/audit.jsonl"
    app_file = "logs/app.log"
    if config is not None:
        try:
            level_name = config.get("logging", "level", fallback=level_name)
        except Exception:
            pass
        try:
            audit_file = config.get("logging", "audit_file", fallback=audit_file)
        except Exception:
            pass
        try:
            app_file = config.get("logging", "app_file", fallback=app_file)
        except Exception:
            pass

    level = getattr(logging, str(level_name).upper(), logging.INFO)
    _logger.setLevel(level)
    _logger.propagate = False

    stream_handler = _find_managed_handler(_MANAGED_STREAM_HANDLER_NAME)
    if stream_handler is None:
        stream_handler = logging.StreamHandler()
        stream_handler.name = _MANAGED_STREAM_HANDLER_NAME
        stream_handler.setFormatter(_build_formatter())
        _logger.addHandler(stream_handler)
    stream_handler.setLevel(level)

    desired_app_path = _resolve_config_path(app_file)
    desired_app_path.parent.mkdir(parents=True, exist_ok=True)
    desired_audit_path = _resolve_config_path(audit_file)
    desired_audit_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = _find_managed_handler(_MANAGED_FILE_HANDLER_NAME)
    current_app_path = None
    if isinstance(file_handler, logging.FileHandler):
        current_app_path = Path(file_handler.baseFilename).resolve()

    if (
        file_handler is not None
        and current_app_path is not None
        and current_app_path != desired_app_path
    ):
        _logger.removeHandler(file_handler)
        file_handler.close()
        file_handler = None

    if file_handler is None:
        file_handler = logging.FileHandler(desired_app_path, encoding="utf-8")
        file_handler.name = _MANAGED_FILE_HANDLER_NAME
        file_handler.setFormatter(_build_formatter())
        _logger.addHandler(file_handler)
    file_handler.setLevel(level)

    _audit = AuditLog(str(desired_audit_path))


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
