from __future__ import annotations

import configparser
from pathlib import Path
from typing import Any

import core.logger as _core_logger
from core.logger import AuditLog


def setup(
    config_or_name: configparser.ConfigParser | str = "Jarvis",
    level: str | int = "INFO",
    log_file: str | Path | None = None,
):
    return _core_logger.setup(config_or_name=config_or_name, level=level, log_file=log_file)


def get(name: str | None = None):
    return _core_logger.get(name=name)


def get_logger(name: str = "Jarvis"):
    return _core_logger.get_logger(name=name)


def audit(event_type: str, payload: dict[str, Any]):
    return _core_logger.audit(event_type=event_type, payload=payload)


def verify_audit():
    return _core_logger.verify_audit()


def __getattr__(name: str):
    if name == "_audit":
        return _core_logger._audit
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "AuditLog",
    "setup",
    "get",
    "get_logger",
    "audit",
    "verify_audit",
]
