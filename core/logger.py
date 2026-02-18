"""
core/logger.py
══════════════
Immutable, append-only structured logging for Jarvis.

Rules:
- Logs are NEVER modified after writing
- Every log entry includes: timestamp, level, module, message
- Separate audit log for state transitions and tool calls
- Human-readable console output
"""

import logging
import json
import os
from datetime import datetime, timezone
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

AUDIT_LOG_PATH = LOG_DIR / "audit.jsonl"
APP_LOG_PATH   = LOG_DIR / "jarvis.log"


class AuditHandler(logging.Handler):
    """Writes structured audit records to append-only JSONL file."""

    def emit(self, record: logging.LogRecord):
        entry = {
            "ts":     datetime.now(timezone.utc).isoformat(),
            "level":  record.levelname,
            "module": record.name,
            "msg":    record.getMessage(),
        }
        # Extra fields (state, tool, risk, action, plan)
        for key in ("state", "tool", "plan", "risk", "action"):
            if hasattr(record, key):
                entry[key] = getattr(record, key)

        with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


def get_logger(name: str) -> logging.Logger:
    """Get a named logger with console + file + audit output."""
    logger = logging.getLogger(f"jarvis.{name}")

    if logger.handlers:
        return logger  # Already configured

    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%H:%M:%S"
    )

    # Console — INFO and above
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)

    # File — full DEBUG log
    file_handler = logging.FileHandler(APP_LOG_PATH, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    # Audit — structured JSONL
    audit_handler = AuditHandler()
    audit_handler.setLevel(logging.INFO)

    logger.addHandler(console)
    logger.addHandler(file_handler)
    logger.addHandler(audit_handler)

    return logger


def audit(logger: logging.Logger, msg: str, **kwargs):
    """Log a structured audit event with extra fields attached."""
    logger.info(msg, extra={k: v for k, v in kwargs.items()})