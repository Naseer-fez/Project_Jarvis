"""Structured JSONL logger for tool executions.

Appends one JSON line per tool call to logs/tool_execution_log.jsonl.

Fields per entry:
    timestamp       ISO-8601 UTC
    tool_name       str
    args_summary    dict (keys only + value truncated)
    success         bool
    duration_ms     float
    error           str | null
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)

# Written relative to PROJECT_ROOT/logs/
_LOG_FILE = Path(__file__).resolve().parents[2] / "logs" / "tool_execution_log.jsonl"


def _summarize_args(args: dict[str, Any], max_val_len: int = 80) -> dict[str, str]:
    """Return a sanitized key→truncated-value dict safe to log."""
    summary: dict[str, str] = {}
    for k, v in (args or {}).items():
        s = str(v)
        summary[str(k)] = s[:max_val_len] + "…" if len(s) > max_val_len else s
    return summary


class ToolExecutionLogger:
    """Thread-safe JSONL logger for tool execution records."""

    def __init__(self, log_file: Path | None = None) -> None:
        self._log_file = log_file or _LOG_FILE
        self._log_file.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        tool_name: str,
        args: dict[str, Any],
        success: bool,
        duration_ms: float,
        error: str | None = None,
    ) -> None:
        entry = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "tool_name": tool_name,
            "args_summary": _summarize_args(args),
            "success": success,
            "duration_ms": round(duration_ms, 2),
            "error": error,
        }
        try:
            with self._log_file.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as exc:  # noqa: BLE001
            _logger.warning("ToolExecutionLogger failed to write: %s", exc)

    def timed_log(self, tool_name: str, args: dict[str, Any]) -> "_TimedContext":
        """Context manager that auto-times and logs on exit."""
        return _TimedContext(self, tool_name, args)


class _TimedContext:
    def __init__(self, logger: ToolExecutionLogger, tool_name: str, args: dict[str, Any]) -> None:
        self._logger = logger
        self._tool_name = tool_name
        self._args = args
        self._start: float = 0.0
        self.success: bool = False
        self.error: str | None = None

    def __enter__(self) -> "_TimedContext":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        duration_ms = (time.perf_counter() - self._start) * 1000
        if exc_val is not None:
            self.error = str(exc_val)
            self.success = False
        self._logger.log(
            tool_name=self._tool_name,
            args=self._args,
            success=self.success,
            duration_ms=duration_ms,
            error=self.error,
        )


# Singleton instance for convenience
execution_logger = ToolExecutionLogger()

__all__ = ["ToolExecutionLogger", "execution_logger"]
