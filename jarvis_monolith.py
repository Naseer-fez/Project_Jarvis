"""
JARVIS MONOLITH
=================
LLM GUIDE: This file contains the entirety of the Jarvis project, merged from 150+ original modular files.
How to navigate:
1. FOLDER STRUCTURE: Before every file's content, there is a marker formatted as `# --- FILE: path/to/file.py ---`.
   Use these markers to map the code back to its original modular location.
2. RENAMED VARIABLES: To avoid namespace collisions during the merge, duplicate class/function names across files
   were prefixed with their original paths (e.g., `ToolResult` from `core/tools/system_automation.py` became
   `core_tools_system_automation_ToolResult`). You can safely trace these to understand execution.
3. INTERNAL IMPORTS: Internal `from core.X import Y` statements were removed. Use direct references instead.
4. WHEN EDITING: If you are suggesting fixes, please specify the ORIGINAL FILE PATH based on the markers,
   rather than line numbers in this monolith, so the user can easily port your fix back to their repository.
"""

from __future__ import annotations

############################################################
# IMPORTS
############################################################

from __future__ import annotations
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Awaitable
from contextvars import Token
from dataclasses import asdict, dataclass
from dataclasses import dataclass
from dataclasses import dataclass, field
from datetime import datetime
from datetime import datetime, timedelta
from datetime import datetime, timedelta, timezone
from datetime import datetime, timezone
from datetime import datetime, timezone, timedelta
from email.mime.text import MIMEText
from enum import Enum
from enum import IntEnum
from functools import lru_cache
from pathlib import Path
from streamlit.elements.lib.file_uploader_utils import enforce_filename_restriction
from streamlit.elements.lib.form_utils import current_form_id
from streamlit.elements.lib.layout_utils import LayoutConfig, validate_width
from streamlit.elements.lib.policies import check_widget_policies, maybe_raise_label_warnings
from streamlit.elements.lib.utils import Key, LabelVisibility, compute_and_register_element_id, get_label_visibility_proto_value, to_key
from streamlit.elements.widgets.file_uploader import _get_upload_files
from streamlit.errors import StreamlitAPIException
from streamlit.proto.AudioInput_pb2 import AudioInput as AudioInputProto
from streamlit.proto.Common_pb2 import FileUploaderState as FileUploaderStateProto
from streamlit.proto.Common_pb2 import UploadedFileInfo as UploadedFileInfoProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import ScriptRunContext, get_script_run_ctx
from streamlit.runtime.state import WidgetArgs, WidgetCallback, WidgetKwargs, register_widget
from streamlit.runtime.uploaded_file_manager import DeletedFile, UploadedFile
from subprocess import CalledProcessError, run
from textwrap import dedent
from typing import Any
from typing import Any, Awaitable, Callable
from typing import Any, Awaitable, Callable, Union
from typing import Any, Callable
from typing import Any, Callable, Dict, Set
from typing import Any, Callable, Dict, Tuple, Type, Union
from typing import Any, Callable, Iterable
from typing import Any, Callable, Optional
from typing import Any, Callable, TypeVar, cast
from typing import Any, Dict
from typing import Any, Dict, List, Set
from typing import Any, Iterable
from typing import Any, List
from typing import Any, Optional
from typing import Awaitable, Callable
from typing import Callable
from typing import Callable, Any
from typing import Dict, List, Any
from typing import Iterable, Any
from typing import List, Dict, Any
from typing import Literal
from typing import Optional
from typing import Optional, TYPE_CHECKING
from typing import Optional, Union
from typing import Protocol, Any, cast
from typing import Sequence, Any
from typing import TYPE_CHECKING
from typing import TYPE_CHECKING, TypeAlias, cast
from urllib.parse import quote
from urllib.parse import quote_plus
from urllib.request import urlopen
import abc
import aiohttp
import aiosqlite
import argparse
import ast
import asyncio
import atexit
import base64
import configparser
import contextlib
import contextvars
import csv
import dataclasses
import datetime
import email as email_lib
import faulthandler
import fnmatch
import hashlib
import hmac
import imaplib
import importlib
import importlib.util
import inspect
import io
import json
import logging
import logging.handlers
import math
import numpy as np
import os
import platform
import queue
import re
import secrets
import shutil
import signal
import smtplib
import sqlite3
import struct
import subprocess
import sys
import threading
import time
import torch
import torch.nn.functional as F
import traceback
import urllib.parse
import urllib.request
import uuid
import warnings


############################################################
# CONSTANTS
############################################################


# --- FILE: audit/audit_logger.py ---

"""Small security-focused helpers for audit payload scrubbing."""

# internal import removed: from __future__ import annotations

import re
import warnings

warnings.warn(
    "The audit.audit_logger module is deprecated. Please use core.logging.logger instead.",
    DeprecationWarning,
    stacklevel=2
)


_ASSIGNMENT_PATTERNS = [
    re.compile(r"(?i)([a-zA-Z0-9_]*(?:password|passwd|token|api[_-]?key))(\s*[:=]\s*)(?:\"[^\"]+\"|'[^']+'|[^\s,;}\]]+)"),
]
_LONG_SECRET = re.compile(r"(?<![a-zA-Z0-9+/=_-])[a-zA-Z0-9+/=_-]{32,}(?![a-zA-Z0-9+/=_-])")


def scrub_secrets(text: str) -> str:
    if text is None:
        return ""
    value = str(text)
    for pattern in _ASSIGNMENT_PATTERNS:
        value = pattern.sub(lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]", value)
    value = _LONG_SECRET.sub("[REDACTED]", value)
    return value




# --- FILE: core/logging/logger.py ---

"""Application logging and append-only audit log support."""

# internal import removed: from __future__ import annotations

import atexit
import hashlib
import json
import logging
import logging.handlers
import queue
import re
import threading
from pathlib import Path
from typing import Any
import contextvars

_trace_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("trace_id", default=None)
_task_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("task_id", default=None)

def set_trace_ids(trace_id: str | None, task_id: str | None) -> tuple[contextvars.Token[str | None], contextvars.Token[str | None]]:
    """Set correlation IDs for the current async context."""
    return _trace_id_var.set(trace_id), _task_id_var.set(task_id)

def reset_trace_ids(trace_token: contextvars.Token[str | None], task_token: contextvars.Token[str | None]) -> None:
    """Restore correlation IDs for the current async context."""
    _trace_id_var.reset(trace_token)
    _task_id_var.reset(task_token)

# Globals for non-blocking logging
_log_queue: queue.Queue = queue.Queue(10000)
_log_listener: FlushingQueueListener | None = None
_queue_handler: logging.handlers.QueueHandler | None = None
_active_handlers: dict[str, logging.Handler] = {}


def redact_sensitive_data(val: Any) -> Any:
    """Recursively redact sensitive patterns (passwords, secrets, tokens) from metadata and strings."""
    if isinstance(val, dict):
        new_dict = {}
        for k, v in val.items():
            k_lower = str(k).lower()
            if any(
                token in k_lower
                for token in (
                    "secret",
                    "token",
                    "password",
                    "passwd",
                    "api_key",
                    "apikey",
                    "access_key",
                    "auth",
                    "credential",
                    "private_key",
                )
            ):
                new_dict[k] = "***REDACTED***"
            else:
                new_dict[k] = redact_sensitive_data(v)
        return new_dict
    elif isinstance(val, list):
        return [redact_sensitive_data(item) for item in val]
    elif isinstance(val, tuple):
        return tuple(redact_sensitive_data(item) for item in val)
    elif isinstance(val, str):
        # Redact OpenAI API keys (sk-...)
        val = re.sub(r"\bsk-[a-zA-Z0-9_-]{20,}\b", "sk-***REDACTED***", val)
        # Redact JWT tokens
        val = re.sub(
            r"\beyJhbGciOi[a-zA-Z0-9-_]+\.[a-zA-Z0-9-_]+\.[a-zA-Z0-9-_]+\b",
            "eyJ***REDACTED***",
            val,
        )
        # Redact generic password/secret key-value patterns
        val = re.sub(
            r"(?i)\b(api_key|password|secret|token|passwd)\b\s*[:=]\s*['\"]?[a-zA-Z0-9_-]{8,}['\"]?",
            r"\1=***REDACTED***",
            val,
        )
        # Redact Authorization and Bearer tokens
        val = re.sub(
            r"(?i)\b(bearer|authorization)\b\s*[:=]?\s*['\"]?[a-zA-Z0-9_\-\.]{15,}['\"]?",
            r"\1=***REDACTED***",
            val,
        )
        return val
    return val


class AuditLog:
    def __init__(self, file_path: str) -> None:
        self.path = Path(file_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._last_hash = "0" * 64
        self._lock = threading.Lock()
        self._queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None

        if self.path.exists():
            try:
                with self._lock:
                    with self.path.open("r", encoding="utf-8", newline="\n") as f:
                        last_line = ""
                        for line in f:
                            if line.strip():
                                last_line = line
                        if last_line:
                            last_payload = json.loads(last_line)
                            if "hash" in last_payload:
                                self._last_hash = last_payload["hash"]
            except Exception:
                pass

        self._start_worker()

    def _start_worker(self) -> None:
        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._write_worker, daemon=True, name="JarvisAuditWorker"
        )
        self._worker_thread.start()

    def _write_worker(self) -> None:
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                record = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                with self._lock:
                    with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception:
                pass
            finally:
                self._queue.task_done()

    def write(self, event_type: str, payload: dict[str, Any]) -> str:
        # Redact the audit payload before storing/hashing to prevent credential leaks in audit trail
        redacted_payload = redact_sensitive_data(payload)
        with self._lock:
            body = {
                "event_type": event_type,
                "payload": redacted_payload,
                "prev_hash": self._last_hash,
            }
            digest = hashlib.sha256(
                json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")
            ).hexdigest()
            record = body | {"hash": digest}
            self._last_hash = digest
            self._queue.put(record)
            return digest

    def stop(self) -> None:
        self._stop_event.set()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=2.0)
            self._worker_thread = None

    def verify(self) -> tuple[bool, int, str]:
        # Flush queue before verifying to ensure all writes are on disk
        self._queue.join()

        with self._lock:
            if not self.path.exists():
                return True, 0, ""

            previous = "0" * 64
            count = 0
            try:
                with self.path.open("r", encoding="utf-8", newline="\n") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            payload = json.loads(line)
                        except Exception:
                            return False, count, "Invalid JSON line in audit log"
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
            except Exception as e:
                return False, count, f"Error reading/parsing audit log: {str(e)}"
            return True, count, ""


_logger = logging.getLogger("Jarvis")
_audit: AuditLog | None = None
_MANAGED_STREAM_HANDLER_NAME = "jarvis_stream"
_MANAGED_FILE_HANDLER_NAME = "jarvis_app_file"


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        try:
            import sys
            # Automatically capture stack trace for errors if not present
            if record.levelno >= logging.ERROR and not record.exc_info and not record.exc_text:
                exc_info = sys.exc_info()
                if exc_info[0] is not None:
                    record.exc_info = exc_info

            # Pre-compute and redact exc_text so it's safely handled by standard formatters
            if record.exc_info and not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
            if record.exc_text:
                record.exc_text = redact_sensitive_data(record.exc_text)

            # Redact message and format arguments on the record before formatting
            record.msg = redact_sensitive_data(record.msg)
            if isinstance(record.args, dict):
                record.args = redact_sensitive_data(record.args)
            elif isinstance(record.args, tuple):
                record.args = tuple(redact_sensitive_data(arg) for arg in record.args)

            trace_id = getattr(record, "trace_id", None) or _trace_id_var.get()
            task_id = getattr(record, "task_id", None) or _task_id_var.get()
            if not trace_id and not task_id:
                return str(redact_sensitive_data(super().format(record)))

            import datetime

            timestamp = (
                datetime.datetime.fromtimestamp(record.created, datetime.timezone.utc)
                .isoformat()
                .replace("+00:00", "Z")
            )

            envelope = {
                "timestamp": timestamp,
                "level": record.levelname,
                "trace_id": trace_id,
                "task_id": task_id,
                "component": record.name,
                "event": getattr(record, "event", "log_message"),
                "metadata": getattr(record, "metadata", {}) or {},
            }
            envelope["metadata"] = redact_sensitive_data(envelope["metadata"])

            if isinstance(envelope["metadata"], dict) and "message" not in envelope["metadata"]:
                envelope["metadata"]["message"] = record.getMessage()

            if record.exc_text:
                envelope["stack_trace"] = record.exc_text

            return json.dumps(envelope, ensure_ascii=False, default=str)
        except Exception as e:
            return f"LOGGING_ERROR: {str(e)}"


class FlushingQueueListener(logging.handlers.QueueListener):
    def handle(self, record: logging.LogRecord) -> None:
        flush_event = getattr(record, "_flush_event", None)
        if flush_event is not None:
            flush_event.set()
            return
        super().handle(record)

    def flush(self) -> None:
        event = threading.Event()
        dummy_record = logging.LogRecord(
            name="flush",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="",
            args=(),
            exc_info=None,
        )
        dummy_record._flush_event = event
        self.queue.put(dummy_record)  # type: ignore[attr-defined]
        event.wait(timeout=2.0)


class JarvisQueueHandler(logging.handlers.QueueHandler):
    def prepare(self, record: logging.LogRecord) -> logging.LogRecord:
        import copy
        import sys
        # Capture exc_info in the calling thread before passing it to the background queue
        if record.levelno >= logging.ERROR and not record.exc_info and not record.exc_text:
            exc_info = sys.exc_info()
            if exc_info[0] is not None:
                record.exc_info = exc_info
        
        # Format the exception text in the calling thread because traceback objects
        # cannot always be safely pickled/copied across boundaries
        if record.exc_info and not record.exc_text:
            import traceback
            record.exc_text = "".join(traceback.format_exception(*record.exc_info))
            
        return copy.copy(record)



def _build_formatter() -> logging.Formatter:
    return JSONFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s")


def _find_managed_handler(name: str) -> logging.Handler | None:
    return _active_handlers.get(name)


def _resolve_config_path(path_value: str) -> Path:
    from core.runtime.paths import _resolve_path

    return _resolve_path(path_value).resolve()


def setup(config=None) -> None:
    global _audit, _log_listener, _queue_handler

    # Reconfigure stdout/stderr encoding to utf-8 if possible to prevent UnicodeEncodeError under Windows console
    import sys
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None:
            reconfigure = getattr(stream, "reconfigure", None)
            if callable(reconfigure):
                try:
                    reconfigure(encoding="utf-8", errors="replace")
                except Exception:
                    pass

    # Stop existing listener if it exists to cleanly re-configure
    if _log_listener is not None:
        _log_listener.stop()
        _log_listener = None

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

    root_logger = logging.getLogger()
    # Reduce spam from third-party libraries by defaulting root logger to WARNING.
    # The handlers will still process _logger's events at the desired level.
    root_logger.setLevel(logging.WARNING)

    _logger.setLevel(level)
    _logger.propagate = True

    # Ensure application modules are captured at the desired level,
    # avoiding the root logger's WARNING suppression.
    logging.getLogger("core").setLevel(level)
    logging.getLogger("dashboard").setLevel(level)

    # Remove any existing handlers from root logger to prevent duplicates
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)

    # Stream Handler Setup
    stream_handler = _active_handlers.get(_MANAGED_STREAM_HANDLER_NAME)
    if stream_handler is None:
        stream_handler = logging.StreamHandler()
        stream_handler.name = _MANAGED_STREAM_HANDLER_NAME
        stream_handler.setFormatter(_build_formatter())
        _active_handlers[_MANAGED_STREAM_HANDLER_NAME] = stream_handler
    stream_handler.setLevel(level)

    # File Handler Setup
    desired_app_path = _resolve_config_path(app_file)
    desired_app_path.parent.mkdir(parents=True, exist_ok=True)
    desired_audit_path = _resolve_config_path(audit_file)
    desired_audit_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = _active_handlers.get(_MANAGED_FILE_HANDLER_NAME)
    current_app_path = None
    if isinstance(file_handler, logging.FileHandler):
        current_app_path = Path(file_handler.baseFilename).resolve()

    if (
        file_handler is not None
        and current_app_path is not None
        and current_app_path != desired_app_path
    ):
        file_handler.close()
        file_handler = None

    if file_handler is None:
        file_handler = logging.handlers.RotatingFileHandler(desired_app_path, maxBytes=10*1024*1024, backupCount=5, encoding="utf-8")
        file_handler.name = _MANAGED_FILE_HANDLER_NAME
        file_handler.setFormatter(_build_formatter())
        _active_handlers[_MANAGED_FILE_HANDLER_NAME] = file_handler
    file_handler.setLevel(level)

    import os
    if "PYTEST_CURRENT_TEST" in os.environ:
        # Avoid async timing issues in unit tests by logging synchronously
        root_logger.addHandler(stream_handler)
        root_logger.addHandler(file_handler)
    else:
        # Re-create and attach the QueueHandler for non-blocking logging
        _queue_handler = JarvisQueueHandler(_log_queue)
        root_logger.addHandler(_queue_handler)

        # Start the background listener thread to dispatch messages to stdout/file handlers
        _log_listener = FlushingQueueListener(
            _log_queue, stream_handler, file_handler, respect_handler_level=True
        )
        _log_listener.start()

    # Re-initialize audit logging safely
    if _audit is not None:
        _audit.stop()
    _audit = AuditLog(str(desired_audit_path))


@atexit.register
def cleanup_logging() -> None:
    global _log_listener, _audit
    if _log_listener is not None:
        _log_listener.stop()
        _log_listener = None
    if _audit is not None:
        _audit.stop()
        _audit = None


def get() -> logging.Logger:
    return _logger


def get_logger(name: str | None = None) -> logging.Logger:
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


def flush() -> None:
    global _log_listener
    if _log_listener is not None:
        _log_listener.flush()


__all__ = ["AuditLog", "_audit", "audit", "get", "get_logger", "setup", "verify_audit", "flush", "set_trace_ids", "reset_trace_ids"]




# --- FILE: core/metrics/confidence.py ---

"""Tiny confidence aggregator used by the agent loop."""

# internal import removed: from __future__ import annotations


class ConfidenceModel:
    def __init__(self) -> None:
        self._scores: dict[str, float] = {}

    def update(self, metric: str, value: float) -> float:
        clamped = max(0.0, min(1.0, float(value)))
        self._scores[str(metric)] = clamped
        return clamped

    def score(self) -> float:
        if not self._scores:
            return 0.5
        return sum(self._scores.values()) / len(self._scores)


__all__ = ["ConfidenceModel"]




# --- FILE: core/metrics/__init__.py ---

"""Metrics helpers."""

# internal import removed: from .confidence import ConfidenceModel

__all__ = ["ConfidenceModel"]



############################################################
# CONFIGURATION
############################################################


# --- FILE: core/config/__init__.py ---

"""Typed, unified config manager for Project Jarvis."""

# internal import removed: from __future__ import annotations

import configparser
import os


class JarvisConfig(configparser.ConfigParser):
    """
    Typed config manager that inherits from configparser.ConfigParser
    to maintain full backward compatibility while exposing typed accessors,
    standardizing env-var overrides, and providing unified fallback lookups.
    """

    def get_str(self, section: str, key: str, fallback: str = "") -> str:
        """Get config string, checking env-var overrides first."""
        env_key = f"JARVIS_{section.upper()}_{key.upper()}"
        if env_key in os.environ:
            return os.environ[env_key]
        env_key_short = f"JARVIS_{key.upper()}"
        if env_key_short in os.environ:
            return os.environ[env_key_short]
        if not self.has_section(section):
            return fallback
        return self.get(section, key, fallback=fallback)

    def get_bool(self, section: str, key: str, fallback: bool = False) -> bool:
        """Get config boolean, checking env-var overrides first."""
        env_key = f"JARVIS_{section.upper()}_{key.upper()}"
        if env_key in os.environ:
            val = os.environ[env_key].lower()
            return val in ("true", "1", "yes", "on", "enable")
        env_key_short = f"JARVIS_{key.upper()}"
        if env_key_short in os.environ:
            val = os.environ[env_key_short].lower()
            return val in ("true", "1", "yes", "on", "enable")
        if not self.has_section(section):
            return fallback
        return self.getboolean(section, key, fallback=fallback)

    def get_int(self, section: str, key: str, fallback: int = 0) -> int:
        """Get config integer, checking env-var overrides first."""
        env_key = f"JARVIS_{section.upper()}_{key.upper()}"
        if env_key in os.environ:
            try:
                return int(os.environ[env_key])
            except ValueError:
                pass
        env_key_short = f"JARVIS_{key.upper()}"
        if env_key_short in os.environ:
            try:
                return int(os.environ[env_key_short])
            except ValueError:
                pass
        if not self.has_section(section):
            return fallback
        return self.getint(section, key, fallback=fallback)


def core_config___init___load_config(config_path: str) -> JarvisConfig:
    """
    Load INI config from an absolute path or relative to PROJECT_ROOT
    into JarvisConfig, with env-var resolution.
    """
    from core.runtime.paths import _resolve_path
    import logging
    
    log = logging.getLogger("jarvis.config")
    config = JarvisConfig()
    path = _resolve_path(config_path)

    if not path.exists():
        env = os.environ.get("JARVIS_ENV", "development").lower()
        msg = f"Config not found: {path}"
        if env == "production":
            log.critical(msg)
            raise SystemExit(2)  # CONFIG_ERROR
        log.warning("%s - using defaults", msg)
        return config

    try:
        with path.open("r", encoding="utf-8") as handle:
            config.read_file(handle)
    except configparser.Error as exc:
        log.critical("Config parse error: %s", exc)
        raise SystemExit(2) from exc
    except OSError as exc:
        log.critical("Config read error: %s", exc)
        raise SystemExit(2) from exc

    return config




# --- FILE: core/config/defaults.py ---

import os

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")



############################################################
# DATA MODELS
############################################################


# --- FILE: core/autonomy/risk_evaluator.py ---

"""Deterministic risk evaluator for planned tool actions without hardcoded tool strings."""

# internal import removed: from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Sequence, Any


class core_autonomy_risk_evaluator_RiskLevel(IntEnum):
    LOW = 0
    MEDIUM = 1
    CONFIRM = 2
    HIGH = 3
    CRITICAL = 4

    # Backward compatibility alias used in legacy paths.
    FORBIDDEN = 4

    def label(self) -> str:
        # Keep FORBIDDEN as the preferred label for CRITICAL so legacy tests
        # that assert "FORBIDDEN in result.summary()" continue to pass.
        if int(self) == int(core_autonomy_risk_evaluator_RiskLevel.CRITICAL):
            return "FORBIDDEN"
        return self.name


@dataclass(frozen=True)
class RiskResult:
    level: core_autonomy_risk_evaluator_RiskLevel
    blocking_actions: list[str] = field(default_factory=list)
    confirm_actions: list[str] = field(default_factory=list)
    high_risk_actions: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    @property
    def is_blocked(self) -> bool:
        return self.level >= core_autonomy_risk_evaluator_RiskLevel.CRITICAL

    @property
    def requires_confirmation(self) -> bool:
        # MEDIUM and above (but below CRITICAL) require confirmation
        return core_autonomy_risk_evaluator_RiskLevel.MEDIUM <= self.level < core_autonomy_risk_evaluator_RiskLevel.CRITICAL

    def summary(self) -> str:
        parts = [f"Risk: {self.level.label()}"]
        if self.blocking_actions:
            parts.append(f"BLOCKED: {', '.join(self.blocking_actions)}")
        if self.confirm_actions:
            parts.append(f"CONFIRM: {', '.join(self.confirm_actions)}")
        if self.high_risk_actions:
            parts.append(f"HIGH: {', '.join(self.high_risk_actions)}")
        if self.reasons:
            parts.append(" | ".join(self.reasons))
        return " - ".join(parts)


class RiskEvaluator:
    """Evaluates a list of action names into LOW/MEDIUM/CONFIRM/HIGH/CRITICAL."""

    def __init__(self, config=None, registry: Any = None) -> None:
        self.registry = registry
        self._critical: set[str] = set()
        self._confirm: set[str] = set()
        self._high: set[str] = set()
        self._medium: set[str] = set()
        self._low: set[str] = set()
        self._cache: dict[str, core_autonomy_risk_evaluator_RiskLevel] = {}
        self._lock = threading.Lock()

        if config is not None:
            self._load_config(config)

    def register_critical_action(self, action: str) -> None:
        """Dynamically register an action as CRITICAL risk level."""
        action_clean = action.strip().lower()
        with self._lock:
            self._critical.add(action_clean)
            self._confirm.discard(action_clean)
            self._high.discard(action_clean)
            self._medium.discard(action_clean)
            self._low.discard(action_clean)
            self._cache.pop(action_clean, None)

    def register_confirm_action(self, action: str) -> None:
        """Dynamically register an action as CONFIRM risk level."""
        action_clean = action.strip().lower()
        with self._lock:
            self._confirm.add(action_clean)
            self._critical.discard(action_clean)
            self._high.discard(action_clean)
            self._medium.discard(action_clean)
            self._low.discard(action_clean)
            self._cache.pop(action_clean, None)

    def register_high_action(self, action: str) -> None:
        """Dynamically register an action as HIGH risk level."""
        action_clean = action.strip().lower()
        with self._lock:
            self._high.add(action_clean)
            self._critical.discard(action_clean)
            self._confirm.discard(action_clean)
            self._medium.discard(action_clean)
            self._low.discard(action_clean)
            self._cache.pop(action_clean, None)

    def register_medium_action(self, action: str) -> None:
        """Dynamically register an action as MEDIUM risk level."""
        action_clean = action.strip().lower()
        with self._lock:
            self._medium.add(action_clean)
            self._critical.discard(action_clean)
            self._confirm.discard(action_clean)
            self._high.discard(action_clean)
            self._low.discard(action_clean)
            self._cache.pop(action_clean, None)

    def register_low_action(self, action: str) -> None:
        """Dynamically register an action as LOW risk level."""
        action_clean = action.strip().lower()
        with self._lock:
            self._low.add(action_clean)
            self._critical.discard(action_clean)
            self._confirm.discard(action_clean)
            self._high.discard(action_clean)
            self._medium.discard(action_clean)
            self._cache.pop(action_clean, None)

    def _load_config(self, config) -> None:
        def _parse(section: str, key: str) -> set[str]:
            raw = config.get(section, key, fallback="")
            return {item.strip().lower() for item in raw.split(",") if item.strip()}

        critical = _parse("risk", "critical_actions") or _parse("risk", "forbidden_actions")
        confirm = _parse("risk", "confirm_actions") or _parse("risk", "user_confirmed_actions")
        high = _parse("risk", "high_risk_actions")
        medium = _parse("risk", "medium_risk_actions")
        low = _parse("risk", "low_risk_actions")

        if critical:
            self._critical.update(critical)
        if confirm:
            self._confirm.update(confirm)
        if high:
            self._high.update(high)
        if medium:
            self._medium.update(medium)
        if low:
            self._low.update(low)

    def evaluate(self, actions: Sequence[str]) -> RiskResult:
        if not actions:
            return RiskResult(level=core_autonomy_risk_evaluator_RiskLevel.LOW, reasons=["No actions - trivial plan"])

        blocking: list[str] = []
        confirm_needed: list[str] = []
        high_risk: list[str] = []
        reasons: list[str] = []
        max_level = core_autonomy_risk_evaluator_RiskLevel.LOW

        for raw_action in actions:
            action = str(raw_action).strip().lower()
            if not action:
                continue

            level = self._cache.get(action)
            if level is None:
                with self._lock:
                    level = self._cache.get(action)
                    if level is None:
                        # 1. Resolve risk dynamically from the Capability Registry if present
                        if self.registry:
                            cap = self.registry.get(action)
                            if cap:
                                level_name = cap.risk_level.name
                                level = getattr(core_autonomy_risk_evaluator_RiskLevel, level_name, core_autonomy_risk_evaluator_RiskLevel.LOW)

                        # 2. Check dynamic/explicit config updates
                        if level is None:
                            if action in self._critical:
                                level = core_autonomy_risk_evaluator_RiskLevel.CRITICAL
                            elif action in self._high:
                                level = core_autonomy_risk_evaluator_RiskLevel.HIGH
                            elif action in self._confirm:
                                level = core_autonomy_risk_evaluator_RiskLevel.CONFIRM
                            elif action in self._medium:
                                level = core_autonomy_risk_evaluator_RiskLevel.MEDIUM
                            elif action in self._low:
                                level = core_autonomy_risk_evaluator_RiskLevel.LOW

                        # 3. Fallback to generic safe keyword patterns (no hardcoded tool name strings)
                        if level is None:
                            critical_kws = {"shell", "exec", "subprocess", "delete_file", "rmdir", "format_disk", "wipe_disk", "serial_send", "serial_write", "physical_actuate"}
                            confirm_kws = {"write", "launch", "send", "click", "drag", "scroll", "type", "press", "hotkey", "focus_window", "clipboard_set", "clipboard_paste", "create_event", "delete_event", "mark_as_read", "create_page", "append_block", "play_track", "create_playlist", "turn_on", "turn_off", "toggle", "set_thermostat", "call_service", "create_issue", "close_issue", "create_gist", "sort_files", "copy_file", "move_file", "create_directory"}
                            high_kws = {"spawn", "popen", "pip_install", "install", "env_write", "system_config", "risky"}
                            medium_kws = {"read", "capture", "sensor", "search", "lookup", "ui_interaction", "key_press", "notification"}

                            if any(kw in action for kw in critical_kws):
                                level = core_autonomy_risk_evaluator_RiskLevel.CRITICAL
                            elif any(kw in action for kw in high_kws):
                                level = core_autonomy_risk_evaluator_RiskLevel.HIGH
                            elif any(kw in action for kw in confirm_kws):
                                level = core_autonomy_risk_evaluator_RiskLevel.CONFIRM
                            elif any(kw in action for kw in medium_kws):
                                level = core_autonomy_risk_evaluator_RiskLevel.MEDIUM
                            else:
                                level = core_autonomy_risk_evaluator_RiskLevel.LOW

                        if len(self._cache) > 1000:
                            self._cache.clear()
                        self._cache[action] = level

            # Apply classification results
            if level == core_autonomy_risk_evaluator_RiskLevel.CRITICAL:
                blocking.append(action)
                max_level = core_autonomy_risk_evaluator_RiskLevel.CRITICAL
                reasons.append(f"'{action}' is critical and blocked")
            elif level == core_autonomy_risk_evaluator_RiskLevel.HIGH:
                high_risk.append(action)
                if max_level < core_autonomy_risk_evaluator_RiskLevel.HIGH:
                    max_level = core_autonomy_risk_evaluator_RiskLevel.HIGH
                reasons.append(f"'{action}' is high-risk")
            elif level == core_autonomy_risk_evaluator_RiskLevel.CONFIRM:
                confirm_needed.append(action)
                if max_level < core_autonomy_risk_evaluator_RiskLevel.CONFIRM:
                    max_level = core_autonomy_risk_evaluator_RiskLevel.CONFIRM
                reasons.append(f"'{action}' requires explicit confirmation")
            elif level == core_autonomy_risk_evaluator_RiskLevel.MEDIUM:
                if max_level < core_autonomy_risk_evaluator_RiskLevel.MEDIUM:
                    max_level = core_autonomy_risk_evaluator_RiskLevel.MEDIUM
                reasons.append(f"'{action}' is medium-risk")

        return RiskResult(
            level=max_level,
            blocking_actions=blocking,
            confirm_actions=confirm_needed,
            high_risk_actions=high_risk,
            reasons=reasons,
        )

    def evaluate_plan(self, plan: dict) -> RiskResult:
        steps = plan.get("steps", []) if isinstance(plan, dict) else []
        actions: list[str] = []

        for step in steps:
            if not isinstance(step, dict):
                continue
            action = step.get("action") or step.get("tool") or step.get("type")
            if action:
                actions.append(str(action))

        return self.evaluate(actions)


__all__ = ["core_autonomy_risk_evaluator_RiskLevel", "RiskResult", "RiskEvaluator"]




# --- FILE: core/state_machine.py ---

"""Finite-state machine used across legacy and current Jarvis flows."""

# internal import removed: from __future__ import annotations

import inspect
import logging
import threading
from datetime import datetime
from enum import Enum
from typing import Callable, Any

logger = logging.getLogger("Jarvis.StateMachine")


class IllegalTransitionError(RuntimeError):
    """Raised when a state transition is not allowed."""


class State(str, Enum):
    IDLE = "IDLE"
    THINKING = "THINKING"
    PLANNING = "PLANNING"
    RISK_EVALUATION = "RISK_EVALUATION"
    AWAITING_CONFIRMATION = "AWAITING_CONFIRMATION"
    APPROVED = "APPROVED"
    CANCELLED = "CANCELLED"
    ACTING = "ACTING"
    OBSERVING = "OBSERVING"
    REFLECTING = "REFLECTING"
    REVIEWING = "REVIEWING"
    EXECUTING = "EXECUTING"
    COMPLETED = "COMPLETED"
    SPEAKING = "SPEAKING"
    LISTENING = "LISTENING"
    TRANSCRIBING = "TRANSCRIBING"
    ERROR = "ERROR"
    ABORTED = "ABORTED"
    SHUTDOWN = "SHUTDOWN"


_ALLOWED_TRANSITIONS: dict[State, set[State]] = {
    State.IDLE: {State.THINKING, State.PLANNING, State.LISTENING, State.SHUTDOWN},
    State.THINKING: {State.IDLE, State.PLANNING, State.ERROR},
    State.PLANNING: {
        State.RISK_EVALUATION,
        State.REVIEWING,
        State.IDLE,
        State.ERROR,
        State.SPEAKING,
    },
    State.RISK_EVALUATION: {
        State.AWAITING_CONFIRMATION,
        State.APPROVED,
        State.CANCELLED,
        State.ACTING,
        State.IDLE,
        State.ERROR,
    },
    State.AWAITING_CONFIRMATION: {
        State.APPROVED,
        State.CANCELLED,
        State.ACTING,
        State.IDLE,
        State.ERROR,
    },
    State.APPROVED: {
        State.EXECUTING,
        State.ACTING,
        State.IDLE,
        State.ERROR,
    },
    State.ACTING: {State.OBSERVING, State.IDLE, State.ERROR},
    State.OBSERVING: {State.ACTING, State.REFLECTING, State.IDLE, State.ERROR},
    State.REFLECTING: {State.SPEAKING, State.IDLE, State.ERROR, State.COMPLETED},
    State.REVIEWING: {State.EXECUTING, State.ABORTED, State.IDLE, State.ERROR},
    State.EXECUTING: {
        State.REFLECTING,
        State.SPEAKING,
        State.COMPLETED,
        State.IDLE,
        State.ERROR,
        State.ABORTED,
    },
    State.COMPLETED: {State.IDLE, State.SHUTDOWN},
    State.CANCELLED: {State.IDLE, State.SHUTDOWN},
    State.SPEAKING: {State.IDLE, State.LISTENING, State.ERROR, State.COMPLETED},
    State.LISTENING: {State.TRANSCRIBING, State.IDLE, State.ERROR},
    State.TRANSCRIBING: {State.PLANNING, State.IDLE, State.ERROR},
    State.ERROR: {State.IDLE, State.SHUTDOWN},
    State.ABORTED: {State.IDLE, State.SHUTDOWN},
    State.SHUTDOWN: set(),
}


class StateGuard:
    """Context manager to temporarily transition to a state, reverting back on exit."""

    def __init__(self, state_machine: StateMachine, target_state: State) -> None:
        self.sm = state_machine
        self.target_state = target_state
        self.previous_state: State | None = None

    def __enter__(self) -> StateMachine:
        self.previous_state = self.sm.state
        self.sm.transition(self.target_state)
        return self.sm

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_type is not None:
            import asyncio
            error_state = State.ERROR
            if issubclass(exc_type, asyncio.CancelledError):
                error_state = State.ABORTED
            
            try:
                if self.sm.state not in {State.ERROR, State.ABORTED, State.COMPLETED, State.CANCELLED, State.SHUTDOWN}:
                    if self.sm.can_transition(error_state):
                        self.sm.transition(error_state)
                    else:
                        self.sm.force_idle()
            except Exception:
                pass
        else:
            try:
                if self.sm.state not in {State.ERROR, State.ABORTED, State.COMPLETED, State.CANCELLED, State.SHUTDOWN}:
                    if self.previous_state and self.sm.can_transition(self.previous_state):
                        self.sm.transition(self.previous_state)
                    else:
                        self.sm.force_idle()
            except Exception:
                pass

    async def __aenter__(self) -> StateMachine:
        return self.__enter__()

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.__exit__(exc_type, exc_val, exc_tb)


class StateMachine:
    def __init__(self, event_bus: Any = None) -> None:
        self._state = State.IDLE
        self._listeners: list[Callable[[State, State], None]] = []
        self.event_bus = event_bus
        self.task_id: str | None = None
        self.diagnostics_mode: bool = False
        self._transition_audit_trail: list[dict[str, Any]] = []
        self._lock = threading.RLock()
        self._pending_notifications: list[tuple[State, State]] = []
        self._notifying = False

    @property
    def state(self) -> State:
        with self._lock:
            return self._state

    def add_listener(self, listener: Callable[[State, State], None]) -> None:
        with self._lock:
            if listener not in self._listeners:
                self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[State, State], None]) -> None:
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    def can_transition(self, new_state: State) -> bool:
        with self._lock:
            try:
                candidate = State(new_state)
            except ValueError:
                return False
            return candidate in _ALLOWED_TRANSITIONS.get(self._state, set())

    def get_valid_transitions(self, state: State | None = None) -> list[State]:
        with self._lock:
            target = State(state) if state is not None else self._state
            return sorted(list(_ALLOWED_TRANSITIONS.get(target, set())), key=lambda s: s.value)

    def get_transition_graph(self) -> dict[str, list[str]]:
        with self._lock:
            return {
                state.value: sorted([t.value for t in targets])
                for state, targets in _ALLOWED_TRANSITIONS.items()
            }

    def _notify(self, old_state: State, new_state: State) -> None:
        with self._lock:
            self._pending_notifications.append((old_state, new_state))
            if self._notifying:
                return
            self._notifying = True

        try:
            while True:
                with self._lock:
                    if not self._pending_notifications:
                        self._notifying = False
                        break
                    old_s, new_s = self._pending_notifications.pop(0)
                    listeners = list(self._listeners)

                # Execute callbacks outside the lock
                for listener in listeners:
                    try:
                        listener(old_s, new_s)
                    except Exception as e:
                        logger.warning("Listener callback failed: %s", e)

                if self.event_bus:
                    try:
                        self.event_bus.publish(
                            "state_transition",
                            {"old_state": old_s.value, "new_state": new_s.value}
                        )
                    except Exception as e:
                        logger.warning("Event bus publish failed: %s", e)
        finally:
            with self._lock:
                if self._notifying:
                    self._notifying = False

    def transition(self, new_state: State) -> State:
        candidate = State(new_state)
        
        # 1. Identify the caller file/line/function
        caller = "unknown"
        try:
            stack = inspect.stack()
            for frame in stack[1:]:
                if "state_machine.py" not in frame.filename:
                    caller = f"{frame.filename}:{frame.lineno} in {frame.function}"
                    break
        except Exception:
            pass

        with self._lock:
            old_state = self._state
            history = [f"{t['from_state']}->{t['to_state']}" for t in self._transition_audit_trail if t["success"]]
            history_str = " -> ".join(history[-5:]) or "None"

            # Log attempt structured debug
            logger.debug(
                "State transition requested",
                extra={
                    "from": old_state.value,
                    "to": candidate.value,
                    "task_id": self.task_id,
                    "caller": caller,
                    "history": history_str,
                }
            )

            # 2. Check validity
            if not self.can_transition(candidate):
                # Audit trail failure entry
                self._transition_audit_trail.append({
                    "timestamp": datetime.now().isoformat(),
                    "from_state": old_state.value,
                    "to_state": candidate.value,
                    "success": False,
                    "caller": caller,
                    "task_id": self.task_id,
                })
                if len(self._transition_audit_trail) > 100:
                    self._transition_audit_trail.pop(0)
                
                # Format extremely informative error
                allowed = self.get_valid_transitions(old_state)
                allowed_str = "\n".join(f"- {s.value}" for s in allowed)
                raise IllegalTransitionError(
                    f"Cannot transition {old_state.value} -> {candidate.value}\n"
                    f"Allowed:\n{allowed_str}"
                )

            # 3. Successful transition
            self._state = candidate
            self._transition_audit_trail.append({
                "timestamp": datetime.now().isoformat(),
                "from_state": old_state.value,
                "to_state": candidate.value,
                "success": True,
                "caller": caller,
                "task_id": self.task_id,
            })
            if len(self._transition_audit_trail) > 100:
                self._transition_audit_trail.pop(0)

        # 4. Notify (outside the lock)
        self._notify(old_state, candidate)
        return candidate

    def reset(self) -> State:
        with self._lock:
            if self._state not in {State.ERROR, State.ABORTED}:
                raise IllegalTransitionError(
                    f"Cannot reset from state {self._state.value}"
                )
            old_state = self._state
            self._state = State.IDLE
            
            self._transition_audit_trail.append({
                "timestamp": datetime.now().isoformat(),
                "from_state": old_state.value,
                "to_state": State.IDLE.value,
                "success": True,
                "caller": "reset()",
                "task_id": self.task_id,
            })
            if len(self._transition_audit_trail) > 100:
                self._transition_audit_trail.pop(0)
            
        self._notify(old_state, State.IDLE)
        return State.IDLE

    def force_idle(self) -> State:
        with self._lock:
            if self._state == State.IDLE:
                return self._state
            old_state = self._state
            self._state = State.IDLE
            
            self._transition_audit_trail.append({
                "timestamp": datetime.now().isoformat(),
                "from_state": old_state.value,
                "to_state": State.IDLE.value,
                "success": True,
                "caller": "force_idle()",
                "task_id": self.task_id,
            })
            if len(self._transition_audit_trail) > 100:
                self._transition_audit_trail.pop(0)
            
        self._notify(old_state, State.IDLE)
        return State.IDLE

    def transition_to(self, target_state: State) -> StateGuard:
        """Return a context manager that temporarily transitions to target_state."""
        return StateGuard(self, target_state)

    def __enter__(self) -> StateMachine:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_type is not None:
            import asyncio
            target_state = State.ERROR
            if issubclass(exc_type, asyncio.CancelledError):
                target_state = State.ABORTED
            try:
                if self._state not in {State.ERROR, State.ABORTED, State.COMPLETED, State.CANCELLED, State.SHUTDOWN}:
                    if self.can_transition(target_state):
                        self.transition(target_state)
                    else:
                        self.force_idle()
            except Exception:
                pass

    async def __aenter__(self) -> StateMachine:
        return self.__enter__()

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.__exit__(exc_type, exc_val, exc_tb)


__all__ = ["IllegalTransitionError", "State", "StateMachine", "StateGuard"]





# --- FILE: core/context/context.py ---

"""
TaskExecutionContext — holds correlation IDs, state machine, execution logs,
and variables isolated to a single task execution flow.
"""

# internal import removed: from __future__ import annotations

import logging
import uuid
from typing import Any
from contextvars import Token
# internal import removed: from core.state_machine import StateMachine, State
# internal import removed: from core.logging.logger import set_trace_ids, reset_trace_ids

logger = logging.getLogger("Jarvis.Context")


class TaskExecutionContext:
    """Isolated execution context container for a task."""

    def __init__(
        self,
        trace_id: str | None = None,
        task_id: str | None = None,
        event_bus: Any = None,
        state_machine: StateMachine | None = None,
    ) -> None:
        self.trace_id = trace_id or uuid.uuid4().hex[:8]
        self.task_id = task_id or uuid.uuid4().hex[:8]
        self.event_bus = event_bus
        self.state_machine = state_machine or StateMachine(event_bus=event_bus)
        self.state_machine.task_id = self.task_id
        self.variables: dict[str, Any] = {}
        self.logs: list[str] = []
        self._trace_token: Token[str | None] | None = None
        self._task_token: Token[str | None] | None = None

    def log(self, message: str, level: str = "INFO") -> None:
        """Log an execution trace message, enriched with correlation IDs."""
        self.logs.append(message)
        log_level = getattr(logging, level.upper(), logging.INFO)
        logger.log(
            log_level,
            message,
            extra={"trace_id": self.trace_id, "task_id": self.task_id},
        )

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a variable value by key."""
        return self.variables.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a variable value by key."""
        self.variables[key] = value

    def __getitem__(self, key: str) -> Any:
        return self.variables[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.variables[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.variables

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "task_id": self.task_id,
            "variables": self.variables,
            "logs": self.logs,
            "state": self.state_machine.state.value if self.state_machine else "unknown",
        }

    async def save_snapshot(self, step_id: str | None = None, metadata: dict[str, Any] | None = None) -> None:
        import json
        from pathlib import Path
        import asyncio
        
        snapshot_dir = Path("logs/traces")
        
        snapshot_data = self.to_dict()
        snapshot_data["step_id"] = step_id
        snapshot_data["metadata"] = metadata or {}
        
        def _write():
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            snapshot_file = snapshot_dir / f"{self.trace_id}.json"
            try:
                snapshot_file.write_text(json.dumps(snapshot_data, indent=2, default=str), encoding="utf-8")
            except Exception as e:
                logger.warning("Failed to save trace snapshot for %s: %s", self.trace_id, e)
                
        await asyncio.to_thread(_write)


    def __enter__(self) -> TaskExecutionContext:
        self._trace_token, self._task_token = set_trace_ids(self.trace_id, self.task_id)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_type is not None:
            import asyncio
            target_state = State.ERROR
            if issubclass(exc_type, asyncio.CancelledError):
                target_state = State.ABORTED
            
            self.log(f"Context exited with exception: {exc_val} (transitioning to {target_state.value})", level="ERROR")
            if self.state_machine:
                try:
                    current_state = self.state_machine.state
                    if current_state not in {State.ERROR, State.ABORTED, State.COMPLETED, State.CANCELLED, State.SHUTDOWN}:
                        if self.state_machine.can_transition(target_state):
                            self.state_machine.transition(target_state)
                        else:
                            self.log(f"Cannot transition to {target_state.value} from {current_state}, forcing IDLE", level="WARNING")
                            self.state_machine.force_idle()
                except Exception as e:
                    self.log(f"Error during context cleanup: {e}", level="ERROR")
        
        if self._trace_token and self._task_token:
            reset_trace_ids(self._trace_token, self._task_token)

    async def __aenter__(self) -> TaskExecutionContext:
        self._trace_token, self._task_token = set_trace_ids(self.trace_id, self.task_id)
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.__exit__(exc_type, exc_val, exc_tb)






# --- FILE: core/capability/base.py ---

"""
core_capability_base_Capability — Base class for all tools in Jarvis.
"""

# internal import removed: from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

# internal import removed: from core.autonomy.risk_evaluator import core_autonomy_risk_evaluator_RiskLevel
# internal import removed: from core.context.context import TaskExecutionContext

logger = logging.getLogger("Jarvis.core_capability_base_Capability")




class core_capability_base_Capability:
    """Base class for all tools and capabilities in Jarvis."""

    name: str = ""
    description: str = ""
    risk_level: core_autonomy_risk_evaluator_RiskLevel = core_autonomy_risk_evaluator_RiskLevel.LOW
    is_write: bool = False

    @property
    def is_write_operation(self) -> bool:
        """Alias for is_write, conforming to the abstract core_capability_base_Capability base interface."""
        return self.is_write

    async def run(self, args: dict[str, Any], context: TaskExecutionContext) -> ToolObservation:
        """Execute the capability logic in the provided task context."""
        raise NotImplementedError


@dataclass
class ToolObservation:
    tool_name: str
    arguments: dict
    execution_status: str       # "success" | "failure"
    output_summary: str
    error_message: Optional[str] = None
    duration_seconds: float = 0.0
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict:
        return {
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "execution_status": self.execution_status,
            "output_summary": self.output_summary,
            "error_message": self.error_message,
            "duration_seconds": round(self.duration_seconds, 3),
            "metadata": dict(self.metadata or {}),
        }





def core_capability_base__normalize_tool_result(result: Any) -> tuple[bool, str, str]:
    if isinstance(result, dict) and "success" in result:
        success = bool(result.get("success", False))
        output = _first_non_empty(
            result.get("output"),
            result.get("data"),
            result.get("metadata"),
        )
        error = str(result.get("error", "") or "")
        if success:
            return True, output or "Tool completed successfully.", ""
        return False, output, error or "Tool returned an error."

    success_attr = getattr(result, "success", None)
    if success_attr is None:
        text = _stringify_payload(result)
        return True, text or "Tool completed successfully.", ""

    success = bool(success_attr)
    output = _first_non_empty(
        getattr(result, "output", None),
        getattr(result, "data", None),
        getattr(result, "metadata", None),
    )
    error = str(getattr(result, "error", "") or "")
    if success:
        return True, output or "Tool completed successfully.", ""
    return False, output, error or _stringify_payload(result) or "Tool returned an error."


def _first_non_empty(*values: Any) -> str:
    for value in values:
        text = _stringify_payload(value)
        if text:
            return text
    return ""


def _stringify_payload(value: Any) -> str:
    if value in (None, "", {}, []):
        return ""
    return str(value)





# --- FILE: core/desktop/contracts.py ---

"""Contracts for desktop actions, observations, and verification results."""

# internal import removed: from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class DesktopActionType(str, Enum):
    LAUNCH_APP = "launch_application"
    FOCUS_WINDOW = "focus_window"
    MOVE_MOUSE = "move_mouse"
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    CLICK_TEXT_ON_SCREEN = "click_text_on_screen"
    CLICK_SCREEN_TARGET = "click_screen_target"
    DOUBLE_CLICK_SCREEN_TARGET = "double_click_screen_target"
    RIGHT_CLICK_SCREEN_TARGET = "right_click_screen_target"
    SCROLL = "scroll"
    DRAG = "drag"
    TYPE_TEXT = "type_text"
    PRESS_KEY = "press_key"
    HOTKEY = "hotkey"
    CLIPBOARD_GET = "clipboard_get"
    CLIPBOARD_SET = "clipboard_set"
    CLIPBOARD_PASTE = "clipboard_paste"


class DesktopRiskTier(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    CONFIRM = "confirm"
    HIGH = "high"
    BLOCKED = "blocked"


class DesktopActionStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    BLOCKED = "blocked"
    NEEDS_APPROVAL = "needs_approval"


@dataclass(frozen=True)
class DesktopAction:
    action_type: DesktopActionType | str
    params: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    expected_change: str = ""
    risk_tier: DesktopRiskTier | str | None = None
    requires_approval: bool | None = None
    action_id: str = field(default_factory=lambda: _new_id("act"))
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def action_name(self) -> str:
        if isinstance(self.action_type, DesktopActionType):
            return self.action_type.value
        return str(self.action_type)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "action_type": self.action_name,
            "description": self.description,
            "params": dict(self.params),
            "expected_change": self.expected_change,
            "risk_tier": str(self.risk_tier.value if isinstance(self.risk_tier, DesktopRiskTier) else self.risk_tier or ""),
            "requires_approval": self.requires_approval,
            "metadata": dict(self.metadata),
        }


@dataclass
class DesktopActionResult:
    action_id: str
    action_type: str
    success: bool
    status: DesktopActionStatus
    output: str = ""
    error: str = ""
    risk_tier: DesktopRiskTier = DesktopRiskTier.MEDIUM
    audit_hash: str = ""
    started_at: float = field(default_factory=time.time)
    ended_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.ended_at - self.started_at)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "action_type": self.action_type,
            "success": self.success,
            "status": self.status.value,
            "output": self.output,
            "error": self.error,
            "risk_tier": self.risk_tier.value,
            "audit_hash": self.audit_hash,
            "duration_seconds": round(self.duration_seconds, 3),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ScreenTarget:
    label: str
    x: int
    y: int
    width: int
    height: int
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class DesktopObservation:
    observation_id: str = field(default_factory=lambda: _new_id("obs"))
    screenshot_path: str = ""
    screenshot_fingerprint: str = ""
    active_window: dict[str, Any] = field(default_factory=dict)
    ocr_text: str = ""
    targets: list[ScreenTarget] = field(default_factory=list)
    confidence: float = 0.0
    low_confidence_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    captured_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "screenshot_path": self.screenshot_path,
            "screenshot_fingerprint": self.screenshot_fingerprint,
            "active_window": dict(self.active_window),
            "ocr_text": self.ocr_text,
            "targets": [target.to_dict() for target in self.targets],
            "confidence": round(self.confidence, 3),
            "low_confidence_reason": self.low_confidence_reason,
            "metadata": dict(self.metadata),
            "captured_at": self.captured_at,
        }


@dataclass(frozen=True)
class DesktopChange:
    changed: bool
    confidence: float
    summary: str
    before_observation_id: str = ""
    after_observation_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "changed": self.changed,
            "confidence": round(self.confidence, 3),
            "summary": self.summary,
            "before_observation_id": self.before_observation_id,
            "after_observation_id": self.after_observation_id,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ApprovalDecision:
    required: bool
    approved: bool
    reason: str = ""
    mode: str = "automatic"

    def to_dict(self) -> dict[str, Any]:
        return {
            "required": self.required,
            "approved": self.approved,
            "reason": self.reason,
            "mode": self.mode,
        }


__all__ = [
    "ApprovalDecision",
    "DesktopAction",
    "DesktopActionResult",
    "DesktopActionStatus",
    "DesktopActionType",
    "DesktopChange",
    "DesktopObservation",
    "DesktopRiskTier",
    "ScreenTarget",
]




# --- FILE: core/registry/registry.py ---

"""
Unified core_capability_base_Capability Registry — replaces tool router and plugin manifest loading,
merging desktop and functional capabilities into a single dynamic registry.
"""

# internal import removed: from __future__ import annotations

import asyncio
import importlib.util
import inspect
import logging
import time
from pathlib import Path
from typing import Any, Callable

# internal import removed: from core.autonomy.risk_evaluator import core_autonomy_risk_evaluator_RiskLevel
# internal import removed: from core.capability.base import core_capability_base_Capability, ToolObservation, core_capability_base__normalize_tool_result
# internal import removed: from core.context.context import TaskExecutionContext

# For desktop capability mapping
# internal import removed: from core.desktop.contracts import DesktopAction, DesktopActionType
# internal import removed: from core.desktop.mission import DesktopMissionExecutor, MissionExecutionRecord

logger = logging.getLogger("Jarvis.Registry")

DESKTOP_TOOL_NAMES: frozenset[str] = frozenset({
    "click",
    "double_click",
    "right_click",
    "click_text_on_screen",
    "click_screen_target",
    "double_click_screen_target",
    "right_click_screen_target",
    "type_text",
    "press_key",
    "hotkey",
    "move_mouse",
    "scroll",
    "drag",
    "focus_window",
    "clipboard_get",
    "clipboard_set",
    "clipboard_paste",
    "launch_application",
})

_ACTION_TYPE_MAP: dict[str, DesktopActionType] = {
    "click": DesktopActionType.CLICK,
    "double_click": DesktopActionType.DOUBLE_CLICK,
    "right_click": DesktopActionType.RIGHT_CLICK,
    "click_text_on_screen": DesktopActionType.CLICK_TEXT_ON_SCREEN,
    "click_screen_target": DesktopActionType.CLICK_SCREEN_TARGET,
    "double_click_screen_target": DesktopActionType.DOUBLE_CLICK_SCREEN_TARGET,
    "right_click_screen_target": DesktopActionType.RIGHT_CLICK_SCREEN_TARGET,
    "type_text": DesktopActionType.TYPE_TEXT,
    "press_key": DesktopActionType.PRESS_KEY,
    "hotkey": DesktopActionType.HOTKEY,
    "move_mouse": DesktopActionType.MOVE_MOUSE,
    "scroll": DesktopActionType.SCROLL,
    "drag": DesktopActionType.DRAG,
    "focus_window": DesktopActionType.FOCUS_WINDOW,
    "clipboard_get": DesktopActionType.CLIPBOARD_GET,
    "clipboard_set": DesktopActionType.CLIPBOARD_SET,
    "clipboard_paste": DesktopActionType.CLIPBOARD_PASTE,
    "launch_application": DesktopActionType.LAUNCH_APP,
}


def _build_desktop_action(
    action_name: str,
    params: dict[str, Any],
    *,
    description: str = "",
    expected_change: str = "",
    requires_approval: bool | None = None,
) -> DesktopAction:
    action_type = _ACTION_TYPE_MAP.get(action_name)
    if action_type is None:
        raise ValueError(f"Unknown desktop action: '{action_name}'")

    return DesktopAction(
        action_type=action_type,
        params=dict(params),
        description=description or f"Agent step: {action_name}",
        expected_change=expected_change,
        requires_approval=requires_approval,
        metadata={"source": "agent_loop", "original_action": action_name},
    )


def _record_to_observation(
    tool_name: str,
    record: MissionExecutionRecord,
) -> ToolObservation:
    # Handle both enum values and string statuses
    status_str = record.status.value if hasattr(record.status, "value") else str(record.status)
    if status_str.lower() in ("succeeded", "success"):
        execution_status = "success"
        output = record.explain()
        error = ""
    else:
        execution_status = "failure"
        output = ""
        error = record.explain()

    metadata: dict[str, Any] = {
        "mission_id": record.mission_id,
        "mission_status": status_str,
        "duration_seconds": record.duration_seconds,
        "steps_completed": sum(1 for s in record.steps if s.status == "succeeded"),
        "steps_total": len(record.steps),
    }

    if record.steps:
        last_step = record.steps[-1]
        action = last_step.action if isinstance(last_step.action, dict) else {}
        arguments = action.get("params", {}) if isinstance(action.get("params", {}), dict) else {}
        if last_step.change:
            metadata["change_detected"] = last_step.change.get("changed", False)
            metadata["change_summary"] = last_step.change.get("summary", "")
        if last_step.observation_after:
            metadata["final_confidence"] = last_step.observation_after.get("confidence", 0.0)
    else:
        arguments = {}

    return ToolObservation(
        tool_name=tool_name,
        arguments=arguments,
        execution_status=execution_status,
        output_summary=output,
        error_message=error,
        metadata=metadata,
    )


class FunctionCapability(core_capability_base_Capability):
    """Adapts a standard python function to the core_capability_base_Capability class interface."""

    def __init__(
        self,
        name: str,
        handler: Callable,
        risk_level: core_autonomy_risk_evaluator_RiskLevel = core_autonomy_risk_evaluator_RiskLevel.LOW,
        is_write: bool = False,
        description: str = "",
    ) -> None:
        self.name = name
        self.handler = handler
        self.risk_level = risk_level
        self.is_write = is_write
        self.description = description or (handler.__doc__ or "").strip()

    async def run(self, args: dict[str, Any], context: TaskExecutionContext) -> ToolObservation:
        sig = inspect.signature(self.handler)
        kwargs = dict(args)
        
        # Pass the task context if the handler accepts it
        if "context" in sig.parameters:
            kwargs["context"] = context

        # Support both coroutine handlers and synchronous blockers
        if inspect.iscoroutinefunction(self.handler):
            result = await self.handler(**kwargs)
        else:
            result = await asyncio.to_thread(self.handler, **kwargs)

        # Normalize outputs into ToolObservation properties
        success, output_summary, error_message = core_capability_base__normalize_tool_result(result)

        return ToolObservation(
            tool_name=self.name,
            arguments=args,
            execution_status="success" if success else "failure",
            output_summary=output_summary,
            error_message=error_message or None,
        )


class DesktopCapability(core_capability_base_Capability):
    """Executes a desktop action through PyAutoGUI / Observe-Act-Verify loop."""

    def __init__(
        self,
        name: str,
        container: Any,
        is_write: bool = True,
        risk_level: core_autonomy_risk_evaluator_RiskLevel = core_autonomy_risk_evaluator_RiskLevel.CONFIRM,
    ) -> None:
        self.name = name
        self.container = container
        self.is_write = is_write
        self.risk_level = risk_level

    async def run(self, args: dict[str, Any], context: TaskExecutionContext) -> ToolObservation:
        # Resolve DesktopMissionExecutor from container on-demand to avoid boot cycles
        desktop_executor = self.container.resolve("desktop_executor") if self.container else None
        desktop_observer = self.container.resolve("desktop_observer") if self.container else None
        
        mission_executor = DesktopMissionExecutor(
            action_executor=desktop_executor,
            observer=desktop_observer,
            max_retries=1,
            min_confidence=0.35,
        )

        requires_approval = None
        if context.get("approval_called") and context.get("approval_result"):
            requires_approval = False

        desktop_action = _build_desktop_action(
            self.name,
            args,
            description=args.get("description", f"Desktop command: {self.name}"),
            expected_change=args.get("expected_change", ""),
            requires_approval=requires_approval,
        )

        record = await mission_executor.run(
            goal=context.variables.get("goal", f"Execute {self.name}"),
            actions=[desktop_action],
            plan_summary=args.get("description", ""),
        )

        return _record_to_observation(self.name, record)


class CapabilityRegistry:
    """Unified Registry for local capabilities, API tools, and dynamically loaded plugins."""

    def __init__(self, container: Any = None) -> None:
        self.container = container
        self._capabilities: dict[str, core_capability_base_Capability] = {}
        self._call_count = 0
        self._observations: list[ToolObservation] = []

    def register(self, name_or_cap: str | core_capability_base_Capability, handler: Callable | None = None) -> None:
        """Register a tool, accepting either a core_capability_base_Capability subclass instance or legacy name/handler."""
        if isinstance(name_or_cap, core_capability_base_Capability):
            name = name_or_cap.name.strip().lower()
            self._capabilities[name] = name_or_cap
            logger.debug(f"Registered core_capability_base_Capability: {name}")
            return

        if handler is None:
            raise ValueError("Handler must be provided for function registration.")

        name = name_or_cap.strip().lower()
        
        # Self-declare properties for functional tools
        is_write = name not in {
            "get_time", "get_system_stats", "list_directory", "read_file",
            "search_memory", "capture_screen", "capture_region",
            "find_text_on_screen", "read_screen_text", "wait_for_text_on_screen",
            "describe_screen", "get_active_window", "clipboard_get",
            "web_search", "web_scrape", "list_hardware_devices",
            "ping_device", "read_sensor",
        }
        
        risk_level = core_autonomy_risk_evaluator_RiskLevel.LOW
        if name in {
            "shell_exec", "shell", "exec", "subprocess", "delete_file", "rm", "rmdir"
        }:
            risk_level = core_autonomy_risk_evaluator_RiskLevel.CRITICAL
        elif name in DESKTOP_TOOL_NAMES or is_write:
            risk_level = core_autonomy_risk_evaluator_RiskLevel.CONFIRM

        if name in DESKTOP_TOOL_NAMES:
            self._capabilities[name] = DesktopCapability(name, container=self.container, is_write=is_write, risk_level=risk_level)
        else:
            self._capabilities[name] = FunctionCapability(name, handler, risk_level=risk_level, is_write=is_write)
        logger.debug(f"Adapted function registry: {name}")

    def get(self, name: str) -> core_capability_base_Capability | None:
        return self._capabilities.get(name.strip().lower())

    def registered_tools(self) -> list[str]:
        return list(self._capabilities.keys())

    def reset_call_count(self) -> None:
        self._call_count = 0

    async def execute(self, tool_name: str, arguments: dict, context: TaskExecutionContext | None = None) -> ToolObservation:
        if context is None:
            context = TaskExecutionContext()

        cap = self.get(tool_name)
        if not cap:
            obs = ToolObservation(
                tool_name=tool_name,
                arguments=arguments,
                execution_status="failure",
                output_summary="",
                error_message=f"No capability registered for tool '{tool_name}'.",
            )
            self._observations.append(obs)
            return obs

        logger.info(f"[CAPABILITY LOG] Executing: {tool_name}({arguments})", extra={"trace_id": context.trace_id, "task_id": context.task_id})
        self._call_count += 1
        start = time.monotonic()

        try:
            obs = await cap.run(arguments, context)
            obs.duration_seconds = time.monotonic() - start
            if obs.execution_status == "success":
                logger.info(f"[CAPABILITY OK] {tool_name} completed.", extra={"trace_id": context.trace_id, "task_id": context.task_id})
            else:
                logger.warning(f"[CAPABILITY FAIL] {tool_name} failed: {obs.error_message}", extra={"trace_id": context.trace_id, "task_id": context.task_id})
        except Exception as e:
            obs = ToolObservation(
                tool_name=tool_name,
                arguments=arguments,
                execution_status="failure",
                output_summary="",
                error_message=str(e),
                duration_seconds=time.monotonic() - start,
            )
            logger.error(f"[CAPABILITY ERROR] {tool_name}: {e}", exc_info=True, extra={"trace_id": context.trace_id, "task_id": context.task_id})

        self._observations.append(obs)
        if len(self._observations) > 1000:
            self._observations = self._observations[-500:]
        return obs

    def get_observations(self) -> list[ToolObservation]:
        return list(self._observations)

    def clear_observations(self) -> None:
        self._observations.clear()

    def load_plugins(self, plugin_dir: str | Path) -> list[str]:
        directory = Path(plugin_dir)
        if not directory.exists() or not directory.is_dir():
            return []

        loaded: list[str] = []
        for path in sorted(directory.glob("*.py")):
            if path.name.startswith("_"):
                continue
            module_name = f"jarvis_plugin_{path.stem}"
            try:
                spec = importlib.util.spec_from_file_location(module_name, path)
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                register_fn = getattr(module, "register", None)
                if callable(register_fn):
                    register_fn(self)
                    loaded.append(path.stem)
            except Exception:
                continue
        return loaded




# --- FILE: core/capability/__init__.py ---

"""Capability base class and helpers."""




# --- FILE: core/context/__init__.py ---

# internal import removed: from core.context.context import TaskExecutionContext

__all__ = ["TaskExecutionContext"]




# --- FILE: core/profile.py ---

"""Persistent user profile engine for Session 3 personalization."""

# internal import removed: from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
import asyncio


class UserProfileEngine:
    PROFILE_PATH = Path("memory/user_profile.json")

    DEFAULTS = {
        "name": "User",
        "communication_style": "casual",
        "expertise_level": "intermediate",
        "preferred_topics": [],
        "timezone": "UTC",
        "language": "en",
        "interaction_count": 0,
        "first_seen": None,
        "last_seen": None,
    }

    _VALID_STYLES = {"casual", "formal", "technical"}
    _VALID_LEVELS = {"beginner", "intermediate", "advanced", "expert"}

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._data = self._fresh_defaults()
        self._load()

    def _fresh_defaults(self) -> dict:
        data = dict(self.DEFAULTS)
        data["preferred_topics"] = []
        return data

    def _load(self) -> None:
        with self._lock:
            try:
                if self.PROFILE_PATH.exists():
                    with open(self.PROFILE_PATH, "r", encoding="utf-8") as f:
                        loaded = json.load(f)
                    if isinstance(loaded, dict):
                        for k, v in loaded.items():
                            if k in self._data:
                                self._data[k] = v

                if not isinstance(self._data.get("preferred_topics"), list):
                    self._data["preferred_topics"] = []
                if self._data.get("communication_style") not in self._VALID_STYLES:
                    self._data["communication_style"] = self.DEFAULTS["communication_style"]
                if self._data.get("expertise_level") not in self._VALID_LEVELS:
                    self._data["expertise_level"] = self.DEFAULTS["expertise_level"]
                if not isinstance(self._data.get("interaction_count"), int):
                    self._data["interaction_count"] = int(self._data.get("interaction_count") or 0)
            except Exception as e:  # noqa: BLE001
                logging.getLogger(__name__).warning(f"Profile load failed: {e}")
                self._data = self._fresh_defaults()

    def save(self) -> None:
        """Atomic write to avoid corruption on interruption."""
        self.PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.PROFILE_PATH.with_suffix(".tmp")
        data_copy = dict(self._data)
        
        def _write():
            with self._lock:
                try:
                    with open(tmp, "w", encoding="utf-8") as f:
                        json.dump(data_copy, f, indent=2, ensure_ascii=False)
                    os.replace(tmp, self.PROFILE_PATH)
                except Exception as e:  # noqa: BLE001
                    logging.getLogger(__name__).error(f"Profile save failed: {e}")

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(asyncio.to_thread(_write))
        except RuntimeError:
            _write()

    def update_from_conversation(self, user_text: str, jarvis_response: str) -> None:
        with self._lock:
            now = datetime.now().isoformat()
            self._data["interaction_count"] = int(self._data.get("interaction_count", 0)) + 1
            self._data["last_seen"] = now
            if self._data.get("first_seen") is None:
                self._data["first_seen"] = now

            lower = (user_text or "").lower()
            for pattern in ("my name is ", "i am ", "i'm ", "call me "):
                if pattern in lower:
                    idx = lower.index(pattern) + len(pattern)
                    remainder = (user_text or "")[idx:].strip()
                    if not remainder:
                        break
                    candidate = remainder.split()[0].strip(".,!?\"'()[]{}")
                    if 2 <= len(candidate) <= 30 and candidate.isalpha():
                        self._data["name"] = candidate
                        break

            _ = jarvis_response
            self.save()

    def apply_delta(self, delta: dict, min_confidence: float = 0.6) -> list:
        """Apply synthesis delta and return list of updated fields."""
        if not isinstance(delta, dict):
            return []

        with self._lock:
            updated: list[str] = []
            for field, val in delta.items():
                if isinstance(val, dict):
                    try:
                        confidence = float(val.get("confidence", 0.0))
                    except (TypeError, ValueError):
                        confidence = 0.0
                    value = val.get("value")
                else:
                    confidence = 1.0
                    value = val

                if confidence < min_confidence or field not in self.DEFAULTS or value is None:
                    continue

                if field == "communication_style":
                    if value not in self._VALID_STYLES:
                        continue
                elif field == "expertise_level":
                    if value not in self._VALID_LEVELS:
                        continue
                elif field == "preferred_topics":
                    if not isinstance(value, list):
                        continue
                    cleaned_topics = []
                    for topic in value:
                        topic_text = str(topic).strip()
                        if topic_text and topic_text not in cleaned_topics:
                            cleaned_topics.append(topic_text)
                        if len(cleaned_topics) >= 10:
                            break
                    value = cleaned_topics
                elif field == "name":
                    text = str(value).strip()
                    if not text or len(text) > 30:
                        continue
                    value = text

                self._data[field] = value
                updated.append(field)

            if updated:
                self.save()
            return updated

    def get_system_prompt_injection(self) -> str:
        """Compact profile context injected into the LLM system prompt."""
        with self._lock:
            parts = [f"User: {self._data['name']}."]
            parts.append(f"Style: {self._data['communication_style']}.")
            parts.append(f"Level: {self._data['expertise_level']}.")
            if self._data.get("preferred_topics"):
                topics = ", ".join(self._data["preferred_topics"][:3])
                parts.append(f"Interests: {topics}.")
            prompt = " ".join(parts)
            words = prompt.split()
            if len(words) > 80:
                return " ".join(words[:80])
            return prompt

    def get_communication_style(self) -> str:
        with self._lock:
            style = self._data.get("communication_style", "casual")
            return {
                "formal": "Be precise and professional. Use formal language.",
                "casual": "Be friendly and conversational. Keep it natural.",
                "technical": "Be detailed and technical. Use correct terminology.",
            }.get(style, "Be helpful and clear.")

    @property
    def interaction_count(self) -> int:
        with self._lock:
            return int(self._data.get("interaction_count", 0))




# --- FILE: core/hardware/device_registry.py ---

"""
core/hardware/device_registry.py
--------------------------------
Registry of hardware devices, supporting dynamic loading and integration with SerialController.
"""

# internal import removed: from __future__ import annotations
import logging
import asyncio
from typing import Dict, List, Any
# internal import removed: from core.hardware.serial_controller import SerialController

logger = logging.getLogger(__name__)

class HardwareDevice:
    def __init__(self, name: str, controller: SerialController) -> None:
        self.name = name
        self.controller = controller

    async def async_send_command(self, command: str, value: str = "") -> str:
        """Send command to device asynchronously using a thread pool."""
        if not self.controller.enabled:
            raise NotImplementedError(f"Hardware serial control is disabled for {self.name}.")
        
        full_command = f"{command} {value}".strip()
        # Offload the blocking serial send to a thread pool
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(None, self.controller.send, full_command)
            return str(result)
        except Exception as e:
            logger.error("Failed to send command to %s: %s", self.name, e, exc_info=True)
            raise

    async def firmware_ping(self) -> bool:
        """Ping the device firmware."""
        try:
            res = await self.async_send_command("PING")
            return "PONG" in res or "OK" in res or res != ""
        except Exception:
            return False


class DeviceRegistry:
    def __init__(self) -> None:
        self._devices: Dict[str, HardwareDevice] = {}
        self._load_from_config()

    def _load_from_config(self) -> None:
        try:
            controller = SerialController()
            # If enabled, register it
            if controller.enabled:
                self.register_device("main_arduino", controller)
        except Exception as e:
            logger.error("Failed to load hardware devices from config: %s", e, exc_info=True)

    def register_device(self, name: str, controller: SerialController) -> None:
        self._devices[name] = HardwareDevice(name, controller)

    def get_device(self, name: str) -> HardwareDevice:
        if name not in self._devices:
            logger.warning("Device '%s' not registered. Creating a default disabled device.", name)
            controller = SerialController()  # Disabled by default
            self._devices[name] = HardwareDevice(name, controller)
        return self._devices[name]

    def list_devices(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": name,
                "connected": device.controller.is_connected,
                "enabled": device.controller.enabled,
            }
            for name, device in self._devices.items()
        ]




# --- FILE: core/memory/context_compressor.py ---

"""
Context compression and optional low-latency focus titling for Jarvis memory.
"""

# internal import removed: from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS = 400
core_memory_context_compressor_DEFAULT_THRESHOLD = 0.30
MAX_PREF_ITEMS = 6
MAX_EPISODE_ITEMS = 3
MAX_CONVO_ITEMS = 2
MAX_VALUE_LEN = 60
MAX_EPISODE_LEN = 100
MAX_CONVO_LEN = 120
TITLE_TIMEOUT_S = 4.0
TITLE_SYSTEM = (
    "You create short memory context titles for a local AI assistant. "
    "Return only a 3-7 word title. No quotes, no bullets, no punctuation-heavy output."
)


class ContextCompressor:
    """Compress recalled memory into a compact block for LLM injection."""

    def __init__(
        self,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        threshold: float = core_memory_context_compressor_DEFAULT_THRESHOLD,
        llm: Any | None = None,
        enable_llm_title: bool = False,
    ) -> None:
        self.max_tokens = max_tokens
        self.threshold = threshold
        self.llm = llm
        self.enable_llm_title = bool(enable_llm_title)

    async def compress(
        self,
        query: str,
        recall_results: dict,
        include_scores: bool = False,
    ) -> str:
        """Build a compact text block from structured recall results with aging and deduplication."""
        import hashlib
        import math
        from datetime import datetime

        # 1. Gather and deduplicate all items
        all_items = []
        seen_hashes = set()
        
        for category, items in recall_results.items():
            for item in items:
                item_copy = dict(item)
                item_copy["_category"] = category
                text = self._get_item_text(item_copy)
                h = hashlib.md5(text.encode("utf-8")).hexdigest()
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    all_items.append(item_copy)

        # 2. Apply temporal memory aging (decay similarity scores by e^(-0.05 * t))
        decayed_items = []
        for item in all_items:
            score = float(item.get("score", 1.0))
            ts = item.get("timestamp", "")
            if ts:
                try:
                    dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                    now = datetime.now(dt.tzinfo)
                    age_days = max(0.0, (now - dt).total_seconds() / (24 * 3600))
                    decay = math.exp(-0.05 * age_days)
                    score *= decay
                except Exception:
                    pass
            item["score"] = score
            decayed_items.append(item)

        # 3. Sort by decayed score and keep top 5
        decayed_items.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        top_5 = decayed_items[:5]

        if not top_5:
            return ""

        # 4. Check if combined text length exceeds limit, and summarize
        combined_text = "\n".join([self._get_item_text(item) for item in top_5])
        if self.llm and self._estimate_tokens(combined_text) > self.max_tokens:
            summary = await self._summarize_context(top_5)
            if summary:
                return f"--- Memory Context ---\nMemory Summary: {summary}\n--- End Memory ---"

        # 5. Format the top 5 high-score entries under categories as fallback
        prefs = [item for item in top_5 if item["_category"] == "preferences"]
        episodes = [item for item in top_5 if item["_category"] == "episodes"]
        convos = [item for item in top_5 if item["_category"] == "conversations"]

        lines: list[str] = []
        token_budget = self.max_tokens

        pref_lines, tokens_used = self._compress_preferences(prefs, include_scores)
        if pref_lines:
            lines.append("Preferences: " + " | ".join(pref_lines))
            token_budget -= tokens_used

        if token_budget > 50:
            episode_lines, tokens_used = self._compress_episodes(episodes, include_scores)
            if episode_lines:
                lines.append("Past events: " + "; ".join(episode_lines))
                token_budget -= tokens_used

        if token_budget > 50:
            convo_lines, _ = self._compress_conversations(convos, include_scores)
            if convo_lines:
                lines.append("Relevant past: " + "; ".join(convo_lines))

        if not lines:
            return ""

        title = await self._generate_focus_title(query, lines)
        if title:
            lines.insert(0, f"Focus: {title}")

        return "--- Memory Context ---\n" + "\n".join(lines) + "\n--- End Memory ---"

    def _get_item_text(self, item: dict) -> str:
        if "key" in item and "value" in item and item["key"]:
            return f"Preference: {item['key']}={item['value']}"
        elif "event" in item:
            return f"Event: {item['event']}"
        elif "user_input" in item:
            return f"Conversation: User: {item['user_input']} -> Assistant: {item['assistant_response']}"
        return str(item.get("document", ""))

    async def _summarize_context(self, top_memories: list[dict]) -> str:
        if not self.llm or not top_memories:
            return ""
        
        memory_strings = [self._get_item_text(item) for item in top_memories]
        context_text = "\n".join(memory_strings)
        prompt = (
            "You are a helpful context summarizer for a personal AI assistant.\n"
            "Summarize the following retrieved memories into a short, cohesive summary (less than 150 words) "
            "that highlights key facts, preferences, and relevant history:\n\n"
            f"{context_text}\n\n"
            "Summary:"
        )
        try:
            summary = await self.llm.complete(
                prompt,
                system="Summarize the assistant's memory context accurately.",
                temperature=0.0,
                task_type="context_summarization",
            )
            return str(summary).strip()
        except Exception as exc:
            logger.debug("Failed to summarize context with LLM: %s", exc)
            return ""

    def _compress_preferences(
        self,
        prefs: list[dict],
        include_scores: bool,
    ) -> tuple[list[str], int]:
        filtered = [item for item in prefs if item.get("score", 0) >= self.threshold]
        filtered = self._deduplicate(filtered, key="key")[:MAX_PREF_ITEMS]
        lines: list[str] = []

        for item in filtered:
            key = self._clean(item.get("key", ""))
            value = self._truncate(self._clean(item.get("value", "")), MAX_VALUE_LEN)
            entry = f"{key}={value}"
            if include_scores:
                entry += f"[{item.get('score', 0):.2f}]"
            lines.append(entry)

        return lines, self._estimate_tokens(" | ".join(lines))

    def _compress_episodes(
        self,
        episodes: list[dict],
        include_scores: bool,
    ) -> tuple[list[str], int]:
        filtered = [item for item in episodes if item.get("score", 0) >= self.threshold]
        filtered = self._deduplicate(filtered, key="event")[:MAX_EPISODE_ITEMS]
        lines: list[str] = []

        for item in filtered:
            event = self._truncate(self._clean(item.get("event", "")), MAX_EPISODE_LEN)
            timestamp = (item.get("timestamp") or "")[:10]
            entry = f"{event}" + (f" ({timestamp})" if timestamp else "")
            if include_scores:
                entry += f"[{item.get('score', 0):.2f}]"
            lines.append(entry)

        return lines, self._estimate_tokens("; ".join(lines))

    def _compress_conversations(
        self,
        convos: list[dict],
        include_scores: bool,
    ) -> tuple[list[str], int]:
        filtered = [item for item in convos if item.get("score", 0) >= self.threshold]
        filtered = filtered[:MAX_CONVO_ITEMS]
        lines: list[str] = []

        for item in filtered:
            user = self._truncate(self._clean(item.get("user_input", "")), MAX_CONVO_LEN)
            assistant = self._truncate(
                self._clean(item.get("assistant_response", "")),
                MAX_CONVO_LEN,
            )
            entry = f'"{user}" -> "{assistant}"'
            if include_scores:
                entry += f"[{item.get('score', 0):.2f}]"
            lines.append(entry)

        return lines, self._estimate_tokens("; ".join(lines))

    async def _generate_focus_title(self, query: str, lines: list[str]) -> str:
        """Generate a short semantic title using the quick-task LLM route."""
        if not self.enable_llm_title or self.llm is None or not lines:
            return ""

        prompt = (
            f"User query: {self._clean(query)}\n"
            f"Memory context:\n{chr(10).join(lines[:3])}\n\n"
            "Return the best short title for this memory context."
        )

        try:
            raw = await asyncio.wait_for(
                self.llm.complete(
                    prompt,
                    system=TITLE_SYSTEM,
                    temperature=0.0,
                    task_type="context_title_generation",
                ),
                timeout=TITLE_TIMEOUT_S
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Context title generation failed: %s", exc)
            return ""

        title = self._clean(raw.splitlines()[0] if raw else "")
        title = title.strip("\"'` .")
        if not title:
            return ""
        return self._truncate(title, 48)

    @staticmethod
    def _clean(text: str) -> str:
        return re.sub(r"\s+", " ", str(text)).strip()

    @staticmethod
    def _truncate(text: str, max_len: int) -> str:
        return text if len(text) <= max_len else text[: max_len - 3] + "..."

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, int(len(text) * 0.75))

    @staticmethod
    def _deduplicate(items: list[dict], key: str) -> list[dict]:
        seen: set[str] = set()
        result: list[dict] = []
        for item in items:
            value = str(item.get(key, ""))
            if value in seen:
                continue
            seen.add(value)
            result.append(item)
        return result

    def explain(self, query: str, recall_results: dict) -> str:
        """Return a human-readable explanation of what would be included."""
        lines = [f"ContextCompressor.explain(query={query[:60]!r})\n"]
        for category, items in recall_results.items():
            lines.append(f"  [{str(category).upper()}] - {len(items)} results:")
            for item in items:
                lines.append(f"    INCLUDED | score={item.get('score', 0):.3f}")
        return "\n".join(lines)




# --- FILE: core/registry/__init__.py ---

"""Registry module initialization."""
# internal import removed: from core.registry.registry import CapabilityRegistry

__all__ = ["CapabilityRegistry"]




# --- FILE: core/registry/base.py ---

"""
core/registry/base.py
─────────────────────
Abstract base class for all dynamically loadable tools and capabilities in Jarvis.
"""

# internal import removed: from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

# internal import removed: from core.capability.base import ToolObservation
# internal import removed: from core.context.context import TaskExecutionContext


class core_registry_base_RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    CONFIRM = "confirm"
    HIGH = "high"
    CRITICAL = "critical"


class core_registry_base_Capability(ABC):
    """Abstract base class for all tools and integrations."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier of the capability."""
        pass

    @property
    @abstractmethod
    def is_write_operation(self) -> bool:
        """True if the tool mutates state, files, or sends outbound payloads."""
        pass

    @property
    @abstractmethod
    def risk_level(self) -> core_registry_base_RiskLevel:
        """The tool risk profile (e.g. LOW, CRITICAL)."""
        pass

    @property
    @abstractmethod
    def schema(self) -> dict[str, Any]:
        """JSON schema defining the expected arguments."""
        pass

    @abstractmethod
    async def run(self, args: dict[str, Any], context: TaskExecutionContext) -> ToolObservation:
        """Asynchronous, non-blocking execution callback."""
        pass




# --- FILE: core/types/common.py ---

"""Common type definitions for Jarvis."""

# internal import removed: from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


IntegrationResult = dict[str, Any]

@dataclass
class core_types_common_ToolResult:
    """Standardised return type for all Jarvis tool functions.

    Attributes:
        success: True if the tool call succeeded.
        data:    Payload on success (arbitrary dict).
        error:   Human-readable error message on failure.
        tool_name: (Optional) Name of the tool.
    """

    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    tool_name: str = ""

    def to_llm_string(self) -> str:
        if self.success:
            return f"{self.tool_name or 'tool'} success: {self.data}"
        return f"{self.tool_name or 'tool'} error: {self.error}"

    def __repr__(self) -> str:
        if self.success:
            return f"core_types_common_ToolResult(success=True, data={self.data})"
        return f"core_types_common_ToolResult(success=False, error={self.error!r})"


class IntegrationRiskLevel(str, Enum):
    READ_ONLY = "READ_ONLY_TOOLS"
    CONFIRM = "CONFIRM_TOOLS"
    HIGH_RISK = "HIGH_RISK_TOOLS"


# Backward compatibility alias
RiskLevel = IntegrationRiskLevel




# --- FILE: core/types/__init__.py ---

"""Core types package."""

# internal import removed: from core.types.common import (
# internal import removed:     IntegrationResult,
# internal import removed:     core_types_common_ToolResult,
# internal import removed:     IntegrationRiskLevel,
# internal import removed:     RiskLevel,
# internal import removed: )

__all__ = [
    "IntegrationResult",
    "core_types_common_ToolResult",
    "IntegrationRiskLevel",
    "RiskLevel",
]




# --- FILE: integrations/registry.py ---

"""Registry and execution router for active Jarvis integrations."""

# internal import removed: from __future__ import annotations

import inspect
import logging
from typing import Any

# internal import removed: from integrations.base import BaseIntegration

logger = logging.getLogger(__name__)


class IntegrationRegistry:
    """Stores active integrations and routes tool execution by name."""

    def __init__(self) -> None:
        self._integrations: dict[str, BaseIntegration] = {}
        self._tool_owner: dict[str, str] = {}

    def register(self, integration: BaseIntegration) -> None:
        if not isinstance(integration, BaseIntegration):
            raise TypeError("integration must inherit BaseIntegration")

        name = (integration.name or integration.__class__.__name__).strip()
        if not name:
            raise ValueError("integration name cannot be empty")

        # Replace previous tool ownership for this integration.
        stale = [tool for tool, owner in self._tool_owner.items() if owner == name]
        for tool_name in stale:
            self._tool_owner.pop(tool_name, None)

        tools = integration.get_tools() or []
        for tool in tools:
            tool_name = str(tool.get("name", "")).strip()
            if not tool_name:
                logger.warning("Integration '%s' declared a nameless tool; skipping entry", name)
                continue

            previous = self._tool_owner.get(tool_name)
            if previous and previous != name:
                logger.warning(
                    "Tool '%s' reassigned from integration '%s' to '%s'",
                    tool_name,
                    previous,
                    name,
                )
            self._tool_owner[tool_name] = name

        self._integrations[name] = integration
        logger.info("Integration registered: %s (%d tools)", name, len(tools))

    def register_safety_rules(
        self,
        autonomy_governor: Any,
        risk_evaluator: Any,
    ) -> None:
        """Scan all registered tools and configure active AutonomyGovernor, RiskEvaluator."""
        tools = self.get_tools() or []
        for tool in tools:
            tool_name = str(tool.get("name", "")).strip()
            if not tool_name:
                continue

            # Risk level string from tool definition
            risk_str = str(tool.get("risk", "")).strip().lower()

            # Determine classification: low/read-only vs write/confirm
            is_write = True
            if risk_str in ("low", "read_only", "read-only"):
                is_write = False

            # Register with AutonomyGovernor
            if autonomy_governor is not None:
                if is_write:
                    autonomy_governor.register_write_tool(tool_name)
                else:
                    autonomy_governor.register_read_only_tool(tool_name)

            # Register with RiskEvaluator
            if risk_evaluator is not None:
                if risk_str == "low":
                    risk_evaluator.register_low_action(tool_name)
                elif risk_str == "medium":
                    risk_evaluator.register_medium_action(tool_name)
                else:  # default/confirm/high
                    risk_evaluator.register_confirm_action(tool_name)

    def get_tools(self) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        for name, integration in self._integrations.items():
            tools = integration.get_tools() or []
            for tool in tools:
                if isinstance(tool, dict):
                    tool_name = tool.get("name")
                    if tool_name and self._tool_owner.get(tool_name) == name:
                        merged.append(dict(tool))
        return merged

    def list_schemas(self) -> list[dict[str, Any]]:
        """Compatibility alias for planner-facing tool schemas."""
        return self.get_tools()

    def get_tool(self, tool_name: str) -> BaseIntegration | None:
        owner = self._tool_owner.get(str(tool_name).strip())
        if not owner:
            return None
        return self._integrations.get(owner)

    def list_tools(self) -> dict[str, str]:
        """Return tool name -> integration owner mapping."""
        return dict(self._tool_owner)

    async def execute(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        owner = self._tool_owner.get(tool_name)
        if not owner:
            return {
                "success": False,
                "data": None,
                "error": f"No integration registered for tool '{tool_name}'",
            }

        integration = self._integrations.get(owner)
        if integration is None:
            return {
                "success": False,
                "data": None,
                "error": f"Integration '{owner}' is not active",
            }

        try:
            execute_fn = getattr(integration, "execute")
            params = list(inspect.signature(execute_fn).parameters.values())
            call_args = args or {}

            if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params):
                result = execute_fn(tool_name, **call_args)
            else:
                result = execute_fn(tool_name, call_args)

            if inspect.isawaitable(result):
                result = await result
        except Exception as exc:  # noqa: BLE001
            logger.exception("Integration '%s' execution failed for tool '%s'", owner, tool_name)
            return {"success": False, "data": None, "error": str(exc)}

        if isinstance(result, dict):
            payload = result
        else:
            payload = {
                "success": bool(getattr(result, "success", False)),
                "data": getattr(result, "data", None),
                "error": getattr(result, "error", f"Integration '{owner}' returned unsupported result"),
            }

        return {
            "success": bool(payload.get("success", False)),
            "data": payload.get("data"),
            "error": payload.get("error"),
        }

    def list_active(self) -> list[str]:
        return sorted(self._integrations.keys())


integration_registry = IntegrationRegistry()
api_registry = integration_registry


__all__ = ["IntegrationRegistry", "integration_registry", "api_registry"]



############################################################
# MEMORY SYSTEM
############################################################


# --- FILE: core/memory/code_indexer_service.py ---

import asyncio
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

class CodeIndexerService:
    """Handles codebase indexing and chunk extraction for hybrid memory."""
    
    def __init__(self, db_pool, semantic_memory, store_episode_cb):
        self.db_pool = db_pool
        self.semantic = semantic_memory
        self.store_episode = store_episode_cb

    async def index_codebase(self, root_path: str, is_hybrid: bool, init_schema_cb: Callable) -> dict[str, int]:
        stats = {
            "indexed_files": 0,
            "indexed_chunks": 0,
            "skipped_files": 0,
            "errors": 0,
        }

        root = Path(root_path)
        if not root.exists():
            stats["errors"] += 1
            logger.warning("Codebase index path does not exist: %s", root_path)
            return stats

        exclude_dirs = {"__pycache__", ".git", ".venv", "venv", "node_modules", "jarvis_env"}

        py_files = []
        for py_file in root.rglob("*.py"):
            if any(part in exclude_dirs for part in py_file.parts):
                continue
            py_files.append(py_file)
            await asyncio.sleep(0)

        chunks_to_index = []

        for py_file in py_files:
            try:
                content = await asyncio.to_thread(py_file.read_text, encoding="utf-8", errors="replace")
                content_hash = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()
                hash_key = f"code_hash::{py_file.resolve()}"

                await init_schema_cb()
                async with self.db_pool.acquire() as conn:
                    async with conn.execute("SELECT value FROM preferences WHERE key=?", (hash_key,)) as cursor:
                        row = await cursor.fetchone()
                    if row and row["value"] == content_hash:
                        stats["skipped_files"] += 1
                        continue

                    async with conn.execute(
                        "INSERT INTO preferences (key, value, updated_at) VALUES (?, ?, ?) "
                        "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                        (hash_key, content_hash, datetime.now().isoformat()),
                    ):
                        pass

                from core.memory.code_indexer import extract_code_chunks
                extracted = extract_code_chunks(str(py_file), content)
                chunks_to_index.extend(extracted)
                stats["indexed_files"] += 1

            except Exception as exc:
                logger.warning("Failed to index %s: %s", py_file, exc)
                stats["errors"] += 1

            while len(chunks_to_index) >= 32:
                batch = chunks_to_index[:32]
                chunks_to_index = chunks_to_index[32:]
                await self._index_chunks_batch(batch, stats, is_hybrid, init_schema_cb)
                await asyncio.sleep(0.01)

            await asyncio.sleep(0.01)

        if chunks_to_index:
            await self._index_chunks_batch(chunks_to_index, stats, is_hybrid, init_schema_cb)

        return stats

    async def _index_chunks_batch(self, batch: list[dict], stats: dict, is_hybrid: bool, init_schema_cb: Callable) -> None:
        try:
            await init_schema_cb()
            async with self.db_pool.acquire() as conn:
                for item in batch:
                    async with conn.execute(
                        "INSERT INTO episodes (event, category, timestamp) VALUES (?, ?, ?)",
                        (f"{item['chunk_id']}\n{item['chunk'][:3000]}", "code", datetime.now().isoformat())
                    ):
                        pass
        except Exception as exc:
            logger.warning("Failed to store batch chunks to SQLite: %s", exc)

        if is_hybrid:
            try:
                events = [f"{item['chunk_id']}\n{item['chunk']}" for item in batch]
                if hasattr(self.semantic, "store_episodes_batch"):
                    await self.semantic.store_episodes_batch(events, category="code")
                else:
                    for event in events:
                        await self.semantic.store_episode(event, category="code")
            except Exception as exc:
                logger.warning("Failed to store batch chunks to Chroma: %s", exc)

        stats["indexed_chunks"] += len(batch)




# --- FILE: core/memory/retriever.py ---

import logging
import re
from typing import Any, Callable

logger = logging.getLogger(__name__)

class MemoryRetriever:
    """Handles hybrid search over semantic memory and SQLite storage."""
    
    def __init__(self, db_pool, semantic_memory):
        self.db_pool = db_pool
        self.semantic = semantic_memory

    @staticmethod
    def query_tokens(query: str) -> list[str]:
        tokens = re.findall(r"[a-z0-9]{3,}", str(query or "").lower())
        return tokens[:10]

    @staticmethod
    def score_text(text: str, tokens: list[str]) -> float:
        if not tokens:
            return 0.5
        lowered = str(text or "").lower()
        hits = sum(1 for token in tokens if token in lowered)
        if hits <= 0:
            return 0.0
        return min(1.0, hits / max(1.0, float(len(tokens))))

    async def recall_preferences(self, query: str, top_k: int, is_hybrid: bool, init_schema_cb: Callable) -> list[dict[str, Any]]:
        if is_hybrid:
            try:
                raw = await self.semantic.recall_preferences(query, top_k=top_k, threshold=0.0)
                return [
                    {
                        "key": item.get("metadata", {}).get("key", ""),
                        "value": item.get("metadata", {}).get("value", ""),
                        "score": item.get("score", 0.0),
                        "document": item.get("document", ""),
                    }
                    for item in raw
                ]
            except Exception as exc:
                logger.debug("Semantic preference recall failed: %s", exc)

        await init_schema_cb()
        tokens = self.query_tokens(query)
        async with self.db_pool.acquire() as conn:
            async with conn.execute(
                "SELECT key, value FROM preferences ORDER BY updated_at DESC LIMIT 200"
            ) as cursor:
                rows = await cursor.fetchall()

        ranked: list[dict[str, Any]] = []
        for row in rows:
            key = str(row["key"] or "")
            val = str(row["value"] or "")
            score = self.score_text(key, tokens)
            if tokens and score < 0.70:
                continue
            ranked.append({"key": key, "value": val, "score": score if tokens else 1.0})

        if tokens:
            ranked.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        return ranked[: max(1, top_k)]

    async def _recall_sqlite_episodes(self, query: str, top_k: int, init_schema_cb: Callable) -> list[dict[str, Any]]:
        tokens = self.query_tokens(query)
        await init_schema_cb()
        async with self.db_pool.acquire() as conn:
            async with conn.execute(
                "SELECT event, category, timestamp FROM episodes ORDER BY timestamp DESC LIMIT 200"
            ) as cursor:
                rows = await cursor.fetchall()

        ranked: list[dict[str, Any]] = []
        for row in rows:
            event = str(row["event"] or "")
            category = str(row["category"] or "")
            haystack = f"{event} {category}".strip()
            score = self.score_text(haystack, tokens)
            if tokens and score <= 0.0:
                continue
            ranked.append({
                "event": event,
                "category": category,
                "timestamp": str(row["timestamp"] or ""),
                "score": score if tokens else 0.4,
                "document": event,
            })

        ranked.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        return ranked[: max(1, top_k)]

    async def _recall_sqlite_conversations(self, query: str, top_k: int, init_schema_cb: Callable) -> list[dict[str, Any]]:
        tokens = self.query_tokens(query)
        await init_schema_cb()
        async with self.db_pool.acquire() as conn:
            async with conn.execute(
                "SELECT user_input, assistant_response, timestamp FROM conversations ORDER BY timestamp DESC LIMIT 200"
            ) as cursor:
                rows = await cursor.fetchall()

        ranked: list[dict[str, Any]] = []
        for row in rows:
            user_text = str(row["user_input"] or "")
            assistant_text = str(row["assistant_response"] or "")
            haystack = f"{user_text} {assistant_text}".strip()
            score = self.score_text(haystack, tokens)
            if tokens and score <= 0.0:
                continue
            ranked.append({
                "user_input": user_text,
                "assistant_response": assistant_text,
                "timestamp": str(row["timestamp"] or ""),
                "score": score if tokens else 0.4,
                "document": f"User: {user_text}\\nAssistant: {assistant_text}",
            })

        ranked.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        return ranked[: max(1, top_k)]

    async def recall_all(self, query: str, top_k: int, is_hybrid: bool, init_schema_cb: Callable) -> dict[str, list[dict[str, Any]]]:
        if is_hybrid:
            try:
                raw = await self.semantic.recall_all(query, top_k=top_k, threshold=0.0)
                return {
                    "preferences": [
                        {
                            "key": item.get("metadata", {}).get("key", ""),
                            "value": item.get("metadata", {}).get("value", ""),
                            "score": item.get("score", 0.0),
                            "document": item.get("document", ""),
                        }
                        for item in raw.get("preferences", [])
                    ],
                    "episodes": [
                        {
                            "event": item.get("metadata", {}).get("event", item.get("document", "")),
                            "category": item.get("metadata", {}).get("category", ""),
                            "timestamp": item.get("metadata", {}).get("timestamp", ""),
                            "score": item.get("score", 0.0),
                            "document": item.get("document", ""),
                        }
                        for item in raw.get("episodes", [])
                    ],
                    "conversations": [
                        {
                            "user_input": item.get("metadata", {}).get("user_input", ""),
                            "assistant_response": item.get("metadata", {}).get("assistant_response", ""),
                            "timestamp": item.get("metadata", {}).get("timestamp", ""),
                            "score": item.get("score", 0.0),
                            "document": item.get("document", ""),
                        }
                        for item in raw.get("conversations", [])
                    ],
                }
            except Exception as exc:
                logger.debug("Semantic recall_all failed: %s", exc)

        return {
            "preferences": await self.recall_preferences(query, top_k=top_k, is_hybrid=is_hybrid, init_schema_cb=init_schema_cb),
            "episodes": await self._recall_sqlite_episodes(query, top_k=top_k, init_schema_cb=init_schema_cb),
            "conversations": await self._recall_sqlite_conversations(query, top_k=top_k, init_schema_cb=init_schema_cb),
        }




# --- FILE: core/memory/semantic_memory.py ---

"""
memory/semantic_memory.py
─────────────────────────
Semantic memory layer for Jarvis using ChromaDB (local vector store)
and sentence-transformers for embedding generation.

Responsibilities:
  - Generate embeddings from text using a local sentence-transformer model
  - Store memories (preferences, episodic events, conversation turns) as vectors
  - Retrieve top-K most relevant memories for any query
  - Provide relevance scoring and threshold filtering
  - Combine with SQLite (long_term.py) for a hybrid exact + semantic recall

Collections (ChromaDB):
  - jarvis_preferences   : key/value preference entries
  - jarvis_episodes      : episodic memory events
  - jarvis_conversations : conversation history turns

Author: Jarvis Session 4
"""

import uuid
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

try:
    import chromadb
    from chromadb.config import Settings
except Exception:
    chromadb = None  # type: ignore
    Settings = None  # type: ignore

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

core_memory_semantic_memory_DEFAULT_MODEL      = "all-MiniLM-L6-v2"   # 80 MB, fast, good quality
DEFAULT_TOP_K      = 5
core_memory_semantic_memory_DEFAULT_THRESHOLD  = 0.30                  # cosine similarity (0–1); lower = more results
CHROMA_PATH        = str(Path(__file__).resolve().parent.parent.parent / "data" / "chroma")

COLLECTION_PREFS   = "jarvis_preferences"
COLLECTION_EPISODES= "jarvis_episodes"
COLLECTION_CONVOS  = "jarvis_conversations"


# ─── SemanticMemory ────────────────────────────────────────────────────────────

class SemanticMemory:
    """
    Local vector-based memory using ChromaDB + sentence-transformers.

    Usage:
        sm = SemanticMemory()
        sm.store_preference("favorite_drink", "coffee")
        results = sm.recall("What do I like to drink?", top_k=3)
    """

    def __init__(
        self,
        chroma_path: str = CHROMA_PATH,
        model_name: str = core_memory_semantic_memory_DEFAULT_MODEL,
        embedding_manager: Any = None,
    ):
        self.chroma_path = chroma_path
        self.model_name  = model_name
        self.embedding_manager = embedding_manager
        self._client: Any = None
        self._collections: Dict[str, Any] = {}
        self._initialized = False

    # ── Init ──────────────────────────────────────────────────────────────────

    async def initialize(self) -> bool:
        """
        Lazy initialization — load model and connect to ChromaDB.
        Returns True on success, False on failure.
        """
        if self._initialized:
            return True

        if chromadb is None or Settings is None:
            logger.warning("ChromaDB is not installed; semantic memory disabled.")
            return False
             
        try:
            if self.embedding_manager is None:
                from core.memory.embeddings import get_embedding_manager
                self.embedding_manager = get_embedding_manager(self.model_name)

            # Embedding manager now lazy-loads on first use via its embed() method.
            # We no longer block startup waiting for it.

            logger.info(f"Connecting to ChromaDB at: {self.chroma_path}")
            self._client = chromadb.PersistentClient(
                path=self.chroma_path,
                settings=Settings(anonymized_telemetry=False),
            )

            # Create or get all collections
            for name in [COLLECTION_PREFS, COLLECTION_EPISODES, COLLECTION_CONVOS]:
                self._collections[name] = self._client.get_or_create_collection(
                    name=name,
                    metadata={"hnsw:space": "cosine"},
                )

            self._initialized = True
            logger.info("SemanticMemory initialized successfully.")
            return True

        except Exception as e:
            logger.error(f"SemanticMemory initialization failed: {e}", exc_info=True)
            return False

    async def _ensure_init(self):
        if not self._initialized:
            if not await self.initialize():
                raise RuntimeError("SemanticMemory is not initialized.")

    async def _embed(self, text: str) -> List[float]:
        """Generate a normalized embedding vector for the given text."""
        from typing import cast
        return cast(list[float], await self.embedding_manager.embed(text))

    def _collection(self, name: str):
        return self._collections[name]

    # ── Store ─────────────────────────────────────────────────────────────────

    async def store_preference(self, key: str, value: str) -> str:
        """
        Upsert a user preference into the vector store.
        The document text is: "key: value" for rich semantic matching.
        Returns the document ID.
        """
        await self._ensure_init()
        doc_id   = f"pref_{key}"
        doc_text = f"{key}: {value}"
        embedding = await self._embed(doc_text)

        await asyncio.to_thread(
            self._collection(COLLECTION_PREFS).upsert,
            ids=[doc_id],
            embeddings=[embedding],
            documents=[doc_text],
            metadatas=[{
                "key":        key,
                "value":      value,
                "updated_at": datetime.now().isoformat(),
            }],
        )
        logger.debug(f"Stored preference: {doc_id} → {doc_text}")
        return doc_id

    async def store_episode(self, event: str, category: str = "general") -> str:
        """
        Store an episodic memory event.
        Returns the document ID.
        """
        await self._ensure_init()
        doc_id    = f"ep_{uuid.uuid4().hex[:12]}"
        embedding = await self._embed(event)

        await asyncio.to_thread(
            self._collection(COLLECTION_EPISODES).add,
            ids=[doc_id],
            embeddings=[embedding],
            documents=[event],
            metadatas=[{
                "category":  category,
                "timestamp": datetime.now().isoformat(),
            }],
        )
        logger.debug(f"Stored episode: {doc_id}")
        return doc_id

    async def store_episodes_batch(
        self,
        events: list[str],
        category: str = "general",
    ) -> list[str]:
        """
        Store a batch of episodic memory events efficiently.
        Returns the list of document IDs.
        """
        if not events:
            return []
        await self._ensure_init()
        embeddings = await self.embedding_manager.embed_batch(events)
        doc_ids = [f"ep_{uuid.uuid4().hex[:12]}" for _ in range(len(events))]

        await asyncio.to_thread(
            self._collection(COLLECTION_EPISODES).add,
            ids=doc_ids,
            embeddings=embeddings,
            documents=events,
            metadatas=[{
                "category":  category,
                "timestamp": datetime.now().isoformat(),
            } for _ in range(len(events))],
        )
        logger.debug(f"Stored {len(events)} episodes in batch")
        return doc_ids


    async def store_conversation_turn(
        self,
        user_input: str,
        assistant_response: str,
        session_id: str = "default",
    ) -> str:
        """
        Store a conversation exchange as a single vector document.
        The document combines user + assistant text for richer context matching.
        Returns the document ID.
        """
        await self._ensure_init()
        doc_id    = f"conv_{uuid.uuid4().hex[:12]}"
        doc_text  = f"User: {user_input}\nAssistant: {assistant_response}"
        embedding = await self._embed(doc_text)

        await asyncio.to_thread(
            self._collection(COLLECTION_CONVOS).add,
            ids=[doc_id],
            embeddings=[embedding],
            documents=[doc_text],
            metadatas=[{
                "user_input":          user_input,
                "assistant_response":  assistant_response,
                "session_id":          session_id,
                "timestamp":           datetime.now().isoformat(),
            }],
        )
        logger.debug(f"Stored conversation turn: {doc_id}")
        return doc_id

    # ── Recall ────────────────────────────────────────────────────────────────

    async def recall_preferences(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        threshold: float = core_memory_semantic_memory_DEFAULT_THRESHOLD,
    ) -> List[Dict]:
        """
        Retrieve the most semantically similar preferences to the query.
        Returns a list of result dicts sorted by relevance score (descending).
        """
        return await self._query_collection(COLLECTION_PREFS, query, top_k, threshold)

    async def recall_episodes(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        threshold: float = core_memory_semantic_memory_DEFAULT_THRESHOLD,
    ) -> List[Dict]:
        """Retrieve the most relevant episodic memories for the query."""
        return await self._query_collection(COLLECTION_EPISODES, query, top_k, threshold)

    async def recall_conversations(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        threshold: float = core_memory_semantic_memory_DEFAULT_THRESHOLD,
    ) -> List[Dict]:
        """Retrieve the most relevant past conversation turns for the query."""
        return await self._query_collection(COLLECTION_CONVOS, query, top_k, threshold)

    async def recall_all(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        threshold: float = core_memory_semantic_memory_DEFAULT_THRESHOLD,
    ) -> Dict[str, List[Dict]]:
        """
        Recall across ALL collections simultaneously.
        Returns a dict with keys: 'preferences', 'episodes', 'conversations'.
        Each value is a sorted list of result dicts.
        """
        results = await asyncio.gather(
            self.recall_preferences(query, top_k, threshold),
            self.recall_episodes(query, top_k, threshold),
            self.recall_conversations(query, top_k, threshold)
        )
        return {
            "preferences":    results[0],
            "episodes":       results[1],
            "conversations":  results[2],
        }

    # ── Core Query ────────────────────────────────────────────────────────────

    async def _query_collection(
        self,
        collection_name: str,
        query: str,
        top_k: int,
        threshold: float,
    ) -> List[Dict]:
        """
        Internal: query a ChromaDB collection, filter by threshold, return results.
        ChromaDB returns cosine *distance* (0=identical, 2=opposite).
        We convert: similarity = 1 - (distance / 2) → range [0, 1]
        """
        await self._ensure_init()
        collection = self._collection(collection_name)

        try:
            # Don't query empty collections — ChromaDB raises on n_results > count
            count = await asyncio.to_thread(collection.count)
            if count == 0:
                return []

            actual_k = min(top_k, count)
            embedding = await self._embed(query)

            results = await asyncio.to_thread(
                collection.query,
                query_embeddings=[embedding],
                n_results=actual_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.warning(
                "Semantic query failed for collection '%s': %s",
                collection_name,
                exc,
            )
            return []

        hits: List[Dict[str, Any]] = []
        if not results["ids"]:
            return hits

        ids       = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for doc_id, doc, meta, dist in zip(ids, documents, metadatas, distances):
            # Convert cosine distance → similarity score [0, 1]
            similarity = max(0.0, 1.0 - (dist / 2.0))
            if similarity >= threshold:
                hits.append({
                    "id":         doc_id,
                    "document":   doc,
                    "metadata":   meta,
                    "score":      round(similarity, 4),
                    "collection": collection_name,
                })

        # Sort highest relevance first
        hits.sort(key=lambda x: x["score"], reverse=True)
        return hits

    # ── Delete ────────────────────────────────────────────────────────────────

    async def delete_preference(self, key: str) -> bool:
        """Delete a preference by key. Returns True if deleted."""
        await self._ensure_init()
        doc_id = f"pref_{key}"
        try:
            await asyncio.to_thread(self._collection(COLLECTION_PREFS).delete, ids=[doc_id])
            logger.debug(f"Deleted preference: {doc_id}")
            return True
        except Exception as e:
            logger.warning(f"Could not delete preference {key}: {e}")
            return False

    async def clear_collection(self, collection_name: str) -> bool:
        """Delete and recreate a collection (full wipe). Use with caution."""
        await self._ensure_init()
        try:
            await asyncio.to_thread(self._client.delete_collection, collection_name)
            self._collections[collection_name] = await asyncio.to_thread(
                self._client.get_or_create_collection,
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"Cleared collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to clear collection {collection_name}: {e}", exc_info=True)
            return False

    # ── Stats ─────────────────────────────────────────────────────────────────

    async def stats(self) -> Dict[str, Any]:
        """Return counts for all collections."""
        if not self._initialized:
            return {"initialized": False}
        prefs_count = await asyncio.to_thread(self._collection(COLLECTION_PREFS).count)
        episodes_count = await asyncio.to_thread(self._collection(COLLECTION_EPISODES).count)
        convos_count = await asyncio.to_thread(self._collection(COLLECTION_CONVOS).count)
        return {
            "initialized":    True,
            "model":          self.model_name,
            "preferences":    prefs_count,
            "episodes":       episodes_count,
            "conversations":  convos_count,
        }

    def is_ready(self) -> bool:
        return self._initialized

    async def close(self) -> None:
        """Close/stop the ChromaDB client if it has one."""
        if self._client is not None:
            try:
                if hasattr(self._client, "_system") and hasattr(self._client._system, "stop"):
                    await asyncio.to_thread(self._client._system.stop)
            except Exception:
                pass
            self._client = None
            self._initialized = False





# --- FILE: core/memory/sqlite_pool.py ---

"""SQLite connection pool for Project Jarvis."""

# internal import removed: from __future__ import annotations

import asyncio
import contextlib
import aiosqlite


class SQLitePool:
    """A simple async-safe connection pool for SQLite."""

    def __init__(self, db_path: str, pool_size: int = 3):
        self.db_path = db_path
        self.pool_size = pool_size
        self._pool: asyncio.Queue[aiosqlite.Connection] | None = None
        self._all_conns: set[aiosqlite.Connection] = set()
        self._in_use_conns: set[aiosqlite.Connection] = set()
        self._closed = False
        self._close_waiter: asyncio.Event | None = None
        self._lock = asyncio.Lock()

    async def _init_pool(self):
        if self._pool is not None:
            return
        pool: asyncio.Queue[aiosqlite.Connection] = asyncio.Queue(maxsize=self.pool_size)
        conns: list[aiosqlite.Connection] = []
        try:
            for _ in range(self.pool_size):
                conn = await aiosqlite.connect(self.db_path, timeout=30.0)
                conns.append(conn)
                conn.row_factory = aiosqlite.Row
                # Configure Write-Ahead Logging (WAL) and performance/concurrency defaults
                async with conn.execute("PRAGMA journal_mode=WAL;"):
                    pass
                async with conn.execute("PRAGMA synchronous=NORMAL;"):
                    pass
                await pool.put(conn)
        except Exception:
            # Roll back/close all opened connections on initialization failure to prevent resource leak
            for conn in conns:
                try:
                    await conn.close()
                except Exception:
                    pass
            raise
        else:
            self._pool = pool
            self._all_conns = set(conns)
            self._closed = False

    @staticmethod
    async def _rollback_quietly(conn: aiosqlite.Connection) -> None:
        try:
            await conn.rollback()
        except Exception:
            pass

    @staticmethod
    async def _close_connection(conn: aiosqlite.Connection) -> None:
        try:
            await conn.close()
        except Exception:
            pass

    @contextlib.asynccontextmanager
    async def acquire(self):
        """Acquire a connection from the pool, committing on success or rolling back on error."""
        if self._closed:
            raise RuntimeError("Database pool is closed")

        async with self._lock:
            if self._pool is None:
                await self._init_pool()

        pool = self._pool
        if pool is None or self._closed:
            raise RuntimeError("Database pool is closed")

        try:
            conn = await pool.get()
        except RuntimeError as e:
            raise RuntimeError("Database pool is closed") from e

        async with self._lock:
            if self._closed or self._pool is None or conn not in self._all_conns:
                self._all_conns.discard(conn)
                should_close = True
            else:
                self._in_use_conns.add(conn)
                should_close = False

        # Double check if pool was closed while waiting.
        if should_close:
            await self._close_connection(conn)
            raise RuntimeError("Database pool was closed during acquire")

        try:
            yield conn
            await conn.commit()
        except asyncio.CancelledError:
            await self._rollback_quietly(conn)
            raise
        except Exception:
            await self._rollback_quietly(conn)
            raise
        finally:
            should_close = False
            close_waiter = None
            async with self._lock:
                self._in_use_conns.discard(conn)
                if not self._closed and self._pool is not None and conn in self._all_conns:
                    self._pool.put_nowait(conn)
                else:
                    self._all_conns.discard(conn)
                    should_close = True

                if self._closed and not self._in_use_conns and not self._all_conns:
                    close_waiter = self._close_waiter
                    self._close_waiter = None

            if should_close:
                await self._close_connection(conn)

            if close_waiter is not None and not close_waiter.is_set():
                close_waiter.set()

    async def close(self):
        """Close all connections in the pool."""
        drained_conns = []
        close_waiter = None
        async with self._lock:
            self._closed = True
            pool = self._pool
            if pool is None:
                return

            # Wake up any tasks currently blocked on get()
            if hasattr(pool, "_getters"):
                for getter in pool._getters:
                    if not getter.done():
                        getter.set_exception(RuntimeError("Database pool is closed"))

            # Drain idle connections from the queue and close them outside the lock.
            while not pool.empty():
                try:
                    conn = pool.get_nowait()
                except asyncio.QueueEmpty:
                    break
                else:
                    drained_conns.append(conn)
                    self._all_conns.discard(conn)

            self._pool = None

            if self._in_use_conns and self._all_conns:
                close_waiter = self._close_waiter or asyncio.Event()
                self._close_waiter = close_waiter
            else:
                self._all_conns.clear()
                self._close_waiter = None

        for conn in drained_conns:
            await self._close_connection(conn)

        if close_waiter is not None:
            await close_waiter.wait()




# --- FILE: core/memory/sqlite_storage.py ---

import json
import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

class SQLiteStorage:
    """Handles raw SQLite operations for memory (preferences, episodes, conversations, actions)."""
    
    def __init__(self, pool):
        self.pool = pool
        self._sqlite_initialized = False

    async def init_schema(self) -> None:
        if self._sqlite_initialized:
            return
        async with self.pool.acquire() as conn:
            async with conn.execute("PRAGMA journal_mode=WAL;"):
                pass
            
            # Handle schema evolution for legacy databases
            for table in ["episodes", "conversations", "actions"]:
                try:
                    async with conn.execute(f"ALTER TABLE {table} ADD COLUMN timestamp TEXT DEFAULT ''"):
                        pass
                except Exception:
                    pass
            try:
                async with conn.execute("ALTER TABLE preferences ADD COLUMN updated_at TEXT DEFAULT ''"):
                    pass
            except Exception:
                pass

            async with conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS preferences (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TEXT
                );

                CREATE TABLE IF NOT EXISTS episodes (
                    id INTEGER PRIMARY KEY,
                    event TEXT,
                    category TEXT,
                    timestamp TEXT
                );

                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY,
                    user_input TEXT,
                    assistant_response TEXT,
                    session_id TEXT,
                    timestamp TEXT
                );

                CREATE TABLE IF NOT EXISTS actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action TEXT NOT NULL,
                    result TEXT,
                    success INTEGER NOT NULL DEFAULT 1,
                    metadata TEXT,
                    timestamp TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_preferences_updated_at ON preferences(updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_episodes_timestamp ON episodes(timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_conversations_timestamp ON conversations(timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_actions_timestamp ON actions(timestamp DESC);
                """
            ):
                pass
            self._sqlite_initialized = True

    async def store_preference(self, key: str, value: str) -> None:
        await self.init_schema()
        now = datetime.now().isoformat()
        async with self.pool.acquire() as conn:
            async with conn.execute(
                "INSERT INTO preferences (key, value, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (key, value, now),
            ):
                pass

    async def store_episode(self, event: str, category: str = "general") -> None:
        await self.init_schema()
        now = datetime.now().isoformat()
        async with self.pool.acquire() as conn:
            async with conn.execute(
                "INSERT INTO episodes (event, category, timestamp) VALUES (?, ?, ?)",
                (event, category, now),
            ):
                pass

    async def store_episodes_batch(self, events: list[str], category: str = "general") -> None:
        await self.init_schema()
        now = datetime.now().isoformat()
        async with self.pool.acquire() as conn:
            async with conn.executemany(
                "INSERT INTO episodes (event, category, timestamp) VALUES (?, ?, ?)",
                [(event, category, now) for event in events],
            ):
                pass

    async def store_conversation(self, user_input: str, assistant_response: str, session_id: str) -> None:
        await self.init_schema()
        now = datetime.now().isoformat()
        async with self.pool.acquire() as conn:
            async with conn.execute(
                "INSERT INTO conversations (user_input, assistant_response, session_id, timestamp) VALUES (?, ?, ?, ?)",
                (user_input, assistant_response, session_id, now),
            ):
                pass

    async def store_action(self, action: str, result: str, success: bool, metadata: dict | None) -> None:
        await self.init_schema()
        async with self.pool.acquire() as conn:
            async with conn.execute(
                "INSERT INTO actions (action, result, success, metadata, timestamp) VALUES (?, ?, ?, ?, ?)",
                (action, result, int(success), json.dumps(metadata or {}), datetime.now().isoformat()),
            ):
                pass

    async def recent_actions(self, limit: int = 20) -> list[dict[str, Any]]:
        await self.init_schema()
        async with self.pool.acquire() as conn:
            async with conn.execute(
                "SELECT action, result, success, metadata, timestamp FROM actions ORDER BY id DESC LIMIT ?",
                (max(1, limit),),
            ) as cursor:
                rows = await cursor.fetchall()
        return [
            {
                "action": r["action"],
                "result": r["result"],
                "success": bool(r["success"]),
                "metadata": json.loads(r["metadata"] or "{}"),
                "timestamp": r["timestamp"],
            }
            for r in rows
        ]





# --- FILE: core/memory/hybrid_memory.py ---

"""Hybrid memory: SQLite for structure plus optional Chroma semantic memory."""

# internal import removed: from __future__ import annotations

import contextlib
import asyncio
from pathlib import Path
from typing import Any
import logging

# internal import removed: from core.memory.semantic_memory import SemanticMemory
# internal import removed: from core.memory.sqlite_pool import SQLitePool
# internal import removed: from core.memory.sqlite_storage import SQLiteStorage
# internal import removed: from core.memory.retriever import MemoryRetriever
# internal import removed: from core.memory.code_indexer_service import CodeIndexerService

logger = logging.getLogger(__name__)

def _looks_like_config(value: Any) -> bool:
    return hasattr(value, "get") and hasattr(value, "has_option")

class HybridMemory:
    def __init__(
        self,
        config_or_db_path: Any = "memory/memory.db",
        chroma_path: str = "data/chroma",
        model_name: str = "all-MiniLM-L6-v2",
        *,
        db_path: str | None = None,
    ):
        if db_path is not None:
            resolved_db = db_path
        elif _looks_like_config(config_or_db_path):
            cfg = config_or_db_path
            resolved_db = cfg.get("memory", "sqlite_file", fallback=cfg.get("memory", "db_path", fallback="memory/memory.db"))
            chroma_path = cfg.get("memory", "chroma_dir", fallback=cfg.get("memory", "chroma_path", fallback=chroma_path))
            model_name = cfg.get("memory", "embedding_model", fallback=model_name)
        else:
            resolved_db = str(config_or_db_path)

        self.db_path = resolved_db
        self.chroma_path = chroma_path
        self.model_name = model_name
        self.mode = "sqlite-only"
        self._llm: Any | None = None
        self._enable_llm_context_titles = True

        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self.semantic = SemanticMemory(chroma_path=self.chroma_path, model_name=self.model_name)
        self._pool = SQLitePool(self.db_path, pool_size=3)
        self._storage = SQLiteStorage(self._pool)
        self._retriever = MemoryRetriever(self._pool, self.semantic)
        self._indexer = CodeIndexerService(self._pool, self.semantic, self.store_episode)

        self._init_lock = asyncio.Lock()
        self._background_tasks: set[asyncio.Task[Any]] = set()

    async def initialize(self, index_path: str = "") -> dict[str, Any]:
        await self._ensure_db_initialized()
        semantic_ready = False
        try:
            semantic_ready = bool(await self.semantic.initialize())
        except Exception as exc:
            logger.warning("Semantic memory initialization failed: %s", exc)
            semantic_ready = False

        self.mode = "hybrid" if semantic_ready else "sqlite-only"
        result: dict[str, Any] = {
            "mode": self.mode,
            "sqlite": True,
            "semantic": semantic_ready,
        }

        if index_path:
            self._track_background_task(
                asyncio.create_task(
                    self.index_codebase(index_path),
                    name="hybrid_memory_code_index",
                )
            )
            result["codebase_index"] = {"status": "background_indexing_started"}

        return result

    def _track_background_task(self, task: asyncio.Task[Any]) -> None:
        self._background_tasks.add(task)
        def _finalize_background_task(done_task: asyncio.Task[Any]) -> None:
            self._background_tasks.discard(done_task)
            with contextlib.suppress(asyncio.CancelledError):
                exc = done_task.exception()
                if exc is not None:
                    logger.warning("HybridMemory background task failed: %s", exc)
        task.add_done_callback(_finalize_background_task)

    def set_llm(self, llm: Any | None, *, enable_context_titles: bool = True) -> None:
        self._llm = llm
        self._enable_llm_context_titles = bool(enable_context_titles)

    async def _ensure_db_initialized(self) -> None:
        async with self._init_lock:
            await self._storage.init_schema()
            
    # Need to keep this private alias used heavily in the older file
    async def _init_sqlite(self) -> None:
        await self._ensure_db_initialized()

    async def store_preference(self, key: str, value: str) -> bool:
        await self._storage.store_preference(key, value)
        if self.mode == "hybrid":
            try:
                await self.semantic.store_preference(key, value)
            except Exception as exc:
                logger.debug("Semantic preference store failed: %s", exc)
        return True

    async def store_episode(self, event: str, category: str = "general") -> bool:
        await self._storage.store_episode(event, category)
        if self.mode == "hybrid":
            try:
                await self.semantic.store_episode(event, category)
            except Exception as exc:
                logger.debug("Semantic episode store failed: %s", exc)
        return True

    async def store_episodes_batch(self, events: list[str], category: str = "general") -> bool:
        await self._storage.store_episodes_batch(events, category)
        if self.mode == "hybrid":
            try:
                if hasattr(self.semantic, "store_episodes_batch"):
                    await self.semantic.store_episodes_batch(events, category=category)
                else:
                    for event in events:
                        await self.semantic.store_episode(event, category=category)
            except Exception as exc:
                logger.debug("Semantic batch store failed: %s", exc)
        return True

    async def store_conversation(self, user_input: str, assistant_response: str, session_id: str = "default") -> bool:
        await self._storage.store_conversation(user_input, assistant_response, session_id)
        if self.mode == "hybrid":
            try:
                await self.semantic.store_conversation_turn(user_input, assistant_response, session_id)
            except Exception as exc:
                logger.debug("Semantic conversation store failed: %s", exc)
        return True

    async def recall_preferences(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        return await self._retriever.recall_preferences(
            query, top_k, self.mode == "hybrid", self._ensure_db_initialized
        )

    async def recall_all(self, query: str, top_k: int = 5) -> dict[str, list[dict[str, Any]]]:
        return await self._retriever.recall_all(
            query, top_k, self.mode == "hybrid", self._ensure_db_initialized
        )

    @staticmethod
    def _query_tokens(query: str) -> list[str]:
        return MemoryRetriever.query_tokens(query)

    @staticmethod
    def _score_text(text: str, tokens: list[str]) -> float:
        return MemoryRetriever.score_text(text, tokens)

    async def _recall_sqlite_episodes(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        return await self._retriever._recall_sqlite_episodes(query, top_k, self._ensure_db_initialized)

    async def _recall_sqlite_conversations(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        return await self._retriever._recall_sqlite_conversations(query, top_k, self._ensure_db_initialized)

    async def build_context_block(self, query: str, n_results: int = 5) -> str:
        try:
            from core.memory.context_compressor import ContextCompressor
            compressor = ContextCompressor(
                threshold=0.0,
                llm=self._llm,
                enable_llm_title=self._enable_llm_context_titles,
            )
            recalled = await self.recall_all(query, top_k=n_results)
            return await compressor.compress(query, recalled)
        except Exception:
            return ""

    async def recall(self, query: str, top_k: int = 5) -> str:
        hits = await self.recall_preferences(query, top_k=top_k)
        if not hits:
            return "I don't know yet. I could not find related facts."
        lines = []
        for item in hits:
            key = item.get("key", "")
            value = item.get("value", "")
            if key or value:
                lines.append(f"{key}: {value}".strip(": "))
        if not lines:
            return "I don't know yet. I could not find related facts."
        return "\\n".join(lines)

    async def store_code_file(self, file_path: str, content: str) -> int:
        from core.memory.code_indexer import extract_code_chunks
        chunks = extract_code_chunks(file_path, content)
        chunks_stored = 0
        for item in chunks:
            chunk_id = item["chunk_id"]
            chunk = item["chunk"]
            metadata = item["metadata"]
            if self.mode == "hybrid":
                try:
                    if hasattr(self.semantic, "store_code_chunk"):
                        await self.semantic.store_code_chunk(chunk_id, chunk, metadata=metadata)
                    else:
                        await self.semantic.store_episode(f"{chunk_id}\n{chunk}", category="code")
                except Exception as exc:
                    logger.debug("Semantic code chunk store failed: %s", exc)
            await self.store_episode(f"{chunk_id}\n{chunk[:3000]}", category="code")
            chunks_stored += 1
        return max(1, chunks_stored)

    async def index_codebase(self, root_path: str) -> dict[str, int]:
        return await self._indexer.index_codebase(
            root_path, self.mode == "hybrid", self._ensure_db_initialized
        )

    def stats(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "sqlite": True,
            "semantic": self.mode == "hybrid",
        }

    async def store_fact(self, key: str, value: str, source: str = "user", **_kwargs) -> None:
        await self.store_preference(key, value)

    async def get_fact(self, key: str):
        await self._ensure_db_initialized()
        async with self._pool.acquire() as conn:
            async with conn.execute("SELECT key, value FROM preferences WHERE key=?", (key,)) as cursor:
                row = await cursor.fetchone()
        if row is None:
            return None
        class _Fact:
            def __init__(self, k, v):
                self.key, self.value = k, v
            def __repr__(self):
                return f"Fact(key={self.key!r}, value={self.value!r})"
        return _Fact(row["key"], row["value"])

    async def list_facts(self, limit: int = 50) -> list:
        await self._ensure_db_initialized()
        async with self._pool.acquire() as conn:
            async with conn.execute(
                "SELECT key, value FROM preferences ORDER BY updated_at DESC LIMIT ?", (max(1, limit),)
            ) as cursor:
                rows = await cursor.fetchall()
        class _Fact:
            def __init__(self, k, v):
                self.key, self.value = k, v
        return [_Fact(r["key"], r["value"]) for r in rows]

    async def count(self) -> int:
        await self._ensure_db_initialized()
        async with self._pool.acquire() as conn:
            async with conn.execute("SELECT COUNT(*) FROM preferences") as cursor:
                row = await cursor.fetchone()
        return row[0] if row else 0

    async def store_action(self, action: str, result: str = "", success: bool = True, metadata: dict | None = None) -> None:
        await self._storage.store_action(action, result, success, metadata)

    async def store_failure(self, action: str, error: str = "", metadata: dict | None = None) -> None:
        await self.store_action(action, result=error, success=False, metadata=metadata)

    async def recent_actions(self, limit: int = 20) -> list[dict[str, Any]]:
        return await self._storage.recent_actions(limit)

    async def set_preference(self, key: str, value: str, category: str = "general", **_kwargs) -> None:
        scoped_key = f"{category}::{key}" if category and category != "general" else key
        await self.store_preference(scoped_key, value)
        await self.store_preference(key, value)

    async def get_preferences(self, category: str = "") -> dict[str, str]:
        prefix = f"{category}::" if category else ""
        await self._ensure_db_initialized()
        async with self._pool.acquire() as conn:
            if prefix:
                async with conn.execute(
                    "SELECT key, value FROM preferences WHERE key LIKE ? ORDER BY updated_at DESC", (f"{prefix}%",)
                ) as cursor:
                    rows = await cursor.fetchall()
                return {r["key"].removeprefix(prefix): r["value"] for r in rows}
            async with conn.execute("SELECT key, value FROM preferences ORDER BY updated_at DESC") as cursor:
                rows = await cursor.fetchall()
            return {r["key"]: r["value"] for r in rows}

    async def close(self) -> None:
        background_tasks = list(self._background_tasks)
        for task in background_tasks:
            task.cancel()
        if background_tasks:
            await asyncio.gather(*background_tasks, return_exceptions=True)
        self._background_tasks.clear()
        if hasattr(self, "_pool") and self._pool is not None:
            await self._pool.close()
        if hasattr(self, "semantic") and self.semantic is not None:
            await self.semantic.close()

__all__ = ["HybridMemory"]




# --- FILE: core/memory/__init__.py ---





# --- FILE: core/memory/code_indexer.py ---

import ast
from typing import Any


def _fallback_file_chunk(
    file_path: str,
    content: str,
    *,
    chunk_type: str,
    error: str | None = None,
) -> dict[str, Any]:
    payload = (content or "").strip() or f"file:{file_path}"
    metadata: dict[str, Any] = {"file": str(file_path), "type": chunk_type}
    if error:
        metadata["error"] = error
    return {
        "chunk_id": f"file:{file_path}",
        "chunk": payload,
        "metadata": metadata,
    }


def extract_code_chunks(file_path: str, content: str) -> list[dict[str, Any]]:
    """
    Parse Python code and extract class/function chunks for semantic retrieval.
    """
    chunks: list[dict[str, Any]] = []
    
    try:
        tree = ast.parse(content)
    except SyntaxError as exc:
        # If there's a syntax error, we just return the whole file as a single chunk
        return [
            _fallback_file_chunk(
                file_path,
                content,
                chunk_type="FileSyntaxError",
                error=str(exc),
            )
        ]

    lines = content.splitlines()

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue

        start = max(0, getattr(node, "lineno", 1) - 1)
        end_lineno_attr = getattr(node, "end_lineno", None)
        end_lineno = int(end_lineno_attr) if end_lineno_attr is not None else getattr(node, "lineno", 1)
        end = min(len(lines), max(start + 1, int(end_lineno)))
        chunk = "\n".join(lines[start:end]).strip()
        if not chunk:
            continue

        chunk_id = f"{file_path}::{getattr(node, 'name', 'anonymous')}"
        metadata = {
            "file": str(file_path),
            "name": getattr(node, "name", ""),
            "type": type(node).__name__,
            "lines": f"{start + 1}-{end}",
        }
        chunks.append({
            "chunk_id": chunk_id,
            "chunk": chunk,
            "metadata": metadata
        })

    if not chunks:
        # If no functions or classes found, return the file as a single chunk
        chunks.append(
            _fallback_file_chunk(file_path, content, chunk_type="File")
        )

    return chunks




# --- FILE: core/memory/embeddings.py ---

"""
core/embeddings.py
───────────────────
Embedding manager for Jarvis.

Handles:
  - Lazy loading of sentence-transformer model (loads once, stays in memory)
  - Single and batch text embedding
  - Cosine similarity computation
  - Cache for repeated text lookups (avoids re-embedding identical strings)
  - Model health checking and warm-up

Available Models (local, no API):
  Model                     Size    Speed    Quality
  ──────────────────────────────────────────────────
  all-MiniLM-L6-v2         80 MB   Fast     Good      ← Default
  all-MiniLM-L12-v2        120 MB  Medium   Better
  all-mpnet-base-v2         420 MB  Slow     Best
  paraphrase-MiniLM-L6-v2  80 MB   Fast     Good (paraphrase-tuned)

Author: Jarvis Session 4
"""
# internal import removed: from __future__ import annotations

import asyncio
import logging
import hashlib
import threading

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


import numpy as np

logger = logging.getLogger(__name__)


# ─── Default Config ───────────────────────────────────────────────────────────

core_memory_embeddings_DEFAULT_MODEL    = "all-MiniLM-L6-v2"
CACHE_SIZE       = 512    # LRU cache: max cached embeddings
WARM_UP_TEXT     = "Hello, I am Jarvis."  # Used to pre-warm the model


# ─── Deterministic Mock Sentence Transformer ──────────────────────────────────

class DeterministicMockSentenceTransformer:
    """
    Zero-dependency, offline deterministic bag-of-words embedding model.
    Generates word vectors using hash-seeded uniform random projections.
    This preserves cosine similarity relations (e.g. sharing words increases similarity),
    enabling all offline tests to run cleanly and extremely fast.
    """
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self._word_vectors: dict[str, np.ndarray] = {}

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

    def _get_word_vector(self, word: str) -> np.ndarray:
        word = word.lower().strip(",.!?\"'()[]{}")
        if not word:
            return np.zeros(self.dimension)
        if word not in self._word_vectors:
            h = hashlib.md5(word.encode('utf-8')).digest()
            seed = int.from_bytes(h[:4], byteorder='big')
            rng = np.random.default_rng(seed)
            # Uniform positive vectors so similarity is always positive in [0, 1]
            vec = rng.uniform(0.1, 1.0, size=self.dimension)
        if word in self._word_vectors:
            self._word_vectors[word] = self._word_vectors.pop(word)
            return self._word_vectors[word]
        
        h = hashlib.md5(word.encode('utf-8')).digest()
        seed = int.from_bytes(h[:4], byteorder='big')
        rng = np.random.default_rng(seed)
        # Uniform positive vectors so similarity is always positive in [0, 1]
        vec = rng.uniform(0.1, 1.0, size=self.dimension)
        if len(self._word_vectors) >= 5000:
            oldest = next(iter(self._word_vectors))
            del self._word_vectors[oldest]
        self._word_vectors[word] = vec / np.linalg.norm(vec)
        return self._word_vectors[word]

    def encode(self, sentences, batch_size=32, **_kwargs):
        if isinstance(sentences, str):
            return self._encode_single(sentences)
        else:
            return np.array([self._encode_single(s) for s in sentences])

    def _encode_single(self, text: str) -> np.ndarray:
        words = text.split()
        if not words:
            return np.zeros(self.dimension)
        vectors = [self._get_word_vector(w) for w in words]
        sum_vec = np.sum(vectors, axis=0)
        norm = np.linalg.norm(sum_vec)
        if norm > 0:
            sum_vec = sum_vec / norm
        from typing import cast
        return cast(np.ndarray, sum_vec)


# ─── EmbeddingManager ─────────────────────────────────────────────────────────

class EmbeddingManager:
    """
    Singleton-style embedding manager.
    Loads the model once and provides a clean interface for the rest of Jarvis.

    Usage:
        em = EmbeddingManager()
        em.initialize()

        vec = em.embed("I like coffee")
        vecs = em.embed_batch(["coffee", "tea", "water"])
        score = em.similarity("I like coffee", "My favorite drink is coffee")
    """

    def __init__(self, model_name: str = core_memory_embeddings_DEFAULT_MODEL):
        self.model_name  = model_name
        self._model: Optional["SentenceTransformer"] = None
        self._initialized = False
        self._embed_count = 0
        self._cache: dict[str, list[float]] = {}
        self._cache_lock = threading.Lock()

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def initialize(self, warm_up: bool = True) -> bool:
        """
        Load the model into memory. Safe to call multiple times.
        Returns True on success.
        """
        if self._initialized:
            return True

        import os
        if os.environ.get("JARVIS_MOCK_EMBEDDINGS") == "1":
            logger.info("JARVIS_MOCK_EMBEDDINGS is enabled. Initializing deterministic mock embedding model.")
            self._model = DeterministicMockSentenceTransformer()  # type: ignore[assignment]
            self._initialized = True
            return True

        try:
            logger.info(f"Loading sentence-transformer model: {self.model_name}")

            def _load_model():
                from sentence_transformers import SentenceTransformer
                return SentenceTransformer(self.model_name)

            self._model = await asyncio.to_thread(_load_model)

            model = self._model
            if model is None:
                raise RuntimeError("Failed to load model")

            if warm_up:
                logger.debug("Warming up embedding model...")
                _ = await asyncio.to_thread(model.encode, WARM_UP_TEXT, normalize_embeddings=True)

            self._initialized = True
            dim = model.get_sentence_embedding_dimension()
            logger.info(f"Embedding model ready. Dimension: {dim}")
            return True

        except Exception as e:
            logger.warning(
                f"Failed to load embedding model '{self.model_name}' ({e}). "
                "Falling back to local deterministic mock embedding model."
            )
            self._model = DeterministicMockSentenceTransformer()  # type: ignore[assignment]
            self._initialized = True
            return True

    def is_ready(self) -> bool:
        return self._initialized

    # ── Core Embedding ─────────────────────────────────────────────────────────

    async def embed(self, text: str, use_cache: bool = True) -> list[float]:
        """
        Generate a normalized embedding for a single text string.
        Caches results to avoid re-embedding identical inputs.

        Args:
            text:      Input text to embed.
            use_cache: Whether to use the in-memory LRU cache.

        Returns:
            List of floats (normalized embedding vector).
        """
        if not self._initialized:
            await self.initialize()

        # Cache key: MD5 hash of (model + text) for safety
        cache_key = hashlib.md5(f"{self.model_name}::{text}".encode(), usedforsecurity=False).hexdigest()

        with self._cache_lock:
            if use_cache and cache_key in self._cache:
                val = self._cache.pop(cache_key)
                self._cache[cache_key] = val
                return val

        model = self._model
        if model is None:
            raise RuntimeError("EmbeddingManager not initialized properly (model is None).")

        vector_array = await asyncio.to_thread(model.encode, text, normalize_embeddings=True)
        from typing import cast
        vector = cast(list[float], vector_array.tolist())
        self._embed_count += 1

        if use_cache:
            with self._cache_lock:
                # Evict oldest if cache is full
                while len(self._cache) >= CACHE_SIZE:
                    oldest = next(iter(self._cache))
                    del self._cache[oldest]
                self._cache[cache_key] = vector

        return vector

    async def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> list[list[float]]:
        """
        Embed a list of texts efficiently in batches.
        Returns a list of normalized embedding vectors.
        """
        if not self._initialized:
            await self.initialize()

        if not texts:
            return []

        model = self._model
        if model is None:
            raise RuntimeError("EmbeddingManager not initialized properly (model is None).")

        vectors = await asyncio.to_thread(
            model.encode,
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        self._embed_count += len(texts)
        from typing import cast
        return cast(list[list[float]], vectors.tolist())

    # ── Similarity ─────────────────────────────────────────────────────────────

    async def similarity(self, text_a: str, text_b: str) -> float:
        """
        Compute cosine similarity between two texts.
        Returns a float in [0, 1] — higher means more similar.

        Since we use normalized embeddings, cosine similarity = dot product.
        """
        vec_a = np.array(await self.embed(text_a))
        vec_b = np.array(await self.embed(text_b))
        return float(np.dot(vec_a, vec_b))

    async def similarity_batch(
        self,
        query: str,
        candidates: list[str],
    ) -> list[tuple[str, float]]:
        """
        Compare one query against many candidates.
        Returns list of (text, score) tuples sorted by score descending.
        """
        if not candidates:
            return []

        query_vec    = np.array(await self.embed(query))
        candidate_vecs = np.array(await self.embed_batch(candidates))

        # Dot products (all normalized → cosine similarity)
        scores = (candidate_vecs @ query_vec).tolist()

        pairs = list(zip(candidates, scores))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs

    async def rank_memories(
        self,
        query: str,
        memory_texts: list[str],
        top_k: int = 5,
        threshold: float = 0.30,
    ) -> list[dict]:
        """
        Rank a list of memory strings by relevance to a query.
        Returns top_k results above the threshold.

        Each result: {"text": str, "score": float, "rank": int}
        """
        pairs = await self.similarity_batch(query, memory_texts)
        results = []
        for rank, (text, score) in enumerate(pairs[:top_k], start=1):
            if score >= threshold:
                results.append({
                    "text":  text,
                    "score": round(score, 4),
                    "rank":  rank,
                })
        return results

    # ── Dimension & Model Info ─────────────────────────────────────────────────

    @property
    def dimension(self) -> Optional[int]:
        """Return the embedding dimension, or None if not initialized."""
        if self._model:
            return self._model.get_sentence_embedding_dimension()
        return None

    def info(self) -> dict:
        """Return model information and usage stats."""
        with self._cache_lock:
            cache_size = len(self._cache)
        return {
            "model":         self.model_name,
            "initialized":   self._initialized,
            "dimension":     self.dimension,
            "embed_count":   self._embed_count,
            "cache_size":    cache_size,
            "cache_capacity": CACHE_SIZE,
        }

    # ── Cache Management ───────────────────────────────────────────────────────

    def clear_cache(self):
        """Clear the embedding cache."""
        with self._cache_lock:
            self._cache.clear()
        logger.debug("Embedding cache cleared.")

    async def preload(self, texts: list[str]):
        """
        Pre-warm the cache with a list of texts.
        Useful on startup to pre-embed stored preferences.
        """
        if not self._initialized:
            logger.warning("Cannot preload: model not initialized.")
            return
        logger.info(f"Preloading {len(texts)} texts into embedding cache...")
        for text in texts:
            await self.embed(text, use_cache=True)
        logger.info("Preload complete.")


# ─── Module-level singleton ────────────────────────────────────────────────────
# Other modules can import this instance directly for convenience.

_default_manager: Optional[EmbeddingManager] = None


def get_embedding_manager(model_name: str = core_memory_embeddings_DEFAULT_MODEL) -> EmbeddingManager:
    """
    Get (or create) the module-level default EmbeddingManager.
    """
    global _default_manager
    if _default_manager is None or _default_manager.model_name != model_name:
        _default_manager = EmbeddingManager(model_name=model_name)
    return _default_manager






############################################################
# TOOLS
############################################################


# --- FILE: core/desktop/actions.py ---

"""Normalized desktop action execution with risk and audit metadata."""

# internal import removed: from __future__ import annotations

import inspect
import time
from typing import Any, Callable

# internal import removed: from core.autonomy.risk_evaluator import RiskEvaluator, core_autonomy_risk_evaluator_RiskLevel, RiskResult
# internal import removed: from core.desktop.contracts import (
# internal import removed:     DesktopAction,
# internal import removed:     DesktopActionResult,
# internal import removed:     DesktopActionStatus,
# internal import removed:     DesktopActionType,
# internal import removed:     DesktopRiskTier,
# internal import removed: )


ActionHandler = Callable[..., Any]

_SENSITIVE_TEXT_MARKERS = (
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "private key",
)


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _stringify(value: Any) -> str:
    if value in (None, "", {}, []):
        return ""
    return str(value)


def core_desktop_actions__normalize_tool_result(result: Any) -> tuple[bool, str, str, dict[str, Any]]:
    if isinstance(result, dict):
        success = bool(result.get("success", False))
        output = _stringify(
            result.get("output")
            or result.get("data")
            or result.get("metadata")
        )
        error = str(result.get("error", "") or "")
        from typing import cast
        metadata: dict[str, Any] = cast(dict, result.get("metadata")) if isinstance(result.get("metadata"), dict) else {}
        data = result.get("data") if isinstance(result.get("data"), dict) else {}
        return success, output, error, {"data": data, **metadata}

    success_attr = getattr(result, "success", None)
    if success_attr is None:
        return True, _stringify(result) or "Action completed successfully.", "", {}

    success = bool(success_attr)
    output = _stringify(
        getattr(result, "output", None)
        or getattr(result, "data", None)
        or getattr(result, "metadata", None)
    )
    error = str(getattr(result, "error", "") or "")
    metadata_raw = getattr(result, "metadata", None)
    data = getattr(result, "data", None)
    normalized_metadata: dict[str, Any] = {}
    if isinstance(data, dict):
        normalized_metadata["data"] = data
    if isinstance(metadata_raw, dict):
        normalized_metadata.update(metadata_raw)
    return success, output, error, normalized_metadata


class DesktopActionExecutor:
    """Execute every desktop operation through one action contract."""

    def __init__(
        self,
        *,
        risk_evaluator: RiskEvaluator | None = None,
        audit_writer: Callable[[str, dict[str, Any]], str] | None = None,
        action_handlers: dict[str | DesktopActionType, ActionHandler] | None = None,
    ) -> None:
        self.risk_evaluator = risk_evaluator or RiskEvaluator()
        if audit_writer is None:
            from core.logging.logger import audit

            audit_writer = audit
        self.audit_writer = audit_writer
        self.action_handlers = self._default_handlers()
        for key, handler in (action_handlers or {}).items():
            action_name = key.value if isinstance(key, DesktopActionType) else str(key)
            self.action_handlers[action_name] = handler

    def evaluate_risk(self, action: DesktopAction) -> tuple[DesktopRiskTier, RiskResult]:
        risk = self.risk_evaluator.evaluate([action.action_name])
        if risk.is_blocked:
            return DesktopRiskTier.BLOCKED, risk
        if risk.level >= core_autonomy_risk_evaluator_RiskLevel.HIGH:
            return DesktopRiskTier.HIGH, risk
        if risk.requires_confirmation:
            return DesktopRiskTier.CONFIRM, risk
        if risk.level >= core_autonomy_risk_evaluator_RiskLevel.MEDIUM:
            return DesktopRiskTier.MEDIUM, risk
        return DesktopRiskTier.LOW, risk

    def requires_approval(self, action: DesktopAction) -> bool:
        if action.requires_approval is not None:
            return action.requires_approval
        risk_tier, risk = self.evaluate_risk(action)
        return risk.requires_confirmation or risk_tier in {DesktopRiskTier.HIGH, DesktopRiskTier.BLOCKED}

    async def execute(
        self,
        action: DesktopAction,
        *,
        approved: bool | None = None,
    ) -> DesktopActionResult:
        started_at = time.time()
        risk_tier, risk = self.evaluate_risk(action)

        if self._contains_sensitive_text(action):
            result = self._result(
                action,
                started_at=started_at,
                risk_tier=DesktopRiskTier.BLOCKED,
                status=DesktopActionStatus.BLOCKED,
                success=False,
                error="Sensitive text entry is blocked by desktop policy.",
                metadata={"risk": risk.summary()},
            )
            return self._audit(action, result)

        if risk.is_blocked:
            result = self._result(
                action,
                started_at=started_at,
                risk_tier=DesktopRiskTier.BLOCKED,
                status=DesktopActionStatus.BLOCKED,
                success=False,
                error=risk.summary(),
                metadata={"blocking_actions": list(risk.blocking_actions)},
            )
            return self._audit(action, result)

        if self.requires_approval(action) and approved is not True:
            result = self._result(
                action,
                started_at=started_at,
                risk_tier=risk_tier,
                status=DesktopActionStatus.NEEDS_APPROVAL,
                success=False,
                error="Desktop action requires user approval.",
                metadata={"risk": risk.summary()},
            )
            return self._audit(action, result)

        handler = self.action_handlers.get(action.action_name)
        if handler is None:
            result = self._result(
                action,
                started_at=started_at,
                risk_tier=risk_tier,
                status=DesktopActionStatus.FAILURE,
                success=False,
                error=f"No desktop action handler registered for '{action.action_name}'.",
                metadata={"risk": risk.summary()},
            )
            return self._audit(action, result)

        try:
            raw_result = await _maybe_await(handler(**dict(action.params)))
            success, output, error, metadata = core_desktop_actions__normalize_tool_result(raw_result)
            result = self._result(
                action,
                started_at=started_at,
                risk_tier=risk_tier,
                status=DesktopActionStatus.SUCCESS if success else DesktopActionStatus.FAILURE,
                success=success,
                output=output,
                error=error,
                metadata={"risk": risk.summary(), **metadata},
            )
            return self._audit(action, result)
        except Exception as exc:  # noqa: BLE001
            result = self._result(
                action,
                started_at=started_at,
                risk_tier=risk_tier,
                status=DesktopActionStatus.FAILURE,
                success=False,
                error=str(exc),
                metadata={"risk": risk.summary()},
            )
            return self._audit(action, result)

    def _audit(self, action: DesktopAction, result: DesktopActionResult) -> DesktopActionResult:
        try:
            result.audit_hash = self.audit_writer(
                "desktop_action",
                {
                    "action": action.to_dict(),
                    "result": result.to_dict(),
                },
            )
        except Exception as exc:  # noqa: BLE001
            result.metadata["audit_error"] = str(exc)
        return result

    @staticmethod
    def _contains_sensitive_text(action: DesktopAction) -> bool:
        if action.action_name != DesktopActionType.TYPE_TEXT.value:
            return False
        text = str(action.params.get("text", "") or "").lower()
        return any(marker in text for marker in _SENSITIVE_TEXT_MARKERS)

    @staticmethod
    def _result(
        action: DesktopAction,
        *,
        started_at: float,
        risk_tier: DesktopRiskTier,
        status: DesktopActionStatus,
        success: bool,
        output: str = "",
        error: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> DesktopActionResult:
        return DesktopActionResult(
            action_id=action.action_id,
            action_type=action.action_name,
            success=success,
            status=status,
            output=output,
            error=error,
            risk_tier=risk_tier,
            started_at=started_at,
            ended_at=time.time(),
            metadata=dict(metadata or {}),
        )

    @staticmethod
    def _default_handlers() -> dict[str, ActionHandler]:
        return {
            DesktopActionType.LAUNCH_APP.value: _launch_application,
            DesktopActionType.FOCUS_WINDOW.value: _focus_window,
            DesktopActionType.MOVE_MOUSE.value: _move_mouse,
            DesktopActionType.CLICK.value: _click,
            DesktopActionType.DOUBLE_CLICK.value: _double_click,
            DesktopActionType.RIGHT_CLICK.value: _right_click,
            DesktopActionType.CLICK_TEXT_ON_SCREEN.value: _click_text_on_screen,
            DesktopActionType.CLICK_SCREEN_TARGET.value: _click_screen_target,
            DesktopActionType.DOUBLE_CLICK_SCREEN_TARGET.value: _double_click_screen_target,
            DesktopActionType.RIGHT_CLICK_SCREEN_TARGET.value: _right_click_screen_target,
            DesktopActionType.SCROLL.value: _scroll,
            DesktopActionType.DRAG.value: _drag,
            DesktopActionType.TYPE_TEXT.value: _type_text,
            DesktopActionType.PRESS_KEY.value: _press_key,
            DesktopActionType.HOTKEY.value: _hotkey,
            DesktopActionType.CLIPBOARD_GET.value: _clipboard_get,
            DesktopActionType.CLIPBOARD_SET.value: _clipboard_set,
            DesktopActionType.CLIPBOARD_PASTE.value: _clipboard_paste,
        }


async def _launch_application(
    target: str | None = None,
    args: list[str] | None = None,
    application: str | None = None,
) -> Any:
    from core.tools.system_automation import async_launch_application

    return await async_launch_application(target=target, args=args, application=application)


async def _click(x: int, y: int, button: str = "left") -> Any:
    from core.tools.gui_control import click

    return await click(x=x, y=y, button=button)


async def _double_click(x: int, y: int) -> Any:
    from core.tools.gui_control import double_click

    return await double_click(x=x, y=y)


async def _right_click(x: int, y: int) -> Any:
    from core.tools.gui_control import right_click

    return await right_click(x=x, y=y)


async def _click_text_on_screen(
    text: str,
    occurrence: int = 1,
    button: str = "left",
    match_mode: str = "contains",
) -> Any:
    from core.tools.gui_control import click_text_on_screen

    return await click_text_on_screen(
        text=text,
        occurrence=occurrence,
        button=button,
        match_mode=match_mode,
    )


async def _click_screen_target(
    target: str,
    occurrence: int = 1,
    button: str = "left",
    match_mode: str = "contains",
    min_confidence: float = 0.2,
) -> Any:
    from core.tools.gui_control import click_screen_target

    return await click_screen_target(
        target=target,
        occurrence=occurrence,
        button=button,
        match_mode=match_mode,
        min_confidence=min_confidence,
    )


async def _double_click_screen_target(
    target: str,
    occurrence: int = 1,
    match_mode: str = "contains",
    min_confidence: float = 0.2,
) -> Any:
    from core.tools.gui_control import double_click_screen_target

    return await double_click_screen_target(
        target=target,
        occurrence=occurrence,
        match_mode=match_mode,
        min_confidence=min_confidence,
    )


async def _right_click_screen_target(
    target: str,
    occurrence: int = 1,
    match_mode: str = "contains",
    min_confidence: float = 0.2,
) -> Any:
    from core.tools.gui_control import right_click_screen_target

    return await right_click_screen_target(
        target=target,
        occurrence=occurrence,
        match_mode=match_mode,
        min_confidence=min_confidence,
    )


async def _type_text(text: str, interval: float = 0.05) -> Any:
    from core.tools.gui_control import type_text

    return await type_text(text=text, interval=interval)


async def _press_key(key: str, presses: int = 1, interval: float = 0.05) -> Any:
    from core.tools.gui_control import press_key

    return await press_key(key=key, presses=presses, interval=interval)


async def _hotkey(keys: list[str] | tuple[str, ...] | str) -> Any:
    from core.tools.gui_control import hotkey

    if isinstance(keys, str):
        key_list = [part.strip() for part in keys.split("+") if part.strip()]
    else:
        key_list = [str(key) for key in keys]
    return await hotkey(*key_list)


async def _move_mouse(x: int, y: int, duration: float = 0.0) -> Any:
    from core.tools.gui_control import move_mouse
    return await move_mouse(x=x, y=y, duration=duration)


async def _scroll(clicks: int, x: int | None = None, y: int | None = None) -> Any:
    from core.tools.gui_control import scroll
    return await scroll(clicks=clicks, x=x, y=y)


async def _drag(
    start_x: int,
    start_y: int,
    end_x: int,
    end_y: int,
    duration: float = 0.2,
    button: str = "left",
) -> Any:
    from core.tools.gui_control import drag
    return await drag(
        start_x=start_x,
        start_y=start_y,
        end_x=end_x,
        end_y=end_y,
        duration=duration,
        button=button,
    )


async def _focus_window(title: str) -> Any:
    from core.tools.gui_control import focus_window
    return focus_window(title=title)


async def _clipboard_get() -> Any:
    from core.tools.gui_control import clipboard_get
    return clipboard_get()


async def _clipboard_set(text: str) -> Any:
    from core.tools.gui_control import clipboard_set
    return clipboard_set(text=text)


async def _clipboard_paste() -> Any:
    from core.tools.gui_control import clipboard_paste
    return await clipboard_paste()


__all__ = ["DesktopActionExecutor"]




# --- FILE: core/desktop/observation.py ---

"""Reusable screen observation and before/after change detection."""

# internal import removed: from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from typing import Any, Callable

# internal import removed: from core.desktop.contracts import DesktopChange, DesktopObservation, ScreenTarget


ObservationHandler = Callable[..., Any]


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _result_success(result: Any) -> bool:
    if isinstance(result, dict):
        return bool(result.get("success", False))
    return bool(getattr(result, "success", False))


def _result_error(result: Any) -> str:
    if isinstance(result, dict):
        return str(result.get("error", "") or "")
    return str(getattr(result, "error", "") or "")


def _result_payload(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        data = result.get("data")
        if isinstance(data, dict):
            return data
        metadata = result.get("metadata")
        if isinstance(metadata, dict):
            return metadata
        return {}

    data = getattr(result, "data", None)
    if isinstance(data, dict):
        return data
    metadata = getattr(result, "metadata", None)
    if isinstance(metadata, dict):
        return metadata
    return {}


def _hash_path(path_value: str) -> str:
    if not path_value:
        return ""
    try:
        path = Path(path_value)
        if not path.is_file():
            return ""
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except Exception:
        return ""


def _target_from_payload(payload: dict[str, Any]) -> ScreenTarget | None:
    try:
        label = str(payload.get("text") or payload.get("label") or "")
        x = int(payload.get("x", 0))
        y = int(payload.get("y", 0))
        width = int(payload.get("w", payload.get("width", 0)))
        height = int(payload.get("h", payload.get("height", 0)))
        confidence = float(payload.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        return None
    if not label and width <= 0 and height <= 0:
        return None
    return ScreenTarget(
        label=label,
        x=x,
        y=y,
        width=width,
        height=height,
        confidence=confidence,
        metadata={k: v for k, v in payload.items() if k not in {"text", "label", "x", "y", "w", "h", "width", "height", "confidence"}},
    )


class DesktopObserver:
    """Capture normalized evidence about the current desktop state."""

    def __init__(
        self,
        *,
        capture_screen: ObservationHandler | None = None,
        active_window: ObservationHandler | None = None,
        ocr: ObservationHandler | None = None,
    ) -> None:
        self._capture_screen = capture_screen
        self._active_window = active_window
        self._ocr = ocr

    async def observe(self, label: str = "") -> DesktopObservation:
        screenshot_path = ""
        screenshot_fingerprint = ""
        active_window: dict[str, Any] = {}
        ocr_text = ""
        targets: list[ScreenTarget] = []
        metadata: dict[str, Any] = {"label": label} if label else {}
        errors: list[str] = []
        confidence = 0.0

        capture = self._capture_screen or self._default_capture_screen
        try:
            result = await _maybe_await(capture())
            if _result_success(result):
                payload = _result_payload(result)
                screenshot_path = str(payload.get("path", "") or "")
                screenshot_fingerprint = str(
                    payload.get("fingerprint", "") or _hash_path(screenshot_path)
                )
                metadata["screenshot"] = {
                    key: value
                    for key, value in payload.items()
                    if key not in {"path", "fingerprint"}
                }
                confidence += 0.45
            else:
                errors.append(_result_error(result) or "screenshot unavailable")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"screenshot failed: {exc}")

        active = self._active_window or self._default_active_window
        try:
            result = await _maybe_await(active())
            if _result_success(result):
                active_window = _result_payload(result)
                confidence += 0.3
            else:
                errors.append(_result_error(result) or "active window unavailable")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"active window failed: {exc}")

        ocr = self._ocr or self._default_ocr
        try:
            result = await _maybe_await(ocr())
            if _result_success(result):
                payload = _result_payload(result)
                ocr_text = str(
                    payload.get("ocr_text")
                    or payload.get("description")
                    or payload.get("text")
                    or ""
                )
                raw_targets = payload.get("matches") or payload.get("lines") or []
                for match in raw_targets:
                    if isinstance(match, dict):
                        target = _target_from_payload(match)
                        if target is not None:
                            targets.append(target)
                if ocr_text or targets:
                    confidence += 0.2
            else:
                errors.append(_result_error(result) or "ocr unavailable")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"ocr failed: {exc}")

        confidence = min(1.0, confidence)
        low_confidence_reason = ""
        if confidence < 0.5:
            low_confidence_reason = "; ".join(error for error in errors if error) or "not enough desktop evidence"
        if errors:
            metadata["observation_errors"] = errors

        return DesktopObservation(
            screenshot_path=screenshot_path,
            screenshot_fingerprint=screenshot_fingerprint,
            active_window=active_window,
            ocr_text=ocr_text,
            targets=targets,
            confidence=confidence,
            low_confidence_reason=low_confidence_reason,
            metadata=metadata,
        )

    def compare(
        self,
        before: DesktopObservation | None,
        after: DesktopObservation | None,
    ) -> DesktopChange:
        if before is None or after is None:
            return DesktopChange(
                changed=False,
                confidence=0.0,
                summary="Missing before or after observation.",
                before_observation_id=getattr(before, "observation_id", ""),
                after_observation_id=getattr(after, "observation_id", ""),
            )

        changed_signals: list[str] = []
        metadata: dict[str, Any] = {}

        if before.screenshot_fingerprint and after.screenshot_fingerprint:
            metadata["screenshot_fingerprint_before"] = before.screenshot_fingerprint
            metadata["screenshot_fingerprint_after"] = after.screenshot_fingerprint
            if before.screenshot_fingerprint != after.screenshot_fingerprint:
                changed_signals.append("screenshot changed")

        before_title = str(before.active_window.get("title", "") or "")
        after_title = str(after.active_window.get("title", "") or "")
        if before_title or after_title:
            metadata["active_window_before"] = before_title
            metadata["active_window_after"] = after_title
            if before_title != after_title:
                changed_signals.append("active window changed")

        if before.ocr_text or after.ocr_text:
            metadata["ocr_before_length"] = len(before.ocr_text)
            metadata["ocr_after_length"] = len(after.ocr_text)
            if before.ocr_text != after.ocr_text:
                changed_signals.append("visible text changed")

        changed = bool(changed_signals)
        if changed:
            confidence = max(before.confidence, after.confidence, 0.7)
            summary = "; ".join(changed_signals)
        else:
            confidence = min(before.confidence, after.confidence)
            summary = "No observable desktop change detected."

        return DesktopChange(
            changed=changed,
            confidence=confidence,
            summary=summary,
            before_observation_id=before.observation_id,
            after_observation_id=after.observation_id,
            metadata=metadata,
        )

    @staticmethod
    def _default_capture_screen() -> Any:
        from core.tools.screen import capture_screen

        return capture_screen()

    @staticmethod
    def _default_active_window() -> Any:
        from core.tools.gui_control import get_active_window

        return get_active_window()

    @staticmethod
    def _default_ocr() -> Any:
        from core.tools.screen import read_screen_text

        return read_screen_text(include_lines=True)


__all__ = ["DesktopObserver"]




# --- FILE: core/desktop/mission.py ---

"""Planner-executor-recovery loop for bounded desktop missions."""

# internal import removed: from __future__ import annotations

import inspect
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable

# internal import removed: from core.desktop.actions import DesktopActionExecutor
# internal import removed: from core.desktop.contracts import (
# internal import removed:     ApprovalDecision,
# internal import removed:     DesktopAction,
# internal import removed:     DesktopActionResult,
# internal import removed:     DesktopActionStatus,
# internal import removed:     DesktopChange,
# internal import removed:     DesktopObservation,
# internal import removed: )
# internal import removed: from core.desktop.observation import DesktopObserver


class DesktopMissionStatus(str, Enum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    NEEDS_USER = "needs_user"
    STOPPED = "stopped"


class RecoveryDecision(str, Enum):
    NONE = "none"
    RETRY = "retry"
    REOBSERVE = "reobserve"
    ASK_USER = "ask_user"
    STOP = "stop"


@dataclass
class MissionStepRecord:
    step_id: str
    action: dict[str, Any]
    observation_before: dict[str, Any] | None = None
    approval: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    observation_after: dict[str, Any] | None = None
    change: dict[str, Any] | None = None
    recovery_decision: str = RecoveryDecision.NONE.value
    attempts: int = 0
    status: str = "pending"
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "action": dict(self.action),
            "observation_before": self.observation_before,
            "approval": self.approval,
            "result": self.result,
            "observation_after": self.observation_after,
            "change": self.change,
            "recovery_decision": self.recovery_decision,
            "attempts": self.attempts,
            "status": self.status,
            "error": self.error,
        }


@dataclass
class MissionExecutionRecord:
    goal: str
    plan: list[dict[str, Any]]
    mission_id: str = field(default_factory=lambda: f"mission_{uuid.uuid4().hex[:12]}")
    status: DesktopMissionStatus = DesktopMissionStatus.RUNNING
    steps: list[MissionStepRecord] = field(default_factory=list)
    final_summary: str = ""
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def close(self, status: DesktopMissionStatus, summary: str) -> None:
        self.status = status
        self.final_summary = summary
        self.ended_at = time.time()

    @property
    def duration_seconds(self) -> float:
        return max(0.0, (self.ended_at or time.time()) - self.started_at)

    def explain(self) -> str:
        if self.final_summary:
            return self.final_summary
        succeeded = sum(1 for step in self.steps if step.status == "succeeded")
        failed = [step for step in self.steps if step.status not in {"succeeded", "pending"}]
        if self.status == DesktopMissionStatus.SUCCEEDED:
            return f"Completed '{self.goal}' with {succeeded} step(s)."
        if failed:
            first = failed[0]
            return f"Stopped '{self.goal}' at step {first.step_id}: {first.error or first.status}."
        return f"Mission '{self.goal}' stopped before completion."

    def to_dict(self) -> dict[str, Any]:
        return {
            "mission_id": self.mission_id,
            "goal": self.goal,
            "plan": list(self.plan),
            "status": self.status.value,
            "steps": [step.to_dict() for step in self.steps],
            "final_summary": self.final_summary,
            "duration_seconds": round(self.duration_seconds, 3),
            "metadata": dict(self.metadata),
        }


class DesktopMissionExecutor:
    """Observe, act, verify, recover, and explain desktop missions."""

    def __init__(
        self,
        *,
        action_executor: DesktopActionExecutor | None = None,
        observer: DesktopObserver | None = None,
        approval_callback: Callable[[DesktopAction, str], Any] | None = None,
        audit_writer: Callable[[str, dict[str, Any]], str] | None = None,
        max_retries: int = 1,
        min_confidence: float = 0.35,
    ) -> None:
        self.action_executor = action_executor or DesktopActionExecutor()
        self.observer = observer or DesktopObserver()
        self.approval_callback = approval_callback
        if audit_writer is None:
            from core.logging.logger import audit

            audit_writer = audit
        self.audit_writer = audit_writer
        self.max_retries = max(0, int(max_retries))
        self.min_confidence = max(0.0, min(1.0, float(min_confidence)))

    async def run(
        self,
        *,
        goal: str,
        actions: Iterable[DesktopAction],
        plan_summary: str = "",
    ) -> MissionExecutionRecord:
        action_list = list(actions)
        record = MissionExecutionRecord(
            goal=goal,
            plan=[action.to_dict() for action in action_list],
            metadata={"plan_summary": plan_summary} if plan_summary else {},
        )
        self._audit("desktop_mission_started", record.to_dict())

        for action in action_list:
            step = MissionStepRecord(step_id=action.action_id, action=action.to_dict())
            record.steps.append(step)

            before = await self._observe_with_recovery(step, "before")
            if before.confidence < self.min_confidence and self.action_executor.requires_approval(action):
                decision = await self._approval(action, "Low-confidence desktop observation before action.")
                step.approval = decision.to_dict()
                if not decision.approved:
                    step.status = "needs_user"
                    step.recovery_decision = RecoveryDecision.ASK_USER.value
                    step.error = decision.reason or "User approval is required before low-confidence desktop action."
                    record.close(DesktopMissionStatus.NEEDS_USER, self._summary_for(record))
                    self._audit("desktop_mission_finished", record.to_dict())
                    return record

            approval = await self._approval_if_required(action)
            step.approval = approval.to_dict()
            if approval.required and not approval.approved:
                step.status = "needs_user"
                step.recovery_decision = RecoveryDecision.ASK_USER.value
                step.error = approval.reason or "User approval is required."
                record.close(DesktopMissionStatus.NEEDS_USER, self._summary_for(record))
                self._audit("desktop_mission_finished", record.to_dict())
                return record

            success = await self._execute_with_verification(step, action, before, approval)
            self._audit("desktop_mission_step", step.to_dict())
            if not success:
                record.close(DesktopMissionStatus.FAILED, self._summary_for(record))
                self._audit("desktop_mission_finished", record.to_dict())
                return record

        record.close(DesktopMissionStatus.SUCCEEDED, self._summary_for(record))
        self._audit("desktop_mission_finished", record.to_dict())
        return record

    async def _observe_with_recovery(self, step: MissionStepRecord, label: str) -> DesktopObservation:
        observation = await self.observer.observe(label)
        if label == "before":
            step.observation_before = observation.to_dict()
        else:
            step.observation_after = observation.to_dict()

        if observation.confidence >= self.min_confidence:
            return observation

        step.recovery_decision = RecoveryDecision.REOBSERVE.value
        second = await self.observer.observe(f"{label}_retry")
        if label == "before":
            step.observation_before = second.to_dict()
        else:
            step.observation_after = second.to_dict()
        return second

    async def _execute_with_verification(
        self,
        step: MissionStepRecord,
        action: DesktopAction,
        before: DesktopObservation,
        approval: ApprovalDecision,
    ) -> bool:
        current_before = before
        last_result: DesktopActionResult | None = None
        last_change: DesktopChange | None = None

        for attempt in range(1, self.max_retries + 2):
            step.attempts = attempt
            result = await self.action_executor.execute(action, approved=approval.approved if approval.required else None)
            after = await self._observe_with_recovery(step, "after")
            change = self.observer.compare(current_before, after)
            step.result = result.to_dict()
            step.change = change.to_dict()
            last_result = result
            last_change = change

            if result.status in {DesktopActionStatus.BLOCKED, DesktopActionStatus.NEEDS_APPROVAL}:
                step.status = result.status.value
                step.recovery_decision = RecoveryDecision.STOP.value
                step.error = result.error
                return False

            expected_change_missing = bool(action.expected_change) and not change.changed
            if result.success and not expected_change_missing:
                step.status = "succeeded"
                if step.recovery_decision == RecoveryDecision.REOBSERVE.value:
                    step.recovery_decision = RecoveryDecision.NONE.value
                return True

            if attempt <= self.max_retries:
                step.recovery_decision = RecoveryDecision.RETRY.value
                current_before = after
                continue

            step.status = "failed"
            if result.success and expected_change_missing:
                step.error = f"No-op detected: expected change '{action.expected_change}' was not observed."
            else:
                step.error = result.error or "Desktop action failed."
            return False

        step.status = "failed"
        if last_result is not None:
            step.result = last_result.to_dict()
        if last_change is not None:
            step.change = last_change.to_dict()
        step.error = step.error or "Desktop action did not complete."
        return False

    async def _approval_if_required(self, action: DesktopAction) -> ApprovalDecision:
        if not self.action_executor.requires_approval(action):
            return ApprovalDecision(required=False, approved=True, reason="Approval not required.")
        return await self._approval(action, "Desktop action requires approval.")

    async def _approval(self, action: DesktopAction, reason: str) -> ApprovalDecision:
        if self.approval_callback is None:
            return ApprovalDecision(
                required=True,
                approved=False,
                reason=reason,
                mode="user_required",
            )
        try:
            result = self.approval_callback(action, reason)
            if inspect.isawaitable(result):
                result = await result
            return ApprovalDecision(
                required=True,
                approved=bool(result),
                reason=reason,
                mode="callback",
            )
        except Exception as exc:  # noqa: BLE001
            return ApprovalDecision(
                required=True,
                approved=False,
                reason=f"Approval callback failed: {exc}",
                mode="callback",
            )

    def _summary_for(self, record: MissionExecutionRecord) -> str:
        succeeded = sum(1 for step in record.steps if step.status == "succeeded")
        total = len(record.steps)
        failed = [step for step in record.steps if step.status not in {"succeeded", "pending"}]

        if not failed and total:
            return f"Completed '{record.goal}' with {succeeded}/{total} desktop step(s) verified."
        if failed:
            first = failed[0]
            return (
                f"Paused '{record.goal}' after {succeeded}/{total} desktop step(s). "
                f"Step {first.step_id} status: {first.status}. {first.error}".strip()
            )
        return f"No desktop steps were executed for '{record.goal}'."

    def _audit(self, event_type: str, payload: dict[str, Any]) -> None:
        try:
            self.audit_writer(event_type, payload)
        except Exception:
            return


__all__ = [
    "DesktopMissionExecutor",
    "DesktopMissionStatus",
    "MissionExecutionRecord",
    "MissionStepRecord",
    "RecoveryDecision",
]




# --- FILE: core/tools/system_automation.py ---

"""
core/tools/system_automation.py
Jarvis V3 - System Automation Tools
All tools are synchronous internally; the dispatcher awaits them via asyncio.to_thread.
"""

import os
import subprocess
import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Tool Result Contract
# ─────────────────────────────────────────────

@dataclass
class core_tools_system_automation_ToolResult:
    success: bool
    output: str = ""
    error: str = ""
    metadata: dict = field(default_factory=dict)

    def to_reflection_payload(self) -> dict:
        """Normalised dict consumed by ReflectionEngine."""
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "metadata": self.metadata,
        }


# ─────────────────────────────────────────────
# Tool Registry  (name -> risk_score)
# ─────────────────────────────────────────────

TOOL_REGISTRY: dict[str, float] = {
    # read-only / informational
    "list_directory":    0.1,
    "read_file":         0.2,
    # state-changing / potentially destructive
    "launch_application": 0.6,
    "execute_shell":      0.7,
    "write_file":         0.8,
    "delete_file":        0.95,
}

SHELL_TIMEOUT = 10   # seconds


# ─────────────────────────────────────────────
# Tool Implementations
# ─────────────────────────────────────────────

def list_directory(path: str) -> core_tools_system_automation_ToolResult:
    """List contents of a directory."""
    try:
        from core.tools.path_utils import _assert_safe_path
        safe_path = _assert_safe_path(path, write_op=False)
        p = Path(safe_path)
        if not p.exists():
            return core_tools_system_automation_ToolResult(False, error=f"Path does not exist: {p}")
        if not p.is_dir():
            return core_tools_system_automation_ToolResult(False, error=f"Not a directory: {p}")
        entries = [
            {"name": e.name, "type": "dir" if e.is_dir() else "file", "size": e.stat().st_size if e.is_file() else None}
            for e in sorted(p.iterdir())
        ]
        lines = "\n".join(
            f"{'[DIR] ':7}{e['name']}" if e["type"] == "dir"
            else f"{'[FILE]':7}{e['name']}  ({e['size']} bytes)"
            for e in entries
        )
        return core_tools_system_automation_ToolResult(True, output=lines or "(empty directory)", metadata={"path": str(p), "count": len(entries)})
    except PermissionError as exc:
        return core_tools_system_automation_ToolResult(False, error=f"Permission denied: {exc}")
    except Exception as exc:
        return core_tools_system_automation_ToolResult(False, error=str(exc))


def read_file(path: str, max_bytes: int = 32_768) -> core_tools_system_automation_ToolResult:
    """Read a text file (capped at max_bytes to protect context window)."""
    try:
        from core.tools.path_utils import _assert_safe_path
        safe_path = _assert_safe_path(path, write_op=False)
        p = Path(safe_path)
        if not p.is_file():
            return core_tools_system_automation_ToolResult(False, error=f"File not found: {p}")
        content = p.read_bytes()[:max_bytes].decode("utf-8", errors="replace")
        truncated = len(p.read_bytes()) > max_bytes
        return core_tools_system_automation_ToolResult(
            True,
            output=content,
            metadata={"path": str(p), "truncated": truncated},
        )
    except PermissionError as exc:
        return core_tools_system_automation_ToolResult(False, error=f"Permission denied: {exc}")
    except Exception as exc:
        return core_tools_system_automation_ToolResult(False, error=str(exc))


def write_file(path: str, content: str, overwrite: bool = False) -> core_tools_system_automation_ToolResult:
    """Write text content to a file. HIGH RISK – requires confirmation."""
    try:
        from core.tools.path_utils import _assert_safe_path
        safe_path = _assert_safe_path(path, write_op=True)
        p = Path(safe_path)
        if p.exists() and not overwrite:
            return core_tools_system_automation_ToolResult(False, error=f"File exists and overwrite=False: {p}")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return core_tools_system_automation_ToolResult(True, output=f"Written {len(content)} chars to {p}", metadata={"path": str(p)})
    except PermissionError as exc:
        return core_tools_system_automation_ToolResult(False, error=f"Permission denied: {exc}")
    except Exception as exc:
        return core_tools_system_automation_ToolResult(False, error=str(exc))


def delete_file(path: str) -> core_tools_system_automation_ToolResult:
    """Delete a file. VERY HIGH RISK – requires confirmation."""
    try:
        from core.tools.path_utils import _assert_safe_path
        safe_path = _assert_safe_path(path, write_op=True)
        p = Path(safe_path)
        if not p.exists():
            return core_tools_system_automation_ToolResult(False, error=f"Path not found: {p}")
        if p.is_dir():
            return core_tools_system_automation_ToolResult(False, error="delete_file does not remove directories. Use a dedicated tool.")
        p.unlink()
        return core_tools_system_automation_ToolResult(True, output=f"Deleted: {p}", metadata={"path": str(p)})
    except PermissionError as exc:
        return core_tools_system_automation_ToolResult(False, error=f"Permission denied: {exc}")
    except Exception as exc:
        return core_tools_system_automation_ToolResult(False, error=str(exc))


def launch_application(
    target: str | None = None,
    args: list[str] | None = None,
    application: str | None = None,
) -> core_tools_system_automation_ToolResult:
    """
    Launch a desktop application or open a file with its default handler.
    Uses os.startfile on Windows; subprocess on other platforms.
    """
    actual_target = target or application
    if not actual_target:
        return core_tools_system_automation_ToolResult(False, error="Either 'target' or 'application' must be provided to launch_application")
    args = args or []

    # Enforce path sandboxing on the target and arguments if they are paths
    if actual_target:
        if "/" in actual_target or "\\" in actual_target or ":" in actual_target or ".." in actual_target:
            try:
                from core.tools.path_utils import _assert_safe_path
                _assert_safe_path(actual_target, write_op=False)
            except Exception as e:
                return core_tools_system_automation_ToolResult(False, error=f"Target path escapes sandbox: {e}")

    if args:
        for arg in args:
            if "/" in arg or "\\" in arg or ":" in arg or ".." in arg:
                if arg.lower().startswith(("http://", "https://")):
                    continue
                try:
                    from core.tools.path_utils import _assert_safe_path
                    _assert_safe_path(arg, write_op=False)
                except Exception as e:
                    return core_tools_system_automation_ToolResult(False, error=f"Argument path escapes sandbox: {e}")

    try:
        if os.name == "nt":
            # os.startfile does not accept extra args, fall through to subprocess for those
            if args:
                subprocess.Popen([actual_target, *args], shell=False)
                return core_tools_system_automation_ToolResult(True, output=f"Launched: {actual_target} {' '.join(args)}")
            os.startfile(actual_target)
            return core_tools_system_automation_ToolResult(True, output=f"Opened: {actual_target}")
        else:
            cmd = ["xdg-open", actual_target] if not args else [actual_target, *args]
            subprocess.Popen(cmd)
            return core_tools_system_automation_ToolResult(True, output=f"Launched: {' '.join(cmd)}")
    except FileNotFoundError:
        return core_tools_system_automation_ToolResult(False, error=f"Executable/file not found: {actual_target}")
    except Exception as exc:
        return core_tools_system_automation_ToolResult(False, error=str(exc))


async def execute_shell(command: str, working_dir: str | None = None) -> core_tools_system_automation_ToolResult:
    """
    Execute a shell command and capture stdout/stderr asynchronously.
    Hard timeout of SHELL_TIMEOUT seconds – never blocks the event loop.
    Uses shell=False (shlex split) to satisfy security policy (no B602).
    """
    import shlex
    try:
        if working_dir:
            from core.tools.path_utils import _assert_safe_path
            try:
                _assert_safe_path(working_dir, write_op=False)
            except Exception as exc:
                return core_tools_system_automation_ToolResult(False, error=f"Working directory escapes sandbox: {exc}")
        cwd = Path(working_dir).expanduser().resolve() if working_dir else None
        # Split command string into a token list — avoids shell=True (B602).
        try:
            cmd_tokens = shlex.split(command, posix=(os.name != "nt"))
        except ValueError:
            return core_tools_system_automation_ToolResult(False, error=f"Invalid shell command syntax: {command}")

        if not cmd_tokens:
            return core_tools_system_automation_ToolResult(False, error="Empty command.")

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd_tokens,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
        except FileNotFoundError:
            return core_tools_system_automation_ToolResult(False, error=f"Executable not found in command: {command}")

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=SHELL_TIMEOUT,
            )
            stdout = stdout_bytes.decode(errors="replace").strip()
            stderr = stderr_bytes.decode(errors="replace").strip()
            returncode = proc.returncode
        except asyncio.TimeoutError:
            try:
                proc.kill()
                await proc.wait()
            except OSError:
                pass
            return core_tools_system_automation_ToolResult(False, error=f"Command timed out after {SHELL_TIMEOUT}s: {command}")

        success = returncode == 0
        return core_tools_system_automation_ToolResult(
            success,
            output=stdout,
            error=stderr,
            metadata={"returncode": returncode, "command": command},
        )
    except Exception as exc:
        return core_tools_system_automation_ToolResult(False, error=str(exc))


# ─────────────────────────────────────────────
# Async wrappers (all blocking calls go to thread pool)
# ─────────────────────────────────────────────

async def async_list_directory(path: str) -> core_tools_system_automation_ToolResult:
    return await asyncio.to_thread(list_directory, path)

async def async_read_file(path: str) -> core_tools_system_automation_ToolResult:
    return await asyncio.to_thread(read_file, path)

async def async_write_file(path: str, content: str, overwrite: bool = False) -> core_tools_system_automation_ToolResult:
    return await asyncio.to_thread(write_file, path, content, overwrite)

async def async_delete_file(path: str) -> core_tools_system_automation_ToolResult:
    return await asyncio.to_thread(delete_file, path)

async def async_launch_application(
    target: str | None = None,
    args: list[str] | None = None,
    application: str | None = None,
) -> core_tools_system_automation_ToolResult:
    return await asyncio.to_thread(launch_application, target=target, args=args, application=application)

async def async_execute_shell(command: str, working_dir: str | None = None) -> core_tools_system_automation_ToolResult:
    return await execute_shell(command, working_dir)





# --- FILE: core/desktop/shortcuts.py ---

"""Simple, safe desktop shortcuts for common open/search commands."""

# internal import removed: from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote_plus

# internal import removed: from core.desktop.actions import DesktopActionExecutor
# internal import removed: from core.desktop.contracts import (
# internal import removed:     DesktopAction,
# internal import removed:     DesktopActionType,
# internal import removed: )
# internal import removed: from core.desktop.mission import (
# internal import removed:     DesktopMissionExecutor,
# internal import removed:     DesktopMissionStatus,
# internal import removed: )
# internal import removed: from core.desktop.observation import DesktopObserver
# internal import removed: from core.tools.system_automation import async_launch_application

INTERACTIVE_VERBS_PATTERN = re.compile(
    r"\b("
    # English
    r"write|type|fill|enter|input|select|click|double-?click|right-?click|middle-?click|press|hotkey|key|drag|drop|scroll|typewrite|keystroke|tap|check|tick|focus|copy|paste|cut|clipboard"
    # Spanish
    r"|escribir|teclear|rellenar|introducir|pulsar|presionar|clic|pinchar|seleccionar|arrastrar|pegar|copiar"
    # French
    r"|écrire|ecrire|taper|saisir|remplir|cliquer|appuyer|presser|sélectionner|selectionner|glisser|coller|copier"
    # German
    r"|schreiben|tippen|eingeben|ausfüllen|ausfullen|klicken|drücken|drucken|auswählen|auswahlen|ziehen|einfügen|einfugen|kopieren"
    r")\b",
    re.IGNORECASE
)


core_desktop_shortcuts_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass(frozen=True)
class DesktopCommandPlan:
    app_label: str
    primary_target: str
    primary_args: list[str] | None = None
    response_text: str = ""


def _supported_apps_message() -> str:
    return (
        "I can open: Microsoft Edge, Visual Studio Code, Notepad, Calculator, or the Jarvis project folder."
    )


def _extract_search_query(lowered: str, original: str) -> str:
    for marker in ("search ", "find "):
        idx = lowered.find(marker)
        if idx == -1:
            continue
        query = original[idx + len(marker):].strip()
        for suffix in (" in that browser", " on that browser", " in the browser"):
            if query.lower().endswith(suffix):
                query = query[: -len(suffix)].strip()
        return query.strip(" .")
    return ""


def plan_desktop_command(user_input: str) -> DesktopCommandPlan | None:
    text = str(user_input or "").strip()
    lowered = text.lower()

    if not text:
        return None

    # If the user request involves keyboard/mouse interactions, bypass the simple launcher
    # and let the full agentic loop plan the execution.
    if INTERACTIVE_VERBS_PATTERN.search(lowered):
        return None

    if "project folder" in lowered and "jarvis" in lowered:
        return DesktopCommandPlan(
            app_label="Jarvis project folder",
            primary_target="explorer.exe",
            primary_args=[str(core_desktop_shortcuts_PROJECT_ROOT)],
            response_text="Opened the Jarvis project folder.",
        )

    if "any app" in lowered and ("open" in lowered or "access" in lowered):
        return DesktopCommandPlan(
            app_label="Notepad",
            primary_target="notepad.exe",
            primary_args=None,
            response_text="Opened Notepad.",
        )

    if "edge" in lowered and any(token in lowered for token in ("search ", "find ")):
        query = _extract_search_query(lowered, text)
        if query:
            return DesktopCommandPlan(
                app_label="Microsoft Edge",
                primary_target="msedge.exe",
                primary_args=[f"https://www.bing.com/search?q={quote_plus(query)}"],
                response_text=f'Opened Microsoft Edge and searched for "{query}".',
            )

    if any(alias in lowered for alias in ("edge", "microst edge", "microsoft edge")) and (
        "open" in lowered
        or "go to" in lowered
        or "access" in lowered
        or "acces" in lowered
        or "launch" in lowered
    ):
        return DesktopCommandPlan(
            app_label="Microsoft Edge",
            primary_target="msedge.exe",
            primary_args=None,
            response_text="Opened Microsoft Edge.",
        )

    if ("notepad" in lowered or "note pad" in lowered) and ("open" in lowered or "launch" in lowered):
        return DesktopCommandPlan(
            app_label="Notepad",
            primary_target="notepad.exe",
            primary_args=None,
            response_text="Opened Notepad.",
        )

    if any(alias in lowered for alias in ("vscode", "vs code", "visual studio code")) and ("open" in lowered or "launch" in lowered):
        return DesktopCommandPlan(
            app_label="Visual Studio Code",
            primary_target="code",
            primary_args=[str(core_desktop_shortcuts_PROJECT_ROOT)],
            response_text="Opened Visual Studio Code.",
        )

    if ("calculator" in lowered or "calc" in lowered) and ("open" in lowered or "launch" in lowered):
        return DesktopCommandPlan(
            app_label="Calculator",
            primary_target="calc.exe",
            primary_args=None,
            response_text="Opened Calculator.",
        )

    if "open " in lowered or "go to " in lowered or "access " in lowered or "acces " in lowered or "launch " in lowered:
        return DesktopCommandPlan(
            app_label="Unsupported app",
            primary_target="",
            primary_args=None,
            response_text=_supported_apps_message(),
        )

    return None


async def handle_desktop_command(user_input: str) -> str | None:
    plan = plan_desktop_command(user_input)
    if plan is None:
        return None
    if not plan.primary_target:
        return plan.response_text

    async def _launch_handler(target: str, args: list[str] | None = None):
        return await async_launch_application(target, args)

    action = DesktopAction(
        action_type=DesktopActionType.LAUNCH_APP,
        description=f"Open {plan.app_label}",
        params={"target": plan.primary_target, "args": plan.primary_args},
        requires_approval=False,
        metadata={"source": "desktop_shortcut", "app_label": plan.app_label},
    )
    action_executor = DesktopActionExecutor(
        action_handlers={DesktopActionType.LAUNCH_APP: _launch_handler}
    )
    mission_executor = DesktopMissionExecutor(
        action_executor=action_executor,
        observer=DesktopObserver(),
        max_retries=0,
        min_confidence=0.0,
    )
    record = await mission_executor.run(
        goal=user_input,
        actions=[action],
        plan_summary=plan.response_text,
    )
    if record.status == DesktopMissionStatus.SUCCEEDED:
        return plan.response_text

    step = record.steps[-1] if record.steps else None
    result = step.result if step else {}
    error = (
        str(result.get("error", "") or "")
        if isinstance(result, dict)
        else ""
    )
    if not error and step is not None:
        error = step.error
    error = error or "Unknown launch failure."
    return f"I couldn't open {plan.app_label}: {error}"




# --- FILE: core/llm/defaults.py ---

"""Centralized LLM defaults for Project Jarvis."""

core_llm_defaults_DEFAULT_MODEL = "deepseek-r1:8b"




# --- FILE: core/llm/model_spec.py ---

"""Model specification registry for adaptive routing.

Each model known to Jarvis is described by a ``ModelSpec`` — a frozen record
of its cost, capabilities and constraints.  ``ModelRegistry`` aggregates
specs and exposes query helpers used by the adaptive ``ModelRouter``.
"""

# internal import removed: from __future__ import annotations

import configparser
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    """Immutable descriptor for a single LLM model."""

    name: str                       # e.g. "mistral:7b", "gemini-2.5-flash"
    provider: str                   # "ollama", "gemini", "groq", "openai", "anthropic"
    tier: int                       # 1 (lightweight) → 3 (heavy reasoning)
    weight: float                   # relative cost weight (0.01 = cheapest local, 1.0 = most expensive cloud)
    max_context_tokens: int = 4096  # context window size
    supports_tools: bool = False    # structured function/tool calling
    supports_vision: bool = False   # multimodal image input
    latency_class: str = "fast"     # "instant" (<200ms), "fast" (<1s), "standard" (<5s), "slow" (>5s)
    reasoning_capability: float = 0.3  # 0.0–1.0 estimated reasoning quality


# ── Built-in model specs ──────────────────────────────────────────────────

_BUILTIN_SPECS: list[ModelSpec] = [
    # ── Ollama Tier 1 — Reflexive ──────────────────────────────────────
    ModelSpec("qwen2.5:0.5b",    "ollama", 1, 0.01, 4096,  False, False, "instant",  0.15),
    ModelSpec("llama3.2:1b",     "ollama", 1, 0.02, 8192,  False, False, "instant",  0.20),
    ModelSpec("qwen2.5:1.5b",    "ollama", 1, 0.02, 4096,  False, False, "instant",  0.20),
    ModelSpec("gemma3:1b",       "ollama", 1, 0.02, 8192,  False, False, "instant",  0.18),
    ModelSpec("gemma2:2b",       "ollama", 1, 0.03, 8192,  False, False, "instant",  0.22),

    # ── Ollama Tier 2 — Execution ──────────────────────────────────────
    ModelSpec("mistral:7b",      "ollama", 2, 0.10, 32768, True,  False, "fast",     0.55),
    ModelSpec("llama3:8b",       "ollama", 2, 0.10, 8192,  True,  False, "fast",     0.55),
    ModelSpec("qwen2.5:7b",      "ollama", 2, 0.10, 32768, True,  False, "fast",     0.55),

    # ── Ollama Tier 3 — Reasoning ──────────────────────────────────────
    ModelSpec("deepseek-r1:8b",  "ollama", 3, 0.15, 32768, True,  False, "standard", 0.75),
    ModelSpec("deepseek-r1:14b", "ollama", 3, 0.25, 32768, True,  False, "standard", 0.82),
    ModelSpec("llama3.3:70b",    "ollama", 3, 0.50, 8192,  True,  False, "slow",     0.88),

    # ── Ollama Vision ──────────────────────────────────────────────────
    ModelSpec("llava",           "ollama", 2, 0.12, 4096,  False, True,  "standard", 0.40),
    ModelSpec("llava:latest",    "ollama", 2, 0.12, 4096,  False, True,  "standard", 0.40),

    # ── Cloud Tier 1 ───────────────────────────────────────────────────
    ModelSpec("gemini-2.0-flash-lite", "gemini",    1, 0.05, 1048576, True,  True,  "fast",     0.45),
    ModelSpec("llama-3.1-8b-instant",  "groq",      1, 0.04, 131072,  True,  False, "instant",  0.40),
    ModelSpec("gpt-4o-mini",           "openai",    1, 0.08, 128000,  True,  True,  "fast",     0.55),
    ModelSpec("claude-3-haiku-20240307","anthropic", 1, 0.06, 200000,  True,  False, "fast",     0.45),

    # ── Cloud Tier 2 ───────────────────────────────────────────────────
    ModelSpec("gemini-2.5-flash",          "gemini",    2, 0.20, 1048576, True,  True,  "fast",     0.80),
    ModelSpec("llama-3.3-70b-versatile",   "groq",      2, 0.15, 131072,  True,  False, "fast",     0.70),
    ModelSpec("gpt-4o",                    "openai",    2, 0.50, 128000,  True,  True,  "standard", 0.82),
    ModelSpec("claude-3-5-sonnet-20241022","anthropic", 2, 0.45, 200000,  True,  False, "standard", 0.85),

    # ── Cloud Tier 3 ───────────────────────────────────────────────────
    ModelSpec("gemini-2.5-pro",                 "gemini",    3, 0.60, 1048576, True,  True,  "standard", 0.92),
    ModelSpec("deepseek-r1-distill-llama-70b",  "groq",      3, 0.30, 131072,  True,  False, "standard", 0.80),
    ModelSpec("o3-mini",                        "openai",    3, 0.70, 128000,  True,  False, "standard", 0.90),
    ModelSpec("claude-sonnet-4-20250514",       "anthropic", 3, 0.80, 200000,  True,  False, "standard", 0.93),
]


@dataclass(frozen=True)
class RoutingDecision:
    """Result of an adaptive routing decision — carries the chosen model + rationale."""

    model: str
    provider: str
    tier: int
    reason: str
    weight: float = 0.0
    fallback_model: str | None = None
    fallback_provider: str | None = None


class ModelRegistry:
    """Queryable catalog of all known model specs.

    Loads built-in defaults, then overlays any ``[models.registry]`` section
    from the Jarvis config file.
    """

    def __init__(self, config: configparser.ConfigParser | None = None) -> None:
        self._specs: dict[str, ModelSpec] = {}
        for spec in _BUILTIN_SPECS:
            self._specs[spec.name] = spec

        # Overlay config overrides
        if config is not None:
            self._load_config_overrides(config)

    # ── Public queries ─────────────────────────────────────────────────

    def get(self, model_name: str) -> ModelSpec | None:
        """Look up a spec by exact name, or by family prefix."""
        spec = self._specs.get(model_name)
        if spec is not None:
            return spec
        # Try family match (e.g. "mistral" → "mistral:7b")
        family = model_name.split(":")[0]
        for name, s in self._specs.items():
            if name.split(":")[0] == family:
                return s
        return None

    def get_tier(self, model_name: str) -> int:
        """Return the tier for a model, defaulting to 2."""
        spec = self.get(model_name)
        return spec.tier if spec else 2

    def get_weight(self, model_name: str) -> float:
        """Return the cost weight for a model, defaulting to 0.5."""
        spec = self.get(model_name)
        return spec.weight if spec else 0.5

    def all_specs(self) -> list[ModelSpec]:
        return list(self._specs.values())

    def by_provider(self, provider: str) -> list[ModelSpec]:
        return [s for s in self._specs.values() if s.provider == provider]

    def by_tier(self, tier: int) -> list[ModelSpec]:
        return sorted(
            [s for s in self._specs.values() if s.tier == tier],
            key=lambda s: s.weight,
        )

    def get_cheapest_capable(
        self,
        *,
        min_reasoning: float = 0.0,
        min_context: int = 0,
        needs_tools: bool = False,
        needs_vision: bool = False,
        available_models: set[str] | None = None,
        providers: set[str] | None = None,
    ) -> list[ModelSpec]:
        """Return models meeting the requirements, sorted cheapest-first.

        Parameters
        ----------
        available_models:
            If provided, restrict to these exact model names (for Ollama
            availability filtering).  Cloud models are included regardless
            unless *providers* is set.
        providers:
            If provided, restrict to these providers.
        """
        candidates: list[ModelSpec] = []
        for spec in self._specs.values():
            if spec.reasoning_capability < min_reasoning:
                continue
            if spec.max_context_tokens < min_context:
                continue
            if needs_tools and not spec.supports_tools:
                continue
            if needs_vision and not spec.supports_vision:
                continue
            if providers and spec.provider not in providers:
                continue
            # For ollama models, check availability
            if spec.provider == "ollama" and available_models is not None:
                if spec.name not in available_models:
                    continue
            candidates.append(spec)

        return sorted(candidates, key=lambda s: (s.weight, -s.reasoning_capability))

    def register(self, spec: ModelSpec) -> None:
        """Add or replace a model spec."""
        self._specs[spec.name] = spec

    # ── Config overlay ─────────────────────────────────────────────────

    def _load_config_overrides(self, config: configparser.ConfigParser) -> None:
        """Parse ``[models.registry]`` entries like ``mistral:7b = tier=2,weight=0.10``."""
        section = "models.registry"
        if not config.has_section(section):
            return
        for model_name in config.options(section):
            raw = config.get(section, model_name, fallback="")
            if not raw.strip():
                continue
            try:
                overrides = self._parse_override(raw)
                existing = self._specs.get(model_name)
                if existing:
                    # Merge overrides into existing spec via replace
                    kwargs = {f.name: getattr(existing, f.name) for f in existing.__dataclass_fields__.values()}
                    kwargs.update(overrides)
                    self._specs[model_name] = ModelSpec(**kwargs)
                else:
                    # Require at least provider for new specs
                    if "provider" not in overrides:
                        logger.warning("Skipping model %s: no provider specified", model_name)
                        continue
                    overrides.setdefault("name", model_name)
                    overrides.setdefault("tier", 2)
                    overrides.setdefault("weight", 0.5)
                    self._specs[model_name] = ModelSpec(**overrides)
            except Exception as exc:
                logger.warning("Failed to parse model registry override for %s: %s", model_name, exc)

    @staticmethod
    def _parse_override(raw: str) -> dict[str, Any]:
        """Parse ``tier=2,weight=0.10,supports_tools=true`` into a dict."""
        result: dict[str, Any] = {}
        for pair in raw.split(","):
            pair = pair.strip()
            if "=" not in pair:
                continue
            key, value = pair.split("=", 1)
            key = key.strip()
            value = value.strip()
            # Type coercion
            if key in ("tier", "max_context_tokens"):
                result[key] = int(value)
            elif key in ("weight", "reasoning_capability"):
                result[key] = float(value)
            elif key in ("supports_tools", "supports_vision"):
                result[key] = value.lower() in ("true", "1", "yes")
            else:
                result[key] = value
        return result


__all__ = ["ModelSpec", "ModelRegistry", "RoutingDecision"]




# --- FILE: core/llm/ollama_client.py ---

"""Pure async HTTP client for local Ollama inference.

Talks to Ollama's /api/generate endpoint.  No cloud fallback,
no memory injection, no profile — just HTTP in, text out.
"""

# internal import removed: from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any
from urllib.request import urlopen

import aiohttp

# internal import removed: from core.config.defaults import OLLAMA_BASE_URL
# internal import removed: from core.llm.defaults import core_llm_defaults_DEFAULT_MODEL

logger = logging.getLogger(__name__)

TIMEOUT_S = 120


def _normalize_base_url(base_url: str) -> str:
    return str(base_url or OLLAMA_BASE_URL).rstrip("/")


def _strip_think(text: str) -> str:
    """Remove <think>…</think> blocks emitted by DeepSeek R1."""
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


def extract_model_names(payload: dict[str, Any] | None) -> list[str]:
    models = payload.get("models", []) if isinstance(payload, dict) else []
    discovered: list[str] = []
    for item in models:
        if not isinstance(item, dict):
            continue
        for key in ("name", "model"):
            value = str(item.get(key, "") or "").strip()
            if value and value not in discovered:
                discovered.append(value)
    return discovered


def list_models_sync(base_url: str = OLLAMA_BASE_URL, timeout_s: float = 3.0) -> list[str]:
    normalized = _normalize_base_url(base_url)
    with urlopen(f"{normalized}/api/tags", timeout=timeout_s) as response:
        payload = json.loads(response.read().decode("utf-8", errors="replace"))
    return extract_model_names(payload)


class OllamaTransientError(aiohttp.ClientError):
    """Raised when Ollama returns a transient HTTP error status."""
    pass


class OllamaClient:
    """Lightweight async client for a local Ollama instance.

    Usage::

        client = OllamaClient()
        reply = await client.complete("say hello", model="mistral:7b")
    """

    def __init__(self, base_url: str = OLLAMA_BASE_URL) -> None:
        self.base_url = _normalize_base_url(base_url)

    async def complete(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.1,
        *,
        model: str = core_llm_defaults_DEFAULT_MODEL,
        keep_think: bool = False,
    ) -> str:
        """Send a prompt to Ollama and return the response text.

        Retries up to 3 times on transient connection errors.
        Raises on timeout, connection refused, or empty response.
        """
        payload: dict = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "top_p": 0.9},
        }
        if system:
            payload["system"] = system

        last_exc: Exception | None = None

        for attempt in range(3):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.base_url}/api/generate",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=TIMEOUT_S, sock_connect=2.0),
                    ) as response:
                        if response.status != 200:
                            if response.status in {400, 401, 403, 404, 422}:
                                raise RuntimeError(
                                    f"Ollama HTTP {response.status} on attempt {attempt + 1}"
                                )
                            else:
                                raise OllamaTransientError(
                                    f"Ollama HTTP {response.status} on attempt {attempt + 1}"
                                )
                        data = await response.json()
                        raw = str(data.get("response", ""))
                        if not raw.strip():
                            raise RuntimeError("Ollama returned empty response")
                        if not keep_think:
                            raw = _strip_think(raw)
                        return raw

            except (asyncio.TimeoutError, aiohttp.ClientError) as exc:
                logger.warning("Ollama connection error or timeout (attempt %d): %s", attempt + 1, exc)
                last_exc = exc
                if attempt < 2:
                    await asyncio.sleep(0.5 * (2 ** attempt))

            except RuntimeError as exc:
                logger.error("Ollama error: %s", exc, exc_info=True)
                last_exc = exc
                break

            except Exception as exc:  # noqa: BLE001
                logger.error("Ollama unexpected failure: %s", exc, exc_info=True)
                last_exc = exc
                break

        raise last_exc or RuntimeError("Ollama request failed after 3 attempts")

    async def list_models(self) -> list[str]:
        """Return currently available Ollama model tags."""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/api/tags",
                timeout=aiohttp.ClientTimeout(total=3),
            ) as response:
                if response.status != 200:
                    raise RuntimeError(f"Ollama HTTP {response.status} while listing models")
                payload = await response.json()
        return extract_model_names(payload)

    async def is_running(self) -> bool:
        """Quick health check — GET the Ollama root endpoint."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.base_url,
                    timeout=aiohttp.ClientTimeout(total=3),
                ) as response:
                    return response.status == 200
        except Exception:
            return False


__all__ = ["OllamaClient", "OllamaTransientError", "extract_model_names", "list_models_sync"]




# --- FILE: core/llm/model_router.py ---

"""Intelligent multi-stage model router for Jarvis.

Supports two strategies:
  - ``adaptive`` (default): cost-minimising selection based on task
    classification signals, model capabilities, availability, and
    historical reliability from telemetry.
  - ``static``: legacy tier-lookup behaviour (backward compatible).

Public surface (all preserved from the original):
  - ``route(task_type)``
  - ``pick_model(task_type)``
  - ``get_best_available(task_type)``
  - ``escalate(current_model)``

New:
  - ``route_adaptive(classification)`` → ``RoutingDecision``
  - ``should_escalate(model, task_type, response)`` → bool
"""

# internal import removed: from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from typing import Iterable, Any

# internal import removed: from core.config.defaults import OLLAMA_BASE_URL
# internal import removed: from core.llm.ollama_client import list_models_sync
# internal import removed: from core.llm.model_spec import ModelRegistry, RoutingDecision

logger = logging.getLogger(__name__)

_MODEL_DISCOVERY_TTL_S = 30.0

# ── Minimum-reasoning thresholds by complexity band ────────────────────
_REASONING_BY_COMPLEXITY = [
    # (max_complexity, min_reasoning_required)
    (0.15, 0.10),   # reflex
    (0.40, 0.25),   # light chat
    (0.60, 0.40),   # standard chat / agentic
    (0.80, 0.60),   # complex
    (1.01, 0.75),   # deep reasoning
]


class ModelRouter:
    """Intelligent multi-stage model router for Jarvis."""

    # ── Legacy tier lists (kept for static mode & backward compat) ─────
    TIERS = {
        1: ["llama3.2:1b", "qwen2.5:0.5b", "qwen2.5:1.5b", "gemma2:2b"],
        2: ["mistral:7b", "llama3:8b", "qwen2.5:7b"],
        3: ["deepseek-r1:8b", "deepseek-r1:14b", "llama3.3:70b"],
    }

    TASK_TIERS = {
        "intent": 1,
        "intent_classification": 1,
        "reflex": 1,
        "web_search_summary": 1,
        "tool_parameter_extraction": 2,
        "context_title_generation": 1,
        "synthesis": 1,
        "summarize": 1,
        "memory_summarization": 1,
        "context_summarization": 1,

        "chat": 2,
        "general": 2,
        "planning": 3,
        "plan": 3,
        "tool_selection": 2,
        "tool_picker": 2,
        "reflection": 2,

        "deep_reasoning": 3,
        "complex_debugging": 3,
        "architecture_generation": 3,
    }

    def __init__(self, config: Any = None) -> None:
        self.config = config
        self._available_ollama_models: set[str] = set()
        self._initial_fetch_done = False
        self._cache_time = 0.0
        self._lock = threading.RLock()
        self._telemetry: Any = None  # set via set_telemetry()

        # ── Strategy ───────────────────────────────────────────────────
        self._strategy = "static"
        self._confidence_threshold = 0.7
        self._max_escalations = 1
        self._cost_preference = "balanced"  # minimum | balanced | quality

        # ── Model registry ─────────────────────────────────────────────
        self._registry = ModelRegistry(config)

        # Override TIERS from config if provided (legacy support)
        if self.config:
            if hasattr(self.config, "has_section") and self.config.has_section("routing"):
                self._strategy = self.config.get("routing", "strategy", fallback="static").strip().lower()
                self._confidence_threshold = float(
                    self.config.get("routing", "confidence_threshold", fallback="0.7")
                )
                self._max_escalations = int(
                    self.config.get("routing", "max_escalations", fallback="1")
                )
                self._cost_preference = str(
                    self.config.get("routing", "cost_preference", fallback="balanced")
                ).strip().lower()

                for tier_num in (1, 2, 3):
                    val = self.config.get("routing", f"tier{tier_num}", fallback="")
                    if val:
                        self.TIERS[tier_num] = [m.strip() for m in val.split(",") if m.strip()]
            elif isinstance(self.config, dict) and "routing" in self.config:
                routing = self.config["routing"]
                if isinstance(routing, dict):
                    self._strategy = str(routing.get("strategy", "static")).strip().lower()
                    self._confidence_threshold = float(routing.get("confidence_threshold", "0.7"))
                    self._max_escalations = int(routing.get("max_escalations", "1"))
                    self._cost_preference = str(routing.get("cost_preference", "balanced")).strip().lower()
                    
                    for tier_num in (1, 2, 3):
                        val = str(routing.get(f"tier{tier_num}", ""))
                        if val:
                            self.TIERS[tier_num] = [m.strip() for m in val.split(",") if m.strip()]

    # ── Telemetry wiring ──────────────────────────────────────────────

    def set_telemetry(self, telemetry: Any) -> None:
        """Connect execution telemetry for adaptive routing feedback."""
        self._telemetry = telemetry

    @property
    def registry(self) -> ModelRegistry:
        return self._registry

    @property
    def strategy(self) -> str:
        return self._strategy

    # ── Public routing API (backward compatible) ──────────────────────

    def route(self, task_type: str) -> str:
        """Determines the target tier for a task and returns the best available model."""
        task = str(task_type or "chat").strip().lower()

        # Vision is a special case
        if task == "vision":
            return self._cfg("vision_model", "llava:latest")

        if self._strategy == "adaptive":
            # Build a minimal classification from task_type
            tier = self.TASK_TIERS.get(task, 2)
            classification = {
                "complexity": {1: 0.1, 2: 0.4, 3: 0.85}.get(tier, 0.4),
                "needs_reasoning": tier >= 3,
                "needs_tools": task in ("tool_picker", "tool_selection", "tool_parameter_extraction"),
                "needs_vision": False,
                "estimated_tokens": {1: 50, 2: 200, 3: 500}.get(tier, 200),
                "context_weight": {1: 0.0, 2: 0.3, 3: 0.6}.get(tier, 0.3),
            }
            decision = self.route_adaptive(classification)
            return decision.model

        # Static fallback
        target_tier = self.TASK_TIERS.get(task, 2)
        return self._pick_model_from_tier(target_tier)

    def escalate(self, current_model: str) -> str:
        """Upgrades the model to a higher tier if possible."""
        current_tier = self._registry.get_tier(current_model)
        next_tier = min(3, current_tier + 1)

        if self._strategy == "adaptive":
            # Pick the cheapest model at the next tier that's available
            candidates = self._registry.get_cheapest_capable(
                min_reasoning=0.0,
                available_models=self._available_ollama_models or None,
            )
            for spec in candidates:
                if spec.tier >= next_tier:
                    resolved = self._resolve_available_variant(spec.name)
                    if resolved or spec.provider != "ollama":
                        return resolved or spec.name
            # Fallback
            return self._pick_model_from_tier(next_tier)

        return self._pick_model_from_tier(next_tier)

    def pick_model(self, task_type: str) -> str:
        """Public entry point — returns the best model name for a task type."""
        return self.route(task_type)

    def get_best_available(self, task_type: str) -> str:
        return self.route(task_type)

    # ── Adaptive routing (new) ────────────────────────────────────────

    def route_adaptive(self, classification: dict[str, Any]) -> RoutingDecision:
        """Cost-optimising model selection using classification signals.

        Algorithm:
            1. Compute minimum capability requirements from classification
            2. Query ModelRegistry for models meeting requirements
            3. Filter by availability (Ollama running, cloud API key present)
            4. Sort by weight (cost) ascending, reliability descending
            5. Select cheapest model with reliability ≥ threshold
            6. If none qualifies, escalate tier
        """
        complexity = float(classification.get("complexity", 0.4))
        needs_reasoning = bool(classification.get("needs_reasoning", False))
        needs_tools = bool(classification.get("needs_tools", False))
        needs_vision = bool(classification.get("needs_vision", False))
        estimated_tokens = int(classification.get("estimated_tokens", 200))

        # 1. Compute minimum reasoning from complexity
        min_reasoning = 0.25  # baseline
        for threshold, required in _REASONING_BY_COMPLEXITY:
            if complexity <= threshold:
                min_reasoning = required
                break

        # Boost for explicit reasoning need
        if needs_reasoning:
            min_reasoning = max(min_reasoning, 0.60)

        # Cost preference adjusts the reasoning floor
        if self._cost_preference == "minimum":
            min_reasoning = max(0.10, min_reasoning - 0.15)
        elif self._cost_preference == "quality":
            min_reasoning = min(1.0, min_reasoning + 0.15)

        # 2. Estimated context (input tokens × 1.5 safety margin)
        min_context = int(estimated_tokens * 1.5)

        # 3. Determine available providers
        available_providers = {"ollama"}
        _cloud_keys = {
            "gemini": "GEMINI_API_KEY",
            "groq": "GROQ_API_KEY",
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
        }
        for provider, env_key in _cloud_keys.items():
            if os.environ.get(env_key):
                available_providers.add(provider)

        # 4. Query registry
        with self._lock:
            self.refresh_available_models()
            candidates = self._registry.get_cheapest_capable(
                min_reasoning=min_reasoning,
                min_context=min_context,
                needs_tools=needs_tools,
                needs_vision=needs_vision,
                available_models=self._available_ollama_models or None,
                providers=available_providers,
            )

        if not candidates:
            # Nothing matches — fall back to static routing
            logger.debug("Adaptive routing found no candidates; falling back to static")
            tier = 2 if complexity < 0.7 else 3
            model = self._pick_model_from_tier(tier)
            return RoutingDecision(
                model=model,
                provider="ollama",
                tier=tier,
                reason="fallback_no_candidates",
            )

        # 5. Apply telemetry-based reliability filtering
        selected = candidates[0]  # cheapest by default
        if self._telemetry is not None:
            for spec in candidates:
                reliability = self._telemetry.get_reliability(spec.name, "overall")
                if reliability >= self._confidence_threshold:
                    selected = spec
                    break
            else:
                # No model meets threshold — pick the most reliable
                best_reliability = -1.0
                for spec in candidates:
                    rel = self._telemetry.get_reliability(spec.name, "overall")
                    if rel > best_reliability:
                        best_reliability = rel
                        selected = spec

        # 6. Determine fallback (next cheapest at higher tier)
        fallback_model = None
        fallback_provider = None
        for spec in candidates:
            if spec.tier > selected.tier and spec.name != selected.name:
                fallback_model = spec.name
                fallback_provider = spec.provider
                break

        reason = (
            f"adaptive: complexity={complexity:.2f}, "
            f"min_reasoning={min_reasoning:.2f}, "
            f"cost_pref={self._cost_preference}, "
            f"candidates={len(candidates)}"
        )
        logger.info(
            "Routing decision: %s (tier %d, weight %.2f) — %s",
            selected.name, selected.tier, selected.weight, reason,
        )

        return RoutingDecision(
            model=selected.name,
            provider=selected.provider,
            tier=selected.tier,
            reason=reason,
            weight=selected.weight,
            fallback_model=fallback_model,
            fallback_provider=fallback_provider,
        )

    def should_escalate(self, model: str, task_type: str, response: str) -> bool:
        """Heuristic check if a response quality is too low and needs retry."""
        if not response or not response.strip():
            return True

        text = response.strip()

        # Suspiciously short response for non-reflex tasks
        tier = self.TASK_TIERS.get(task_type, 2)
        if tier >= 2 and len(text) < 20:
            return True

        # Model refused / error patterns
        refusal_markers = [
            "i cannot", "i can't", "i'm unable", "as an ai",
            "i don't have access", "error:", "exception:",
        ]
        lower = text.lower()
        if any(marker in lower for marker in refusal_markers):
            return True

        # Telemetry-based: if model has low reliability for this task type
        if self._telemetry is not None:
            reliability = self._telemetry.get_reliability(model, task_type)
            if reliability < 0.3 and reliability > 0.0:
                return True

        return False

    # ── Legacy static routing helpers ─────────────────────────────────

    def _pick_model_from_tier(self, tier: int) -> str:
        with self._lock:
            self.refresh_available_models()
            for candidate in self.TIERS.get(tier, []):
                resolved = self._resolve_available_variant(candidate)
                if resolved:
                    return resolved

            # Fallback to config or default if tier models are unavailable
            if tier == 1:
                return self._cfg("quick_model", self._cfg("chat_model", "llama3.2:1b"))
            elif tier == 3:
                return self._cfg("reasoning_model", self._cfg("chat_model", "deepseek-r1:8b"))
            else:
                return self._cfg("chat_model", "mistral:7b")

    # ── Availability ──────────────────────────────────────────────────

    def is_available(self, model_name: str) -> bool:
        with self._lock:
            self.refresh_available_models()
            model = str(model_name or "").strip()
            if not model or not self._available_ollama_models:
                return False
            if model in self._available_ollama_models:
                return True
            return self._resolve_available_variant(model) is not None

    def list_available(self) -> list[str]:
        with self._lock:
            self.refresh_available_models()
            return sorted(self._available_ollama_models)

    def refresh_available_models(
        self,
        base_url: str | None = None,
        *,
        force: bool = False,
        timeout_s: float = 3.0,
    ) -> list[str]:
        with self._lock:
            now = time.time()
            if (
                not force
                and self._available_ollama_models
                and (now - self._cache_time) < _MODEL_DISCOVERY_TTL_S
            ):
                return self.list_available_without_refresh()

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop is not None and loop.is_running():
                # If cache is completely empty, do a synchronous fetch first
                # to prevent race conditions during early startup queries.
                if not self._initial_fetch_done:
                    try:
                        discovered = list_models_sync(
                            base_url=base_url or self._ollama_base_url(),
                            timeout_s=timeout_s,
                        )
                        self.set_available_models(discovered)
                        self._initial_fetch_done = True
                        return self.list_available_without_refresh()
                    except Exception:
                        self._initial_fetch_done = True
                        pass

                if not force:
                    self._cache_time = now - _MODEL_DISCOVERY_TTL_S + 5.0

                def _bg_update():
                    try:
                        discovered = list_models_sync(
                            base_url=base_url or self._ollama_base_url(),
                            timeout_s=timeout_s,
                        )
                        self.set_available_models(discovered)
                    except Exception:
                        pass

                loop.run_in_executor(None, _bg_update)
                return self.list_available_without_refresh()
            else:
                try:
                    discovered = list_models_sync(
                        base_url=base_url or self._ollama_base_url(),
                        timeout_s=timeout_s,
                    )
                    self.set_available_models(discovered)
                    self._initial_fetch_done = True
                except Exception:
                    self._initial_fetch_done = True
                    pass
                return self.list_available_without_refresh()

    def list_available_without_refresh(self) -> list[str]:
        with self._lock:
            return sorted(self._available_ollama_models)

    def set_available_models(self, models: Iterable[str]) -> None:
        with self._lock:
            self._available_ollama_models = {str(model) for model in models if str(model)}
            self._cache_time = time.time()

    # ── Internal helpers ──────────────────────────────────────────────

    def _cfg(self, key: str, default: str) -> str:
        if self.config is None:
            return default
        try:
            return str(self.config.get("models", key, fallback=default))
        except Exception:
            return default

    def _ollama_base_url(self) -> str:
        if self.config is not None:
            try:
                return str(self.config.get("ollama", "base_url", fallback=OLLAMA_BASE_URL))
            except Exception:
                pass
        return os.environ.get("OLLAMA_BASE_URL", OLLAMA_BASE_URL)

    def _resolve_available_variant(self, model_name: str) -> str | None:
        with self._lock:
            if not self._available_ollama_models:
                return None

            model = str(model_name or "").strip()
            if model in self._available_ollama_models:
                return model

            family = model.split(":", 1)[0]
            latest = f"{family}:latest"
            if latest in self._available_ollama_models:
                return latest

            for candidate in sorted(self._available_ollama_models):
                if candidate.split(":", 1)[0] == family:
                    return candidate
            return None


__all__ = ["ModelRouter"]




# --- FILE: core/llm/client.py ---

"""Async LLM client — single entry point for all Jarvis LLM calls.

Architecture:
    LLMClientV2.complete(prompt, task_type)
        → ModelRouter.pick_model(task_type)   → model name
        → OllamaClient.complete(prompt, model) → response (or raise)
        → CloudLLMClient.complete(prompt)      → fallback response
"""

# internal import removed: from __future__ import annotations

import asyncio
import logging
import re
import time
from pathlib import Path
from typing import Any

# internal import removed: from core.config.defaults import OLLAMA_BASE_URL
# internal import removed: from core.llm.ollama_client import OllamaClient
# internal import removed: from core.llm.model_router import ModelRouter
# internal import removed: from core.llm.defaults import core_llm_defaults_DEFAULT_MODEL

logger = logging.getLogger(__name__)

try:
    from core.llm.cloud_client import CloudLLMClient
except Exception:  # pragma: no cover - cloud fallback is optional
    CloudLLMClient = None  # type: ignore

JARVIS_SYSTEM = (
    "You are Jarvis, a local personal AI assistant.\n"
    "You are concise, technical, and truthful.\n"
    "You run on the user's local machine."
)


def _strip_fences(text: str) -> str:
    cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", (text or "").strip())
    cleaned = re.sub(r"\n?```$", "", cleaned)
    return cleaned.strip()


_WORKSPACE_CACHE: dict[str, dict[str, Any]] = {}

def _get_workspace_map(path: str, max_depth: int = 3, max_files: int = 50) -> str:
    """Build a compact directory view to ground model responses in local files."""
    global _WORKSPACE_CACHE
    now = time.time()

    if path in _WORKSPACE_CACHE and now - _WORKSPACE_CACHE[path]["time"] < 60:
        return str(_WORKSPACE_CACHE[path]["data"])

    if len(_WORKSPACE_CACHE) > 50:
        _WORKSPACE_CACHE.clear()

    root = Path(path)
    if not root.exists() or not root.is_dir():
        return ""

    lines: list[str] = []
    count = 0
    ignored = {"__pycache__", ".git", "node_modules", ".venv", "venv", "jarvis_env"}

    def _walk(current: Path, depth: int) -> None:
        nonlocal count
        if depth > max_depth or count >= max_files:
            return
        try:
            for item in sorted(current.iterdir()):
                if count >= max_files:
                    if lines and lines[-1] != "... (truncated)":
                        lines.append("... (truncated)")
                    break
                if item.name in ignored or item.name.startswith("."):
                    continue
                indent = "  " * (depth - 1) if depth > 0 else ""
                marker = "[DIR]" if item.is_dir() else "[FILE]"
                if depth > 0:
                    lines.append(f"{indent}{marker} {item.name}")
                    count += 1
                if item.is_dir():
                    _walk(item, depth + 1)
        except PermissionError:
            pass

    _walk(root, 0)
    result = "\n".join(lines)
    _WORKSPACE_CACHE[path] = {"time": time.time(), "data": result}
    return result


class LLMClientV2:
    """Public interface — all LLM calls in Jarvis enter here.

    Wiring (in order):
        1. ``ModelRouter.pick_model(task_type)`` → model name
        2. ``OllamaClient.complete(prompt, model=…)`` → try local first
        3. ``CloudLLMClient.complete(prompt)`` → fallback if Ollama fails
    """

    def __init__(
        self,
        hybrid_memory: Any = None,
        model: str = core_llm_defaults_DEFAULT_MODEL,
        profile: Any = None,
        base_url: str = OLLAMA_BASE_URL,
        max_concurrent: int = 4,
    ) -> None:
        self.memory = hybrid_memory
        self.model = model
        self.profile = profile
        self.base_url = base_url
        self._semaphore = asyncio.Semaphore(max_concurrent)

        # ── Component wiring ─────────────────────────────────────────────
        self._ollama = OllamaClient(base_url=base_url)
        self.model_router: ModelRouter | None = None

        self._cloud_client = None
        import os
        if (
            CloudLLMClient is not None
            and str(os.environ.get("CLOUD_LLM_FALLBACK_ENABLED", "true")).lower() == "true"
        ):
            try:
                self._cloud_client = CloudLLMClient()
            except Exception as e:
                logger.warning("Could not init CloudLLMClient: %s", e)

    def set_router(self, router: ModelRouter) -> None:
        self.model_router = router

    def set_telemetry(self, telemetry: Any) -> None:
        """Connect execution telemetry for recording LLM call metrics."""
        self._telemetry = telemetry

    # ── Core complete() — the single path every prompt takes ─────────────

    async def complete(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.1,
        task_type: str = "chat",
        keep_think: bool = False,
        classification: dict[str, Any] | None = None,
    ) -> str:
        """Text completion: ModelRouter → OllamaClient → CloudLLMClient fallback.

        Steps:
            1. Ask ModelRouter for the best model name for this task_type.
            2. Call OllamaClient.complete() with that model.
            3. If response quality is poor, auto-escalate once.
            4. If Ollama raises ANY exception, fall back to CloudLLMClient
               with the correct tier.
            5. Record telemetry for every call.
        """
        async with self._semaphore:
            # Step 1 — pick model (adaptive if classification provided)
            routing_decision = None
            if self.model_router is not None:
                if classification is not None and hasattr(self.model_router, "route_adaptive"):
                    routing_decision = self.model_router.route_adaptive(classification)
                    model_to_use = routing_decision.model
                else:
                    model_to_use = self.model_router.pick_model(task_type)
            else:
                model_to_use = self.model

            tier = 2  # default tier for cloud fallback
            if routing_decision is not None:
                tier = routing_decision.tier
            elif self.model_router is not None and hasattr(self.model_router, "_registry"):
                tier = self.model_router._registry.get_tier(model_to_use)

            # Step 2 — try Ollama
            t0 = time.time()
            response = ""
            try:
                response = await self._ollama.complete(
                    prompt,
                    system=system,
                    temperature=temperature,
                    model=model_to_use,
                    keep_think=keep_think,
                )
                latency_ms = (time.time() - t0) * 1000
                self._record_telemetry(model_to_use, task_type, latency_ms, prompt, response, True)

                # Step 3 — auto-escalate if quality is poor
                if (
                    self.model_router is not None
                    and hasattr(self.model_router, "should_escalate")
                    and self.model_router.should_escalate(model_to_use, task_type, response)
                ):
                    escalated_model = self.model_router.escalate(model_to_use)
                    if escalated_model != model_to_use:
                        logger.info(
                            "Auto-escalating from %s to %s (poor quality detected)",
                            model_to_use, escalated_model,
                        )
                        t1 = time.time()
                        try:
                            response = await self._ollama.complete(
                                prompt,
                                system=system,
                                temperature=temperature,
                                model=escalated_model,
                                keep_think=keep_think,
                            )
                            latency_ms = (time.time() - t1) * 1000
                            self._record_telemetry(
                                escalated_model, task_type, latency_ms, prompt, response, True
                            )
                        except Exception as esc_exc:
                            logger.warning("Escalated model %s also failed: %s", escalated_model, esc_exc)

                return response

            except Exception as exc:
                latency_ms = (time.time() - t0) * 1000
                self._record_telemetry(model_to_use, task_type, latency_ms, prompt, "", False)
                logger.warning("Ollama failed (%s). Attempting cloud fallback.", exc)

            # Step 4 — cloud fallback with correct tier
            if self._cloud_client is not None:
                t0 = time.time()
                try:
                    response = await self._cloud_client.complete(
                        prompt, system=system, temperature=temperature, tier=tier
                    )
                    latency_ms = (time.time() - t0) * 1000
                    cloud_model = f"cloud_tier{tier}"
                    self._record_telemetry(cloud_model, task_type, latency_ms, prompt, response, True)
                    return response
                except Exception as cloud_exc:
                    latency_ms = (time.time() - t0) * 1000
                    self._record_telemetry(f"cloud_tier{tier}", task_type, latency_ms, prompt, "", False)
                    logger.error("Cloud fallback also failed: %s", cloud_exc, exc_info=True)

            return ""

    def _record_telemetry(
        self,
        model: str,
        task_type: str,
        latency_ms: float,
        prompt: str,
        response: str,
        success: bool,
    ) -> None:
        """Record call metrics to telemetry if available."""
        telemetry = getattr(self, "_telemetry", None)
        if telemetry is None:
            return
        try:
            # Rough token estimation: ~4 chars per token
            input_tokens = max(1, len(prompt) // 4)
            output_tokens = max(0, len(response) // 4)
            telemetry.record(
                model=model,
                task_type=task_type,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                success=success,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Telemetry recording failed: %s", exc)

    # ── JSON completion helper ───────────────────────────────────────────


    # ── Chat interface (backward-compat for controller / agent loop) ─────

    async def chat_async(
        self,
        messages: list[dict[str, Any]],
        query_for_memory: str = "",
        profile_summary: str = "",
        workspace_path: str = "",
        trace_id: str | None = None,
        task_type: str = "chat",
    ) -> str:
        """Async version — use this inside any async context (agent loop, controller)."""
        if trace_id:
            logger.info("Client chat_async starting", extra={"trace_id": trace_id})

        system = await self._build_system(
            query=query_for_memory,
            profile=profile_summary,
            workspace_path=workspace_path,
        )
        prompt = self._messages_to_prompt(messages)
        return await self.complete(prompt, system=system, task_type=task_type) or ""

    def chat(
        self,
        messages: list[dict[str, Any]],
        query_for_memory: str = "",
        profile_summary: str = "",
        workspace_path: str = "",
        trace_id: str | None = None,
        task_type: str = "chat",
    ) -> str:
        """Sync bridge — ONLY call from truly synchronous, non-async contexts."""
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                asyncio.run,
                self.chat_async(messages, query_for_memory, profile_summary, workspace_path, trace_id=trace_id, task_type=task_type)
            )
            return future.result()


    # ── Health check ─────────────────────────────────────────────────────


    # ── Internal helpers ─────────────────────────────────────────────────

    async def _build_system(self, query: str = "", profile: str = "", workspace_path: str = "") -> str:
        parts = [JARVIS_SYSTEM]

        profile_obj = getattr(self, "profile", None)
        if profile_obj is not None:
            profile_injection = ""
            style_instruction = ""
            try:
                profile_injection = str(profile_obj.get_system_prompt_injection() or "").strip()
            except Exception as exc:  # noqa: BLE001
                logger.debug("Profile injection failed: %s", exc)
            try:
                style_instruction = str(profile_obj.get_communication_style() or "").strip()
            except Exception as exc:  # noqa: BLE001
                logger.debug("Profile style injection failed: %s", exc)

            combined = " ".join(part for part in (profile_injection, style_instruction) if part).strip()
            if combined:
                words = combined.split()
                if len(words) > 120:
                    combined = " ".join(words[:120])
                parts.append(f"\nPROFILE GUIDANCE:\n{combined}")

        if profile:
            parts.append(f"\nUSER PROFILE:\n{profile}")

        if workspace_path:
            workspace_map = _get_workspace_map(workspace_path)
            if workspace_map:
                parts.append(f"\nWORKSPACE:\n{workspace_map}")

        if query and self.memory is not None and hasattr(self.memory, "build_context_block"):
            try:
                context = await self.memory.build_context_block(query, n_results=3)
            except TypeError:
                context = await self.memory.build_context_block(query)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Memory context injection failed: %s", exc)
                context = ""

            if context:
                parts.append(f"\nRELEVANT MEMORY:\n{context}")

        return "\n".join(parts)

    @staticmethod
    def _messages_to_prompt(messages: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for message in messages:
            role = str(message.get("role", "user")).strip().capitalize()
            content = str(message.get("content", ""))
            lines.append(f"{role}: {content}")
        return "\n".join(lines)


__all__ = ["LLMClientV2"]




# --- FILE: core/llm/telemetry.py ---

"""Session-scoped execution telemetry for LLM routing decisions.

Tracks per-model execution stats (latency, reliability, cost, quality)
using a sliding-window approach to bound memory usage.  Thread-safe.

Usage::

    tel = RoutingTelemetry()
    tel.record("mistral:7b", "chat", latency_ms=320, input_tokens=128,
               output_tokens=64, success=True, quality_score=0.9)
    stats = tel.get_model_stats("mistral:7b")
"""

# internal import removed: from __future__ import annotations

import json
import logging
import math
import threading
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_WINDOW_SIZE = 100

# ── Relative cost weights per token (unitless, for comparison only) ──────────
# Local models are effectively free; cloud models vary.  These are rough
# multipliers applied to (input_tokens + output_tokens) to produce a
# comparable "cost units" number.  Extend as needed.
_DEFAULT_COST_WEIGHTS: dict[str, float] = {
    # Local / Ollama — near-zero real cost
    "llama3.2:1b":      0.01,
    "qwen2.5:0.5b":     0.01,
    "qwen2.5:1.5b":     0.01,
    "gemma2:2b":         0.01,
    "mistral:7b":        0.02,
    "llama3:8b":         0.02,
    "qwen2.5:7b":        0.02,
    "deepseek-r1:8b":    0.03,
    "deepseek-r1:14b":   0.05,
    "llama3.3:70b":      0.10,
    # Cloud models — higher weight
    "gemini":            0.50,
    "groq":              0.30,
    "openai":            1.00,
    "anthropic":         1.00,
}

_DEFAULT_WEIGHT = 0.05  # fallback for unknown models


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass(slots=True)
class _CallRecord:
    """Single LLM call record kept in the sliding window."""

    model: str
    task_type: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    success: bool
    quality_score: float | None = None


@dataclass(slots=True)
class ModelStats:
    """Aggregate statistics for a single model."""

    model: str
    total_calls: int = 0
    success_count: int = 0
    failure_count: int = 0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    avg_quality: float = 0.5
    success_rate: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0


# ── Core tracker ─────────────────────────────────────────────────────────────

class RoutingTelemetry:
    """Session-scoped telemetry tracker for LLM routing decisions.

    Keeps the last ``_WINDOW_SIZE`` records per model to cap memory.
    All public methods are thread-safe.
    """

    def __init__(
        self,
        *,
        cost_weights: dict[str, float] | None = None,
        window_size: int = _WINDOW_SIZE,
    ) -> None:
        self._lock = threading.Lock()
        self._windows: dict[str, deque[_CallRecord]] = {}
        self._window_size = window_size
        self._cost_weights = cost_weights or _DEFAULT_COST_WEIGHTS

    # ── Recording ────────────────────────────────────────────────────────

    def record(
        self,
        model: str,
        task_type: str,
        latency_ms: float,
        input_tokens: int,
        output_tokens: int,
        success: bool,
        quality_score: float | None = None,
    ) -> None:
        """Record a single LLM call result."""
        rec = _CallRecord(
            model=model,
            task_type=task_type,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            success=success,
            quality_score=quality_score,
        )
        with self._lock:
            window = self._windows.setdefault(
                model, deque(maxlen=self._window_size),
            )
            window.append(rec)

    # ── Query helpers ────────────────────────────────────────────────────

    def get_model_stats(self, model: str) -> ModelStats:
        """Return aggregate stats for *model* across its sliding window."""
        with self._lock:
            records = list(self._windows.get(model, []))

        if not records:
            return ModelStats(model=model)

        total = len(records)
        successes = sum(1 for r in records if r.success)
        failures = total - successes

        latencies = [r.latency_ms for r in records]
        avg_lat = sum(latencies) / total
        p95_lat = _percentile(latencies, 95)

        quality_vals = [r.quality_score for r in records if r.quality_score is not None]
        avg_q = sum(quality_vals) / len(quality_vals) if quality_vals else 0.5

        return ModelStats(
            model=model,
            total_calls=total,
            success_count=successes,
            failure_count=failures,
            avg_latency_ms=round(avg_lat, 2),
            p95_latency_ms=round(p95_lat, 2),
            avg_quality=round(avg_q, 4),
            success_rate=round(successes / total, 4),
            total_input_tokens=sum(r.input_tokens for r in records),
            total_output_tokens=sum(r.output_tokens for r in records),
        )

    def get_reliability(self, model: str, task_type: str) -> float:
        """Return 0.0–1.0 reliability for *model* on a specific *task_type*."""
        with self._lock:
            records = [
                r for r in self._windows.get(model, [])
                if r.task_type == task_type
            ]
        if not records:
            return 0.0
        return round(sum(1 for r in records if r.success) / len(records), 4)

    def get_avg_latency(self, model: str) -> float:
        """Return average latency in ms across the sliding window."""
        with self._lock:
            records = list(self._windows.get(model, []))
        if not records:
            return 0.0
        return round(sum(r.latency_ms for r in records) / len(records), 2)

    def get_cost_estimate(self, model: str) -> float:
        """Return estimated relative cost based on recorded tokens × model weight."""
        with self._lock:
            records = list(self._windows.get(model, []))
        if not records:
            return 0.0
        weight = self._resolve_cost_weight(model)
        total_tokens = sum(r.input_tokens + r.output_tokens for r in records)
        return round(total_tokens * weight, 4)

    def summary(self) -> dict[str, Any]:
        """Full stats summary for every tracked model."""
        with self._lock:
            models = list(self._windows.keys())
        return {m: asdict(self.get_model_stats(m)) for m in models}

    # ── Persistence ──────────────────────────────────────────────────────

    def save_to_file(self, path: str | Path) -> None:
        """Persist all records as JSONL (one JSON object per line)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            all_records = [
                rec
                for window in self._windows.values()
                for rec in window
            ]
        try:
            with path.open("w", encoding="utf-8") as fh:
                for rec in all_records:
                    line = json.dumps({
                        "model": rec.model,
                        "task_type": rec.task_type,
                        "latency_ms": rec.latency_ms,
                        "input_tokens": rec.input_tokens,
                        "output_tokens": rec.output_tokens,
                        "success": rec.success,
                        "quality_score": rec.quality_score,
                    }, ensure_ascii=False)
                    fh.write(line + "\n")
            logger.debug("Saved %d telemetry records to %s", len(all_records), path)
        except OSError as exc:
            logger.warning("Failed to save telemetry to %s: %s", path, exc)

    @classmethod
    def load_from_file(cls, path: str | Path, **kwargs: Any) -> RoutingTelemetry:
        """Load telemetry from a JSONL file, returning a new instance.

        Extra *kwargs* are forwarded to the ``RoutingTelemetry`` constructor.
        """
        instance = cls(**kwargs)
        path = Path(path)
        if not path.is_file():
            logger.debug("Telemetry file %s not found — starting fresh", path)
            return instance

        loaded = 0
        try:
            with path.open("r", encoding="utf-8") as fh:
                for lineno, line in enumerate(fh, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        instance.record(
                            model=str(obj["model"]),
                            task_type=str(obj["task_type"]),
                            latency_ms=float(obj["latency_ms"]),
                            input_tokens=int(obj["input_tokens"]),
                            output_tokens=int(obj["output_tokens"]),
                            success=bool(obj["success"]),
                            quality_score=(
                                float(obj["quality_score"])
                                if obj.get("quality_score") is not None
                                else None
                            ),
                        )
                        loaded += 1
                    except (KeyError, ValueError, TypeError) as exc:
                        logger.warning(
                            "Skipping malformed telemetry line %d: %s", lineno, exc,
                        )
        except OSError as exc:
            logger.warning("Failed to read telemetry from %s: %s", path, exc)

        logger.debug("Loaded %d telemetry records from %s", loaded, path)
        return instance

    # ── Internal ─────────────────────────────────────────────────────────

    def _resolve_cost_weight(self, model: str) -> float:
        """Look up cost weight, falling back to family prefix matching."""
        if model in self._cost_weights:
            return self._cost_weights[model]
        # Try family prefix (e.g. "mistral:7b" → "mistral")
        family = model.split(":")[0]
        if family in self._cost_weights:
            return self._cost_weights[family]
        return _DEFAULT_WEIGHT


# ── Utilities ────────────────────────────────────────────────────────────────

def _percentile(values: list[float], pct: int) -> float:
    """Compute the *pct*-th percentile of *values* (nearest-rank method)."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = max(0, math.ceil(len(sorted_vals) * pct / 100) - 1)
    return sorted_vals[idx]


__all__ = ["RoutingTelemetry", "ModelStats"]




# --- FILE: core/tools/path_utils.py ---

import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

ALLOWED_DIRECTORIES = [
    (_PROJECT_ROOT / "workspace").resolve(),
    (_PROJECT_ROOT / "outputs").resolve(),
]

# Project root sandbox - all resolved paths must stay inside here.
_SANDBOX_ROOT = _PROJECT_ROOT

def _assert_safe_path(path_str: str, write_op: bool = False) -> Path:
    """Raise PermissionError / ValueError if path is outside the sandbox."""
    # Block path traversal sequences before resolution
    if ".." in str(Path(path_str)):
        raise PermissionError(f"Path traversal blocked: {path_str}")

    resolved = Path(path_str).resolve()
    sandbox = _SANDBOX_ROOT

    resolved_str = str(resolved)
    sandbox_str = str(sandbox)
    if os.name == "nt":
        resolved_str = resolved_str.lower()
        sandbox_str = sandbox_str.lower()

    # Must be inside project sandbox
    if not (resolved_str == sandbox_str or resolved_str.startswith(sandbox_str + os.sep) or resolved_str.startswith(sandbox_str + "/")):
        raise PermissionError(f"Path outside sandbox: {resolved}")

    # Symlink must not escape sandbox
    if resolved.is_symlink():
        link_target = resolved.resolve()
        link_target_str = str(link_target)
        if os.name == "nt":
            link_target_str = link_target_str.lower()
        if not (link_target_str == sandbox_str or link_target_str.startswith(sandbox_str + os.sep) or link_target_str.startswith(sandbox_str + "/")):
            raise PermissionError(f"Symlink escapes sandbox: {link_target}")

    # Also check legacy ALLOWED_DIRECTORIES for backward compatibility
    target = resolved
    if write_op:
        allowed_dirs = ALLOWED_DIRECTORIES
    else:
        allowed_dirs = ALLOWED_DIRECTORIES + [
            (_PROJECT_ROOT / "config").resolve(),
            (_PROJECT_ROOT / "data").resolve(),
            (_PROJECT_ROOT / "logs").resolve(),
            (_PROJECT_ROOT / "core").resolve(),
        ]
    
    target_str = str(target)
    if os.name == "nt":
        target_str = target_str.lower()

    for allowed in allowed_dirs:
        allowed_str = str(allowed)
        if os.name == "nt":
            allowed_str = allowed_str.lower()
        if target_str == allowed_str or target_str.startswith(allowed_str + os.sep) or target_str.startswith(allowed_str + "/"):
            return target
        try:
            target.relative_to(allowed)
            return target
        except ValueError:
            continue
    raise ValueError(f"Path '{path_str}' is outside the sandbox. Allowed: {[str(d) for d in allowed_dirs]}")




# --- FILE: core/tools/builtin_tools.py ---

"""
Built-in tools for Jarvis.
All tools are async coroutines and sandboxed to allowed directories.
"""

import asyncio
import json
import logging
import os
import platform
from pathlib import Path

# internal import removed: from core.tools.path_utils import _assert_safe_path, _PROJECT_ROOT

logger = logging.getLogger("Jarvis.Tools")


# ── System tools ────────────────────────────────────────────────────────────

async def get_time() -> str:
    """Returns current local time and date."""
    import datetime
    now = datetime.datetime.now()
    return now.strftime("Current time: %H:%M:%S on %A, %B %d, %Y")


async def get_system_stats() -> str:
    """Returns basic system resource usage."""
    try:
        import psutil
        cpu = psutil.cpu_percent(interval=0.5)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        return (
            f"CPU: {cpu}% | "
            f"Memory: {mem.percent}% used ({mem.available // 1024 // 1024} MB free) | "
            f"Disk: {disk.percent}% used ({disk.free // 1024 // 1024 // 1024} GB free)"
        )
    except ImportError:
        return f"Platform: {platform.system()} {platform.release()} | (install psutil for detailed stats)"


# ── File tools ──────────────────────────────────────────────────────────────

async def list_directory(path: str = "./workspace") -> str:
    """Lists files in a sandboxed directory."""
    safe = _assert_safe_path(path, write_op=False)
    
    def _list_dir():
        if not safe.exists():
            return f"Directory '{path}' does not exist."
        entries = sorted(safe.iterdir(), key=lambda p: (p.is_file(), p.name))
        lines = []
        for e in entries:
            tag = "[DIR] " if e.is_dir() else "[FILE]"
            size = f" ({e.stat().st_size} bytes)" if e.is_file() else ""
            lines.append(f"{tag} {e.name}{size}")
        return "\n".join(lines) if lines else "(empty directory)"
        
    return await asyncio.to_thread(_list_dir)


async def read_file(path: str = "./workspace/test.txt") -> str:
    """Reads a text file from the sandbox."""
    if not path:
        path = "./workspace/test.txt"
    safe = _assert_safe_path(path, write_op=False)
    
    def _read_file():
        if not safe.exists():
            return f"File '{path}' not found."
        if not safe.is_file():
            return f"'{path}' is not a file."
        size = os.path.getsize(safe)
        if size > 10 * 1024 * 1024:   # 10 MB hard limit
            raise ValueError(f"File too large: {size} bytes (max 10MB)")
        if size > 100_000:
            return f"File too large ({size} bytes). Max 100KB."
        return safe.read_text(encoding="utf-8", errors="replace")
        
    return await asyncio.to_thread(_read_file)


async def write_file_safe(path: str = "./workspace/test.txt", content: str = "Hello World") -> str:
    """Writes content to a file in the sandbox (creates if needed)."""
    if not path:
        path = "./workspace/test.txt"
    safe = _assert_safe_path(path, write_op=True)
    
    def _write_file():
        safe.parent.mkdir(parents=True, exist_ok=True)
        safe.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} characters to '{path}'."
        
    return await asyncio.to_thread(_write_file)


# ── Memory tools ─────────────────────────────────────────────────────────────

_memory_store: list[dict] = []  # In-process simple memory


async def search_memory(query: str, limit: int = 5) -> str:
    """Simple keyword search over in-session memory."""
    query_lower = query.lower()
    matches = [
        m for m in _memory_store
        if query_lower in m.get("content", "").lower()
    ]
    if not matches:
        return f"No memory entries found matching '{query}'."
    results = matches[-limit:]
    return "\n".join(
        f"[{m.get('timestamp', '?')}] {m['content']}" for m in results
    )


async def log_event(content: str = "", category: str = "general") -> str:
    """Logs an event to in-session memory and the outputs log file."""
    import datetime
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "category": category,
        "content": content,
    }
    _memory_store.append(entry)
    # Cap to prevent unbounded memory growth
    if len(_memory_store) > 1000:
        _memory_store.pop(0)

    def _write_log():
        log_path = Path("./outputs/memory_log.jsonl")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
            
    await asyncio.to_thread(_write_log)

    return f"Event logged: [{category}] {content}"


# Global LLM and Config reference for tool usage
_LLM_CLIENT = None
_CONFIG = None


def _fallback_classify_file(file_path: Path) -> str:
    ext = file_path.suffix.lower()
    if ext in {".py", ".js", ".jsx", ".ts", ".tsx", ".html", ".css", ".go", ".java", ".cpp", ".c", ".h", ".sh", ".ps1", ".bat"}:
        return "code"
    if ext in {".txt", ".md", ".pdf", ".docx", ".doc", ".rtf"}:
        return "documentation"
    if ext in {".csv", ".xlsx", ".xls", ".json", ".xml", ".yaml", ".yml", ".ini"}:
        return "data"
    if ext in {".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp", ".bmp"}:
        return "images"
    if ext in {".mp3", ".wav", ".mp4", ".mkv", ".avi", ".mov"}:
        return "media"
    if ext in {".log", ".err", ".out"}:
        return "logs"
    if ext in {".zip", ".tar", ".gz", ".rar", ".7z"}:
        return "archives"
    return "others"


async def sort_files(directory: str = "./workspace", output_dir: str = "./workspace") -> str:
    """
    Sorts files in a sandboxed directory into subfolders according to their content using LLM classification.
    """
    safe_dir = _assert_safe_path(directory, write_op=False)
    safe_output = _assert_safe_path(output_dir, write_op=True)

    def _sort():
        if not safe_dir.exists():
            return f"Source directory '{directory}' does not exist."
        if not safe_dir.is_dir():
            return f"Source path '{directory}' is not a directory."
    
        # List all files (excluding directories) sorted alphabetically
        files = sorted([e for e in safe_dir.iterdir() if e.is_file()], key=lambda x: x.name)
        if not files:
            return f"No files found to sort in '{directory}'."
    
        sorted_count = 0
        results = []
    
        for file_path in files:
            # Avoid sorting configuration or special system files in project root
            if file_path.name in {".env", "jarvis_env", "jarvis_voice_section.ini", "desktop_automation_report.md"}:
                continue
            
            category = _fallback_classify_file(file_path)
    
            # Ensure category folder exists
            target_folder = safe_output / category
            try:
                target_folder.mkdir(parents=True, exist_ok=True)
                target_file_path = target_folder / file_path.name
                
                # Move the file
                import shutil
                shutil.move(str(file_path), str(target_file_path))
                sorted_count += 1
                results.append(f"{file_path.name} -> {category}/")
            except Exception as exc:
                results.append(f"FAILED to move {file_path.name}: {exc}")
    
        return f"Successfully sorted {sorted_count}/{len(files)} files.\nDetails:\n" + "\n".join(results)
        
    return await asyncio.to_thread(_sort)


async def find_files(pattern: str, directory: str = "./workspace") -> str:
    """Finds files matching a wildcard pattern in a sandboxed directory (recursive)."""
    safe_dir = _assert_safe_path(directory, write_op=False)
    
    def _find():
        if not safe_dir.exists():
            return f"Directory '{directory}' does not exist."
        
        matches = []
        ignored_dirs = {"__pycache__", ".git", "node_modules", ".venv", "venv", "jarvis_env"}
        
        for root, dirs, files in os.walk(safe_dir):
            dirs[:] = [d for d in dirs if d not in ignored_dirs]
            
            try:
                _assert_safe_path(root, write_op=False)
            except PermissionError:
                continue
                
            import fnmatch
            for filename in fnmatch.filter(files, pattern):
                file_path = Path(root) / filename
                try:
                    rel_path = file_path.relative_to(safe_dir)
                    matches.append(f"[FILE] {rel_path} ({file_path.stat().st_size} bytes)")
                except Exception:
                    pass
                    
            for dirname in fnmatch.filter(dirs, pattern):
                dir_path = Path(root) / dirname
                try:
                    rel_path = dir_path.relative_to(safe_dir)
                    matches.append(f"[DIR]  {rel_path}")
                except Exception:
                    pass
                    
        if not matches:
            return f"No matches found for '{pattern}' in '{directory}'."
        return "\n".join(matches)
        
    return await asyncio.to_thread(_find)


async def copy_file(source: str, destination: str) -> str:
    """Copies a file from source to destination in the sandbox."""
    safe_src = _assert_safe_path(source, write_op=False)
    safe_dst = _assert_safe_path(destination, write_op=True)
    
    def _copy():
        if not safe_src.exists():
            return f"Source file '{source}' does not exist."
        if not safe_src.is_file():
            return f"Source path '{source}' is not a file."
            
        safe_dst.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(str(safe_src), str(safe_dst))
        return f"Successfully copied '{source}' to '{destination}'."
        
    return await asyncio.to_thread(_copy)


async def move_file(source: str, destination: str) -> str:
    """Moves a file or directory from source to destination in the sandbox."""
    safe_src = _assert_safe_path(source, write_op=True)
    safe_dst = _assert_safe_path(destination, write_op=True)
    
    def _move():
        if not safe_src.exists():
            return f"Source '{source}' does not exist."
            
        safe_dst.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.move(str(safe_src), str(safe_dst))
        return f"Successfully moved '{source}' to '{destination}'."
        
    return await asyncio.to_thread(_move)


async def create_directory(path: str = "./workspace") -> str:
    """Creates a new directory (and any parent directories) in the sandbox."""
    if not path:
        path = "./workspace"
    safe = _assert_safe_path(path, write_op=True)
    
    def _create():
        safe.mkdir(parents=True, exist_ok=True)
        return f"Successfully created directory '{path}'."
        
    return await asyncio.to_thread(_create)


async def fast_search(path: str = "all", query: str = "", content: str = "", threads: int = 8, case_sensitive: bool = False, no_skip: bool = False, max_results: int = 1000) -> str:
    """
    Search files by name pattern and/or by file content (grep) across the PC/drive.
    Highly optimized multi-threaded execution.
    """
    from core.tools.fast_search_tool import run_fast_search
    from core.tools.path_utils import _assert_safe_path, ALLOWED_DIRECTORIES

    try:
        if path == "all":
            path_to_check = [
                str(d) for d in ALLOWED_DIRECTORIES + [
                    (_PROJECT_ROOT / "config").resolve(),
                    (_PROJECT_ROOT / "data").resolve(),
                    (_PROJECT_ROOT / "logs").resolve(),
                    (_PROJECT_ROOT / "core").resolve(),
                ]
            ]
        else:
            safe_path = _assert_safe_path(path, write_op=False)
            path_to_check = [str(safe_path)]
    except (PermissionError, ValueError) as e:
        return f"Error: Search path is outside the sandbox. {e}"

    try:
        res = await run_fast_search(path_to_check, query, content, threads, case_sensitive, no_skip, max_results)
        results = res.get("results", [])
        summary = res.get("summary", {})
        
        output = [
            f"Search completed using {res.get('engine', 'unknown')} engine.",
            f"Elapsed time: {summary.get('elapsed')}",
            f"Files scanned: {summary.get('files_scanned')}",
            f"Folders scanned: {summary.get('dirs_scanned')}",
            f"Matches found: {len(results)}\n",
            "Matches:"
        ]
        for item in results[:100]: # display first 100
            if item["type"] == "file":
                output.append(f"[FILE] {item['path']}")
            else:
                output.append(f"[MATCH] {item['path']}:{item['line']}: {item['text']}")
        if len(results) > 100:
            output.append(f"... and {len(results) - 100} more matches.")
        return "\n".join(output)
    except Exception as e:
        return f"Error executing fast search: {e}"


async def convert_file_format(source_path: str, target_format: str, output_path: str | None = None) -> str:
    """
    Convert a file from its current format to target_format (e.g. webp, pdf, html, csv, json, xlsx, mp3, wav, mp4).
    Dynamically installs missing libraries on demand.
    """
    from core.tools.universal_converter import perform_conversion
    try:
        safe_src = _assert_safe_path(source_path, write_op=False)
        if output_path:
            safe_dst = _assert_safe_path(output_path, write_op=True)
        else:
            ext_dst = f".{target_format.lower().lstrip('.')}"
            dst_path = Path(source_path).with_suffix(ext_dst)
            safe_dst = _assert_safe_path(str(dst_path), write_op=True)
    except (PermissionError, ValueError) as e:
        return f"Error: Path outside sandbox. {e}"

    try:
        dest_path = perform_conversion(str(safe_src), target_format, str(safe_dst))
        return f"File successfully converted! Saved to: {dest_path}"
    except Exception as e:
        return f"Error converting file: {e}"


def register_all_tools(router, llm=None, config=None) -> None:
    """Register all built-in tools with a CapabilityRegistry instance."""
    global _LLM_CLIENT, _CONFIG
    _LLM_CLIENT = llm
    _CONFIG = config
    allow_gui_automation = False
    allow_app_launch = True
    if config is not None:
        try:
            allow_gui_automation = config.getboolean(
                "execution",
                "allow_gui_automation",
                fallback=False,
            )
        except Exception:
            allow_gui_automation = False
        try:
            allow_app_launch = config.getboolean(
                "execution",
                "allow_app_launch",
                fallback=True,
            )
        except Exception:
            allow_app_launch = True

    from core.tools.system_automation import (
        async_delete_file,
        async_execute_shell,
        async_launch_application,
        async_write_file,
    )
    # ── Core tools ─────────────────────────────────────────────────────────
    router.register("get_time", get_time)
    router.register("get_system_stats", get_system_stats)
    router.register("list_directory", list_directory)
    router.register("read_file", read_file)
    router.register("write_file", async_write_file)
    router.register("delete_file", async_delete_file)
    router.register("sort_files", sort_files)
    router.register("find_files", find_files)
    router.register("copy_file", copy_file)
    router.register("move_file", move_file)
    router.register("create_directory", create_directory)
    if allow_app_launch:
        router.register("launch_application", async_launch_application)
    router.register("execute_shell", async_execute_shell)
    router.register("write_file_safe", write_file_safe)
    router.register("search_memory", search_memory)
    router.register("log_event", log_event)
    router.register("fast_search", fast_search)
    router.register("convert_file_format", convert_file_format)

    # ── Hardware tools (Session 7) ─────────────────────────────────────────
    try:
        from core.tools.hardware_tools import (
            send_hardware_command,
            read_sensor,
            list_hardware_devices,
            ping_device,
        )
        router.register("send_hardware_command", send_hardware_command)
        router.register("read_sensor", read_sensor)
        router.register("list_hardware_devices", list_hardware_devices)
        router.register("ping_device", ping_device)
        logger.info("Hardware tools registered (Session 7)")
    except Exception as e:
        logger.warning("Hardware tools unavailable: %s", e)

    # ── Screen tools (Session 7) ───────────────────────────────────────────
    try:
        from core.tools.screen import (
            capture_screen,
            capture_region,
            describe_screen,
            find_text_on_screen,
            read_screen_text,
            wait_for_text_on_screen,
        )
        from core.tools.gui_control import get_active_window
        router.register("capture_screen", capture_screen)
        router.register("capture_region", capture_region)
        router.register("find_text_on_screen", find_text_on_screen)
        router.register("read_screen_text", read_screen_text)
        router.register("wait_for_text_on_screen", wait_for_text_on_screen)
        router.register("describe_screen", describe_screen)
        router.register("get_active_window", get_active_window)
        logger.info("Screen tools registered (Session 7)")
    except Exception as e:
        logger.warning("Screen tools unavailable: %s", e)

    # ── GUI control tools (Session 7) ──────────────────────────────────────
    if allow_gui_automation:
        try:
            from core.tools.gui_control import (
                click,
                click_screen_target,
                click_text_on_screen,
                clipboard_get,
                clipboard_paste,
                clipboard_set,
                double_click,
                double_click_screen_target,
                drag,
                focus_window,
                move_mouse,
                press_key,
                right_click,
                right_click_screen_target,
                scroll,
                type_text,
                hotkey,
            )
            router.register("click", click)
            router.register("double_click", double_click)
            router.register("right_click", right_click)
            router.register("type_text", type_text)
            router.register("hotkey", hotkey)
            router.register("press_key", press_key)
            router.register("move_mouse", move_mouse)
            router.register("scroll", scroll)
            router.register("drag", drag)
            router.register("focus_window", focus_window)
            router.register("clipboard_get", clipboard_get)
            router.register("clipboard_set", clipboard_set)
            router.register("clipboard_paste", clipboard_paste)
            router.register("click_text_on_screen", click_text_on_screen)
            router.register("click_screen_target", click_screen_target)
            router.register("double_click_screen_target", double_click_screen_target)
            router.register("right_click_screen_target", right_click_screen_target)
            logger.info("GUI control tools registered (Session 7)")
        except Exception as e:
            logger.warning("GUI control tools unavailable: %s", e)
    else:
        logger.info("GUI control tools skipped because allow_gui_automation=false")

    # ── Web Research tools ─────────────────────────────────────────────────
    try:
        from core.tools.web_tools import (
            configure_web_tools,
            web_search,
            web_scrape,
        )
        configure_web_tools(config=config, llm=llm)
        router.register("web_search", web_search)
        router.register("web_scrape", web_scrape)
        logger.info("Web research tools registered")
    except Exception as e:
        logger.warning("Web research tools unavailable: %s", e)

    logger.info("Registered %d tools total: %s", len(router.registered_tools()), router.registered_tools())




# --- FILE: core/desktop/__init__.py ---

"""Closed-loop desktop reliability primitives."""

# internal import removed: from core.desktop.actions import DesktopActionExecutor
# internal import removed: from core.desktop.contracts import (
# internal import removed:     ApprovalDecision,
# internal import removed:     DesktopAction,
# internal import removed:     DesktopActionResult,
# internal import removed:     DesktopActionStatus,
# internal import removed:     DesktopActionType,
# internal import removed:     DesktopChange,
# internal import removed:     DesktopObservation,
# internal import removed:     DesktopRiskTier,
# internal import removed:     ScreenTarget,
# internal import removed: )
# internal import removed: from core.desktop.mission import (
# internal import removed:     DesktopMissionExecutor,
# internal import removed:     DesktopMissionStatus,
# internal import removed:     MissionExecutionRecord,
# internal import removed:     MissionStepRecord,
# internal import removed:     RecoveryDecision,
# internal import removed: )
# internal import removed: from core.desktop.observation import DesktopObserver

__all__ = [
    "ApprovalDecision",
    "DesktopAction",
    "DesktopActionExecutor",
    "DesktopActionResult",
    "DesktopActionStatus",
    "DesktopActionType",
    "DesktopChange",
    "DesktopMissionExecutor",
    "DesktopMissionStatus",
    "DesktopObservation",
    "DesktopObserver",
    "DesktopRiskTier",
    "MissionExecutionRecord",
    "MissionStepRecord",
    "RecoveryDecision",
    "ScreenTarget",
]




# --- FILE: core/hardware/__init__.py ---

"""Hardware compatibility package."""

# internal import removed: from .serial_controller import SerialController
# internal import removed: from .device_registry import DeviceRegistry, HardwareDevice

__all__ = ["SerialController", "DeviceRegistry", "HardwareDevice"]




# --- FILE: core/hardware/serial_controller.py ---

"""Minimal serial controller with a disabled-by-default safety posture."""

# internal import removed: from __future__ import annotations


class SerialController:
    def __init__(
        self,
        config=None,
        port: str | None = None,
        baud_rate: int | None = None,
        timeout: float = 1.0,
    ) -> None:
        self.config = config
        self.timeout = timeout
        self._serial = None
        self.enabled = False
        self.default_port = port
        self.baud_rate = int(baud_rate or 115200)

        if config is not None:
            try:
                self.enabled = config.getboolean("hardware", "enabled", fallback=False)
            except Exception:
                self.enabled = False
            try:
                self.default_port = self.default_port or config.get(
                    "hardware",
                    "default_port",
                    fallback=None,
                )
            except Exception:
                pass
            try:
                self.baud_rate = int(
                    config.get("hardware", "baud_rate", fallback=str(self.baud_rate))
                )
            except Exception:
                pass

        if self.enabled and self.default_port:
            self.connect(self.default_port)

    @property
    def is_connected(self) -> bool:
        return bool(self._serial is not None and getattr(self._serial, "is_open", False))

    def connect(self, port: str | None = None):
        if not self.enabled:
            raise NotImplementedError("Hardware serial control is disabled by config.")

        target_port = port or self.default_port
        if not target_port:
            raise ValueError("No serial port configured.")

        import serial

        self._serial = serial.Serial(
            target_port,
            baudrate=self.baud_rate,
            timeout=self.timeout,
        )
        self.default_port = target_port
        return self

    def send(self, command: str):
        if not self.enabled:
            raise NotImplementedError("Hardware serial control is disabled by config.")
        if not self.is_connected or self._serial is None:
            raise RuntimeError("Serial controller is not connected.")

        payload = f"{command}\n".encode("utf-8")
        self._serial.write(payload)
        if hasattr(self._serial, "flush"):
            self._serial.flush()
        if hasattr(self._serial, "readline"):
            return self._serial.readline().decode("utf-8", errors="replace").strip()
        return "OK"

    def close(self) -> None:
        if self._serial is not None and hasattr(self._serial, "close"):
            self._serial.close()


__all__ = ["SerialController"]




# --- FILE: core/llm/cloud_client.py ---

# internal import removed: from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


class CloudLLMClient:
    """Best-effort cloud fallback across a small provider chain, with Tier-aware routing."""

    PROVIDERS = ["gemini", "groq", "openai", "anthropic"]

    # Tiered models for each provider
    MODELS = {
        "gemini": {
            1: "gemini-2.0-flash-lite",
            2: "gemini-2.5-flash",
            3: "gemini-2.5-pro",
        },
        "groq": {
            1: "llama-3.1-8b-instant",
            2: "llama-3.3-70b-versatile",
            3: "deepseek-r1-distill-llama-70b",
        },
        "openai": {
            1: "gpt-4o-mini",
            2: "gpt-4o",
            3: "o3-mini",
        },
        "anthropic": {
            1: "claude-3-haiku-20240307",
            2: "claude-3-5-sonnet-20241022",
            3: "claude-sonnet-4-20250514",
        },
    }

    # Tier-aware provider ordering: cheap-first for tiers 1-2, quality-first for tier 3
    PROVIDER_ORDER = {
        1: ["groq", "gemini", "openai", "anthropic"],
        2: ["groq", "gemini", "openai", "anthropic"],
        3: ["anthropic", "openai", "gemini", "groq"],
    }

    def __init__(self) -> None:
        provider_keys = {
            "gemini": "GEMINI_API_KEY",
            "groq": "GROQ_API_KEY",
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
        }
        self._available = [
            provider
            for provider in self.PROVIDERS
            if os.environ.get(provider_keys[provider])
        ]
        if not self._available:
            logger.warning("No cloud LLM providers configured. Cloud fallback disabled.")
        self.last_usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0}

    async def complete(self, prompt: str, system: str = "", temperature: float = 0.1, tier: int = 2) -> str:
        ordered = self.PROVIDER_ORDER.get(tier, self.PROVIDERS)
        providers = [p for p in ordered if p in self._available]

        for provider in providers:
            try:
                model = self.MODELS[provider].get(tier, self.MODELS[provider][2])
                response, usage = await self._call(provider, prompt, system, temperature, model)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Cloud provider '%s' failed: %s", provider, exc)
                continue

            if response:
                self.last_usage = usage
                logger.info("Cloud LLM response from '%s' using model '%s'", provider, model)
                return response

        raise RuntimeError(f"All cloud LLM providers failed for tier {tier}.")

    async def _call(
        self, provider: str, prompt: str, system: str, temperature: float, model: str,
    ) -> tuple[str, dict[str, int]]:
        if provider == "gemini":
            return await self._call_gemini(prompt, system, temperature, model)
        if provider == "groq":
            return await self._call_groq(prompt, system, temperature, model)
        if provider == "openai":
            return await self._call_openai(prompt, system, temperature, model)
        if provider == "anthropic":
            return await self._call_anthropic(prompt, system, temperature, model)
        return "", {"input_tokens": 0, "output_tokens": 0}

    async def _call_groq(
        self, prompt: str, system: str, temperature: float, model: str,
    ) -> tuple[str, dict[str, int]]:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.environ['GROQ_API_KEY']}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": temperature,
                    "max_tokens": 2048,
                },
                timeout=aiohttp.ClientTimeout(total=45),
            ) as resp:
                data = await resp.json()
        usage = self._extract_openai_usage(data)
        if not data.get("choices"):
            logger.debug("Groq response missing choices: %s", data)
            return "", usage
        return str(data["choices"][0]["message"]["content"]), usage

    async def _call_openai(
        self, prompt: str, system: str, temperature: float, model: str,
    ) -> tuple[str, dict[str, int]]:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": temperature,
                },
                timeout=aiohttp.ClientTimeout(total=45),
            ) as resp:
                data = await resp.json()
        usage = self._extract_openai_usage(data)
        if not data.get("choices"):
            logger.debug("OpenAI response missing choices: %s", data)
            return "", usage
        return str(data["choices"][0]["message"]["content"]), usage

    async def _call_anthropic(
        self, prompt: str, system: str, temperature: float, model: str,
    ) -> tuple[str, dict[str, int]]:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "max_tokens": 2048,
                    "system": system,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                },
                timeout=aiohttp.ClientTimeout(total=45),
            ) as resp:
                data = await resp.json()
        usage_data = data.get("usage", {})
        usage = {
            "input_tokens": usage_data.get("input_tokens", 0),
            "output_tokens": usage_data.get("output_tokens", 0),
        }
        if not data.get("content"):
            logger.debug("Anthropic response missing content: %s", data)
            return "", usage
        return str(data["content"][0]["text"]), usage

    async def _call_gemini(
        self, prompt: str, system: str, temperature: float, model: str,
    ) -> tuple[str, dict[str, int]]:
        import aiohttp

        api_key = os.environ["GEMINI_API_KEY"]
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={api_key}"
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature},
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=45),
            ) as resp:
                data = await resp.json()
        usage_meta = data.get("usageMetadata", {})
        usage = {
            "input_tokens": usage_meta.get("promptTokenCount", 0),
            "output_tokens": usage_meta.get("candidatesTokenCount", 0),
        }
        try:
            return str(data["candidates"][0]["content"]["parts"][0]["text"]), usage
        except (KeyError, IndexError):
            logger.debug("Gemini response missing content: %s", data)
            return "", usage

    @staticmethod
    def _extract_openai_usage(data: dict) -> dict[str, int]:
        """Extract usage tokens from OpenAI-compatible API responses (OpenAI, Groq)."""
        usage_data = data.get("usage", {})
        return {
            "input_tokens": usage_data.get("prompt_tokens", 0),
            "output_tokens": usage_data.get("completion_tokens", 0),
        }


__all__ = ["CloudLLMClient"]




# --- FILE: core/llm/__init__.py ---

"""
core/llm/__init__.py
═════════════════════
LLM subsystem for Jarvis — adaptive multi-tier architecture.

Components:
    LLMClientV2    — Public interface, all LLM calls enter here
    OllamaClient   — Local Ollama HTTP client (tried first, no API cost)
    CloudLLMClient — Cloud fallback chain: Gemini → Groq → OpenAI → Anthropic
    ModelRouter    — Adaptive task → model routing (cost-optimising)
    ModelSpec      — Model capability/cost descriptors
    ModelRegistry  — Queryable catalog of all known models
    RoutingDecision — Result of an adaptive routing decision
    RoutingTelemetry — Per-model execution stats tracker
"""

# internal import removed: from core.llm.client import LLMClientV2
# internal import removed: from core.llm.ollama_client import OllamaClient, OllamaTransientError
# internal import removed: from core.llm.cloud_client import CloudLLMClient
# internal import removed: from core.llm.model_router import ModelRouter
# internal import removed: from core.llm.model_spec import ModelSpec, ModelRegistry, RoutingDecision
# internal import removed: from core.llm.telemetry import RoutingTelemetry

__all__ = [
    "LLMClientV2",
    "OllamaClient",
    "OllamaTransientError",
    "CloudLLMClient",
    "ModelRouter",
    "ModelSpec",
    "ModelRegistry",
    "RoutingDecision",
    "RoutingTelemetry",
]




# --- FILE: core/plugins/__init__.py ---

"""Unified plugin catalog stub for Jarvis extensions."""

# internal import removed: from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("Jarvis.Plugins")


class PluginCatalog:
    """Mock/Stub of PluginCatalog to support the Dashboard summary API after manifest removal."""
    def __init__(self, root: str | Path | None = None, config: Any | None = None, enabled_scopes: set[str] | None = None) -> None:
        self.root = Path(root or "core/plugins")
        self.enabled_scopes = enabled_scopes or set()
        self.errors: dict[str, str] = {}

    def summary(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "count": 0,
            "enabled_scopes": sorted(list(self.enabled_scopes)),
            "plugins": [],
            "errors": {},
        }


class PluginManifest:
    pass


class PluginManifestError(ValueError):
    pass


def load_plugin_manifest(plugin_dir: Any) -> Any:
    return None


__all__ = [
    "PluginCatalog",
    "PluginManifest",
    "PluginManifestError",
    "load_plugin_manifest",
]




# --- FILE: core/tools/__init__.py ---





# --- FILE: core/tools/gui_control.py ---

"""Desktop GUI automation tools for Jarvis."""

# internal import removed: from __future__ import annotations
# internal import removed: from core.types.common import core_types_common_ToolResult

import asyncio
import configparser
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

GUI_AUDIT_DIR = Path("outputs/gui_audit")
_CONFIG_PATH = Path("config/jarvis.ini")

_FORBIDDEN_KEYWORDS: tuple[str, ...] = (
    "password",
    "passwd",
    "secret",
    "token",
    "apikey",
    "api_key",
)

_LAST_CLICK_TIME: float = 0.0
_CLICK_SAFETY_DELAY: float = 0.3


def _save_audit_screenshot(label: str) -> str:
    try:
        import pyautogui

        GUI_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time() * 1000)
        path = GUI_AUDIT_DIR / f"{timestamp}_{label}.png"
        pyautogui.screenshot().save(str(path))
        return str(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Audit screenshot failed (%s): %s", label, exc)
        return ""


def _validate_coords(x: int, y: int) -> bool:
    try:
        import pyautogui

        width, height = pyautogui.size()
        return bool(0 <= x < width and 0 <= y < height)
    except Exception:
        return True


def core_tools_gui_control__require_pyautogui():
    try:
        import pyautogui

        return pyautogui
    except ImportError as exc:
        raise ImportError("pyautogui not installed - run: pip install pyautogui") from exc


def _contains_sensitive_text(text: str) -> str | None:
    lowered = str(text or "").lower()
    for keyword in _FORBIDDEN_KEYWORDS:
        if keyword in lowered:
            return keyword
    return None


def _tool_result_payload(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        data = result.get("data")
        return data if isinstance(data, dict) else {}
    data = getattr(result, "data", None)
    return data if isinstance(data, dict) else {}


def _extract_json_object(raw: str) -> dict[str, Any]:
    candidate = str(raw or "").strip()
    fenced = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
    if fenced is not None:
        candidate = fenced.group(0)
    parsed = json.loads(candidate)
    if not isinstance(parsed, dict):
        raise ValueError("Vision locator did not return a JSON object.")
    return parsed


def _match_center(match: dict[str, Any]) -> tuple[int, int]:
    x = int(match.get("x", 0))
    y = int(match.get("y", 0))
    width = int(match.get("w", match.get("width", 0)) or 0)
    height = int(match.get("h", match.get("height", 0)) or 0)
    return x + max(0, width // 2), y + max(0, height // 2)


def _runtime_config() -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    if _CONFIG_PATH.is_file():
        config.read(_CONFIG_PATH, encoding="utf-8")
    return config


def _vision_locate_target(target: str):
    from core.tools.screen import capture_screen
    from core.tools.vision import VisionTool

    screen_result = capture_screen()
    if not screen_result.success:
        return core_types_common_ToolResult(success=False, error=screen_result.error)

    screenshot_path = str(screen_result.data.get("path", "") or "")
    if not screenshot_path:
        return core_types_common_ToolResult(success=False, error="Vision locator could not capture a screenshot.")

    prompt = (
        "Locate the UI target described below in this screenshot.\n"
        f"Target: {target}\n"
        "Return strict JSON only with keys: found, x, y, width, height, confidence, reason.\n"
        "Use absolute image pixels for x and y as the center point of the target.\n"
        "If you cannot find it, return found=false."
    )

    try:
        raw = VisionTool(_runtime_config()).analyze(screenshot_path, prompt)
        parsed = _extract_json_object(raw)
    except Exception as exc:  # noqa: BLE001
        return core_types_common_ToolResult(success=False, error=f"Vision locator failed: {exc}")

    found = bool(parsed.get("found", True))
    x = int(parsed.get("x", 0) or 0)
    y = int(parsed.get("y", 0) or 0)
    confidence = float(parsed.get("confidence", 0.0) or 0.0)
    if not found:
        return core_types_common_ToolResult(
            success=False,
            data={"target": target, "confidence": confidence, "reason": str(parsed.get("reason", "") or "")},
            error=f"Target '{target}' was not found by the vision locator.",
        )
    if not _validate_coords(x, y):
        return core_types_common_ToolResult(success=False, error=f"Vision returned out-of-bounds coordinates ({x}, {y}).")
    return core_types_common_ToolResult(
        success=True,
        data={
            "target": target,
            "x": x,
            "y": y,
            "width": int(parsed.get("width", 0) or 0),
            "height": int(parsed.get("height", 0) or 0),
            "confidence": confidence,
            "reason": str(parsed.get("reason", "") or ""),
            "method": "vision",
            "screenshot_path": screenshot_path,
        },
    )


async def click(x: int, y: int, button: str = "left"):
    """Click at screen coordinates with bounds checks and audit capture."""

    global _LAST_CLICK_TIME
    try:
        pag = core_tools_gui_control__require_pyautogui()
    except ImportError as exc:
        return core_types_common_ToolResult(success=False, error=str(exc))

    if not _validate_coords(x, y):
        return core_types_common_ToolResult(success=False, error=f"Coordinates ({x}, {y}) are outside screen bounds")

    await asyncio.to_thread(_save_audit_screenshot, "before_click")
    await asyncio.sleep(_CLICK_SAFETY_DELAY)
    _LAST_CLICK_TIME = time.time()
    await asyncio.to_thread(pag.click, x, y, button=button)
    await asyncio.to_thread(_save_audit_screenshot, "after_click")
    logger.info("click(%d, %d, button=%s)", x, y, button)
    return core_types_common_ToolResult(success=True, data={"action": "click", "x": x, "y": y, "button": button})


async def double_click(x: int, y: int):
    """Double-click at screen coordinates."""

    try:
        pag = core_tools_gui_control__require_pyautogui()
    except ImportError as exc:
        return core_types_common_ToolResult(success=False, error=str(exc))

    if not _validate_coords(x, y):
        return core_types_common_ToolResult(success=False, error=f"Coordinates ({x}, {y}) are outside screen bounds")

    await asyncio.to_thread(_save_audit_screenshot, "before_dblclick")
    await asyncio.sleep(_CLICK_SAFETY_DELAY)
    await asyncio.to_thread(pag.doubleClick, x, y)
    await asyncio.to_thread(_save_audit_screenshot, "after_dblclick")
    logger.info("double_click(%d, %d)", x, y)
    return core_types_common_ToolResult(success=True, data={"action": "double_click", "x": x, "y": y})


async def right_click(x: int, y: int):
    """Right-click at screen coordinates."""

    try:
        pag = core_tools_gui_control__require_pyautogui()
    except ImportError as exc:
        return core_types_common_ToolResult(success=False, error=str(exc))

    if not _validate_coords(x, y):
        return core_types_common_ToolResult(success=False, error=f"Coordinates ({x}, {y}) are outside screen bounds")

    await asyncio.to_thread(_save_audit_screenshot, "before_rightclick")
    await asyncio.sleep(_CLICK_SAFETY_DELAY)
    await asyncio.to_thread(pag.rightClick, x, y)
    await asyncio.to_thread(_save_audit_screenshot, "after_rightclick")
    logger.info("right_click(%d, %d)", x, y)
    return core_types_common_ToolResult(success=True, data={"action": "right_click", "x": x, "y": y})


async def click_text_on_screen(
    text: str,
    *,
    occurrence: int = 1,
    button: str = "left",
    match_mode: str = "contains",
):
    """Locate visible text on screen and click its center."""
    from core.tools.screen import find_text_on_screen

    result = find_text_on_screen(text, match_mode=match_mode)
    if not result.success:
        return result

    matches = list(result.data.get("matches", []))
    if not matches:
        return core_types_common_ToolResult(
            success=False,
            data={"query": text, "match_mode": match_mode},
            error=f"No visible text matched '{text}'.",
        )

    index = max(0, int(occurrence) - 1)
    if index >= len(matches):
        return core_types_common_ToolResult(
            success=False,
            data={"query": text, "matches_found": len(matches)},
            error=f"Requested match {occurrence}, but only {len(matches)} match(es) were found for '{text}'.",
        )

    match = matches[index]
    center_x, center_y = _match_center(match)
    click_result = await click(center_x, center_y, button=button)
    if not click_result.success:
        return click_result

    return core_types_common_ToolResult(
        success=True,
        data={
            **_tool_result_payload(click_result),
            "query": text,
            "match_mode": match_mode,
            "occurrence": occurrence,
            "matched_text": str(match.get("text", "") or ""),
            "match": match,
            "method": "ocr_text",
        },
    )


async def _resolve_target_coordinates(
    target: str,
    *,
    occurrence: int = 1,
    match_mode: str = "contains",
    min_confidence: float = 0.2,
):
    """Resolve coordinates for a described screen target, trying OCR first, then Vision.
    Returns:
        (x, y, resolved_metadata, error_result)
        If resolved successfully: (x, y, metadata, None)
        If failed: (None, None, None, core_types_common_ToolResult)
    """
    from core.tools.screen import find_text_on_screen

    # 1. Try OCR text locator first
    text_res = await asyncio.to_thread(find_text_on_screen, target, match_mode=match_mode)
    text_error = ""
    if text_res.success:
        matches = list(text_res.data.get("matches", []))
        index = max(0, int(occurrence) - 1)
        if matches and index < len(matches):
            match = matches[index]
            center_x, center_y = _match_center(match)
            return center_x, center_y, {
                "matched_text": str(match.get("text", "") or ""),
                "match": match,
                "target": target,
                "occurrence": occurrence,
                "match_mode": match_mode,
                "method": "ocr_text",
            }, None
        else:
            text_error = f"No visible text matched '{target}'." if not matches else f"Requested match {occurrence}, but only {len(matches)} match(es) were found."
    else:
        text_error = text_res.error or "OCR text lookup failed."

    # 2. Try Vision locator fallback
    vision_result = await asyncio.to_thread(_vision_locate_target, target)
    if not vision_result.success:
        error = text_error or vision_result.error
        if text_error and vision_result.error:
            error = f"{text_error} Vision fallback: {vision_result.error}"
        return None, None, None, core_types_common_ToolResult(
            success=False,
            data={"target": target, "text_locator_error": text_error, "vision_locator_error": vision_result.error},
            error=error,
        )

    confidence = float(vision_result.data.get("confidence", 0.0) or 0.0)
    if confidence < float(min_confidence):
        return None, None, None, core_types_common_ToolResult(
            success=False,
            data=dict(vision_result.data),
            error=(
                f"Vision confidence for '{target}' was {confidence:.2f}, below the minimum "
                f"threshold of {float(min_confidence):.2f}."
            ),
        )

    x = int(vision_result.data.get("x", 0) or 0)
    y = int(vision_result.data.get("y", 0) or 0)
    return x, y, {
        **dict(vision_result.data),
        "target": target,
        "method": "vision",
    }, None


async def click_screen_target(
    target: str,
    *,
    occurrence: int = 1,
    button: str = "left",
    match_mode: str = "contains",
    min_confidence: float = 0.2,
):
    """Click a described screen target without hard-coded coordinates."""

    x, y, resolved, err = await _resolve_target_coordinates(
        target,
        occurrence=occurrence,
        match_mode=match_mode,
        min_confidence=min_confidence,
    )
    if err is not None:
        return err

    click_result = await click(x, y, button=button)
    if not click_result.success:
        return click_result

    return core_types_common_ToolResult(
        success=True,
        data={
            **_tool_result_payload(click_result),
            **resolved,
        },
    )


async def double_click_screen_target(
    target: str,
    *,
    occurrence: int = 1,
    match_mode: str = "contains",
    min_confidence: float = 0.2,
):
    """Double-click a described screen target without hard-coded coordinates."""

    x, y, resolved, err = await _resolve_target_coordinates(
        target,
        occurrence=occurrence,
        match_mode=match_mode,
        min_confidence=min_confidence,
    )
    if err is not None:
        return err

    click_result = await double_click(x, y)
    if not click_result.success:
        return click_result

    return core_types_common_ToolResult(
        success=True,
        data={
            **_tool_result_payload(click_result),
            **resolved,
        },
    )


async def right_click_screen_target(
    target: str,
    *,
    occurrence: int = 1,
    match_mode: str = "contains",
    min_confidence: float = 0.2,
):
    """Right-click a described screen target without hard-coded coordinates."""

    x, y, resolved, err = await _resolve_target_coordinates(
        target,
        occurrence=occurrence,
        match_mode=match_mode,
        min_confidence=min_confidence,
    )
    if err is not None:
        return err

    click_result = await right_click(x, y)
    if not click_result.success:
        return click_result

    return core_types_common_ToolResult(
        success=True,
        data={
            **_tool_result_payload(click_result),
            **resolved,
        },
    )


async def move_mouse(x: int, y: int, duration: float = 0.0):

    try:
        pag = core_tools_gui_control__require_pyautogui()
    except ImportError as exc:
        return core_types_common_ToolResult(success=False, error=str(exc))

    if not _validate_coords(x, y):
        return core_types_common_ToolResult(success=False, error=f"Coordinates ({x}, {y}) are outside screen bounds")

    await asyncio.to_thread(pag.moveTo, x, y, duration=max(0.0, float(duration)))
    return core_types_common_ToolResult(success=True, data={"action": "move_mouse", "x": x, "y": y, "duration": duration})


async def scroll(clicks: int, x: int | None = None, y: int | None = None):

    try:
        pag = core_tools_gui_control__require_pyautogui()
    except ImportError as exc:
        return core_types_common_ToolResult(success=False, error=str(exc))

    if x is not None and y is not None:
        if not _validate_coords(int(x), int(y)):
            return core_types_common_ToolResult(success=False, error=f"Coordinates ({x}, {y}) are outside screen bounds")
        await asyncio.to_thread(pag.moveTo, int(x), int(y))
    await asyncio.to_thread(pag.scroll, int(clicks))
    return core_types_common_ToolResult(success=True, data={"action": "scroll", "clicks": int(clicks), "x": x, "y": y})


async def drag(
    start_x: int,
    start_y: int,
    end_x: int,
    end_y: int,
    *,
    duration: float = 0.2,
    button: str = "left",
):

    try:
        pag = core_tools_gui_control__require_pyautogui()
    except ImportError as exc:
        return core_types_common_ToolResult(success=False, error=str(exc))

    if not _validate_coords(start_x, start_y) or not _validate_coords(end_x, end_y):
        return core_types_common_ToolResult(success=False, error="Drag coordinates are outside screen bounds")

    await asyncio.to_thread(pag.moveTo, start_x, start_y)
    await asyncio.to_thread(pag.dragTo, end_x, end_y, duration=max(0.0, float(duration)), button=button)
    return core_types_common_ToolResult(
        success=True,
        data={
            "action": "drag",
            "start_x": start_x,
            "start_y": start_y,
            "end_x": end_x,
            "end_y": end_y,
            "button": button,
            "duration": duration,
        },
    )


async def type_text(text: str, interval: float = 0.05):

    forbidden = _contains_sensitive_text(text)
    if forbidden is not None:
        logger.warning("type_text refused because it contained '%s'", forbidden)
        return core_types_common_ToolResult(success=False, error=f"Refused: text contains sensitive keyword '{forbidden}'")

    try:
        pag = core_tools_gui_control__require_pyautogui()
    except ImportError as exc:
        return core_types_common_ToolResult(success=False, error=str(exc))

    await asyncio.sleep(_CLICK_SAFETY_DELAY)
    await asyncio.to_thread(pag.typewrite, text, interval=max(0.0, float(interval)))
    logger.info("type_text: typed %d character(s)", len(text))
    return core_types_common_ToolResult(success=True, data={"action": "type_text", "length": len(text)})


async def press_key(key: str, presses: int = 1, interval: float = 0.05):

    try:
        pag = core_tools_gui_control__require_pyautogui()
    except ImportError as exc:
        return core_types_common_ToolResult(success=False, error=str(exc))

    await asyncio.sleep(0.1)
    await asyncio.to_thread(pag.press, str(key), presses=max(1, int(presses)), interval=max(0.0, float(interval)))
    return core_types_common_ToolResult(
        success=True,
        data={"action": "press_key", "key": str(key), "presses": max(1, int(presses))},
    )


async def hotkey(*keys: str):

    try:
        pag = core_tools_gui_control__require_pyautogui()
    except ImportError as exc:
        return core_types_common_ToolResult(success=False, error=str(exc))

    await asyncio.sleep(0.1)
    await asyncio.to_thread(pag.hotkey, *keys)
    logger.info("hotkey: %s", "+".join(keys))
    return core_types_common_ToolResult(success=True, data={"action": "hotkey", "keys": list(keys)})


def focus_window(title: str):

    try:
        import pygetwindow as gw
    except ImportError:
        return core_types_common_ToolResult(success=False, error="pygetwindow not installed - run: pip install pygetwindow")

    windows = gw.getWindowsWithTitle(title)
    if not windows:
        return core_types_common_ToolResult(success=False, error=f"No window found matching '{title}'.")
    window = windows[0]
    window.activate()
    return core_types_common_ToolResult(success=True, data={"title": getattr(window, "title", title)})


def get_active_window():

    try:
        import pygetwindow as gw
    except ImportError:
        return core_types_common_ToolResult(success=False, error="pygetwindow not installed - run: pip install pygetwindow")

    window = gw.getActiveWindow()
    if window is None:
        return core_types_common_ToolResult(success=True, data={"title": None, "message": "No active window detected"})
    return core_types_common_ToolResult(
        success=True,
        data={
            "title": window.title,
            "x": window.left,
            "y": window.top,
            "width": window.width,
            "height": window.height,
        },
    )


def clipboard_get():

    try:
        import pyperclip
    except ImportError:
        return core_types_common_ToolResult(success=False, error="pyperclip not installed - run: pip install pyperclip")

    text = pyperclip.paste()
    return core_types_common_ToolResult(success=True, data={"length": len(text), "text": text})


def clipboard_set(text: str):

    forbidden = _contains_sensitive_text(text)
    if forbidden is not None:
        return core_types_common_ToolResult(success=False, error=f"Refused: text contains sensitive keyword '{forbidden}'")

    try:
        import pyperclip
    except ImportError:
        return core_types_common_ToolResult(success=False, error="pyperclip not installed - run: pip install pyperclip")

    pyperclip.copy(text)
    return core_types_common_ToolResult(success=True, data={"length": len(text)})


async def clipboard_paste():

    result = await hotkey("ctrl", "v")
    if not result.success:
        return result
    return core_types_common_ToolResult(success=True, data={"action": "clipboard_paste"})


async def get_mouse_position():

    try:
        pag = core_tools_gui_control__require_pyautogui()
    except ImportError as exc:
        return core_types_common_ToolResult(success=False, error=str(exc))

    x, y = await asyncio.to_thread(pag.position)
    return core_types_common_ToolResult(success=True, data={"action": "get_mouse_position", "x": x, "y": y})


async def mouse_down(button: str = "left"):

    try:
        pag = core_tools_gui_control__require_pyautogui()
    except ImportError as exc:
        return core_types_common_ToolResult(success=False, error=str(exc))

    await asyncio.to_thread(pag.mouseDown, button=button)
    logger.info("mouse_down(button=%s)", button)
    return core_types_common_ToolResult(success=True, data={"action": "mouse_down", "button": button})


async def mouse_up(button: str = "left"):

    try:
        pag = core_tools_gui_control__require_pyautogui()
    except ImportError as exc:
        return core_types_common_ToolResult(success=False, error=str(exc))

    await asyncio.to_thread(pag.mouseUp, button=button)
    logger.info("mouse_up(button=%s)", button)
    return core_types_common_ToolResult(success=True, data={"action": "mouse_up", "button": button})


async def key_down(key: str):

    try:
        pag = core_tools_gui_control__require_pyautogui()
    except ImportError as exc:
        return core_types_common_ToolResult(success=False, error=str(exc))

    await asyncio.to_thread(pag.keyDown, str(key))
    logger.info("key_down(key=%s)", key)
    return core_types_common_ToolResult(success=True, data={"action": "key_down", "key": str(key)})


async def key_up(key: str):

    try:
        pag = core_tools_gui_control__require_pyautogui()
    except ImportError as exc:
        return core_types_common_ToolResult(success=False, error=str(exc))

    await asyncio.to_thread(pag.keyUp, str(key))
    logger.info("key_up(key=%s)", key)
    return core_types_common_ToolResult(success=True, data={"action": "key_up", "key": str(key)})


async def middle_click(x: int, y: int):
    """Middle-click at screen coordinates."""

    try:
        pag = core_tools_gui_control__require_pyautogui()
    except ImportError as exc:
        return core_types_common_ToolResult(success=False, error=str(exc))

    if not _validate_coords(x, y):
        return core_types_common_ToolResult(success=False, error=f"Coordinates ({x}, {y}) are outside screen bounds")

    await asyncio.to_thread(_save_audit_screenshot, "before_middleclick")
    await asyncio.sleep(_CLICK_SAFETY_DELAY)
    await asyncio.to_thread(pag.middleClick, x, y)
    await asyncio.to_thread(_save_audit_screenshot, "after_middleclick")
    logger.info("middle_click(%d, %d)", x, y)
    return core_types_common_ToolResult(success=True, data={"action": "middle_click", "x": x, "y": y})


__all__ = [
    "click",
    "click_screen_target",
    "click_text_on_screen",
    "clipboard_get",
    "clipboard_paste",
    "clipboard_set",
    "double_click",
    "double_click_screen_target",
    "drag",
    "focus_window",
    "get_active_window",
    "get_mouse_position",
    "hotkey",
    "key_down",
    "key_up",
    "middle_click",
    "mouse_down",
    "mouse_up",
    "move_mouse",
    "press_key",
    "right_click",
    "right_click_screen_target",
    "scroll",
    "type_text",
]




# --- FILE: core/tools/auto_clicker.py ---

"""Desktop Auto-Clicker utility for continuously finding and clicking dynamic UI elements."""

import argparse
import asyncio
import logging
import sys

# internal import removed: from core.tools.gui_control import click_screen_target

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AutoClicker")


async def run_auto_clicker(target: str, interval: float, continuous: bool, min_confidence: float) -> None:
    """Run the auto clicker loop."""
    logger.info("Starting Auto-Clicker for target: '%s'", target)
    logger.info("Interval: %.1f seconds | Continuous: %s", interval, continuous)
    
    attempts = 0
    while True:
        attempts += 1
        logger.debug("Attempt %d: Searching for target...", attempts)
        try:
            # We use click_screen_target which combines OCR and Vision
            result = await click_screen_target(
                target=target,
                occurrence=1,
                button="left",
                match_mode="contains",
                min_confidence=min_confidence,
            )
            
            if result.success:
                logger.info("Successfully clicked target! Result: %s", result.data)
                if not continuous:
                    logger.info("Continuous mode is disabled. Exiting after successful click.")
                    break
            else:
                logger.debug("Target not found or click failed: %s", result.error)
                
        except Exception as e:
            logger.error("Error during auto-clicker loop: %s", e, exc_info=True)
            
        logger.debug("Waiting %.1f seconds before next check...", interval)
        await asyncio.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="Desktop Auto-Clicker using Vision and OCR.")
    parser.add_argument("-t", "--target", type=str, required=True, help="Description or text of the target to click.")
    parser.add_argument("-i", "--interval", type=float, default=5.0, help="Seconds to wait between checks.")
    parser.add_argument("-c", "--continuous", action="store_true", help="Keep running even after successfully clicking.")
    parser.add_argument("--min-confidence", type=float, default=0.2, help="Minimum confidence threshold for Vision matching.")
    
    args = parser.parse_args()
    
    try:
        asyncio.run(run_auto_clicker(
            target=args.target,
            interval=args.interval,
            continuous=args.continuous,
            min_confidence=args.min_confidence,
        ))
    except KeyboardInterrupt:
        logger.info("Auto-Clicker stopped by user (Ctrl+C).")
        sys.exit(0)


# main block removed: if __name__ == "__main__":
# main block removed:     main()




# --- FILE: core/tools/fast_search_tool.py ---

import os
import queue
import threading
import fnmatch
import asyncio
from pathlib import Path


# ----------------------------------------------------------------------
# Fast Python Multi-threaded Traversal & Grep Fallback Engine
# ----------------------------------------------------------------------
class PythonSearchEngine:
    def __init__(self, start_paths, query=None, content_query=None, num_threads=16, case_sensitive=False, no_skip=False, max_results=2000):
        self.start_paths = [Path(p) for p in start_paths]
        self.query = query
        self.content_query = content_query
        self.num_threads = num_threads
        self.case_sensitive = case_sensitive
        self.no_skip = no_skip
        self.max_results = max_results
        
        self.q: queue.Queue[str] = queue.Queue()
        self.active_workers = 0
        self.active_workers_lock = threading.Lock()
        self.results = []
        self.results_lock = threading.Lock()
        self.done_event = threading.Event()
        
        self.files_scanned = 0
        self.dirs_scanned = 0
        
        self.skip_dirs = {
            "$recycle.bin",
            "system volume information",
            "node_modules",
            ".git",
            ".venv",
            "venv",
            "appdata",
            "winsxs",
            "servicing",
            "windows\\temp",
            "microsoft",
            "recovery"
        }

    def should_skip(self, path):
        if self.no_skip:
            return False
        p_lower = str(path).lower()
        for skip in self.skip_dirs:
            if skip in p_lower:
                return True
        return False

    def is_binary(self, filepath):
        try:
            with open(filepath, 'rb') as f:
                chunk = f.read(1024)
                return b'\x00' in chunk
        except OSError:
            return True

    def search_file_content(self, filepath):
        if not self.content_query:
            return
        try:
            if os.path.getsize(filepath) > 20 * 1024 * 1024:  # 20MB limit
                return
            if self.is_binary(filepath):
                return
            
            target = self.content_query if self.case_sensitive else self.content_query.lower()
            # Open file with error-handling to prevent decode failures from stopping walk
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    search_line = line if self.case_sensitive else line.lower()
                    if target in search_line:
                        with self.results_lock:
                            if len(self.results) < self.max_results:
                                self.results.append({
                                    "type": "match",
                                    "path": str(filepath),
                                    "line": line_num,
                                    "text": line.strip()
                                })
                            else:
                                self.done_event.set()
                                return
        except OSError:
            pass

    def worker(self):
        while not self.done_event.is_set():
            try:
                # Short timeout to periodically check done_event
                current_dir = self.q.get(timeout=0.05)
            except queue.Empty:
                with self.active_workers_lock:
                    if self.active_workers == 0:
                        self.done_event.set()
                        break
                continue
            
            with self.active_workers_lock:
                self.active_workers += 1
            
            self.dirs_scanned += 1
            try:
                with os.scandir(current_dir) as it:
                    for entry in it:
                        if self.done_event.is_set():
                            break
                        try:
                            if entry.is_dir(follow_symlinks=False):
                                if not self.should_skip(entry.path):
                                    self.q.put(entry.path)
                            elif entry.is_file(follow_symlinks=False):
                                self.files_scanned += 1
                                filename = entry.name
                                match = False
                                if not self.query:
                                    match = True
                                else:
                                    if self.case_sensitive:
                                        match = fnmatch.fnmatch(filename, self.query)
                                    else:
                                        match = fnmatch.fnmatch(filename.lower(), self.query.lower())
                                
                                if match:
                                    if self.content_query:
                                        self.search_file_content(entry.path)
                                    else:
                                        with self.results_lock:
                                            if len(self.results) < self.max_results:
                                                self.results.append({
                                                    "type": "file",
                                                    "path": str(entry.path)
                                                })
                                            else:
                                                self.done_event.set()
                                                break
                        except OSError:
                            pass
            except OSError:
                pass
            finally:
                with self.active_workers_lock:
                    self.active_workers -= 1
                self.q.task_done()

    def run(self):
        # Initialize queue with root paths
        for path in self.start_paths:
            if path.exists():
                if path.is_dir():
                    self.q.put(str(path))
                else:
                    self.files_scanned += 1
                    filename = path.name
                    match = False
                    if not self.query:
                        match = True
                    else:
                        if self.case_sensitive:
                            match = fnmatch.fnmatch(filename, self.query)
                        else:
                            match = fnmatch.fnmatch(filename.lower(), self.query.lower())
                    if match:
                        if self.content_query:
                            self.search_file_content(path)
                        else:
                            self.results.append({
                                "type": "file",
                                "path": str(path)
                            })
        
        threads = []
        for _ in range(self.num_threads):
            t = threading.Thread(target=self.worker)
            t.start()
            threads.append(t)
            
        for t in threads:
            t.join()
            
        return {
            "results": self.results,
            "files_scanned": self.files_scanned,
            "dirs_scanned": self.dirs_scanned,
            "matches_count": len(self.results)
        }

# ----------------------------------------------------------------------
# Helper: Get all Windows Logical Drives
# ----------------------------------------------------------------------
def get_windows_drives():
    import ctypes
    drives = []
    bitmask = ctypes.windll.kernel32.GetLogicalDrives()
    for letter in range(26):
        if bitmask & (1 << letter):
            drive = f"{chr(65 + letter)}:\\"
            drives.append(drive)
    return drives

# ----------------------------------------------------------------------
# Fast Search Runner (Subprocess for C++ or Fallback to Python)
# ----------------------------------------------------------------------
async def run_fast_search(path="all", query="", content="", threads=8, case_sensitive=False, no_skip=False, max_results=1000):
    """
    Search files by name pattern and/or by file content (grep) using C++ executable.
    If the executable is not compiled or fails to run, it falls back to a high-performance Python threaded crawler.
    """
    # 1. Determine roots
    if path == "all":
        roots = get_windows_drives() if os.name == "nt" else ["/"]
    elif isinstance(path, list):
        roots = path
    else:
        roots = [path]

    # Resolve binary path
    project_root = Path(__file__).resolve().parent.parent.parent
    exe_path = project_root / "core" / "tools" / "fast_search" / "fast_search.exe"
    
    # Try using the compiled C++ binary first
    if exe_path.exists():
        args = []
        for r in roots:
            args.extend(["--path", str(r)])
        if query:
            args.extend(["--query", query])
        if content:
            args.extend(["--content", content])
        args.extend(["--threads", str(threads)])
        args.extend(["--max-results", str(max_results)])
        if case_sensitive:
            args.append("--case")
        if no_skip:
            args.append("--no-skip")

        try:
            process = await asyncio.create_subprocess_exec(
                str(exe_path),
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            if process.returncode == 0:
                import re
                ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
                output_lines = stdout.decode('utf-8', errors='replace').splitlines()
                results = []
                summary = {}
                for line in output_lines:
                    line_clean = ansi_escape.sub('', line)
                    if line_clean.startswith("[FILE] "):
                        results.append({
                            "type": "file",
                            "path": line_clean[7:].strip()
                        })
                    elif line_clean.startswith("[MATCH] "):
                        # Format: [MATCH] path:line: content
                        parts = line_clean[8:].split(':', 2)
                        if len(parts) >= 3:
                            results.append({
                                "type": "match",
                                "path": parts[0].strip(),
                                "line": parts[1].strip(),
                                "text": parts[2].strip()
                            })
                    elif "Files scanned" in line_clean:
                        summary["files_scanned"] = line_clean.split(":")[-1].strip()
                    elif "Folders scanned" in line_clean:
                        summary["dirs_scanned"] = line_clean.split(":")[-1].strip()
                    elif "Elapsed time" in line_clean:
                        summary["elapsed"] = line_clean.split(":")[-1].strip()

                return {
                    "engine": "cpp",
                    "results": results,
                    "summary": summary
                }
        except OSError:
            # Fall back silently to Python if process launch fails
            pass

    # 2. Python fallback engine
    engine = PythonSearchEngine(
        start_paths=roots,
        query=query if query else None,
        content_query=content if content else None,
        num_threads=threads * 2, # Spin more threads for Python to hide I/O latency
        case_sensitive=case_sensitive,
        no_skip=no_skip,
        max_results=max_results
    )
    
    start_time = asyncio.get_event_loop().time()
    # Run in standard executor to avoid blocking the asyncio event loop
    loop = asyncio.get_running_loop()
    res = await loop.run_in_executor(None, engine.run)
    elapsed = asyncio.get_event_loop().time() - start_time
    
    res["engine"] = "python"
    res["summary"] = {
        "files_scanned": res["files_scanned"],
        "dirs_scanned": res["dirs_scanned"],
        "elapsed": f"{elapsed:.4f} seconds"
    }
    return res

# main block removed: if __name__ == "__main__":
# main block removed:     # Test script locally
# main block removed:     async def test():
# main block removed:         print("Testing fast search Python/C++ tool...")
# main block removed:         res = await run_fast_search(path=".", query="*.py", threads=4)
# main block removed:         print(f"Engine used: {res['engine']}")
# main block removed:         print(f"Summary: {res['summary']}")
# main block removed:         print(f"Results Count: {len(res['results'])}")
# main block removed:         print("First 3 results:")
# main block removed:         for r in res['results'][:3]:
# main block removed:             print(r)
# main block removed:     asyncio.run(test())




# --- FILE: core/tools/hardware_tools.py ---

"""
core/tools/hardware_tools.py
------------------------------
Async tool functions for interacting with registered hardware devices via
the DeviceRegistry / SerialController layer.

All functions return a core_types_common_ToolResult so they integrate cleanly with the ToolRouter.
"""

# internal import removed: from __future__ import annotations
# internal import removed: from core.types.common import core_types_common_ToolResult

import logging

logger = logging.getLogger(__name__)

# Lazy singleton — created once on first import.
_registry = None


def _get_registry():
    global _registry
    if _registry is None:
        from core.hardware.device_registry import DeviceRegistry
        _registry = DeviceRegistry()
    return _registry


# ── Tool functions ────────────────────────────────────────────────────────────

async def send_hardware_command(
    device_name: str, command: str, value: str = ""
):
    """Send an arbitrary command to a registered hardware device.

    Args:
        device_name: Registered device name (e.g. ``"main_arduino"``).
        command:     Command string (e.g. ``"LIGHT"``).
        value:       Optional value string (e.g. ``"ON"``).
    """
    try:
        device = _get_registry().get_device(device_name)
        result = await device.async_send_command(command, value)
        return core_types_common_ToolResult(success=True, data=result)
    except Exception as e:
        logger.error("send_hardware_command failed: %s", e, exc_info=True)
        return core_types_common_ToolResult(success=False, error=str(e))


async def read_sensor(device_name: str, sensor_type: str = "all"):
    """Request a sensor reading from a registered device.

    Args:
        device_name: Registered device name.
        sensor_type: Sensor identifier (e.g. ``"TEMPERATURE"``). Defaults to ``"all"``.
    """
    try:
        device = _get_registry().get_device(device_name)
        result = await device.async_send_command("READ", sensor_type)
        return core_types_common_ToolResult(success=True, data=result)
    except Exception as e:
        logger.error("read_sensor failed: %s", e, exc_info=True)
        return core_types_common_ToolResult(success=False, error=str(e))


async def list_hardware_devices():
    """Return a list of all registered hardware devices with their status."""
    try:
        devices = _get_registry().list_devices()
        return core_types_common_ToolResult(success=True, data={"devices": devices})
    except Exception as e:
        logger.error("list_hardware_devices failed: %s", e, exc_info=True)
        return core_types_common_ToolResult(success=False, error=str(e))


async def ping_device(device_name: str):
    """Ping a registered device to check if its firmware is responsive.

    Args:
        device_name: Registered device name.
    """
    try:
        device = _get_registry().get_device(device_name)
        alive = await device.firmware_ping()
        return core_types_common_ToolResult(success=True, data={"alive": alive, "device": device_name})
    except Exception as e:
        logger.error("ping_device failed: %s", e, exc_info=True)
        return core_types_common_ToolResult(success=False, error=str(e))


__all__ = [
    "send_hardware_command",
    "read_sensor",
    "list_hardware_devices",
    "ping_device",
]




# --- FILE: core/tools/screen.py ---

"""Screen capture, OCR, and screen-state tools for Jarvis."""

# internal import removed: from __future__ import annotations
# internal import removed: from core.types.common import core_types_common_ToolResult

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SCREENSHOT_DIR = Path("outputs/screenshots")
_OCR_MAX_TEXT_CHARS = 4000
_TESSERACT_WARN_LOGGED = False



def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_dir() -> None:
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)


def core_tools_screen__require_pyautogui():
    try:
        import pyautogui

        return pyautogui
    except ImportError as exc:
        raise ImportError("pyautogui not installed") from exc


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _capture_image(*, region: tuple[int, int, int, int] | None = None):
    pyautogui = core_tools_screen__require_pyautogui()
    return pyautogui.screenshot(region=region)


def _ocr_words_from_data(data: dict[str, Any]) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    count = len(data.get("text", []))
    for idx in range(count):
        text = _clean_text(data["text"][idx])
        if not text:
            continue
        width = _safe_int(data.get("width", [0])[idx])
        height = _safe_int(data.get("height", [0])[idx])
        words.append(
            {
                "text": text,
                "x": _safe_int(data.get("left", [0])[idx]),
                "y": _safe_int(data.get("top", [0])[idx]),
                "w": width,
                "h": height,
                "width": width,
                "height": height,
                "confidence": _safe_float(data.get("conf", [0])[idx]),
                "block_num": _safe_int(data.get("block_num", [0])[idx]),
                "par_num": _safe_int(data.get("par_num", [0])[idx]),
                "line_num": _safe_int(data.get("line_num", [0])[idx]),
                "source": "word",
            }
        )
    return words


def _ocr_lines_from_words(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for word in words:
        key = (
            _safe_int(word.get("block_num")),
            _safe_int(word.get("par_num")),
            _safe_int(word.get("line_num")),
        )
        grouped.setdefault(key, []).append(word)

    lines: list[dict[str, Any]] = []
    for line_words in grouped.values():
        ordered = sorted(line_words, key=lambda item: (item.get("y", 0), item.get("x", 0)))
        if not ordered:
            continue
        left = min(_safe_int(item.get("x")) for item in ordered)
        top = min(_safe_int(item.get("y")) for item in ordered)
        right = max(_safe_int(item.get("x")) + _safe_int(item.get("w")) for item in ordered)
        bottom = max(_safe_int(item.get("y")) + _safe_int(item.get("h")) for item in ordered)
        text = _clean_text(" ".join(str(item.get("text", "")) for item in ordered))
        if not text:
            continue
        lines.append(
            {
                "text": text,
                "x": left,
                "y": top,
                "w": max(0, right - left),
                "h": max(0, bottom - top),
                "width": max(0, right - left),
                "height": max(0, bottom - top),
                "confidence": round(
                    sum(_safe_float(item.get("confidence")) for item in ordered) / max(1, len(ordered)),
                    3,
                ),
                "word_count": len(ordered),
                "source": "line",
            }
        )
    return sorted(lines, key=lambda item: (item.get("y", 0), item.get("x", 0)))


def _match_entries(
    entries: list[dict[str, Any]],
    query: str,
    *,
    match_mode: str = "contains",
) -> list[dict[str, Any]]:
    normalized_query = _clean_text(query).casefold()
    if not normalized_query:
        return []

    matches: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int, int, int]] = set()
    for entry in entries:
        haystack = _clean_text(entry.get("text", "")).casefold()
        if not haystack:
            continue

        matched = False
        if match_mode == "exact":
            matched = haystack == normalized_query
        elif match_mode == "starts_with":
            matched = haystack.startswith(normalized_query)
        else:
            matched = normalized_query in haystack

        if not matched:
            continue

        key = (
            _clean_text(entry.get("text", "")),
            _safe_int(entry.get("x")),
            _safe_int(entry.get("y")),
            _safe_int(entry.get("w", entry.get("width"))),
            _safe_int(entry.get("h", entry.get("height"))),
        )
        if key in seen:
            continue
        seen.add(key)
        matches.append(
            {
                "text": _clean_text(entry.get("text", "")),
                "x": _safe_int(entry.get("x")),
                "y": _safe_int(entry.get("y")),
                "w": _safe_int(entry.get("w", entry.get("width"))),
                "h": _safe_int(entry.get("h", entry.get("height"))),
                "width": _safe_int(entry.get("width", entry.get("w"))),
                "height": _safe_int(entry.get("height", entry.get("h"))),
                "confidence": _safe_float(entry.get("confidence")),
                "source": str(entry.get("source", "word") or "word"),
            }
        )
    return matches


def capture_screen():
    """Take a full-screen screenshot and save it to outputs/screenshots/."""

    try:
        _ensure_dir()
        path = SCREENSHOT_DIR / f"{_ts()}.png"
        img = _capture_image()
        img.save(str(path))
        logger.info("Screenshot saved: %s", path)
        return core_types_common_ToolResult(success=True, data={"path": str(path), "width": img.width, "height": img.height})
    except Exception as exc:  # noqa: BLE001
        logger.error("capture_screen failed: %s", exc, exc_info=True)
        return core_types_common_ToolResult(success=False, error=str(exc))


def capture_region(x: int, y: int, width: int, height: int):
    """Screenshot a specific screen region."""

    try:
        _ensure_dir()
        path = SCREENSHOT_DIR / f"{_ts()}_region.png"
        img = _capture_image(region=(x, y, width, height))
        img.save(str(path))
        logger.info("Region screenshot saved: %s", path)
        return core_types_common_ToolResult(success=True, data={"path": str(path), "x": x, "y": y, "width": width, "height": height})
    except Exception as exc:  # noqa: BLE001
        logger.error("capture_region failed: %s", exc, exc_info=True)
        return core_types_common_ToolResult(success=False, error=str(exc))


def read_screen_text(
    query: str = "",
    *,
    match_mode: str = "contains",
    include_words: bool = False,
    include_lines: bool = False,
    max_text_chars: int = _OCR_MAX_TEXT_CHARS,
):
    """Read visible screen text with OCR and optional query matching."""

    try:
        import os
        import sys
        import pytesseract
        from PIL import Image  # noqa: F401

        # Configure Tesseract path dynamically
        if getattr(sys, "frozen", False):
            # Packaged desktop app mode: use bundled Tesseract binary from temporary directory
            base_dir = getattr(sys, '_MEIPASS')
            tesseract_dir = os.path.join(base_dir, "bin", "tesseract")
            os.environ["TESSDATA_PREFIX"] = os.path.join(tesseract_dir, "tessdata")
            pytesseract.pytesseract.tesseract_cmd = os.path.join(tesseract_dir, "tesseract.exe")
        else:
            # Development mode: check if custom TESSERACT_CMD env variable is set
            tesseract_cmd = os.environ.get("TESSERACT_CMD")
            if tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            else:
                # Check for the local bundled folder inside the workspace
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                local_bundled = os.path.join(project_root, "bin", "tesseract", "tesseract.exe")
                if os.path.exists(local_bundled):
                    pytesseract.pytesseract.tesseract_cmd = local_bundled
                    os.environ["TESSDATA_PREFIX"] = os.path.join(project_root, "bin", "tesseract", "tessdata")
    except ImportError as exc:
        return core_types_common_ToolResult(success=False, error=f"Missing dependency: {exc}")

    try:
        img = _capture_image()
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        words = _ocr_words_from_data(data)
        lines = _ocr_lines_from_words(words)
        full_text = "\n".join(line["text"] for line in lines)

        matches: list[dict[str, Any]] = []
        if _clean_text(query):
            matches.extend(_match_entries(lines, query, match_mode=match_mode))
            matches.extend(_match_entries(words, query, match_mode=match_mode))

        payload: dict[str, Any] = {
            "query": query,
            "text": full_text[:max_text_chars],
            "matches": matches,
            "match_count": len(matches),
            "line_count": len(lines),
            "word_count": len(words),
        }
        if include_lines:
            payload["lines"] = lines
        if include_words:
            payload["words"] = words

        logger.info("read_screen_text(query=%r): %d match(es)", query, len(matches))
        return core_types_common_ToolResult(success=True, data=payload)
    except Exception as exc:  # noqa: BLE001
        global _TESSERACT_WARN_LOGGED
        err_msg = str(exc)
        if "tesseract is not installed" in err_msg or "TesseractNotFoundError" in type(exc).__name__:
            if not _TESSERACT_WARN_LOGGED:
                logger.warning(
                    "read_screen_text: Tesseract OCR is not installed or not in your PATH. "
                    "Screen-aware features will be limited. See README.MD to set it up."
                )
                _TESSERACT_WARN_LOGGED = True
            else:
                logger.debug("read_screen_text failed (Tesseract not installed): %s", exc)
        else:
            logger.error("read_screen_text failed: %s", exc, exc_info=True)
        return core_types_common_ToolResult(success=False, error=err_msg)


def find_text_on_screen(text: str, match_mode: str = "contains"):
    """Search for visible screen text using phrase-aware OCR matching."""

    result = read_screen_text(
        query=text,
        match_mode=match_mode,
        include_lines=False,
        include_words=False,
    )
    if not result.success:
        return result

    return core_types_common_ToolResult(
        success=True,
        data={
            "query": text,
            "match_mode": match_mode,
            "matches": list(result.data.get("matches", [])),
            "text": str(result.data.get("text", "") or ""),
        },
    )


async def wait_for_text_on_screen(
    text: str,
    *,
    timeout_seconds: float = 10.0,
    poll_interval_seconds: float = 0.5,
    match_mode: str = "contains",
):
    """Poll OCR until the requested text appears on screen."""

    timeout_value = max(0.1, float(timeout_seconds))
    deadline = time.monotonic() + timeout_value
    started_at = time.monotonic()
    attempts = 0
    last_error = ""

    while time.monotonic() < deadline:
        attempts += 1
        result = find_text_on_screen(text, match_mode=match_mode)
        if result.success and result.data.get("matches"):
            payload = dict(result.data)
            payload["attempts"] = attempts
            payload["elapsed_seconds"] = round(max(0.0, time.monotonic() - started_at), 3)
            return core_types_common_ToolResult(success=True, data=payload)
        if not result.success:
            last_error = result.error
        await asyncio.sleep(max(0.1, float(poll_interval_seconds)))

    return core_types_common_ToolResult(
        success=False,
        data={
            "query": text,
            "attempts": attempts,
            "elapsed_seconds": round(max(0.0, time.monotonic() - started_at), 3),
        },
        error=last_error or f"Timed out waiting for '{text}' to appear on screen.",
    )


def describe_screen(llm_client=None):
    """Describe the current screen contents."""

    try:
        img = _capture_image()
    except Exception as exc:  # noqa: BLE001
        return core_types_common_ToolResult(success=False, error=f"Screenshot failed: {exc}")

    if llm_client is not None:
        try:
            import base64
            import io

            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            b64 = base64.b64encode(buffer.getvalue()).decode()
            description = llm_client.complete(
                prompt="Describe what you see on this screen briefly.",
                images=[b64],
                task_type="vision",
            )
            return core_types_common_ToolResult(success=True, data={"description": description})
        except Exception:
            pass

    ocr_result = read_screen_text()
    if ocr_result.success:
        return core_types_common_ToolResult(success=True, data={"ocr_text": str(ocr_result.data.get("text", "") or "")})
    if "Missing dependency" in ocr_result.error:
        return core_types_common_ToolResult(success=True, data={"description": "Screen captured but no OCR backend is available."})
    logger.warning("describe_screen OCR failed: %s", ocr_result.error)
    return core_types_common_ToolResult(success=True, data={"description": "Screen captured but OCR failed."})


__all__ = [
    "capture_screen",
    "capture_region",
    "describe_screen",
    "find_text_on_screen",
    "read_screen_text",
    "wait_for_text_on_screen",
]




# --- FILE: core/tools/universal_converter.py ---

import sys
import subprocess
import csv
import json
from pathlib import Path

# ----------------------------------------------------------------------
# Self-Bootstrapping Dependency Manager
# ----------------------------------------------------------------------
def ensure_package(package_name, import_name=None):
    if import_name is None:
        import_name = package_name
    try:
        __import__(import_name)
    except ImportError:
        print(f"[Universal Converter] Installing missing dependency: {package_name}...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", package_name], check=True)
            print(f"[Universal Converter] Successfully installed {package_name}.")
        except Exception as e:
            raise RuntimeError(f"Failed to install package '{package_name}' for conversion: {e}")

# ----------------------------------------------------------------------
# Conversion Implementations
# ----------------------------------------------------------------------
def convert_image(src: Path, dst: Path):
    ensure_package("pillow", "PIL")
    from PIL import Image
    with Image.open(src) as img:
        # Convert RGBA to RGB for formats that do not support transparency (like JPG/BMP)
        if dst.suffix.lower() in [".jpg", ".jpeg", ".bmp"] and img.mode in ("RGBA", "LA", "P"):
            converted_img = img.convert("RGB")
            converted_img.save(dst)
        else:
            img.save(dst)

def csv_to_json(src: Path, dst: Path):
    with open(src, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    with open(dst, 'w', encoding='utf-8') as f:
        json.dump(rows, f, indent=2)

def json_to_csv(src: Path, dst: Path):
    with open(src, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        if isinstance(data, dict):
            # If it's a single dictionary, wrap it in a list
            data = [data]
        else:
            raise ValueError("JSON data must be an array of objects to convert to CSV.")
    if not data:
        raise ValueError("JSON data is empty.")
        
    headers = list(data[0].keys())
    with open(dst, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in data:
            writer.writerow(row)

def json_to_yaml(src: Path, dst: Path):
    ensure_package("pyyaml", "yaml")
    import yaml
    with open(src, 'r', encoding='utf-8') as f:
        data = json.load(f)
    with open(dst, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, default_flow_style=False)

def yaml_to_json(src: Path, dst: Path):
    ensure_package("pyyaml", "yaml")
    import yaml
    with open(src, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    with open(dst, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def md_to_html(src: Path, dst: Path):
    ensure_package("markdown")
    import markdown
    with open(src, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    html = markdown.markdown(text)
    full_html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Converted Document</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      line-height: 1.6;
      max-width: 800px;
      margin: 40px auto;
      padding: 0 20px;
      color: #333;
      background-color: #fafafa;
    }}
    h1, h2, h3 {{ color: #111; border-bottom: 1px solid #eaeaea; padding-bottom: 0.3em; }}
    pre {{ background: #f6f8fa; padding: 16px; border-radius: 6px; overflow-x: auto; font-family: monospace; }}
    code {{ font-family: monospace; background: rgba(27,31,35,0.05); padding: .2em .4em; border-radius: 3px; font-size: 85%; }}
    blockquote {{ border-left: 4px solid #dfe2e5; color: #6a737d; padding-left: 1em; margin-left: 0; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 16px; }}
    th, td {{ border: 1px solid #dfe2e5; padding: 6px 13px; }}
    th {{ background-color: #f6f8fa; }}
  </style>
</head>
<body>
  {html}
</body>
</html>"""
    with open(dst, 'w', encoding='utf-8') as f:
        f.write(full_html)

def txt_to_html(src: Path, dst: Path):
    with open(src, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    import html
    escaped_text = html.escape(text)
    full_html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Plain Text View</title>
</head>
<body style="font-family: monospace; padding: 20px; background: #fafafa; line-height: 1.4;">
  <pre>{escaped_text}</pre>
</body>
</html>"""
    with open(dst, 'w', encoding='utf-8') as f:
        f.write(full_html)

def txt_to_pdf(src: Path, dst: Path):
    ensure_package("fpdf2", "fpdf")
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)
    
    with open(src, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # Replace non-latin1 characters to avoid encoding crashes
            line_clean = line.encode('latin-1', 'replace').decode('latin-1').strip('\n')
            pdf.cell(0, 6, txt=line_clean, ln=1)
            
    pdf.output(str(dst))

def pdf_to_txt(src: Path, dst: Path):
    ensure_package("pypdf")
    from pypdf import PdfReader
    reader = PdfReader(src)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
        
    with open(dst, 'w', encoding='utf-8') as f:
        f.write(text)

def data_to_excel(src: Path, dst: Path):
    ensure_package("pandas")
    ensure_package("openpyxl")
    import pandas as pd
    if src.suffix.lower() == ".csv":
        df = pd.read_csv(src)
    elif src.suffix.lower() == ".json":
        df = pd.read_json(src)
    else:
        raise ValueError(f"Cannot convert format {src.suffix} directly to Excel. Must be CSV or JSON.")
    df.to_excel(dst, index=False)

def convert_media_ffmpeg(src: Path, dst: Path):
    try:
        cmd = ["ffmpeg", "-y", "-i", str(src), str(dst)]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode('utf-8', errors='replace'))
    except FileNotFoundError:
        raise RuntimeError("FFmpeg is not installed or not in system PATH. Audio/Video conversion requires FFmpeg installed.")

# ----------------------------------------------------------------------
# Universal Entry Point
# ----------------------------------------------------------------------
def perform_conversion(source_path: str, target_format: str, output_path: str | None = None) -> str:
    """
    Main entry point to convert files.
    :param source_path: Path of the input file.
    :param target_format: Target format extension (e.g. 'webp', 'pdf', 'csv', 'json').
    :param output_path: Optional custom output file path.
    :return: Path of the converted file.
    """
    src = Path(source_path).resolve()
    if not src.exists():
        raise FileNotFoundError(f"Source file does not exist: {source_path}")
    if not src.is_file():
        raise ValueError(f"Source path is not a file: {source_path}")
        
    ext_src = src.suffix.lower()
    ext_dst = f".{target_format.lower().lstrip('.')}"
    
    if ext_src == ext_dst:
        raise ValueError(f"Source and target format are the same ({ext_src}). No conversion needed.")
        
    if output_path:
        dst = Path(output_path).resolve()
    else:
        dst = src.with_suffix(ext_dst)
        
    dst.parent.mkdir(parents=True, exist_ok=True)
    
    # Define route map
    image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff"}
    audio_video_exts = {".mp3", ".wav", ".ogg", ".m4a", ".flac", ".mp4", ".avi", ".mkv", ".webm", ".mov"}
    
    # Conversion Dispatcher
    try:
        # 1. Images
        if ext_src in image_exts and ext_dst in image_exts:
            convert_image(src, dst)
            
        # 2. Data conversions
        elif ext_src == ".csv" and ext_dst == ".json":
            csv_to_json(src, dst)
        elif ext_src == ".json" and ext_dst == ".csv":
            json_to_csv(src, dst)
        elif ext_src == ".json" and ext_dst in (".yaml", ".yml"):
            json_to_yaml(src, dst)
        elif ext_src in (".yaml", ".yml") and ext_dst == ".json":
            yaml_to_json(src, dst)
        elif ext_src in (".csv", ".json") and ext_dst in (".xlsx", ".xls"):
            data_to_excel(src, dst)
            
        # 3. Documents
        elif ext_src == ".md" and ext_dst in (".html", ".htm"):
            md_to_html(src, dst)
        elif ext_src == ".txt" and ext_dst in (".html", ".htm"):
            txt_to_html(src, dst)
        elif ext_src == ".txt" and ext_dst == ".pdf":
            txt_to_pdf(src, dst)
        elif ext_src == ".pdf" and ext_dst == ".txt":
            pdf_to_txt(src, dst)
            
        # 4. Audio/Video (using FFmpeg)
        elif ext_src in audio_video_exts and ext_dst in audio_video_exts:
            convert_media_ffmpeg(src, dst)
            
        # 5. Generic FFmpeg fallback for other extensions if source/destination might be media
        elif ext_src in audio_video_exts or ext_dst in audio_video_exts:
            convert_media_ffmpeg(src, dst)
            
        else:
            # Catch-all: Try using FFmpeg for everything else if it might work
            try:
                convert_media_ffmpeg(src, dst)
            except Exception as ffmpeg_err:
                raise ValueError(
                    f"Unsupported conversion format pair: {ext_src} -> {ext_dst}. "
                    f"FFmpeg fallback also failed: {ffmpeg_err}"
                )
                
        return str(dst)
        
    except Exception as e:
        raise RuntimeError(f"Error converting {src.name} to {ext_dst}: {e}")

# ----------------------------------------------------------------------
# CLI Runner
# ----------------------------------------------------------------------
# main block removed: if __name__ == "__main__":
# main block removed:     if len(sys.argv) < 3:
# main block removed:         print("Universal File Converter CLI")
# main block removed:         print("Usage: python universal_converter.py <source_file> <target_format> [output_file]")
# main block removed:         sys.exit(1)
# main block removed:         
# main block removed:     src_file = sys.argv[1]
# main block removed:     tgt_fmt = sys.argv[2]
# main block removed:     out_file = sys.argv[3] if len(sys.argv) > 3 else None
# main block removed:     
# main block removed:     try:
# main block removed:         res = perform_conversion(src_file, tgt_fmt, out_file)
# main block removed:         print(f"Success! Converted file saved to: {res}")
# main block removed:     except Exception as e:
# main block removed:         print(f"Error: {e}")
# main block removed:         sys.exit(1)




# --- FILE: core/tools/vision.py ---

"""
core/tools/vision.py — Vision (image analysis) tool.

Wraps Ollama's LLaVA model for image understanding.
Tests can patch _call_llava() to inject fake responses.
"""

# internal import removed: from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}


class VisionTool:
    """Analyze images using LLaVA or a test stub."""

    def __init__(self, config: Any) -> None:
        self._config = config
        self._model = self._get("vision_model", "llava")
        self._base_url = self._get("base_url", "http://localhost:11434")

    def _get(self, key: str, default: str) -> str:
        for section in ("ollama", "vision"):
            try:
                return str(self._config.get(section, key, fallback=default))
            except Exception:  # noqa: BLE001
                pass
        return default

    def analyze(self, image_path: str, prompt: str = "Describe this image.") -> str:
        """
        Analyze *image_path* using LLaVA and return a text description.

        Raises:
            FileNotFoundError: if the image file does not exist.
            ValueError: if the file extension is not a supported image format.
        """
        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        if path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported image format: {path.suffix!r}. "
                f"Supported: {sorted(_SUPPORTED_EXTENSIONS)}"
            )

        return self._call_llava(str(path), prompt)

    def _call_llava(self, image_path: str, prompt: str) -> str:
        """Call the LLaVA model. Override in tests."""
        try:
            import requests
            import base64

            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()

            with requests.post(
                f"{self._base_url}/api/generate",
                json={
                    "model": self._model,
                    "prompt": prompt,
                    "images": [img_b64],
                    "stream": False,
                },
                timeout=30,
            ) as resp:
                resp.raise_for_status()
                return str(resp.json().get("response", ""))
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLaVA call failed: %s", exc)
            return f"[Vision unavailable: {exc}]"


__all__ = ["VisionTool"]




# --- FILE: core/tools/web_tools.py ---

"""
Configurable web search and web scraping tools for Jarvis.
"""

# internal import removed: from __future__ import annotations

import asyncio
import configparser
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Any, cast

try:
    from duckduckgo_search import DDGS
except ImportError:  # pragma: no cover - optional dependency at runtime
    DDGS = None

try:
    import requests
except ImportError:  # pragma: no cover - optional dependency at runtime
    requests = None  # type: ignore[assignment]

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover - optional dependency at runtime
    BeautifulSoup = None  # type: ignore

logger = logging.getLogger("Jarvis.WebTools")

core_tools_web_tools_PROJECT_ROOT = Path(__file__).resolve().parents[2]
core_tools_web_tools_DEFAULT_CONFIG_PATH = core_tools_web_tools_PROJECT_ROOT / "config" / "jarvis.ini"
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}
QUERY_EXTRACTION_SYSTEM = (
    "You convert conversational requests into concise web search queries. "
    "Return only the query text, no quotes, no bullets, no explanation."
)
SEARCH_SUMMARY_SYSTEM = (
    "You summarize web search results for a local AI assistant. "
    "Write 2-4 short sentences grounded only in the provided results. "
    "Do not invent facts. Mention uncertainty if results conflict."
)


class SupportsQuickLLM(Protocol):
    """Minimal protocol used by the web tools for fast internal tasks."""

    async def complete(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.0,
        task_type: str = "chat",
        keep_think: bool = False,
    ) -> str:
        """Return a plain-text completion."""


@dataclass(frozen=True)
class SearchSettings:
    """Runtime settings for the web tools."""

    enabled: bool = True
    provider: str = "auto"
    default_max_results: int = 5
    summarize_results: bool = True
    auto_extract_query: bool = True
    provider_timeout_s: float = 8.0
    scrape_timeout_s: float = 10.0
    quick_task_timeout_s: float = 4.0
    max_scrape_chars: int = 8000
    ddgs_region: str = "wt-wt"
    ddgs_safesearch: str = "moderate"
    tavily_api_key: str = ""

    @classmethod
    def from_sources(
        cls,
        config: configparser.ConfigParser | None = None,
    ) -> "SearchSettings":
        """Load settings from config and environment variables."""
        resolved_config = config if config is not None else _load_default_config()

        def _resolve(
            env_key: str,
            opt: str,
            parser_type: str,
            fallback: Any,
            section: str = "web_search",
        ) -> Any:
            env_val = os.environ.get(env_key)
            if env_val is not None and env_val.strip():
                raw = env_val.strip()
                if parser_type == "bool":
                    return raw.lower() in {"1", "true", "yes", "on"}
                elif parser_type == "int":
                    try:
                        return int(raw)
                    except ValueError:
                        pass
                elif parser_type == "float":
                    try:
                        return float(raw)
                    except ValueError:
                        pass
                else:
                    return raw

            if resolved_config is not None and resolved_config.has_section(section):
                if parser_type == "bool":
                    try:
                        return resolved_config.getboolean(
                            section, opt, fallback=fallback
                        )
                    except ValueError:
                        pass
                elif parser_type == "int":
                    try:
                        return resolved_config.getint(section, opt, fallback=fallback)
                    except ValueError:
                        pass
                elif parser_type == "float":
                    try:
                        return resolved_config.getfloat(section, opt, fallback=fallback)
                    except ValueError:
                        pass
                else:
                    return resolved_config.get(section, opt, fallback=fallback)

            return fallback

        fallback_enabled = _resolve(
            "", "allow_web_search", "bool", True, section="execution"
        )
        enabled = _resolve("WEB_SEARCH_ENABLED", "enabled", "bool", fallback_enabled)
        provider = str(_resolve("WEB_SEARCH_PROVIDER", "provider", "str", "auto")).lower()
        default_max_results = _resolve(
            "WEB_SEARCH_DEFAULT_MAX_RESULTS", "default_max_results", "int", 5
        )
        summarize_results = _resolve(
            "WEB_SEARCH_SUMMARIZE_RESULTS", "summarize_results", "bool", True
        )
        auto_extract_query = _resolve(
            "WEB_SEARCH_AUTO_EXTRACT_QUERY", "auto_extract_query", "bool", True
        )
        provider_timeout_s = _resolve(
            "WEB_SEARCH_PROVIDER_TIMEOUT_S", "provider_timeout_s", "float", 8.0
        )
        scrape_timeout_s = _resolve(
            "WEB_SEARCH_SCRAPE_TIMEOUT_S", "scrape_timeout_s", "float", 10.0
        )
        quick_task_timeout_s = _resolve(
            "WEB_SEARCH_QUICK_TASK_TIMEOUT_S", "quick_task_timeout_s", "float", 4.0
        )
        max_scrape_chars = _resolve(
            "WEB_SEARCH_MAX_SCRAPE_CHARS", "max_scrape_chars", "int", 8000
        )
        ddgs_region = _resolve("WEB_SEARCH_DDGS_REGION", "ddgs_region", "str", "wt-wt")
        ddgs_safesearch = _resolve(
            "WEB_SEARCH_DDGS_SAFESEARCH", "ddgs_safesearch", "str", "moderate"
        )
        tavily_api_key = _resolve("TAVILY_API_KEY", "tavily_api_key", "str", "")

        return cls(
            enabled=enabled,
            provider=provider or "auto",
            default_max_results=max(1, min(default_max_results, 10)),
            summarize_results=summarize_results,
            auto_extract_query=auto_extract_query,
            provider_timeout_s=max(1.0, provider_timeout_s),
            scrape_timeout_s=max(1.0, scrape_timeout_s),
            quick_task_timeout_s=max(1.0, quick_task_timeout_s),
            max_scrape_chars=max(500, max_scrape_chars),
            ddgs_region=ddgs_region or "wt-wt",
            ddgs_safesearch=ddgs_safesearch or "moderate",
            tavily_api_key=str(tavily_api_key).strip(),
        )


@dataclass(frozen=True)
class SearchResult:
    """Normalized search result returned by a search provider."""

    title: str
    url: str
    snippet: str
    provider: str


class WebToolService:
    """Configurable service backing the public web tool functions."""

    def __init__(
        self,
        settings: SearchSettings,
        llm: SupportsQuickLLM | None = None,
    ) -> None:
        self.settings = settings
        self.llm = llm

    async def web_search(self, query: str, max_results: int = 5) -> str:
        """Perform a web search and return a source-grounded summary."""
        if not self.settings.enabled:
            return "Web search is disabled by configuration."

        raw_query = _normalize_whitespace(query)
        if not raw_query:
            return "Search failed: query is empty."

        result_limit = max(
            1, min(int(max_results or self.settings.default_max_results), 10)
        )
        effective_query = await self._extract_search_query(raw_query)

        try:
            results = await self._search(effective_query, result_limit)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Web search failed for %r: %s", effective_query, exc, exc_info=True
            )
            return f"Search failed: {exc}"

        if not results:
            return f"No results found for query: {effective_query}"

        summary = await self._summarize_results(raw_query, effective_query, results)
        return self._format_search_output(
            original_query=raw_query,
            effective_query=effective_query,
            results=results,
            summary=summary,
        )

    async def web_scrape(self, url: str, max_chars: int = 8000) -> str:
        """Fetch and extract readable text from a web page."""
        if requests is None or BeautifulSoup is None:
            return (
                "Error: requests and beautifulsoup4 must be installed for web scraping."
            )

        target_url = str(url or "").strip()
        if not target_url:
            return "Scraping failed: url is empty."

        max_chars = max(500, int(max_chars or self.settings.max_scrape_chars))

        try:
            text = await asyncio.to_thread(self._scrape_page, target_url)
        except Exception as exc:  # noqa: BLE001
            logger.error("Web scrape failed for %s: %s", target_url, exc, exc_info=True)
            return f"Scraping failed: {exc}"

        if not text:
            return "Failed to extract readable text from the page."
        if len(text) > max_chars:
            return text[:max_chars] + f"\n\n...[Truncated, total {len(text)} chars]..."
        return text

    async def _search(self, query: str, max_results: int) -> list[SearchResult]:
        """Search with the configured provider chain."""
        errors: list[str] = []
        for provider in self._provider_chain():
            try:
                if provider == "tavily":
                    return await asyncio.to_thread(
                        self._search_with_tavily,
                        query,
                        max_results,
                    )
                if provider == "ddgs":
                    return await asyncio.to_thread(
                        self._search_with_ddgs,
                        query,
                        max_results,
                    )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{provider}: {exc}")
                logger.warning("Search provider %s failed: %s", provider, exc)

        if errors:
            raise RuntimeError("; ".join(errors))
        return []

    def _provider_chain(self) -> list[str]:
        provider = (self.settings.provider or "auto").strip().lower()
        if provider == "tavily":
            return ["tavily", "ddgs"]
        if provider == "ddgs":
            return ["ddgs"]

        providers: list[str] = []
        if self.settings.tavily_api_key:
            providers.append("tavily")
        providers.append("ddgs")
        return providers

    def _search_with_ddgs(self, query: str, max_results: int) -> list[SearchResult]:
        if DDGS is None:
            raise RuntimeError("ddgs package is not installed.")

        with DDGS() as ddgs:
            try:
                raw_results = list(
                    ddgs.text(
                        query,
                        region=self.settings.ddgs_region,
                        safesearch=self.settings.ddgs_safesearch,
                        max_results=max_results,
                    )
                )
            except TypeError:
                raw_results = list(ddgs.text(query, max_results=max_results))

        return [
            SearchResult(
                title=_normalize_whitespace(str(item.get("title", "No Title"))),
                url=_normalize_whitespace(str(item.get("href", "No URL"))),
                snippet=_normalize_whitespace(str(item.get("body", ""))),
                provider="ddgs",
            )
            for item in raw_results
            if item
        ]

    def _search_with_tavily(self, query: str, max_results: int) -> list[SearchResult]:
        if requests is None:
            raise RuntimeError("requests package is not installed.")
        if not self.settings.tavily_api_key:
            raise RuntimeError("Tavily API key is not configured.")

        payload = {
            "api_key": self.settings.tavily_api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": max_results,
            "include_answer": False,
            "include_raw_content": False,
        }
        with requests.post(
            "https://api.tavily.com/search",
            json=cast(Any, payload),
            timeout=self.settings.provider_timeout_s,
            headers=DEFAULT_HEADERS,
        ) as response:
            response.raise_for_status()
            data = response.json()
        raw_results = data.get("results", [])

        return [
            SearchResult(
                title=_normalize_whitespace(str(item.get("title", "No Title"))),
                url=_normalize_whitespace(str(item.get("url", "No URL"))),
                snippet=_normalize_whitespace(str(item.get("content", ""))),
                provider="tavily",
            )
            for item in raw_results
            if item
        ]

    def _scrape_page(self, url: str) -> str:
        if requests is None or BeautifulSoup is None:
            raise RuntimeError("requests and beautifulsoup4 must be installed.")

        with requests.get(
            url,
            headers=DEFAULT_HEADERS,
            timeout=self.settings.scrape_timeout_s,
        ) as response:
            response.raise_for_status()
            content = response.content

        soup = BeautifulSoup(content, "html.parser")
        for element in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            element.decompose()

        text = soup.get_text(separator="\n")
        lines = (line.strip() for line in text.splitlines())
        chunks = (piece.strip() for line in lines for piece in line.split("  "))
        return "\n".join(chunk for chunk in chunks if chunk)

    async def _extract_search_query(self, raw_query: str) -> str:
        cleaned = _basic_query_cleanup(raw_query)
        if not self.settings.auto_extract_query or self.llm is None:
            return cleaned
        if not _needs_query_extraction(cleaned):
            return cleaned

        prompt = (
            "User request:\n"
            f"{raw_query}\n\n"
            "Return the single best web search query for this request."
        )
        try:
            response = await asyncio.wait_for(
                self.llm.complete(
                    prompt,
                    system=QUERY_EXTRACTION_SYSTEM,
                    temperature=0.0,
                    task_type="tool_parameter_extraction",
                ),
                timeout=self.settings.quick_task_timeout_s,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Search query extraction failed: %s", exc)
            return cleaned

        extracted = _clean_llm_line(response)
        return extracted or cleaned

    async def _summarize_results(
        self,
        original_query: str,
        effective_query: str,
        results: list[SearchResult],
    ) -> str:
        if not self.settings.summarize_results:
            return ""
        if self.llm is None:
            return _fallback_summary(results)

        serialized_results = "\n".join(
            f"{idx}. {item.title}\nURL: {item.url}\nSnippet: {item.snippet}"
            for idx, item in enumerate(results, start=1)
        )
        prompt = (
            f"Original request: {original_query}\n"
            f"Search query used: {effective_query}\n\n"
            f"Results:\n{serialized_results}"
        )
        try:
            response = await asyncio.wait_for(
                self.llm.complete(
                    prompt,
                    system=SEARCH_SUMMARY_SYSTEM,
                    temperature=0.1,
                    task_type="web_search_summary",
                ),
                timeout=self.settings.quick_task_timeout_s,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Search result summarization failed: %s", exc)
            return _fallback_summary(results)

        return _normalize_whitespace(response) or _fallback_summary(results)

    @staticmethod
    def _format_search_output(
        *,
        original_query: str,
        effective_query: str,
        results: list[SearchResult],
        summary: str,
    ) -> str:
        lines = [f"Search query used: {effective_query}"]
        if effective_query != original_query:
            lines.append(f"Original request: {original_query}")
        if summary:
            lines.append(f"Summary: {summary}")
        lines.append("Sources:")

        for idx, result in enumerate(results, start=1):
            lines.append(f"{idx}. {result.title}")
            lines.append(f"URL: {result.url}")
            if result.snippet:
                lines.append(f"Snippet: {result.snippet}")

        return "\n".join(lines)


_SERVICE: WebToolService | None = None


def configure_web_tools(
    *,
    config: configparser.ConfigParser | None = None,
    llm: SupportsQuickLLM | None = None,
) -> WebToolService:
    """Configure the process-wide web tool service used by the tool router."""
    global _SERVICE
    _SERVICE = WebToolService(SearchSettings.from_sources(config), llm=llm)
    return _SERVICE


def _get_service() -> WebToolService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = WebToolService(SearchSettings.from_sources())
    return _SERVICE


async def web_search(query: str, max_results: int = 5) -> str:
    """Perform a web search using the configured provider chain."""
    return await _get_service().web_search(query, max_results=max_results)


async def web_scrape(url: str, max_chars: int = 8000) -> str:
    """Fetch and extract readable text from a webpage."""
    return await _get_service().web_scrape(url, max_chars=max_chars)


def _load_default_config() -> configparser.ConfigParser | None:
    if not core_tools_web_tools_DEFAULT_CONFIG_PATH.exists():
        return None

    parser = configparser.ConfigParser()
    try:
        with core_tools_web_tools_DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as handle:
            parser.read_file(handle)
    except OSError:
        return None
    return parser


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def _basic_query_cleanup(text: str) -> str:
    cleaned = _normalize_whitespace(text)
    cleaned = re.sub(
        r"^(please\s+)?(search|look up|find|check|google|browse|search for)\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\s+(for me|online|on the web)$", "", cleaned, flags=re.IGNORECASE
    )
    return cleaned.strip(" .?!")


def _needs_query_extraction(query: str) -> bool:
    if len(query.split()) <= 6 and not re.search(r"[,:;?!]", query):
        return False
    return any(
        token in query.lower()
        for token in (
            "please",
            "latest",
            "search",
            "look up",
            "find",
            "check",
            "what is",
            "who is",
            "summarize",
        )
    )


def _clean_llm_line(text: str) -> str:
    cleaned = _normalize_whitespace(text)
    if not cleaned:
        return ""

    cleaned = re.sub(r"^['\"`]+|['\"`]+$", "", cleaned)

    if cleaned.startswith("{") and cleaned.endswith("}"):
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                for key in ("query", "search_query", "value"):
                    value = parsed.get(key)
                    if isinstance(value, str) and value.strip():
                        cleaned = value.strip()
                        break
        except json.JSONDecodeError:
            pass

    first_line = cleaned.splitlines()[0].strip()
    first_line = re.sub(r"^[\-\d\.\)\s]+", "", first_line)
    return first_line[:160].strip(" .")


def _fallback_summary(results: list[SearchResult]) -> str:
    top_titles = [result.title for result in results[:3] if result.title]
    if not top_titles:
        return ""
    return "Top matches: " + "; ".join(top_titles)


__all__ = [
    "SearchResult",
    "SearchSettings",
    "WebToolService",
    "configure_web_tools",
    "web_scrape",
    "web_search",
]




# --- FILE: core/voice/__init__.py ---





# --- FILE: core/voice/audio.py ---

import os
from functools import lru_cache
from subprocess import CalledProcessError, run
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

def exact_div(x, y):
    assert x % y == 0
    return x // y

# hard-coded audio hyperparameters
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk
N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)  # 3000 frames in a mel spectrogram input

N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2  # the initial convolutions has stride 2
FRAMES_PER_SECOND = exact_div(SAMPLE_RATE, HOP_LENGTH)  # 10ms per audio frame
TOKENS_PER_SECOND = exact_div(SAMPLE_RATE, N_SAMPLES_PER_TOKEN)  # 20ms per audio token


def load_audio(file: str, sr: int = SAMPLE_RATE):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """

    # This launches a subprocess to decode audio while down-mixing
    # and resampling as necessary.  Requires the ffmpeg CLI in PATH.
    # fmt: off
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", file,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-"
    ]
    # fmt: on
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(
                dim=axis, index=torch.arange(length, device=array.device)
            )

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)

    return array


@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int) -> torch.Tensor:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:

        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
            mel_128=librosa.filters.mel(sr=16000, n_fft=400, n_mels=128),
        )
    """
    assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}"

    filters_path = os.path.join(os.path.dirname(__file__), "assets", "mel_filters.npz")
    with np.load(filters_path, allow_pickle=False) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)


def log_mel_spectrogram(
    audio: Union[str, np.ndarray, torch.Tensor],
    n_mels: int = 80,
    padding: int = 0,
    device: Optional[Union[str, torch.device]] = None,
):
    """
    Compute the log-Mel spectrogram of

    Parameters
    ----------
    audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
        The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz

    n_mels: int
        The number of Mel-frequency filters, only 80 and 128 are supported

    padding: int
        Number of zero samples to pad to the right

    device: Optional[Union[str, torch.device]]
        If given, the audio tensor is moved to this device before STFT

    Returns
    -------
    torch.Tensor, shape = (n_mels, n_frames)
        A Tensor that contains the Mel spectrogram
    """
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)

    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    filters = mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec





# --- FILE: core/voice/audio_input.py ---

# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022-2026)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# internal import removed: from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent
from typing import TYPE_CHECKING, TypeAlias, cast

from streamlit.elements.lib.file_uploader_utils import enforce_filename_restriction
from streamlit.elements.lib.form_utils import current_form_id
from streamlit.elements.lib.layout_utils import LayoutConfig, validate_width
from streamlit.elements.lib.policies import (
    check_widget_policies,
    maybe_raise_label_warnings,
)
from streamlit.elements.lib.utils import (
    Key,
    LabelVisibility,
    compute_and_register_element_id,
    get_label_visibility_proto_value,
    to_key,
)
from streamlit.elements.widgets.file_uploader import _get_upload_files
from streamlit.errors import StreamlitAPIException
from streamlit.proto.AudioInput_pb2 import AudioInput as AudioInputProto
from streamlit.proto.Common_pb2 import FileUploaderState as FileUploaderStateProto
from streamlit.proto.Common_pb2 import UploadedFileInfo as UploadedFileInfoProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import ScriptRunContext, get_script_run_ctx
from streamlit.runtime.state import (
    WidgetArgs,
    WidgetCallback,
    WidgetKwargs,
    register_widget,
)
from streamlit.runtime.uploaded_file_manager import DeletedFile, UploadedFile

if TYPE_CHECKING:
    from streamlit.delta_generator import DeltaGenerator
    from streamlit.elements.lib.layout_utils import WidthWithoutContent

SomeUploadedAudioFile: TypeAlias = UploadedFile | DeletedFile | None

# Allowed sample rates for audio recording
ALLOWED_SAMPLE_RATES = {8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000}


@dataclass
class AudioInputSerde:
    def serialize(
        self,
        audio_file: SomeUploadedAudioFile,
    ) -> FileUploaderStateProto:
        state_proto = FileUploaderStateProto()

        if audio_file is None or isinstance(audio_file, DeletedFile):
            return state_proto

        file_info: UploadedFileInfoProto = state_proto.uploaded_file_info.add()
        file_info.file_id = audio_file.file_id
        file_info.name = audio_file.name
        file_info.size = audio_file.size
        file_info.file_urls.CopyFrom(audio_file._file_urls)

        return state_proto

    def deserialize(
        self, ui_value: FileUploaderStateProto | None
    ) -> SomeUploadedAudioFile:
        upload_files = _get_upload_files(ui_value)
        return_value = None if len(upload_files) == 0 else upload_files[0]
        if return_value is not None and not isinstance(return_value, DeletedFile):
            enforce_filename_restriction(return_value.name, [".wav"])
        return return_value


class AudioInputMixin:
    @gather_metrics("audio_input")
    def audio_input(
        self,
        label: str,
        *,
        sample_rate: int | None = 16000,
        key: Key | None = None,
        help: str | None = None,
        on_change: WidgetCallback | None = None,
        args: WidgetArgs | None = None,
        kwargs: WidgetKwargs | None = None,
        disabled: bool = False,
        label_visibility: LabelVisibility = "visible",
        width: WidthWithoutContent = "stretch",
    ) -> UploadedFile | None:
        r"""Display a widget that returns an audio recording from the user's microphone.

        Parameters
        ----------
        label : str
            A short label explaining to the user what this widget is used for.
            The label can optionally contain GitHub-flavored Markdown of the
            following types: Bold, Italics, Strikethroughs, Inline Code, Links,
            and Images. Images display like icons, with a max height equal to
            the font height.

            Unsupported Markdown elements are unwrapped so only their children
            (text contents) render. Display unsupported elements as literal
            characters by backslash-escaping them. E.g.,
            ``"1\. Not an ordered list"``.

            See the ``body`` parameter of |st.markdown|_ for additional,
            supported Markdown directives.

            For accessibility reasons, you should never set an empty label, but
            you can hide it with ``label_visibility`` if needed. In the future,
            we may disallow empty labels by raising an exception.

            .. |st.markdown| replace:: ``st.markdown``
            .. _st.markdown: https://docs.streamlit.io/develop/api-reference/text/st.markdown

        sample_rate : int or None
            The target sample rate for the audio recording in Hz. This
            defaults to ``16000``, which is optimal for speech recognition.

            The following values are supported: ``8000`` (telephone quality),
            ``11025``, ``16000`` (speech-recognition quality), ``22050``,
            ``24000``, ``32000``, ``44100``, ``48000`` (high-quality), or
            ``None``. If this is ``None``, the widget uses the browser's
            default sample rate (typically 44100 or 48000 Hz).

        key : str or int
            An optional string or integer to use as the unique key for the widget.
            If this is omitted, a key will be generated for the widget
            based on its content. No two widgets may have the same key.

        help : str or None
            A tooltip that gets displayed next to the widget label. Streamlit
            only displays the tooltip when ``label_visibility="visible"``. If
            this is ``None`` (default), no tooltip is displayed.

            The tooltip can optionally contain GitHub-flavored Markdown,
            including the Markdown directives described in the ``body``
            parameter of ``st.markdown``.

        on_change : callable
            An optional callback invoked when this audio input's value
            changes.

        args : list or tuple
            An optional list or tuple of args to pass to the callback.

        kwargs : dict
            An optional dict of kwargs to pass to the callback.

        disabled : bool
            An optional boolean that disables the audio input if set to
            ``True``. Default is ``False``.

        label_visibility : "visible", "hidden", or "collapsed"
            The visibility of the label. The default is ``"visible"``. If this
            is ``"hidden"``, Streamlit displays an empty spacer instead of the
            label, which can help keep the widget aligned with other widgets.
            If this is ``"collapsed"``, Streamlit displays no label or spacer.

        width : "stretch" or int
            The width of the audio input widget. This can be one of the following:

            - ``"stretch"`` (default): The width of the widget matches the
              width of the parent container.
            - An integer specifying the width in pixels: The widget has a
              fixed width. If the specified width is greater than the width of
              the parent container, the width of the widget matches the width
              of the parent container.

        Returns
        -------
        None or UploadedFile
            The ``UploadedFile`` class is a subclass of ``BytesIO``, and
            therefore is "file-like". This means you can pass an instance of it
            anywhere a file is expected. The MIME type for the audio data is
            ``audio/wav``.

            .. Note::
                The resulting ``UploadedFile`` is subject to the size
                limitation configured in ``server.maxUploadSize``. If you
                expect large sound files, update the configuration option
                appropriately.

        Examples
        --------
        *Example 1:* Record a voice message and play it back.*

        The default sample rate of 16000 Hz is optimal for speech recognition.

        >>> import streamlit as st
        >>>
        >>> audio_value = st.audio_input("Record a voice message")
        >>>
        >>> if audio_value:
        ...     st.audio(audio_value)

        .. output::
           https://doc-audio-input.streamlit.app/
           height: 260px

        *Example 2:* Record high-fidelity audio and play it back.*

        Higher sample rates can create higher-quality, larger audio files. This
        might require a nicer microphone to fully appreciate the difference.

        >>> import streamlit as st
        >>>
        >>> audio_value = st.audio_input("Record high quality audio", sample_rate=48000)
        >>>
        >>> if audio_value:
        ...     st.audio(audio_value)

        .. output::
           https://doc-audio-input-high-rate.streamlit.app/
           height: 260px

        """
        # Validate sample_rate parameter
        if sample_rate is not None and sample_rate not in ALLOWED_SAMPLE_RATES:
            raise StreamlitAPIException(
                f"Invalid sample_rate: {sample_rate}. "
                f"Must be one of {sorted(ALLOWED_SAMPLE_RATES)} Hz, or None for browser default."
            )

        ctx = get_script_run_ctx()
        return self._audio_input(
            label=label,
            sample_rate=sample_rate,
            key=key,
            help=help,
            on_change=on_change,
            args=args,
            kwargs=kwargs,
            disabled=disabled,
            label_visibility=label_visibility,
            width=width,
            ctx=ctx,
        )

    def _audio_input(
        self,
        label: str,
        sample_rate: int | None = 16000,
        key: Key | None = None,
        help: str | None = None,
        on_change: WidgetCallback | None = None,
        args: WidgetArgs | None = None,
        kwargs: WidgetKwargs | None = None,
        *,  # keyword-only arguments:
        disabled: bool = False,
        label_visibility: LabelVisibility = "visible",
        width: WidthWithoutContent = "stretch",
        ctx: ScriptRunContext | None = None,
    ) -> UploadedFile | None:
        key = to_key(key)

        check_widget_policies(
            self.dg,
            key,
            on_change,
            default_value=None,
            writes_allowed=False,
        )
        maybe_raise_label_warnings(label, label_visibility)

        element_id = compute_and_register_element_id(
            "audio_input",
            user_key=key,
            # Treat the provided key as the main identity.
            key_as_main_identity=True,
            dg=self.dg,
            label=label,
            help=help,
            width=width,
            sample_rate=sample_rate,
        )

        audio_input_proto = AudioInputProto()
        audio_input_proto.id = element_id
        audio_input_proto.label = label
        audio_input_proto.form_id = current_form_id(self.dg)
        audio_input_proto.disabled = disabled
        audio_input_proto.label_visibility.value = get_label_visibility_proto_value(
            label_visibility
        )

        # Set sample_rate in protobuf if specified
        if sample_rate is not None:
            audio_input_proto.sample_rate = sample_rate

        if label and help is not None:
            audio_input_proto.help = dedent(help)

        validate_width(width)
        layout_config = LayoutConfig(width=width)

        serde = AudioInputSerde()

        audio_input_state = register_widget(
            audio_input_proto.id,
            on_change_handler=on_change,
            args=args,
            kwargs=kwargs,
            deserializer=serde.deserialize,
            serializer=serde.serialize,
            ctx=ctx,
            value_type="file_uploader_state_value",
        )

        self.dg._enqueue("audio_input", audio_input_proto, layout_config=layout_config)

        if isinstance(audio_input_state.value, DeletedFile):
            return None
        return audio_input_state.value

    @property
    def dg(self) -> DeltaGenerator:
        """Get our DeltaGenerator."""
        return cast("DeltaGenerator", self)





# --- FILE: core/voice/audio_playback.py ---

"""Audio playback using ffplay."""

import shutil
import subprocess
from typing import Optional


class AudioPlayer:
    """Plays raw audio using ffplay."""

    def __init__(self, sample_rate: int) -> None:
        """Initializes audio player."""
        self.sample_rate = sample_rate
        self._proc: Optional[subprocess.Popen] = None

    def __enter__(self):
        """Starts ffplay subprocess and returns player."""
        self._proc = subprocess.Popen(
            [
                "ffplay",
                "-nodisp",
                "-autoexit",
                "-f",
                "s16le",
                "-sample_rate",
                str(self.sample_rate),
                "-ch_layout",
                "mono",
                "-",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stops ffplay subprocess."""
        if self._proc:
            try:
                if self._proc.stdin:
                    self._proc.stdin.close()
            except Exception:
                pass
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    self._proc.kill()
                    self._proc.wait()
                except Exception:
                    pass

    def play(self, audio_bytes: bytes) -> None:
        """Plays raw audio using ffplay."""
        assert self._proc is not None
        assert self._proc.stdin is not None

        self._proc.stdin.write(audio_bytes)
        self._proc.stdin.flush()

    @staticmethod
    def is_available() -> bool:
        """Returns true if ffplay is available."""
        return bool(shutil.which("ffplay"))





# --- FILE: core/voice/audio_utils.py ---

# Copyright 2023 The HuggingFace Team. All rights reserved.
import datetime
import platform
import subprocess

import numpy as np
from typing import Any


def ffmpeg_read(bpayload: bytes, sampling_rate: int) -> np.ndarray:
    """
    Helper function to read an audio file through ffmpeg.
    """
    ar = f"{sampling_rate}"
    ac = "1"
    format_for_conversion = "f32le"
    ffmpeg_command = [
        "ffmpeg",
        "-i",
        "pipe:0",
        "-ac",
        ac,
        "-ar",
        ar,
        "-f",
        format_for_conversion,
        "-hide_banner",
        "-loglevel",
        "quiet",
        "pipe:1",
    ]

    try:
        with subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE) as ffmpeg_process:
            output_stream = ffmpeg_process.communicate(bpayload)
    except FileNotFoundError as error:
        raise ValueError("ffmpeg was not found but is required to load audio files from filename") from error
    out_bytes = output_stream[0]
    audio = np.frombuffer(out_bytes, np.float32)
    if audio.shape[0] == 0:
        raise ValueError(
            "Soundfile is either not in the correct format or is malformed. Ensure that the soundfile has "
            "a valid audio file extension (e.g. wav, flac or mp3) and is not corrupted. If reading from a remote "
            "URL, ensure that the URL is the full address to **download** the audio file."
        )
    return audio


def ffmpeg_microphone(
    sampling_rate: int,
    chunk_length_s: float,
    format_for_conversion: str = "f32le",
    ffmpeg_input_device: str | None = None,
    ffmpeg_additional_args: list[str] | None = None,
):
    """
    Helper function to read audio from a microphone using ffmpeg. The default input device will be used unless another
    input device is specified using the `ffmpeg_input_device` argument. Uses 'alsa' on Linux, 'avfoundation' on MacOS and
    'dshow' on Windows.

    Arguments:
        sampling_rate (`int`):
            The sampling_rate to use when reading the data from the microphone. Try using the model's sampling_rate to
            avoid resampling later.
        chunk_length_s (`float` or `int`):
            The length of the maximum chunk of audio to be sent returned.
        format_for_conversion (`str`, defaults to `f32le`):
            The name of the format of the audio samples to be returned by ffmpeg. The standard is `f32le`, `s16le`
            could also be used.
        ffmpeg_input_device (`str`, *optional*):
            The identifier of the input device to be used by ffmpeg (i.e. ffmpeg's '-i' argument). If unset,
            the default input device will be used. See `https://www.ffmpeg.org/ffmpeg-devices.html#Input-Devices`
            for how to specify and list input devices.
        ffmpeg_additional_args (`list[str]`, *optional*):
            Additional arguments to pass to ffmpeg, can include arguments like -nostdin for running as a background
            process. For example, to pass -nostdin to the ffmpeg process, pass in ["-nostdin"]. If passing in flags
            with multiple arguments, use the following convention (eg ["flag", "arg1", "arg2]).

    Returns:
        A generator yielding audio chunks of `chunk_length_s` seconds as `bytes` objects of length
        `int(round(sampling_rate * chunk_length_s)) * size_of_sample`.
    """
    ar = f"{sampling_rate}"
    ac = "1"
    if format_for_conversion == "s16le":
        size_of_sample = 2
    elif format_for_conversion == "f32le":
        size_of_sample = 4
    else:
        raise ValueError(f"Unhandled format `{format_for_conversion}`. Please use `s16le` or `f32le`")

    system = platform.system()

    if system == "Linux":
        format_ = "alsa"
        input_ = ffmpeg_input_device or "default"
    elif system == "Darwin":
        format_ = "avfoundation"
        input_ = ffmpeg_input_device or ":default"
    elif system == "Windows":
        format_ = "dshow"
        input_ = ffmpeg_input_device or _get_microphone_name()

    ffmpeg_additional_args = [] if ffmpeg_additional_args is None else ffmpeg_additional_args

    ffmpeg_command = [
        "ffmpeg",
        "-f",
        format_,
        "-i",
        input_,
        "-ac",
        ac,
        "-ar",
        ar,
        "-f",
        format_for_conversion,
        "-fflags",
        "nobuffer",
        "-hide_banner",
        "-loglevel",
        "quiet",
        "pipe:1",
    ]

    ffmpeg_command.extend(ffmpeg_additional_args)

    chunk_len = int(round(sampling_rate * chunk_length_s)) * size_of_sample
    iterator = _ffmpeg_stream(ffmpeg_command, chunk_len)
    yield from iterator


def ffmpeg_microphone_live(
    sampling_rate: int,
    chunk_length_s: float,
    stream_chunk_s: int | None = None,
    stride_length_s: tuple[float, float] | float | None = None,
    format_for_conversion: str = "f32le",
    ffmpeg_input_device: str | None = None,
    ffmpeg_additional_args: list[str] | None = None,
):
    """
    Helper function to read audio from a microphone using ffmpeg. This will output `partial` overlapping chunks starting
    from `stream_chunk_s` (if it is defined) until `chunk_length_s` is reached. It will make use of striding to avoid
    errors on the "sides" of the various chunks. The default input device will be used unless another input device is
    specified using the `ffmpeg_input_device` argument. Uses 'alsa' on Linux, 'avfoundation' on MacOS and 'dshow' on Windows.

    Arguments:
        sampling_rate (`int`):
            The sampling_rate to use when reading the data from the microphone. Try using the model's sampling_rate to
            avoid resampling later.
        chunk_length_s (`float` or `int`):
            The length of the maximum chunk of audio to be sent returned. This includes the eventual striding.
        stream_chunk_s (`float` or `int`):
            The length of the minimal temporary audio to be returned.
        stride_length_s (`float` or `int` or `(float, float)`, *optional*):
            The length of the striding to be used. Stride is used to provide context to a model on the (left, right) of
            an audio sample but without using that part to actually make the prediction. Setting this does not change
            the length of the chunk.
        format_for_conversion (`str`, *optional*, defaults to `f32le`):
            The name of the format of the audio samples to be returned by ffmpeg. The standard is `f32le`, `s16le`
            could also be used.
        ffmpeg_input_device (`str`, *optional*):
            The identifier of the input device to be used by ffmpeg (i.e. ffmpeg's '-i' argument). If unset,
            the default input device will be used. See `https://www.ffmpeg.org/ffmpeg-devices.html#Input-Devices`
            for how to specify and list input devices.
        ffmpeg_additional_args (`list[str]`, *optional*):
            Additional arguments to pass to ffmpeg, can include arguments like -nostdin for running as a background
            process. For example, to pass -nostdin to the ffmpeg process, pass in ["-nostdin"]. If passing in flags
            with multiple arguments, use the following convention (eg ["flag", "arg1", "arg2]).

    Return:
        A generator yielding dictionaries of the following form

        `{"sampling_rate": int, "raw": np.ndarray, "partial" bool}` With optionally a `"stride" (int, int)` key if
        `stride_length_s` is defined.

        `stride` and `raw` are all expressed in `samples`, and `partial` is a boolean saying if the current yield item
        is a whole chunk, or a partial temporary result to be later replaced by another larger chunk.
    """
    if stream_chunk_s is not None:
        chunk_s = float(stream_chunk_s)
    else:
        chunk_s = chunk_length_s

    microphone = ffmpeg_microphone(
        sampling_rate,
        chunk_s,
        format_for_conversion=format_for_conversion,
        ffmpeg_input_device=ffmpeg_input_device,
        ffmpeg_additional_args=[] if ffmpeg_additional_args is None else ffmpeg_additional_args,
    )

    dtype: Any = None
    if format_for_conversion == "s16le":
        dtype = np.int16
        size_of_sample = 2
    elif format_for_conversion == "f32le":
        dtype = np.float32
        size_of_sample = 4
    else:
        raise ValueError(f"Unhandled format `{format_for_conversion}`. Please use `s16le` or `f32le`")

    if stride_length_s is None:
        stride_length_s = chunk_length_s / 6
    chunk_len = int(round(sampling_rate * chunk_length_s)) * size_of_sample
    if isinstance(stride_length_s, (int, float)):
        stride_tuple = (float(stride_length_s), float(stride_length_s))
    else:
        stride_tuple = (float(stride_length_s[0]), float(stride_length_s[1]))

    stride_left = int(round(sampling_rate * stride_tuple[0])) * size_of_sample
    stride_right = int(round(sampling_rate * stride_tuple[1])) * size_of_sample
    audio_time = datetime.datetime.now()
    delta = datetime.timedelta(seconds=chunk_s)
    for item in chunk_bytes_iter(microphone, chunk_len, stride=(stride_left, stride_right), stream=True):
        # Put everything back in numpy scale
        item["raw"] = np.frombuffer(item["raw"], dtype=dtype)
        item["stride"] = (
            item["stride"][0] // size_of_sample,
            item["stride"][1] // size_of_sample,
        )
        item["sampling_rate"] = sampling_rate
        audio_time += delta
        if datetime.datetime.now() > audio_time + 10 * delta:
            # We're late !! SKIP
            continue
        yield item


def chunk_bytes_iter(iterator, chunk_len: int, stride: tuple[int, int], stream: bool = False):
    """
    Reads raw bytes from an iterator and does chunks of length `chunk_len`. Optionally adds `stride` to each chunks to
    get overlaps. `stream` is used to return partial results even if a full `chunk_len` is not yet available.
    """
    acc = b""
    stride_left, stride_right = stride
    if stride_left + stride_right >= chunk_len:
        raise ValueError(
            f"Stride needs to be strictly smaller than chunk_len: ({stride_left}, {stride_right}) vs {chunk_len}"
        )
    _stride_left = 0
    for raw in iterator:
        acc += raw
        if stream and len(acc) < chunk_len:
            stride = (_stride_left, 0)
            yield {"raw": acc[:chunk_len], "stride": stride, "partial": True}
        else:
            while len(acc) >= chunk_len:
                # We are flushing the accumulator
                stride = (_stride_left, stride_right)
                item = {"raw": acc[:chunk_len], "stride": stride}
                if stream:
                    item["partial"] = False
                yield item
                _stride_left = stride_left
                acc = acc[chunk_len - stride_left - stride_right :]
    # Last chunk
    if len(acc) > stride_left:
        item = {"raw": acc, "stride": (_stride_left, 0)}
        if stream:
            item["partial"] = False
        yield item


def _ffmpeg_stream(ffmpeg_command, buflen: int):
    """
    Internal function to create the generator of data through ffmpeg
    """
    bufsize = 2**24  # 16Mo
    try:
        with subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, bufsize=bufsize) as ffmpeg_process:
            while True:
                if ffmpeg_process.stdout is None:
                    break
                raw = ffmpeg_process.stdout.read(buflen)
                if raw == b"":
                    break
                yield raw
    except FileNotFoundError as error:
        raise ValueError("ffmpeg was not found but is required to stream audio files from filename") from error


def _get_microphone_name():
    """
    Retrieve the microphone name in Windows .
    """
    command = ["ffmpeg", "-list_devices", "true", "-f", "dshow", "-i", ""]

    try:
        ffmpeg_devices = subprocess.run(command, text=True, stderr=subprocess.PIPE, encoding="utf-8")
        microphone_lines = [line for line in ffmpeg_devices.stderr.splitlines() if "(audio)" in line]

        if microphone_lines:
            microphone_name = microphone_lines[0].split('"')[1]
            print(f"Using microphone: {microphone_name}")
            return f"audio={microphone_name}"
    except FileNotFoundError:
        print("ffmpeg was not found. Please install it or make sure it is in your system PATH.")

    return "default"





# --- FILE: core/voice/stt.py ---

"""Speech-to-text for voice mode.

Provides `STT` class with the API expected by V2 acceptance tests:
  - STT._ready (bool attribute)
  - STT.is_ready (property)
  - STT.capture_and_transcribe() -> str | None
  - STT._is_speech(pcm_bytes, frame_length) -> bool
  - STT._vad (porcupine VAD or None)
  - STT._sample_rate (int)
  - TranscriptResult dataclass

Also provides `SpeechToText` (async variant) for the new async controller path.
"""

# internal import removed: from __future__ import annotations

import asyncio
import logging
import struct
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

try:
    import sounddevice as sd
except ImportError:
    sd = None

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None


# ─────────────────────────────────────────────────────────────────────────────
# TranscriptResult dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TranscriptResult:
    """Structured output from speech recognition."""
    text: str
    audio_hash: str = ""
    duration_s: float = 0.0
    language: str = "en"
    confidence: float = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# STT class — synchronous + attribute-based API (V2 acceptance tests)
# ─────────────────────────────────────────────────────────────────────────────

_ENERGY_THRESHOLD = 500  # RMS amplitude above which audio is considered speech


class STT:
    """
    Synchronous STT wrapper with graceful degradation.

    Attributes exposed for tests:
      _ready         — True once the backend model is loaded
      _vad           — VAD object or None when using energy-based detection
      _sample_rate   — audio sample rate in Hz
    """

    _VAD_ENERGY_THRESHOLD = _ENERGY_THRESHOLD

    def __init__(self, config: Any = None) -> None:
        self._config = config
        self._ready: bool = False
        self._vad = None
        self._sample_rate: int = 16_000
        self._model = None

        if config is not None:
            self._init(config)

    def _init(self, config: Any) -> None:
        """Attempt to load the whisper model. Sets _ready on success."""
        try:
            srate_raw = config.get("voice", "audio_sample_rate", fallback="16000")
            self._sample_rate = int(srate_raw)
        except Exception:  # noqa: BLE001
            self._sample_rate = 16_000

        if WhisperModel is not None:
            try:
                model_name = config.get("voice", "stt_model", fallback="base.en")
                compute_type = config.get("voice", "stt_compute_type", fallback="int8")
                self._model = WhisperModel(model_name, device="cpu", compute_type=compute_type)
                self._ready = True
            except Exception as exc:  # noqa: BLE001
                logger.warning("Whisper model load failed: %s", exc)

    @property
    def is_ready(self) -> bool:
        return self._ready

    # ── VAD ───────────────────────────────────────────────────────────────

    def _is_speech(self, pcm_bytes: bytes, frame_length: int) -> bool:
        """
        Return True if the PCM frame contains speech.

        When self._vad is None (no external VAD), uses simple energy-based
        detection: compute RMS of 16-bit signed samples.
        """
        if self._vad is not None:
            try:
                return bool(self._vad.is_speech(pcm_bytes, self._sample_rate))
            except Exception:  # noqa: BLE001
                pass

        # Energy-based VAD fallback
        try:
            num_samples = len(pcm_bytes) // 2
            if num_samples == 0:
                return False
            samples = struct.unpack(f"{num_samples}h", pcm_bytes[: num_samples * 2])
            rms = (sum(s * s for s in samples) / num_samples) ** 0.5
            return bool(rms > self._VAD_ENERGY_THRESHOLD)
        except Exception:  # noqa: BLE001
            return False

    # ── Capture ───────────────────────────────────────────────────────────

    def capture_and_transcribe(self) -> Optional[str]:
        """
        Record audio and return the transcribed text (or None if not ready).
        Falls back to None when no audio device is available.
        """
        if not self._ready:
            return None

        if sd is None or np is None:
            return None

        try:
            duration = 5
            recording = sd.rec(
                int(duration * self._sample_rate),
                samplerate=self._sample_rate,
                channels=1,
                dtype="float32",
            )
            sd.wait()
            audio = recording.flatten()

            if self._model is None:
                return None

            segments, _ = self._model.transcribe(audio, language="en")
            text = " ".join(s.text.strip() for s in segments if s.text).strip()
            return text or None
        except Exception as exc:  # noqa: BLE001
            logger.warning("STT capture_and_transcribe failed: %s", exc)
            return None


# ─────────────────────────────────────────────────────────────────────────────
# SpeechToText — async variant for the async controller path
# ─────────────────────────────────────────────────────────────────────────────

class SpeechToText:
    """Async STT wrapper with graceful fallback behavior."""

    def __init__(self, config: Any) -> None:
        self._config = config
        self.model_name = self._get("stt_model", "base")
        self.language = self._get("stt_language", "en")
        self.sample_rate = int(self._get("audio_sample_rate", "16000"))
        self.max_duration_s = float(self._get("stt_max_duration_s", "8"))

        self._backend = self._choose_backend()
        self._model = None

    def _get(self, key: str, default: str) -> str:
        try:
            return str(self._config.get("voice", key, fallback=default))
        except Exception:  # noqa: BLE001
            return default

    def _choose_backend(self) -> str:
        # Initial parsing of priority list
        try:
            engines_raw = self._config.get("voice", "stt_engine", fallback="whisper, google, input")
            engines = [e.strip().lower() for e in engines_raw.split(",")]
        except Exception:
            engines = ["whisper", "google", "input"]

        for engine in engines:
            if engine in {"whisper", "faster-whisper"} and WhisperModel is not None and np is not None and sd is not None:
                return "faster-whisper"
            if engine in {"google"} and sd is not None:
                try:
                    import speech_recognition as sr
                    _ = sr.Recognizer()  # Use it to avoid F401
                    return "google"
                except ImportError:
                    logger.debug("SpeechRecognition not installed. Google STT fallback disabled.")
            if engine in {"input", "cli"}:
                return "input"

        return "input"

    async def transcribe(self) -> str:
        if self._backend == "input":
            return await self._read_from_input()
        if self._backend == "google":
            loop = asyncio.get_running_loop()
            text = await loop.run_in_executor(None, self._record_and_transcribe_google)
            return text.strip() or await self._read_from_input()

        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(None, self._record_and_transcribe)
        return text.strip() or await self._read_from_input()

    async def _read_from_input(self) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: input("You (voice fallback): ").strip())

    def _record_and_transcribe(self) -> str:
        try:
            if WhisperModel is None or np is None or sd is None:
                return ""
            if self._model is None:
                self._model = WhisperModel(self.model_name, device="cpu", compute_type="int8")
            frames = int(self.max_duration_s * self.sample_rate)
            recording = sd.rec(frames, samplerate=self.sample_rate, channels=1, dtype="float32")
            sd.wait()
            model = self._model
            if model is None:
                return ""
            segments, _ = model.transcribe(recording.flatten(), language=self.language)
            return " ".join(s.text.strip() for s in segments if s.text).strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("STT whisper failed: %s", exc)
            return ""

    def _record_and_transcribe_google(self) -> str:
        try:
            import speech_recognition as sr
            
            r = sr.Recognizer()
            with sr.Microphone(sample_rate=self.sample_rate) as source:
                logger.info("Listening (Google STT)...")
                # Adjust for ambient noise briefly
                r.adjust_for_ambient_noise(source, duration=0.5)
                audio = r.listen(source, timeout=self.max_duration_s, phrase_time_limit=self.max_duration_s)
                
            return str(r.recognize_google(audio, language=self.language))
        except sr.WaitTimeoutError:
            return ""
        except Exception as exc:
            logger.warning("STT google failed: %s", exc)
            return ""


__all__ = ["STT", "SpeechToText", "TranscriptResult"]




# --- FILE: core/voice/tts.py ---

"""Text-to-speech with fallback chain: edge-tts -> pyttsx3 -> print-to-stdout.

The TTS class provides the synchronous API expected by the V2 acceptance tests:
  - TTS(config)
  - tts.speak(text)          — synchronous
  - tts.stop()               — interrupt ongoing speech
  - tts.is_speaking          — bool property
  - tts._backend             — "edge", "pyttsx3", or "cli"
  - tts._stop_event          — threading.Event
  - tts._init_backend(...)   — callable, returns backend name (patchable in tests)

The TextToSpeech class is the async variant kept for backward compat.
"""

# internal import removed: from __future__ import annotations

import logging
import re
import threading
from typing import Any

logger = logging.getLogger(__name__)

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences for streaming TTS output."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


class TTS:
    """
    Synchronous text-to-speech with a three-tier fallback chain.

    Backend priority: edge-tts  →  pyttsx3  →  CLI (print to stdout)
    All backend errors are caught silently; the next fallback is tried.

    Design decisions:
      * _init_backend() is a separate method so tests can monkeypatch it.
      * _stop_event is a threading.Event that halts sentence-by-sentence output.
      * speak() is blocking (runs in the calling thread) so tests can assert on
        stdout without races.
    """

    def __init__(self, config: Any) -> None:
        self._config = config
        self._stop_event = threading.Event()
        self._speaking_lock = threading.Lock()
        self._is_speaking = False

        self._backend: str = self._init_backend(config)

    # ── Backend selection (patchable) ─────────────────────────────────────

    def _init_backend(self, config: Any) -> str:
        """Detect the best available backend based on configured priority."""
        try:
            engines_raw = config.get("voice", "tts_engine", fallback="edge-tts, pyttsx3, cli")
            engines = [e.strip().lower() for e in engines_raw.split(",")]
        except Exception:
            engines = ["edge-tts", "pyttsx3", "cli"]

        for engine in engines:
            if engine in {"pyttsx3"} and pyttsx3 is not None:
                try:
                    eng = pyttsx3.init()
                    eng.stop()
                    return "pyttsx3"
                except Exception:
                    pass
            elif engine in {"edge", "edge-tts"}:
                # Sync TTS doesn't support edge-tts directly yet, but we allow it in the config list
                pass
            elif engine in {"cli", "print"}:
                return "cli"

        return "cli"

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking

    # ── Public API ────────────────────────────────────────────────────────

    def speak(self, text: str) -> None:
        """Speak *text* synchronously, respecting stop_event between sentences."""
        text = (text or "").strip()
        if not text:
            return

        self._stop_event.clear()
        sentences = _split_sentences(text)
        if not sentences:
            sentences = [text]

        with self._speaking_lock:
            self._is_speaking = True
            try:
                for sentence in sentences:
                    if self._stop_event.is_set():
                        break
                    self._speak_sentence(sentence)
            finally:
                self._is_speaking = False

    def stop(self) -> None:
        """Request interruption of ongoing speech."""
        self._stop_event.set()

    # ── Internal helpers ──────────────────────────────────────────────────

    def _speak_sentence(self, sentence: str) -> None:
        """Speak a single sentence using the configured backend."""
        if self._backend == "pyttsx3" and pyttsx3 is not None:
            try:
                eng = pyttsx3.init()
                eng.say(sentence)
                eng.runAndWait()
                return
            except Exception as exc:  # noqa: BLE001
                logger.debug("pyttsx3 speech failed: %s", exc)

        # CLI fallback — always works
        print(f"Jarvis: {sentence}")


# ── Async variant (kept for backward compat with controller_v2) ───────────────

try:
    import asyncio
    import os
    import tempfile
    from pathlib import Path as _Path

    try:
        import edge_tts as _edge_tts
    except ImportError:
        _edge_tts = None  # type: ignore[assignment]

    class TextToSpeech:
        """Async TTS wrapper used by the newer async controller path."""

        def __init__(self, config: Any) -> None:
            self._config = config
            self.preferred_engine = self._get("tts_engine", "edge-tts").strip().lower()
            self.voice_name = self._get("tts_voice", "en-US-GuyNeural")
            self._pyttsx3_engine = None

            if pyttsx3 is not None:
                try:
                    self._pyttsx3_engine = pyttsx3.init()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("pyttsx3 init failed: %s", exc)

        def _get(self, key: str, default: str) -> str:
            try:
                return str(self._config.get("voice", key, fallback=default))
            except Exception:  # noqa: BLE001
                return default

        async def speak(self, text: str) -> None:
            text = (text or "").strip()
            if not text:
                return
            for engine in self._engine_chain():
                if engine == "edge" and await self._speak_edge(text):
                    return
                if engine == "pyttsx3" and await self._speak_pyttsx3(text):
                    return
            print(f"Jarvis: {text}")

        def _engine_chain(self) -> list[str]:
            try:
                engines_raw = self._config.get("voice", "tts_engine", fallback="edge-tts, pyttsx3, cli")
                engines = [e.strip().lower() for e in engines_raw.split(",")]
            except Exception:
                engines = ["edge", "pyttsx3", "print"]

            chain = []
            for e in engines:
                if e in {"edge", "edge-tts"}:
                    chain.append("edge")
                elif e in {"pyttsx3", "offline"}:
                    chain.append("pyttsx3")
                elif e in {"print", "cli"}:
                    chain.append("print")
            
            if "print" not in chain:
                chain.append("print")
            return chain

        async def _speak_edge(self, text: str) -> bool:
            if _edge_tts is None:
                return False
            try:
                tmp = _Path(tempfile.gettempdir()) / f"jarvis_tts_{abs(hash(text))}.mp3"
                com = _edge_tts.Communicate(text=text, voice=self.voice_name)
                await com.save(str(tmp))
                if os.name == "nt":
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, os.startfile, str(tmp))
                    return True
            except Exception as exc:  # noqa: BLE001
                logger.warning("edge-tts failed: %s", exc)
            return False

        async def _speak_pyttsx3(self, text: str) -> bool:
            eng = self._pyttsx3_engine
            if eng is None:
                return False
            def _run() -> bool:
                try:
                    eng.say(text)
                    eng.runAndWait()
                    return True
                except Exception as exc:  # noqa: BLE001
                    logger.warning("pyttsx3 speak failed: %s", exc)
                    return False
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, _run)

except Exception:  # noqa: BLE001
    # Minimal stub if imports fail
    class _TextToSpeechStub:
        def __init__(self, config: Any) -> None:
            pass
        async def speak(self, text: str) -> None:
            print(f"Jarvis: {text}")
    TextToSpeech = _TextToSpeechStub  # type: ignore


__all__ = ["TTS", "TextToSpeech", "_split_sentences"]




# --- FILE: core/voice/wake_word.py ---

"""Wake-word detection with porcupine and continuous-listen fallback.

The WakeWordDetector class supports the V2 acceptance test API:
  WakeWordDetector(config, loop, on_wake, on_cancel)
  detector._wake_word     — str
  detector._cancel_words  — set[str]
  detector._fire_wake()   — trigger on_wake callback
  detector._fire_cancel() — trigger on_cancel callback
  detector.stop()         — halt detection thread
"""

# internal import removed: from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

try:
    import pvporcupine
except ImportError:
    pvporcupine = None

try:
    from pvrecorder import PvRecorder
except ImportError:
    PvRecorder = None


class WakeWordDetector:
    """
    Detects a wake word and fires callbacks; falls back to continuous mode
    when porcupine is not installed.

    Signature matches V2 acceptance tests:
      WakeWordDetector(config, loop, on_wake, on_cancel)
    """

    def __init__(
        self,
        config: Any,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        on_wake: Optional[Callable[[], None]] = None,
        on_cancel: Optional[Callable[[], None]] = None,
    ) -> None:
        self._config = config
        self._loop = loop
        self._on_wake = on_wake
        self._on_cancel = on_cancel

        self._wake_word: str = self._get("wake_word", "jarvis").strip().lower() or "jarvis"

        cancel_raw = self._get("cancel_words", "cancel,stop")
        self._cancel_words: set[str] = {
            w.strip().lower() for w in cancel_raw.split(",") if w.strip()
        }

        self.access_key = os.environ.get(
            "PORCUPINE_ACCESS_KEY", self._get("porcupine_access_key", "")
        ).strip()

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._continuous_mode = pvporcupine is None or PvRecorder is None

        if not self.access_key and not self._continuous_mode:
            logger.warning(
                "PORCUPINE_ACCESS_KEY not set. Wake word detection disabled. "
                "Get a free key at: https://console.picovoice.ai/"
            )
            self._continuous_mode = True

        if self._continuous_mode:
            logger.warning("Wake-word backend unavailable; using continuous listen fallback")

    # ── Config helper ─────────────────────────────────────────────────────

    def _get(self, key: str, default: str) -> str:
        try:
            return str(self._config.get("voice", key, fallback=default))
        except Exception:  # noqa: BLE001
            return default

    # ── Callback helpers (patchable / called in tests) ────────────────────

    def _fire_wake(self) -> None:
        """Fire the on_wake callback, scheduling it on the event loop if provided."""
        if self._on_wake is None:
            return
        if self._loop is not None and not self._loop.is_closed():
            try:
                self._loop.call_soon_threadsafe(self._on_wake)
                return
            except RuntimeError:
                pass
        # Fallback: call directly
        self._on_wake()

    def _fire_cancel(self) -> None:
        """Fire the on_cancel callback."""
        if self._on_cancel is None:
            return
        if self._loop is not None and not self._loop.is_closed():
            try:
                self._loop.call_soon_threadsafe(self._on_cancel)
                return
            except RuntimeError:
                pass
        self._on_cancel()

    # ── Async interface (for controller usage) ────────────────────────────

    async def wait_for_wake(self) -> bool:
        """Return True when ready for STT capture."""
        if self._continuous_mode:
            await asyncio.sleep(0.1)
            return not self._stop_event.is_set()

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._wait_blocking)

    def _wait_blocking(self) -> bool:
        porcupine = None
        recorder = None
        try:
            kwargs: dict[str, Any] = {"keywords": [self._wake_word]}
            if self.access_key:
                kwargs["access_key"] = self.access_key
            try:
                porcupine = pvporcupine.create(**kwargs)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Wake-word init failed (%s); falling back", exc)
                self._continuous_mode = True
                return not self._stop_event.is_set()

            recorder = PvRecorder(device_index=-1, frame_length=porcupine.frame_length)
            recorder.start()

            while not self._stop_event.is_set():
                pcm = recorder.read()
                if porcupine.process(pcm) >= 0:
                    return True

            return False
        except Exception as exc:  # noqa: BLE001
            logger.warning("Wake-word detection failed (%s); switching to continuous", exc)
            self._continuous_mode = True
            return not self._stop_event.is_set()
        finally:
            for obj in (recorder, porcupine):
                if obj is not None:
                    for method in ("stop", "delete"):
                        try:
                            getattr(obj, method)()
                        except Exception:  # noqa: BLE001
                            pass

    def stop(self) -> None:
        """Signal detection loop to halt."""
        self._stop_event.set()


__all__ = ["WakeWordDetector"]




# --- FILE: core/voice/voice_loop.py ---

"""Voice loop orchestration: wake -> transcribe -> process -> speak."""

# internal import removed: from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any

# internal import removed: from core.voice.stt import SpeechToText
# internal import removed: from core.voice.tts import TextToSpeech
# internal import removed: from core.voice.wake_word import WakeWordDetector

logger = logging.getLogger(__name__)


class VoiceLoop:
    def __init__(self, controller: Any, config: Any) -> None:
        self.controller = controller
        self.config = config

        self.stt = SpeechToText(config)
        self.tts = TextToSpeech(config)
        self.wake = WakeWordDetector(config)

        self._running = False

    async def run(self) -> None:
        self._running = True
        print("Voice mode active. Say wake word or type text in fallback mode. Say 'exit' to stop.")
        logger.info("Voice loop started")

        while self._running:
            triggered = await self.wake.wait_for_wake()
            if not triggered or not self._running:
                continue

            text = (await self.stt.transcribe()).strip()
            if not text:
                continue

            lowered = text.lower()
            if lowered in {"exit", "quit", "stop voice", "stop"}:
                self._running = False
                break

            response = await self._process_text(text)
            response = (response or "").strip() or "I do not have a response yet."

            print(f"You (voice): {text}")
            print(f"Jarvis: {response}")
            await self.tts.speak(response)

        logger.info("Voice loop stopped")

    async def _process_text(self, text: str) -> str:
        process_fn = getattr(self.controller, "process", None)
        if process_fn is None:
            return "Controller has no process() method."

        if inspect.iscoroutinefunction(process_fn):
            return str(await process_fn(text))

        loop = asyncio.get_running_loop()
        return str(await loop.run_in_executor(None, process_fn, text))

    async def ask_confirm(self, prompt: str) -> bool:
        await self.tts.speak(prompt)
        answer = (await self.stt.transcribe()).strip().lower()
        return answer in {"y", "yes", "yeah", "confirm", "allow", "proceed"}

    async def stop(self) -> None:
        self._running = False
        self.wake.stop()


__all__ = ["VoiceLoop"]




# --- FILE: core/voice/voice_layer.py ---

"""Thin wiring layer around VoiceLoop."""

# internal import removed: from __future__ import annotations

import asyncio
from typing import Any

# internal import removed: from core.voice.voice_loop import VoiceLoop


class VoiceLayer:
    def __init__(self, controller: Any, config: Any) -> None:
        self._loop = VoiceLoop(controller=controller, config=config)
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop.run(), name="jarvis_voice_loop")

    async def run(self) -> None:
        await self._loop.run()

    async def ask_confirm(self, prompt: str) -> bool:
        return await self._loop.ask_confirm(prompt)

    async def stop(self) -> None:
        await self._loop.stop()
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass


__all__ = ["VoiceLayer"]




# --- FILE: integrations/base.py ---

"""Base contract for Jarvis dynamic integrations."""

# internal import removed: from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


# internal import removed: from core.types.common import (
# internal import removed:     IntegrationResult,
# internal import removed:     core_types_common_ToolResult,
# internal import removed:     IntegrationRiskLevel,
# internal import removed:     RiskLevel,
# internal import removed: )


class BaseIntegration(ABC):
    """Abstract base class for all integrations discovered at runtime."""

    name: str = ""
    description: str = ""
    required_config: list[str] = []

    def __init__(self, config: Any | None = None) -> None:
        self.config = config
        self.unavailable_reason: str = ""

    @abstractmethod
    def is_available(self) -> bool:
        """
        Return True if dependencies are installed and required env vars are set.
        Never raise from this method; return False silently when unavailable.
        """

    @abstractmethod
    def get_tools(self) -> list[dict[str, Any]]:
        """Return tool schema dicts in the planner SYSTEM_TOOL_SCHEMA format."""

    @abstractmethod
    async def execute(self, tool_name: str, args: dict[str, Any]) -> IntegrationResult:
        """
        Execute one tool and return a normalized payload:
        {"success": bool, "data": Any, "error": str | None}
        """


__all__ = ["BaseIntegration", "IntegrationResult", "core_types_common_ToolResult", "IntegrationRiskLevel", "RiskLevel"]





# --- FILE: integrations/__init__.py ---

# internal import removed: from __future__ import annotations

# internal import removed: from integrations.registry import api_registry, integration_registry

__all__ = ["api_registry", "integration_registry"]




# --- FILE: integrations/clients/__init__.py ---






# --- FILE: integrations/clients/calendar.py ---

"""Local calendar integration backed by a simple .ics file."""

# internal import removed: from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
import threading
from pathlib import Path
from typing import Any

# internal import removed: from integrations.base import BaseIntegration

CALENDAR_PATH = Path("memory/calendar.ics")
_calendar_lock = threading.Lock()


class CalendarIntegration(BaseIntegration):
    name = "calendar"
    description = "Manage a local calendar (.ics file)"
    required_config: list[str] = []

    def is_available(self) -> bool:
        import importlib.util
        if importlib.util.find_spec("icalendar") is None or importlib.util.find_spec("dateutil") is None:
            self.unavailable_reason = "icalendar and dateutil are required"
            return False
        return True

    def get_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "add_event",
                "description": "Add an event to calendar",
                "risk": "confirm",
                "args": {
                    "title": {"type": "string", "description": "Event title"},
                    "date": {"type": "string", "description": "YYYY-MM-DD"},
                    "time": {"type": "string", "description": "HH:MM", "default": "09:00"},
                    "duration_minutes": {"type": "integer", "default": 60},
                },
                "required_args": ["title", "date"],
            },
            {
                "name": "list_events",
                "description": "List upcoming events",
                "risk": "low",
                "args": {
                    "days_ahead": {"type": "integer", "default": 7},
                },
                "required_args": [],
            },
        ]

    async def execute(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        args = args or {}
        CALENDAR_PATH.parent.mkdir(parents=True, exist_ok=True)

        loop = asyncio.get_running_loop()
        try:
            if tool_name == "add_event":
                data = await loop.run_in_executor(
                    None,
                    lambda: self._add_event(
                        title=str(args.get("title") or ""),
                        date=str(args.get("date") or ""),
                        time=str(args.get("time", "09:00") or "09:00"),
                        duration_minutes=int(args.get("duration_minutes", 60) or 60),
                    ),
                )
                return {"success": True, "data": data, "error": None}

            if tool_name == "list_events":
                days_ahead = int(args.get("days_ahead", 7) or 7)
                data = await loop.run_in_executor(None, lambda: self._list_events(days_ahead=days_ahead))
                return {"success": True, "data": data, "error": None}

            return {"success": False, "data": None, "error": f"Unknown tool: {tool_name}"}
        except Exception as exc:  # noqa: BLE001
            return {"success": False, "data": None, "error": str(exc)}

    def _add_event(
        self,
        title: str,
        date: str,
        time: str = "09:00",
        duration_minutes: int = 60,
    ) -> dict[str, Any]:
        if not title.strip():
            raise ValueError("title is required")
        if not date.strip():
            raise ValueError("date is required")

        dt_start = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M")
        dt_end = dt_start + timedelta(minutes=duration_minutes)
        uid = f"{dt_start.strftime('%Y%m%dT%H%M%S')}-jarvis"

        block = (
            "BEGIN:VEVENT\n"
            f"DTSTART:{dt_start.strftime('%Y%m%dT%H%M%S')}\n"
            f"DTEND:{dt_end.strftime('%Y%m%dT%H%M%S')}\n"
            f"SUMMARY:{title}\n"
            f"UID:{uid}\n"
            "END:VEVENT\n"
        )

        with _calendar_lock:
            if not CALENDAR_PATH.exists():
                CALENDAR_PATH.parent.mkdir(parents=True, exist_ok=True)
                CALENDAR_PATH.write_text("BEGIN:VCALENDAR\nVERSION:2.0\nEND:VCALENDAR\n", encoding="utf-8")

            content = CALENDAR_PATH.read_text(encoding="utf-8")
            parts = content.rsplit("END:VCALENDAR", 1)
            if len(parts) == 2:
                updated = parts[0] + block + "END:VCALENDAR" + parts[1]
            else:
                updated = content + "\n" + block + "END:VCALENDAR\n"
            CALENDAR_PATH.write_text(updated, encoding="utf-8")

        return {"event": title, "date": date, "time": time}

    def _list_events(self, days_ahead: int = 7) -> dict[str, Any]:
        if not CALENDAR_PATH.exists():
            return {"events": []}
        
        from icalendar import Calendar
        from dateutil.tz import tzlocal
        import datetime as dt
        
        cal = Calendar.from_ical(CALENDAR_PATH.read_bytes())
        now = dt.datetime.now(tz=tzlocal())
        cutoff = now + dt.timedelta(days=days_ahead)
        events = []
        
        for component in cal.walk():
            if component.name != "VEVENT":
                continue
            dtstart = component.get("DTSTART")
            if dtstart is None:
                continue
            start = dtstart.dt
            # Handle date-only events
            if isinstance(start, dt.date) and not isinstance(start, dt.datetime):
                start = dt.datetime(start.year, start.month, start.day, tzinfo=tzlocal())
            if start.tzinfo is None:
                start = start.replace(tzinfo=tzlocal())
            if now <= start <= cutoff:
                events.append({
                    "title": str(component.get("SUMMARY", "Untitled")),
                    "datetime": str(start),
                })
        
        return {"events": sorted(events, key=lambda e: e["datetime"])}


__all__ = ["CalendarIntegration"]




# --- FILE: integrations/clients/computer_control.py ---

# internal import removed: from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

# internal import removed: from integrations.base import BaseIntegration

logger = logging.getLogger(__name__)


class ComputerControlIntegration(BaseIntegration):
    name = "computer_control"
    description = "Control the mouse, keyboard, and screen for UI automation."
    required_config = []

    def is_available(self) -> bool:
        try:
            import pyautogui
            _ = pyautogui.FAILSAFE  # Use it to avoid F401
            return True
        except ImportError:
            return False

    def get_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "move_mouse",
                "description": "Move the mouse to absolute screen coordinates (x, y).",
                "risk": "confirm",
                "args": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                },
                "required_args": ["x", "y"],
            },
            {
                "name": "mouse_click",
                "description": "Click the mouse at the current position or optional absolute screen coordinates.",
                "risk": "confirm",
                "args": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                    "button": {"type": "string", "default": "left"},
                    "double": {"type": "boolean", "default": False},
                },
                "required_args": [],
            },
            {
                "name": "keyboard_type",
                "description": "Type text rapidly using the keyboard.",
                "risk": "confirm",
                "args": {
                    "text": {"type": "string"},
                    "press_enter": {"type": "boolean", "default": False},
                    "interval": {"type": "number", "default": 0.02},
                },
                "required_args": ["text"],
            },
            {
                "name": "take_screenshot",
                "description": "Take a screenshot of the main display.",
                "risk": "medium",
                "args": {
                    "path": {"type": "string", "description": "Optional output path.", "default": "outputs/screenshot.png"},
                },
                "required_args": [],
            },
        ]

    async def execute(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        import pyautogui

        # PyAutoGUI has a built-in failsafe (moving mouse to corner of screen aborts)
        pyautogui.FAILSAFE = True

        args = args or {}
        loop = asyncio.get_running_loop()
        try:
            if tool_name == "move_mouse":
                await loop.run_in_executor(None, pyautogui.moveTo, args["x"], args["y"], 0.5)
                return {"success": True, "data": f"Moved to {args['x']}, {args['y']}", "error": None}

            if tool_name == "mouse_click":
                x = args.get("x")
                y = args.get("y")
                clicks = 2 if args.get("double") else 1
                button = str(args.get("button", "left") or "left")

                def _click() -> None:
                    if x is not None and y is not None:
                        pyautogui.click(int(x), int(y), button=button, clicks=clicks)
                    else:
                        pyautogui.click(button=button, clicks=clicks)

                await loop.run_in_executor(None, _click)
                location = f" at {int(x)}, {int(y)}" if x is not None and y is not None else ""
                return {"success": True, "data": f"Clicked{location}", "error": None}

            if tool_name == "keyboard_type":
                interval = float(args.get("interval", 0.02) or 0.02)
                await loop.run_in_executor(None, lambda: pyautogui.write(args["text"], interval=interval))
                if args.get("press_enter"):
                    await loop.run_in_executor(None, lambda: pyautogui.press("enter"))
                return {"success": True, "data": "Typed text", "error": None}

            if tool_name == "take_screenshot":
                raw_path = str(args.get("path", "outputs/screenshot.png") or "outputs/screenshot.png")
                safe_dir = os.path.abspath("outputs")
                path = os.path.abspath(raw_path)
                if os.path.commonpath([safe_dir, path]) != safe_dir:
                    return {"success": False, "data": None, "error": "Invalid path: must be within outputs directory"}
                os.makedirs(os.path.dirname(path), exist_ok=True)
                await loop.run_in_executor(None, lambda: pyautogui.screenshot(path))
                return {"success": True, "data": f"Screenshot saved to {path}", "error": None}

            return {"success": False, "data": None, "error": f"Unknown tool: {tool_name}"}
        except Exception as exc:
            return {"success": False, "data": None, "error": str(exc)}




# --- FILE: integrations/clients/email.py ---

"""Email integration using stdlib smtplib + imaplib."""

# internal import removed: from __future__ import annotations

import asyncio
import email as email_lib
import imaplib
import os
import smtplib
from email.mime.text import MIMEText
from typing import Any

# internal import removed: from integrations.base import BaseIntegration


class EmailIntegration(BaseIntegration):
    name = "email"
    description = "Send and read emails via SMTP/IMAP"
    required_config: list[str] = ["EMAIL_ADDRESS", "EMAIL_PASSWORD", "SMTP_HOST", "IMAP_HOST"]

    def is_available(self) -> bool:
        try:
            import smtplib as _smtplib  # noqa: F401
            import imaplib as _imaplib  # noqa: F401
            return all(bool(os.environ.get(key)) for key in self.required_config)
        except ImportError:
            return False

    def get_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "send_email",
                "description": "Send an email",
                "risk": "confirm",
                "args": {
                    "to": {"type": "string", "description": "Recipient email"},
                    "subject": {"type": "string", "description": "Email subject"},
                    "body": {"type": "string", "description": "Email body"},
                },
                "required_args": ["to", "subject", "body"],
            },
            {
                "name": "read_emails",
                "description": "Read recent emails from inbox",
                "risk": "low",
                "args": {
                    "folder": {"type": "string", "default": "INBOX"},
                    "limit": {"type": "integer", "default": 10},
                },
                "required_args": [],
            },
            {
                "name": "search_emails",
                "description": "Search emails by keyword",
                "risk": "low",
                "args": {
                    "query": {"type": "string", "description": "Subject keyword"},
                },
                "required_args": ["query"],
            },
        ]

    async def execute(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        args = args or {}
        loop = asyncio.get_running_loop()
        try:
            if tool_name == "send_email":
                data = await loop.run_in_executor(
                    None,
                    lambda: self._send_email(
                        to=str(args.get("to") or ""),
                        subject=str(args.get("subject") or ""),
                        body=str(args.get("body") or ""),
                    ),
                )
                return {"success": True, "data": data, "error": None}

            if tool_name == "read_emails":
                folder = str(args.get("folder", "INBOX") or "INBOX")
                limit = max(1, int(args.get("limit", 10) or 10))
                emails_data = await loop.run_in_executor(None, lambda: self._read_emails(folder=folder, limit=limit))
                return {"success": True, "data": {"emails": emails_data}, "error": None}

            if tool_name == "search_emails":
                query = str(args.get("query", "")).strip()
                if not query:
                    return {"success": False, "data": None, "error": "query is required"}
                data = await loop.run_in_executor(None, lambda: self._search_emails(query=query))
                return {"success": True, "data": data, "error": None}

            return {"success": False, "data": None, "error": f"Unknown tool: {tool_name}"}
        except Exception as exc:  # noqa: BLE001
            return {"success": False, "data": None, "error": str(exc)}

    def _send_email(self, to: str, subject: str, body: str) -> dict[str, Any]:
        if not to.strip():
            raise ValueError("to is required")

        addr = os.environ["EMAIL_ADDRESS"]
        pwd = os.environ["EMAIL_PASSWORD"]
        host = os.environ["SMTP_HOST"]
        port = int(os.environ.get("SMTP_PORT", "587"))

        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = addr
        msg["To"] = to

        with smtplib.SMTP(host, port, timeout=10) as client:
            client.starttls()
            client.login(addr, pwd)
            client.send_message(msg)

        return {"sent_to": to}

    def _read_emails(self, folder: str = "INBOX", limit: int = 10) -> list[dict[str, Any]]:
        addr = os.environ["EMAIL_ADDRESS"]
        pwd = os.environ["EMAIL_PASSWORD"]
        host = os.environ["IMAP_HOST"]

        with imaplib.IMAP4_SSL(host, timeout=10) as client:
            client.login(addr, pwd)
            client.select(folder)
            _, data = client.search(None, "ALL")
            ids = (data[0] if data else b"").split()[-limit:]

            results: list[dict[str, Any]] = []
            for email_id in reversed(ids):
                _, fetched = client.fetch(email_id, "(RFC822)")
                if not fetched or not fetched[0] or not isinstance(fetched[0], tuple):
                    continue
                msg = email_lib.message_from_bytes(fetched[0][1])
                results.append(
                    {
                        "from": msg.get("From"),
                        "subject": msg.get("Subject"),
                        "date": msg.get("Date"),
                    }
                )

        return results

    def _search_emails(self, query: str) -> dict[str, Any]:
        addr = os.environ["EMAIL_ADDRESS"]
        pwd = os.environ["EMAIL_PASSWORD"]
        host = os.environ["IMAP_HOST"]
        safe_query = query.replace('"', "")

        with imaplib.IMAP4_SSL(host, timeout=10) as client:
            client.login(addr, pwd)
            client.select("INBOX")
            _, data = client.search(None, f'SUBJECT "{safe_query}"')
            ids = (data[0] if data else b"").split()
            return {
                "matches": len(ids),
                "ids": [item.decode("utf-8", errors="ignore") for item in ids[-10:]],
            }


__all__ = ["EmailIntegration"]




# --- FILE: integrations/clients/github.py ---

"""GitHub integration via PyGithub.

Required env vars:
    GITHUB_TOKEN

Optional env vars:
    GITHUB_DEFAULT_REPO
"""

# internal import removed: from __future__ import annotations

import asyncio
import os
from typing import Any, Iterable

# internal import removed: from integrations.base import BaseIntegration

_DEFAULT_LIMIT = 20
_MAX_LIMIT = 100


class GitHubIntegration(BaseIntegration):
    """Read repository state and perform confirm-gated write actions on GitHub."""

    name = "github"
    description = "Inspect GitHub repositories, pull requests, issues, gists, and code search"
    required_config: list[str] = ["GITHUB_TOKEN"]

    def is_available(self) -> bool:
        try:
            import github  # noqa: F401
        except ImportError:
            self.unavailable_reason = "PyGithub not installed"
            return False

        if not os.environ.get("GITHUB_TOKEN"):
            self.unavailable_reason = "Missing env var: GITHUB_TOKEN"
            return False
        return True

    def get_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "list_open_issues",
                "description": "List open GitHub issues, optionally filtered by label, assignee, or milestone",
                "risk": "low",
                "args": {
                    "repo": {"type": "string", "description": "Repository like owner/name", "default": ""},
                    "label": {"type": "string", "description": "Optional label filter", "default": ""},
                    "assignee": {"type": "string", "description": "Optional assignee login filter", "default": ""},
                    "milestone": {
                        "type": "string",
                        "description": "Optional milestone title or number",
                        "default": "",
                    },
                    "limit": {"type": "integer", "default": 20},
                },
                "required_args": [],
            },
            {
                "name": "create_issue",
                "description": "Create a GitHub issue in the target repository",
                "risk": "confirm",
                "args": {
                    "repo": {"type": "string", "description": "Repository like owner/name", "default": ""},
                    "title": {"type": "string", "description": "Issue title"},
                    "body": {"type": "string", "description": "Issue body", "default": ""},
                    "labels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional label names",
                        "default": [],
                    },
                },
                "required_args": ["title"],
            },
            {
                "name": "close_issue",
                "description": "Close a GitHub issue by number",
                "risk": "confirm",
                "args": {
                    "repo": {"type": "string", "description": "Repository like owner/name", "default": ""},
                    "issue_number": {"type": "integer", "description": "GitHub issue number"},
                },
                "required_args": ["issue_number"],
            },
            {
                "name": "list_open_prs",
                "description": "List open pull requests in the target repository",
                "risk": "low",
                "args": {
                    "repo": {"type": "string", "description": "Repository like owner/name", "default": ""},
                    "limit": {"type": "integer", "default": 20},
                },
                "required_args": [],
            },
            {
                "name": "get_pr_diff",
                "description": "Return a summarized view of changed files and patch excerpts for a pull request",
                "risk": "low",
                "args": {
                    "repo": {"type": "string", "description": "Repository like owner/name", "default": ""},
                    "pr_number": {"type": "integer", "description": "Pull request number"},
                    "max_files": {"type": "integer", "default": 20},
                    "max_patch_chars": {"type": "integer", "default": 6000},
                },
                "required_args": ["pr_number"],
            },
            {
                "name": "create_gist",
                "description": "Create a GitHub gist from a code snippet or text file",
                "risk": "confirm",
                "args": {
                    "filename": {"type": "string", "description": "Gist filename"},
                    "content": {"type": "string", "description": "Full gist file content"},
                    "description": {"type": "string", "description": "Optional gist description", "default": ""},
                    "public": {"type": "boolean", "description": "Whether the gist is public", "default": False},
                },
                "required_args": ["filename", "content"],
            },
            {
                "name": "search_code",
                "description": "Search GitHub code, optionally scoped to a repository",
                "risk": "low",
                "args": {
                    "query": {"type": "string", "description": "Code search query"},
                    "repo": {"type": "string", "description": "Optional repository like owner/name", "default": ""},
                    "limit": {"type": "integer", "default": 10},
                },
                "required_args": ["query"],
            },
        ]

    async def execute(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        args = args or {}

        operations = {
            "list_open_issues": self._list_open_issues,
            "create_issue": self._create_issue,
            "close_issue": self._close_issue,
            "list_open_prs": self._list_open_prs,
            "get_pr_diff": self._get_pr_diff,
            "create_gist": self._create_gist,
            "search_code": self._search_code,
        }

        operation = operations.get(tool_name)
        if operation is None:
            return {"success": False, "data": None, "error": f"Unknown tool: {tool_name}"}

        try:
            loop = asyncio.get_running_loop()
            data = await loop.run_in_executor(None, lambda: operation(args))
            return {"success": True, "data": data, "error": None}
        except Exception as exc:  # noqa: BLE001
            return {"success": False, "data": None, "error": str(exc)}

    def _get_client(self):
        from github import Github

        return Github(os.environ["GITHUB_TOKEN"], per_page=_MAX_LIMIT)

    def _make_input_file_content(self, content: str):
        from github import InputFileContent

        return InputFileContent(content)

    def _resolve_repo_name(self, args: dict[str, Any]) -> str:
        repo_name = str(args.get("repo", "") or "").strip()
        if repo_name:
            return repo_name

        repo_name = str(os.environ.get("GITHUB_DEFAULT_REPO", "") or "").strip()
        if repo_name:
            return repo_name

        raise ValueError("repo is required when GITHUB_DEFAULT_REPO is not configured")

    def _get_repo(self, client, args: dict[str, Any]):
        return client.get_repo(self._resolve_repo_name(args))

    def _coerce_limit(self, raw_value: Any, *, default: int, max_value: int = _MAX_LIMIT) -> int:
        try:
            value = int(raw_value)
        except (ValueError, TypeError):
            value = default
        return max(1, min(max_value, value))

    def _take(self, items: Iterable[Any], limit: int) -> list[Any]:
        result: list[Any] = []
        for item in items:
            result.append(item)
            if len(result) >= limit:
                break
        return result

    def _matches_issue_filters(self, issue: Any, args: dict[str, Any]) -> bool:
        label = str(args.get("label", "") or "").strip().lower()
        assignee = str(args.get("assignee", "") or "").strip().lower()
        milestone = str(args.get("milestone", "") or "").strip()

        if label:
            labels = {str(getattr(item, "name", "")).strip().lower() for item in getattr(issue, "labels", [])}
            if label not in labels:
                return False

        if assignee:
            assignees = {
                str(getattr(item, "login", "")).strip().lower()
                for item in getattr(issue, "assignees", [])
                if getattr(item, "login", None)
            }
            if assignee not in assignees:
                return False

        if milestone:
            issue_milestone = getattr(issue, "milestone", None)
            if issue_milestone is None:
                return False

            milestone_title = str(getattr(issue_milestone, "title", "") or "")
            milestone_number = str(getattr(issue_milestone, "number", "") or "")
            if milestone not in {milestone_title, milestone_number}:
                return False

        return True

    def _list_open_issues(self, args: dict[str, Any]) -> dict[str, Any]:
        client = self._get_client()
        repo = self._get_repo(client, args)
        limit = self._coerce_limit(args.get("limit", _DEFAULT_LIMIT), default=_DEFAULT_LIMIT)

        issues = self._take(repo.get_issues(state="open"), max(limit * 3, limit))
        filtered = [issue for issue in issues if self._matches_issue_filters(issue, args)][:limit]

        return {
            "repo": repo.full_name,
            "issues": [
                {
                    "number": issue.number,
                    "title": issue.title,
                    "url": issue.html_url,
                    "state": issue.state,
                    "author": getattr(getattr(issue, "user", None), "login", None),
                    "labels": [getattr(label, "name", "") for label in getattr(issue, "labels", [])],
                    "assignees": [getattr(user, "login", "") for user in getattr(issue, "assignees", [])],
                    "milestone": getattr(getattr(issue, "milestone", None), "title", None),
                }
                for issue in filtered
            ],
            "count": len(filtered),
        }

    def _create_issue(self, args: dict[str, Any]) -> dict[str, Any]:
        title = str(args.get("title", "") or "").strip()
        if not title:
            raise ValueError("title is required")

        client = self._get_client()
        repo = self._get_repo(client, args)
        
        raw_labels = args.get("labels")
        if raw_labels is None:
            raw_labels = []
        labels = [str(item).strip() for item in raw_labels if str(item).strip()]

        create_kwargs: dict[str, Any] = {
            "title": title,
            "body": str(args.get("body", "") or ""),
        }
        if labels:
            create_kwargs["labels"] = labels

        issue = repo.create_issue(**create_kwargs)

        return {
            "repo": repo.full_name,
            "number": issue.number,
            "title": issue.title,
            "url": issue.html_url,
            "state": issue.state,
        }

    def _close_issue(self, args: dict[str, Any]) -> dict[str, Any]:
        if "issue_number" not in args:
            raise ValueError("issue_number is required")

        issue_number = int(args["issue_number"])
        client = self._get_client()
        repo = self._get_repo(client, args)
        issue = repo.get_issue(number=issue_number)
        issue.edit(state="closed")

        return {
            "repo": repo.full_name,
            "number": issue.number,
            "title": issue.title,
            "state": issue.state,
            "url": issue.html_url,
        }

    def _list_open_prs(self, args: dict[str, Any]) -> dict[str, Any]:
        client = self._get_client()
        repo = self._get_repo(client, args)
        limit = self._coerce_limit(args.get("limit", _DEFAULT_LIMIT), default=_DEFAULT_LIMIT)
        pulls = self._take(repo.get_pulls(state="open", sort="created", direction="desc"), limit)

        return {
            "repo": repo.full_name,
            "pull_requests": [
                {
                    "number": pull.number,
                    "title": pull.title,
                    "url": pull.html_url,
                    "state": pull.state,
                    "draft": bool(getattr(pull, "draft", False)),
                    "author": getattr(getattr(pull, "user", None), "login", None),
                    "head": getattr(getattr(pull, "head", None), "ref", None),
                    "base": getattr(getattr(pull, "base", None), "ref", None),
                }
                for pull in pulls
            ],
            "count": len(pulls),
        }

    def _truncate_patch(self, patch: str, remaining_chars: int) -> tuple[str, int]:
        if remaining_chars <= 0:
            return "", 0

        if len(patch) <= remaining_chars:
            return patch, remaining_chars - len(patch)

        marker = "\n... [truncated]"
        allowed = max(0, remaining_chars - len(marker))
        excerpt = patch[:allowed] + marker
        return excerpt, 0

    def _get_pr_diff(self, args: dict[str, Any]) -> dict[str, Any]:
        if "pr_number" not in args:
            raise ValueError("pr_number is required")

        client = self._get_client()
        repo = self._get_repo(client, args)
        pull = repo.get_pull(number=int(args["pr_number"]))

        max_files = self._coerce_limit(args.get("max_files", 20), default=20, max_value=100)
        remaining_patch_chars = self._coerce_limit(
            args.get("max_patch_chars", 6000),
            default=6000,
            max_value=20000,
        )

        files = []
        truncated = False
        for changed_file in self._take(pull.get_files(), max_files):
            patch = str(getattr(changed_file, "patch", "") or "")
            excerpt, remaining_patch_chars = self._truncate_patch(patch, remaining_patch_chars)
            if patch and excerpt != patch:
                truncated = True
            elif patch and remaining_patch_chars == 0 and excerpt:
                truncated = True
            elif patch and not excerpt:
                truncated = True

            files.append(
                {
                    "filename": changed_file.filename,
                    "status": changed_file.status,
                    "additions": changed_file.additions,
                    "deletions": changed_file.deletions,
                    "changes": changed_file.changes,
                    "patch_excerpt": excerpt,
                }
            )
            if remaining_patch_chars <= 0:
                break

        if getattr(pull, "changed_files", 0) > len(files):
            truncated = True

        return {
            "repo": repo.full_name,
            "number": pull.number,
            "title": pull.title,
            "url": pull.html_url,
            "state": pull.state,
            "author": getattr(getattr(pull, "user", None), "login", None),
            "merged": bool(getattr(pull, "merged", False)),
            "mergeable": getattr(pull, "mergeable", None),
            "commits": getattr(pull, "commits", None),
            "changed_files": getattr(pull, "changed_files", len(files)),
            "additions": getattr(pull, "additions", None),
            "deletions": getattr(pull, "deletions", None),
            "files": files,
            "truncated": truncated,
        }

    def _create_gist(self, args: dict[str, Any]) -> dict[str, Any]:
        filename = str(args.get("filename", "") or "").strip()
        if not filename:
            raise ValueError("filename is required")

        if "content" not in args or not str(args.get("content", "")).strip():
            raise ValueError("content is required")

        client = self._get_client()
        user = client.get_user()
        gist = user.create_gist(
            public=bool(args.get("public", False)),
            files={filename: self._make_input_file_content(str(args.get("content", "")))},
            description=str(args.get("description", "") or ""),
        )

        return {
            "id": gist.id,
            "url": gist.html_url,
            "public": gist.public,
            "description": gist.description,
        }

    def _search_code(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query", "") or "").strip()
        if not query:
            raise ValueError("query is required")

        limit = self._coerce_limit(args.get("limit", 10), default=10, max_value=50)
        repo_name = str(args.get("repo", "") or "").strip() or str(os.environ.get("GITHUB_DEFAULT_REPO", "") or "").strip()
        effective_query = f"{query} repo:{repo_name}" if repo_name else query

        client = self._get_client()
        results = self._take(client.search_code(effective_query), limit)

        return {
            "query": effective_query,
            "results": [
                {
                    "name": getattr(result, "name", None),
                    "path": getattr(result, "path", None),
                    "repository": getattr(getattr(result, "repository", None), "full_name", None),
                    "url": getattr(result, "html_url", None),
                    "sha": getattr(result, "sha", None),
                }
                for result in results
            ],
            "count": len(results),
        }


__all__ = ["GitHubIntegration"]




# --- FILE: integrations/clients/gmail.py ---

"""Gmail integration via Gmail API v1 (async aiohttp).

Uses the same Google OAuth credentials as google_calendar.py.

Required env vars:
    GOOGLE_CLIENT_ID
    GOOGLE_CLIENT_SECRET
    GOOGLE_REFRESH_TOKEN

Rules:
- Email content is ALWAYS truncated to 2 000 chars before any LLM injection
- summarize_unread uses task_type="synthesis"
- No raw email headers injected blindly into context
"""

# internal import removed: from __future__ import annotations

import aiohttp
import base64
import os
from email.mime.text import MIMEText
from typing import Any

# internal import removed: from integrations.base import BaseIntegration

integrations_clients_gmail__TOKEN_URL = "https://oauth2.googleapis.com/token"
_GMAIL_BASE = "https://gmail.googleapis.com/gmail/v1/users/me"
_MAX_BODY_CHARS = 2000  # Truncation guard before any LLM injection


class GmailIntegration(BaseIntegration):
    """Gmail API v1 integration — async, token-refreshing, truncation-safe."""

    name = "gmail"
    description = "Read, send, and manage Gmail messages"
    required_config: list[str] = [
        "GOOGLE_CLIENT_ID",
        "GOOGLE_CLIENT_SECRET",
        "GOOGLE_REFRESH_TOKEN",
    ]

    def is_available(self) -> bool:
        try:
            import aiohttp  # noqa: F401
        except ImportError:
            self.unavailable_reason = "aiohttp not installed"
            return False
        if not all(bool(os.environ.get(k)) for k in self.required_config):
            missing = [k for k in self.required_config if not os.environ.get(k)]
            self.unavailable_reason = f"Missing env vars: {missing}"
            return False
        return True

    def get_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "list_unread",
                "description": "List unread emails from Gmail inbox",
                "risk": "low",
                "args": {
                    "max_results": {
                        "type": "integer",
                        "description": "Max emails to return",
                        "default": 10,
                    },
                },
                "required_args": [],
            },
            {
                "name": "send_gmail",
                "description": "Send an email via Gmail",
                "risk": "confirm",
                "args": {
                    "to": {"type": "string", "description": "Recipient email address"},
                    "subject": {"type": "string", "description": "Email subject"},
                    "body": {"type": "string", "description": "Plain-text email body"},
                },
                "required_args": ["to", "subject", "body"],
            },
            {
                "name": "summarize_unread",
                "description": "Fetch unread emails and return truncated content for LLM summarization",
                "risk": "low",
                "args": {
                    "max_results": {"type": "integer", "default": 5},
                },
                "required_args": [],
            },
            {
                "name": "mark_as_read",
                "description": "Mark a Gmail message as read by its message ID",
                "risk": "confirm",
                "args": {
                    "message_id": {
                        "type": "string",
                        "description": "Gmail message ID",
                    },
                },
                "required_args": ["message_id"],
            },
        ]

    async def execute(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        args = args or {}
        try:
            token = await self._refresh_access_token()
            if tool_name == "list_unread":
                return await self._list_unread(token, int(args.get("max_results", 10) or 10))
            if tool_name == "send_gmail":
                return await self._send_gmail(
                    token,
                    to=str(args.get("to") or ""),
                    subject=str(args.get("subject") or ""),
                    body=str(args.get("body") or ""),
                )
            if tool_name == "summarize_unread":
                return await self._summarize_unread(token, int(args.get("max_results", 5) or 5))
            if tool_name == "mark_as_read":
                return await self._mark_as_read(token, str(args.get("message_id") or ""))
            return {"success": False, "data": None, "error": f"Unknown tool: {tool_name}"}
        except Exception as exc:  # noqa: BLE001
            return {"success": False, "data": None, "error": str(exc)}

    # ── OAuth ─────────────────────────────────────────────────────────────────

    async def _refresh_access_token(self) -> str:
        import aiohttp

        payload = {
            "client_id": os.environ["GOOGLE_CLIENT_ID"],
            "client_secret": os.environ["GOOGLE_CLIENT_SECRET"],
            "refresh_token": os.environ["GOOGLE_REFRESH_TOKEN"],
            "grant_type": "refresh_token",
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(integrations_clients_gmail__TOKEN_URL, data=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                data = await resp.json()
                if "access_token" not in data:
                    raise RuntimeError(f"Token refresh failed: {data.get('error', 'unknown')}")
                return str(data["access_token"])

    # ── Tool implementations ──────────────────────────────────────────────────

    async def _list_unread(self, token: str, max_results: int) -> dict[str, Any]:
        import aiohttp

        headers = {"Authorization": f"Bearer {token}"}
        params: dict[str, str | int] = {"q": "is:unread", "maxResults": min(max_results, 50)}

        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{_GMAIL_BASE}/messages",
                headers=headers,
                params=params,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                data = await resp.json()
                if resp.status != 200:
                    return {
                        "success": False,
                        "data": None,
                        "error": data.get("error", {}).get("message", str(resp.status)),
                    }

            messages = data.get("messages", [])
            summaries = []
            for m in messages[:max_results]:
                meta = await self._get_message_meta(session, headers, m["id"])
                summaries.append(meta)

        return {"success": True, "data": {"unread": summaries, "total": data.get("resultSizeEstimate", 0)}, "error": None}

    async def _get_message_meta(self, session: Any, headers: dict, message_id: str) -> dict[str, Any]:
        async with session.get(
            f"{_GMAIL_BASE}/messages/{message_id}",
            headers=headers,
            params={"format": "metadata", "metadataHeaders": ["From", "Subject", "Date"]},
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            data = await resp.json()
            if resp.status != 200:
                return {
                    "id": message_id,
                    "from": "",
                    "subject": "Error fetching message",
                    "date": "",
                    "snippet": data.get("error", {}).get("message", f"HTTP {resp.status}"),
                }

        header_map: dict[str, str] = {}
        for h in data.get("payload", {}).get("headers", []):
            header_map[h["name"]] = h["value"]

        return {
            "id": message_id,
            "from": header_map.get("From", ""),
            "subject": header_map.get("Subject", "")[:200],  # truncate subject
            "date": header_map.get("Date", ""),
            "snippet": data.get("snippet", "")[:_MAX_BODY_CHARS],
        }

    async def _summarize_unread(self, token: str, max_results: int) -> dict[str, Any]:
        """Return truncated email content safe for LLM summarization."""
        result = await self._list_unread(token, max_results)
        if not result["success"]:
            return result
        # Content is already truncated in _get_message_meta; add metadata
        emails = result["data"]["unread"]
        safe_content = [
            {
                "from": e["from"],
                "subject": e["subject"],
                "snippet": e["snippet"][:_MAX_BODY_CHARS],
                "id": e["id"],
            }
            for e in emails
        ]
        return {
            "success": True,
            "data": {
                "emails_for_summary": safe_content,
                "count": len(safe_content),
                "task_type": "synthesis",  # hint for LLM router
            },
            "error": None,
        }

    async def _send_gmail(self, token: str, to: str, subject: str, body: str) -> dict[str, Any]:
        import aiohttp

        if not to.strip():
            return {"success": False, "data": None, "error": "to is required"}

        msg = MIMEText(body, "plain")
        msg["To"] = to
        msg["Subject"] = subject
        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")

        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        payload = {"raw": raw}

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{_GMAIL_BASE}/messages/send",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                data = await resp.json()
                if resp.status not in (200, 201):
                    return {
                        "success": False,
                        "data": None,
                        "error": data.get("error", {}).get("message", str(resp.status)),
                    }
                return {"success": True, "data": {"message_id": data.get("id")}, "error": None}

    async def _mark_as_read(self, token: str, message_id: str) -> dict[str, Any]:
        import aiohttp

        if not message_id.strip():
            return {"success": False, "data": None, "error": "message_id is required"}

        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        payload = {"removeLabelIds": ["UNREAD"]}

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{_GMAIL_BASE}/messages/{message_id}/modify",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status == 200:
                    return {"success": True, "data": {"marked_read": message_id}, "error": None}
                data = await resp.json()
                return {
                    "success": False,
                    "data": None,
                    "error": data.get("error", {}).get("message", str(resp.status)),
                }


__all__ = ["GmailIntegration"]




# --- FILE: integrations/clients/google_calendar.py ---

"""Google Calendar integration via Google Calendar API v3.

Uses fully async aiohttp for all HTTP calls. OAuth is handled via
refresh token — no browser popup at runtime.

Required env vars:
    GOOGLE_CLIENT_ID
    GOOGLE_CLIENT_SECRET
    GOOGLE_REFRESH_TOKEN
"""

# internal import removed: from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta
from typing import Any

# internal import removed: from integrations.base import BaseIntegration

integrations_clients_google_calendar__TOKEN_URL = "https://oauth2.googleapis.com/token"
_CALENDAR_BASE = "https://www.googleapis.com/calendar/v3"


class GoogleCalendarIntegration(BaseIntegration):
    """Google Calendar v3 integration (async, RFC3339, OAuth refresh)."""

    name = "google_calendar"
    description = "Create, list, and delete Google Calendar events"
    required_config: list[str] = [
        "GOOGLE_CLIENT_ID",
        "GOOGLE_CLIENT_SECRET",
        "GOOGLE_REFRESH_TOKEN",
    ]

    def is_available(self) -> bool:
        try:
            import aiohttp  # noqa: F401
        except ImportError:
            self.unavailable_reason = "aiohttp not installed"
            return False
        if not all(bool(os.environ.get(k)) for k in self.required_config):
            missing = [k for k in self.required_config if not os.environ.get(k)]
            self.unavailable_reason = f"Missing env vars: {missing}"
            return False
        return True

    def get_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "create_event",
                "description": "Create a new event in Google Calendar",
                "risk": "confirm",
                "args": {
                    "summary": {"type": "string", "description": "Event title"},
                    "start": {
                        "type": "string",
                        "description": "Start datetime in ISO-8601 (e.g. 2026-03-15T10:00:00)",
                    },
                    "end": {
                        "type": "string",
                        "description": "End datetime in ISO-8601",
                    },
                    "description": {
                        "type": "string",
                        "description": "Event description",
                        "default": "",
                    },
                    "timezone": {
                        "type": "string",
                        "description": "IANA timezone (e.g. Asia/Kolkata)",
                        "default": "UTC",
                    },
                    "calendar_id": {
                        "type": "string",
                        "description": "Calendar ID (default: primary)",
                        "default": "primary",
                    },
                },
                "required_args": ["summary", "start", "end"],
            },
            {
                "name": "list_events",
                "description": "List upcoming Google Calendar events",
                "risk": "low",
                "args": {
                    "days_ahead": {"type": "integer", "description": "Look ahead N days", "default": 7},
                    "max_results": {"type": "integer", "description": "Max events to return", "default": 10},
                    "calendar_id": {"type": "string", "default": "primary"},
                },
                "required_args": [],
            },
            {
                "name": "delete_event",
                "description": "Delete a Google Calendar event by its event ID",
                "risk": "confirm",
                "args": {
                    "event_id": {"type": "string", "description": "Google Calendar event ID"},
                    "calendar_id": {"type": "string", "default": "primary"},
                },
                "required_args": ["event_id"],
            },
            {
                "name": "find_free_slot",
                "description": "Find the next free time slot of a given duration",
                "risk": "low",
                "args": {
                    "duration_minutes": {
                        "type": "integer",
                        "description": "Required free slot duration in minutes",
                        "default": 60,
                    },
                    "days_ahead": {"type": "integer", "default": 7},
                    "calendar_id": {"type": "string", "default": "primary"},
                },
                "required_args": [],
            },
        ]

    async def execute(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        args = args or {}
        try:
            token = await self._refresh_access_token()
            if tool_name == "create_event":
                return await self._create_event(token, args)
            if tool_name == "list_events":
                return await self._list_events(token, args)
            if tool_name == "delete_event":
                return await self._delete_event(token, args)
            if tool_name == "find_free_slot":
                return await self._find_free_slot(token, args)
            return {"success": False, "data": None, "error": f"Unknown tool: {tool_name}"}
        except Exception as exc:  # noqa: BLE001
            return {"success": False, "data": None, "error": str(exc)}

    async def _refresh_access_token(self) -> str:
        import aiohttp

        payload = {
            "client_id": os.environ["GOOGLE_CLIENT_ID"],
            "client_secret": os.environ["GOOGLE_CLIENT_SECRET"],
            "refresh_token": os.environ["GOOGLE_REFRESH_TOKEN"],
            "grant_type": "refresh_token",
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(integrations_clients_google_calendar__TOKEN_URL, data=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                data = await resp.json()
                if "access_token" not in data:
                    raise RuntimeError(f"Token refresh failed: {data.get('error', 'unknown')}")
                return str(data["access_token"])

    def _to_rfc3339(self, dt_str: str, tz: str = "UTC") -> str:
        """Parse ISO-8601 string and return RFC3339 with timezone offset."""
        # Try with offset first, then naive
        for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S"):
            try:
                dt = datetime.strptime(dt_str.strip(), fmt)
                # If naive, we don't attach UTC, allowing Google Calendar to use the timeZone field
                return dt.isoformat()
            except ValueError:
                continue
        raise ValueError(f"Cannot parse datetime: {dt_str!r}")

    async def _create_event(self, token: str, args: dict[str, Any]) -> dict[str, Any]:
        import aiohttp

        summary = str(args.get("summary") or "").strip()
        if not summary:
            return {"success": False, "data": None, "error": "summary is required"}

        tz = str(args.get("timezone", "UTC") or "UTC")
        start_str = str(args.get("start") or "")
        end_str = str(args.get("end") or "")
        cal_id = str(args.get("calendar_id", "primary") or "primary")

        event_body = {
            "summary": summary,
            "description": str(args.get("description", "") or ""),
            "start": {"dateTime": self._to_rfc3339(start_str), "timeZone": tz},
            "end": {"dateTime": self._to_rfc3339(end_str), "timeZone": tz},
        }

        url = f"{_CALENDAR_BASE}/calendars/{cal_id}/events"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json=event_body, headers=headers, timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                data = await resp.json()
                if resp.status not in (200, 201):
                    return {
                        "success": False,
                        "data": None,
                        "error": data.get("error", {}).get("message", str(resp.status)),
                    }
                return {"success": True, "data": {"event_id": data["id"], "link": data.get("htmlLink")}, "error": None}

    async def _list_events(self, token: str, args: dict[str, Any]) -> dict[str, Any]:
        import aiohttp

        days_ahead = int(args.get("days_ahead", 7) or 7)
        max_results = min(50, int(args.get("max_results", 10) or 10))
        cal_id = str(args.get("calendar_id", "primary") or "primary")

        now = datetime.now(tz=timezone.utc)
        time_min = now.isoformat()
        time_max = (now + timedelta(days=days_ahead)).isoformat()

        url = f"{_CALENDAR_BASE}/calendars/{cal_id}/events"
        headers = {"Authorization": f"Bearer {token}"}
        params: dict[str, str | int] = {
            "timeMin": time_min,
            "timeMax": time_max,
            "maxResults": max_results,
            "singleEvents": "true",
            "orderBy": "startTime",
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, headers=headers, params=params, timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                data = await resp.json()
                if resp.status != 200:
                    return {
                        "success": False,
                        "data": None,
                        "error": data.get("error", {}).get("message", str(resp.status)),
                    }
                events = [
                    {
                        "id": item["id"],
                        "summary": item.get("summary", "Untitled"),
                        "start": item.get("start", {}).get("dateTime") or item.get("start", {}).get("date"),
                        "end": item.get("end", {}).get("dateTime") or item.get("end", {}).get("date"),
                    }
                    for item in data.get("items", [])
                ]
                return {"success": True, "data": {"events": events}, "error": None}

    async def _delete_event(self, token: str, args: dict[str, Any]) -> dict[str, Any]:
        import aiohttp

        event_id = str(args.get("event_id") or "").strip()
        if not event_id:
            return {"success": False, "data": None, "error": "event_id is required"}
        cal_id = str(args.get("calendar_id", "primary") or "primary")

        url = f"{_CALENDAR_BASE}/calendars/{cal_id}/events/{event_id}"
        headers = {"Authorization": f"Bearer {token}"}

        async with aiohttp.ClientSession() as session:
            async with session.delete(url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status == 204:
                    return {"success": True, "data": {"deleted": event_id}, "error": None}
                body = await resp.json()
                return {
                    "success": False,
                    "data": None,
                    "error": body.get("error", {}).get("message", str(resp.status)),
                }

    async def _find_free_slot(self, token: str, args: dict[str, Any]) -> dict[str, Any]:
        """Find the earliest free slot of the requested duration using freebusy query."""
        import aiohttp

        duration_min = int(args.get("duration_minutes", 60) or 60)
        days_ahead = int(args.get("days_ahead", 7) or 7)
        cal_id = str(args.get("calendar_id", "primary") or "primary")

        now = datetime.now(tz=timezone.utc)
        time_max = (now + timedelta(days=days_ahead)).isoformat()

        url = f"{_CALENDAR_BASE}/freeBusy"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        body = {
            "timeMin": now.isoformat(),
            "timeMax": time_max,
            "items": [{"id": cal_id}],
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=body, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                data = await resp.json()
                if resp.status != 200:
                    return {
                        "success": False,
                        "data": None,
                        "error": data.get("error", {}).get("message", str(resp.status)),
                    }

        busy_slots = data.get("calendars", {}).get(cal_id, {}).get("busy", [])
        # Find first gap of >= duration_min minutes
        cursor = now
        for slot in busy_slots:
            busy_start = datetime.fromisoformat(slot["start"].replace("Z", "+00:00"))
            gap = (busy_start - cursor).total_seconds() / 60
            if gap >= duration_min:
                return {
                    "success": True,
                    "data": {
                        "free_start": cursor.isoformat(),
                        "free_end": (cursor + timedelta(minutes=duration_min)).isoformat(),
                    },
                    "error": None,
                }
            busy_end = datetime.fromisoformat(slot["end"].replace("Z", "+00:00"))
            if busy_end > cursor:
                cursor = busy_end

        # Gap after all busy slots
        return {
            "success": True,
            "data": {
                "free_start": cursor.isoformat(),
                "free_end": (cursor + timedelta(minutes=duration_min)).isoformat(),
            },
            "error": None,
        }


__all__ = ["GoogleCalendarIntegration"]




# --- FILE: integrations/clients/home_assistant.py ---

"""Home Assistant integration via the REST API.

Required env vars:
    HOME_ASSISTANT_URL
    HOME_ASSISTANT_TOKEN
"""

# internal import removed: from __future__ import annotations

import os
import time
from typing import Any
from urllib.parse import quote

# internal import removed: from integrations.base import BaseIntegration

_ENTITY_CACHE_TTL_SECONDS = 60
_SENSITIVE_DOMAINS = {"lock", "alarm_control_panel"}


class HomeAssistantIntegration(BaseIntegration):
    """Read entity state and call Home Assistant services."""

    name = "home_assistant"
    description = "Inspect entities and control smart-home devices through Home Assistant"
    required_config: list[str] = ["HOME_ASSISTANT_URL", "HOME_ASSISTANT_TOKEN"]

    def __init__(self, config: Any | None = None) -> None:
        super().__init__(config=config)
        self._entity_cache: list[dict[str, Any]] = []
        self._entity_cache_at: float = 0.0

    def is_available(self) -> bool:
        try:
            import aiohttp  # noqa: F401
        except ImportError:
            self.unavailable_reason = "aiohttp not installed"
            return False

        if not all(bool(os.environ.get(key)) for key in self.required_config):
            missing = [key for key in self.required_config if not os.environ.get(key)]
            self.unavailable_reason = f"Missing env vars: {missing}"
            return False
        return True

    def get_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "get_entity_state",
                "description": "Get the current state and friendly name for a Home Assistant entity",
                "risk": "low",
                "args": {
                    "entity_id": {"type": "string", "description": "Entity ID like light.kitchen"},
                },
                "required_args": ["entity_id"],
            },
            {
                "name": "turn_on_entity",
                "description": "Turn on a Home Assistant light, switch, fan, or similar entity",
                "risk": "confirm",
                "args": {
                    "entity_id": {"type": "string", "description": "Single entity ID", "default": ""},
                    "entity_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of entity IDs",
                        "default": [],
                    },
                    "area_id": {"type": "string", "description": "Optional Home Assistant area ID", "default": ""},
                    "device_id": {"type": "string", "description": "Optional Home Assistant device ID", "default": ""},
                    "domain": {"type": "string", "description": "Required when targeting an area or device", "default": ""},
                    "service_data": {
                        "type": "object",
                        "description": "Optional extra service data like brightness_pct",
                        "default": {},
                    },
                },
                "required_args": [],
            },
            {
                "name": "turn_off_entity",
                "description": "Turn off a Home Assistant light, switch, fan, or similar entity",
                "risk": "confirm",
                "args": {
                    "entity_id": {"type": "string", "description": "Single entity ID", "default": ""},
                    "entity_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of entity IDs",
                        "default": [],
                    },
                    "area_id": {"type": "string", "description": "Optional Home Assistant area ID", "default": ""},
                    "device_id": {"type": "string", "description": "Optional Home Assistant device ID", "default": ""},
                    "domain": {"type": "string", "description": "Required when targeting an area or device", "default": ""},
                    "service_data": {
                        "type": "object",
                        "description": "Optional extra service data",
                        "default": {},
                    },
                },
                "required_args": [],
            },
            {
                "name": "toggle_entity",
                "description": "Toggle a Home Assistant light, switch, fan, or similar entity",
                "risk": "confirm",
                "args": {
                    "entity_id": {"type": "string", "description": "Single entity ID", "default": ""},
                    "entity_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of entity IDs",
                        "default": [],
                    },
                    "area_id": {"type": "string", "description": "Optional Home Assistant area ID", "default": ""},
                    "device_id": {"type": "string", "description": "Optional Home Assistant device ID", "default": ""},
                    "domain": {"type": "string", "description": "Required when targeting an area or device", "default": ""},
                    "service_data": {
                        "type": "object",
                        "description": "Optional extra service data",
                        "default": {},
                    },
                },
                "required_args": [],
            },
            {
                "name": "set_thermostat",
                "description": "Set a climate entity target temperature",
                "risk": "confirm",
                "args": {
                    "entity_id": {"type": "string", "description": "Climate entity ID like climate.living_room"},
                    "temperature": {"type": "number", "description": "Target temperature"},
                    "hvac_mode": {
                        "type": "string",
                        "description": "Optional HVAC mode like heat, cool, or auto",
                        "default": "",
                    },
                },
                "required_args": ["entity_id", "temperature"],
            },
            {
                "name": "call_service",
                "description": "Call a specific Home Assistant service for a targeted entity, area, or device",
                "risk": "confirm",
                "args": {
                    "domain": {"type": "string", "description": "Service domain like light or media_player"},
                    "service": {"type": "string", "description": "Service name like turn_on or volume_set"},
                    "entity_id": {"type": "string", "description": "Optional single entity ID", "default": ""},
                    "entity_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of entity IDs",
                        "default": [],
                    },
                    "area_id": {"type": "string", "description": "Optional area ID", "default": ""},
                    "device_id": {"type": "string", "description": "Optional device ID", "default": ""},
                    "service_data": {
                        "type": "object",
                        "description": "Optional extra Home Assistant service fields",
                        "default": {},
                    },
                },
                "required_args": ["domain", "service"],
            },
            {
                "name": "list_entities",
                "description": "List Home Assistant entities, optionally filtered by domain",
                "risk": "low",
                "args": {
                    "domain": {"type": "string", "description": "Optional domain filter like light", "default": ""},
                    "include_attributes": {
                        "type": "boolean",
                        "description": "Include raw Home Assistant attributes in the response",
                        "default": False,
                    },
                },
                "required_args": [],
            },
        ]

    async def execute(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        args = args or {}
        try:
            if tool_name == "get_entity_state":
                return await self._get_entity_state(args)
            if tool_name == "turn_on_entity":
                return await self._entity_service("turn_on", args)
            if tool_name == "turn_off_entity":
                return await self._entity_service("turn_off", args)
            if tool_name == "toggle_entity":
                return await self._entity_service("toggle", args)
            if tool_name == "set_thermostat":
                return await self._set_thermostat(args)
            if tool_name == "call_service":
                return await self._call_service(args)
            if tool_name == "list_entities":
                return await self._list_entities(args)
            return {"success": False, "data": None, "error": f"Unknown tool: {tool_name}"}
        except Exception as exc:  # noqa: BLE001
            return {"success": False, "data": None, "error": str(exc)}

    def _base_url(self) -> str:
        value = str(os.environ["HOME_ASSISTANT_URL"]).strip().rstrip("/")
        if value.endswith("/api"):
            value = value[:-4]
        return value

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {os.environ['HOME_ASSISTANT_TOKEN']}",
            "Content-Type": "application/json",
        }

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_payload: dict[str, Any] | None = None,
    ) -> Any:
        import aiohttp

        url = f"{self._base_url()}{path}"
        timeout = aiohttp.ClientTimeout(total=15)
        async with aiohttp.ClientSession() as session:
            request_fn = getattr(session, method.lower())
            async with request_fn(
                url,
                headers=self._headers(),
                json=json_payload,
                timeout=timeout,
            ) as resp:
                data = await self._read_response(resp)
                if resp.status >= 400:
                    raise RuntimeError(self._extract_error_message(resp.status, data))
                return data

    async def _read_response(self, resp: Any) -> Any:
        if getattr(resp, "status", None) == 204:
            return {}
        return await resp.json(content_type=None)

    def _extract_error_message(self, status: int, data: Any) -> str:
        if isinstance(data, dict):
            for key in ("message", "error", "detail"):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    return value
        if isinstance(data, str) and data.strip():
            return data
        return f"Home Assistant API request failed with HTTP {status}"

    def _invalidate_entity_cache(self) -> None:
        self._entity_cache = []
        self._entity_cache_at = 0.0

    async def _get_states(self, *, force_refresh: bool = False) -> list[dict[str, Any]]:
        is_fresh = (time.monotonic() - self._entity_cache_at) < _ENTITY_CACHE_TTL_SECONDS
        if not force_refresh and self._entity_cache and is_fresh:
            return self._entity_cache

        data = await self._request("get", "/api/states")
        if not isinstance(data, list):
            raise RuntimeError("Unexpected Home Assistant states response")

        self._entity_cache = [item for item in data if isinstance(item, dict)]
        self._entity_cache_at = time.monotonic()
        return self._entity_cache

    def _extract_entity_ids(self, args: dict[str, Any]) -> list[str]:
        ids: list[str] = []

        raw_entity_id = args.get("entity_id", "")
        if isinstance(raw_entity_id, list):
            ids.extend(str(item).strip() for item in raw_entity_id if str(item).strip())
        else:
            value = str(raw_entity_id or "").strip()
            if value:
                ids.append(value)

        raw_entity_ids = args.get("entity_ids", [])
        if isinstance(raw_entity_ids, list):
            ids.extend(str(item).strip() for item in raw_entity_ids if str(item).strip())

        deduped: list[str] = []
        seen: set[str] = set()
        for entity_id in ids:
            if entity_id not in seen:
                deduped.append(entity_id)
                seen.add(entity_id)
        return deduped

    def _build_target(self, args: dict[str, Any]) -> dict[str, Any]:
        target: dict[str, Any] = {}
        entity_ids = self._extract_entity_ids(args)

        if entity_ids:
            target["entity_id"] = entity_ids[0] if len(entity_ids) == 1 else entity_ids

        area_id = str(args.get("area_id", "") or "").strip()
        if area_id:
            target["area_id"] = area_id

        device_id = str(args.get("device_id", "") or "").strip()
        if device_id:
            target["device_id"] = device_id

        return target

    def _infer_domain(self, args: dict[str, Any]) -> str:
        explicit = str(args.get("domain", "") or "").strip().lower()
        if explicit:
            return explicit

        entity_ids = self._extract_entity_ids(args)
        if not entity_ids:
            return ""
        entity_id = entity_ids[0]
        return entity_id.split(".", 1)[0].strip().lower() if "." in entity_id else ""

    def _contains_sensitive_domain(self, args: dict[str, Any], *, explicit_domain: str = "") -> bool:
        domains: set[str] = set()
        if explicit_domain:
            domains.add(explicit_domain.strip().lower())

        for entity_id in self._extract_entity_ids(args):
            if "." in entity_id:
                domains.add(entity_id.split(".", 1)[0].strip().lower())

        return any(domain in _SENSITIVE_DOMAINS for domain in domains)

    def _normalize_service_data(self, args: dict[str, Any]) -> dict[str, Any]:
        service_data = args.get("service_data", {})
        if isinstance(service_data, dict):
            return dict(service_data)
        return {}

    def _format_entity(self, item: dict[str, Any], *, include_attributes: bool = False) -> dict[str, Any]:
        entity_id = str(item.get("entity_id", ""))
        attributes = item.get("attributes") or {}
        payload = {
            "entity_id": entity_id,
            "domain": entity_id.split(".", 1)[0] if "." in entity_id else "",
            "state": item.get("state"),
            "friendly_name": attributes.get("friendly_name") or entity_id,
            "last_changed": item.get("last_changed"),
            "last_updated": item.get("last_updated"),
        }
        if include_attributes:
            payload["attributes"] = attributes
        return payload

    async def _get_entity_state(self, args: dict[str, Any]) -> dict[str, Any]:
        entity_id = str(args.get("entity_id", "") or "").strip()
        if not entity_id:
            return {"success": False, "data": None, "error": "entity_id is required"}

        states = await self._get_states()
        for item in states:
            if str(item.get("entity_id", "")) == entity_id:
                return {"success": True, "data": self._format_entity(item, include_attributes=True), "error": None}

        encoded_entity_id = quote(entity_id, safe="")
        data = await self._request("get", f"/api/states/{encoded_entity_id}")
        if not isinstance(data, dict):
            raise RuntimeError("Unexpected Home Assistant entity response")
        return {"success": True, "data": self._format_entity(data, include_attributes=True), "error": None}

    async def _entity_service(self, service: str, args: dict[str, Any]) -> dict[str, Any]:
        domain = self._infer_domain(args)
        target = self._build_target(args)
        if not target:
            return {
                "success": False,
                "data": None,
                "error": "Provide entity_id, entity_ids, area_id, or device_id",
            }
        if not domain:
            return {
                "success": False,
                "data": None,
                "error": "domain is required when the target is an area or device",
            }
        if self._contains_sensitive_domain(args, explicit_domain=domain):
            return {
                "success": False,
                "data": None,
                "error": "Sensitive domains must use explicit confirm-gated services instead of the convenience helpers",
            }

        payload = self._normalize_service_data(args)
        payload.update(target)
        data = await self._request("post", f"/api/services/{domain}/{service}", json_payload=payload)
        self._invalidate_entity_cache()
        return {
            "success": True,
            "data": {
                "service": f"{domain}.{service}",
                "result": self._format_service_result(data),
            },
            "error": None,
        }

    async def _set_thermostat(self, args: dict[str, Any]) -> dict[str, Any]:
        entity_id = str(args.get("entity_id", "") or "").strip()
        if not entity_id:
            return {"success": False, "data": None, "error": "entity_id is required"}
        if "temperature" not in args:
            return {"success": False, "data": None, "error": "temperature is required"}

        payload: dict[str, Any] = {
            "entity_id": entity_id,
            "temperature": args["temperature"],
        }
        hvac_mode = str(args.get("hvac_mode", "") or "").strip()
        if hvac_mode:
            payload["hvac_mode"] = hvac_mode

        data = await self._request("post", "/api/services/climate/set_temperature", json_payload=payload)
        self._invalidate_entity_cache()
        return {
            "success": True,
            "data": {
                "service": "climate.set_temperature",
                "result": self._format_service_result(data),
            },
            "error": None,
        }

    async def _call_service(self, args: dict[str, Any]) -> dict[str, Any]:
        domain = str(args.get("domain", "") or "").strip().lower()
        service = str(args.get("service", "") or "").strip()
        if not domain:
            return {"success": False, "data": None, "error": "domain is required"}
        if not service:
            return {"success": False, "data": None, "error": "service is required"}

        target = self._build_target(args)
        if not target:
            return {
                "success": False,
                "data": None,
                "error": "Targeted service calls require entity_id, entity_ids, area_id, or device_id",
            }

        payload = self._normalize_service_data(args)
        payload.update(target)
        data = await self._request("post", f"/api/services/{domain}/{service}", json_payload=payload)
        self._invalidate_entity_cache()
        return {
            "success": True,
            "data": {
                "service": f"{domain}.{service}",
                "result": self._format_service_result(data),
            },
            "error": None,
        }

    async def _list_entities(self, args: dict[str, Any]) -> dict[str, Any]:
        domain = str(args.get("domain", "") or "").strip().lower()
        include_attributes = bool(args.get("include_attributes", False))
        states = await self._get_states()

        if domain:
            states = [item for item in states if str(item.get("entity_id", "")).startswith(f"{domain}.")]

        entities = [self._format_entity(item, include_attributes=include_attributes) for item in states]
        return {
            "success": True,
            "data": {
                "entities": entities,
                "count": len(entities),
                "cached": True,
            },
            "error": None,
        }

    def _format_service_result(self, data: Any) -> Any:
        if not isinstance(data, list):
            return data

        formatted: list[dict[str, Any]] = []
        for item in data:
            if isinstance(item, dict) and item.get("entity_id"):
                formatted.append(self._format_entity(item, include_attributes=False))
            elif isinstance(item, dict):
                formatted.append(dict(item))
        return formatted


__all__ = ["HomeAssistantIntegration"]




# --- FILE: integrations/clients/notion.py ---

"""Notion integration via official Notion API v1 (async aiohttp).

Required env vars:
    NOTION_API_KEY  — Notion internal integration token

Rules:
- Schema validation before writing anything
- No raw LLM JSON written without validation
- All reads are low-risk; all writes are confirm-risk
"""

# internal import removed: from __future__ import annotations

import os
from typing import Any

# internal import removed: from integrations.base import BaseIntegration

_NOTION_BASE = "https://api.notion.com/v1"
_NOTION_VERSION = "2022-06-28"


class NotionIntegration(BaseIntegration):
    """Notion API integration — pages, databases, blocks (async aiohttp)."""

    name = "notion"
    description = "Create and query Notion pages, databases, and blocks"
    required_config: list[str] = ["NOTION_API_KEY"]

    def is_available(self) -> bool:
        try:
            import aiohttp  # noqa: F401
        except ImportError:
            self.unavailable_reason = "aiohttp not installed"
            return False
        if not os.environ.get("NOTION_API_KEY"):
            self.unavailable_reason = "Missing env var: NOTION_API_KEY"
            return False
        return True

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {os.environ['NOTION_API_KEY']}",
            "Notion-Version": _NOTION_VERSION,
            "Content-Type": "application/json",
        }

    def get_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "create_page",
                "description": "Create a new Notion page under a parent page or database",
                "risk": "confirm",
                "args": {
                    "parent_id": {
                        "type": "string",
                        "description": "Parent page or database ID",
                    },
                    "title": {
                        "type": "string",
                        "description": "Page title",
                    },
                    "content": {
                        "type": "string",
                        "description": "Optional plain-text content for page body",
                        "default": "",
                    },
                    "parent_type": {
                        "type": "string",
                        "description": "'page_id' or 'database_id'",
                        "default": "page_id",
                    },
                },
                "required_args": ["parent_id", "title"],
            },
            {
                "name": "query_database",
                "description": "Query a Notion database and return matching pages",
                "risk": "low",
                "args": {
                    "database_id": {
                        "type": "string",
                        "description": "Notion database ID",
                    },
                    "filter_property": {
                        "type": "string",
                        "description": "Optional property name to filter on",
                        "default": "",
                    },
                    "filter_value": {
                        "type": "string",
                        "description": "Optional filter value (text contains)",
                        "default": "",
                    },
                    "page_size": {"type": "integer", "default": 10},
                },
                "required_args": ["database_id"],
            },
            {
                "name": "append_block",
                "description": "Append a text block to an existing Notion page",
                "risk": "confirm",
                "args": {
                    "page_id": {
                        "type": "string",
                        "description": "Notion page ID to append to",
                    },
                    "text": {
                        "type": "string",
                        "description": "Text content to append",
                    },
                    "block_type": {
                        "type": "string",
                        "description": "Block type: paragraph, bulleted_list_item, heading_2",
                        "default": "paragraph",
                    },
                },
                "required_args": ["page_id", "text"],
            },
            {
                "name": "get_page",
                "description": "Retrieve metadata and top-level blocks from a Notion page",
                "risk": "low",
                "args": {
                    "page_id": {
                        "type": "string",
                        "description": "Notion page ID",
                    },
                },
                "required_args": ["page_id"],
            },
        ]

    async def execute(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        args = args or {}
        try:
            if tool_name == "create_page":
                return await self._create_page(args)
            if tool_name == "query_database":
                return await self._query_database(args)
            if tool_name == "append_block":
                return await self._append_block(args)
            if tool_name == "get_page":
                return await self._get_page(args)
            return {"success": False, "data": None, "error": f"Unknown tool: {tool_name}"}
        except Exception as exc:  # noqa: BLE001
            return {"success": False, "data": None, "error": str(exc)}

    # ── Validation helpers ────────────────────────────────────────────────────

    @staticmethod
    def _validate_block_type(block_type: str) -> str:
        allowed = {"paragraph", "bulleted_list_item", "numbered_list_item", "heading_1", "heading_2", "heading_3", "to_do", "quote"}
        return block_type if block_type in allowed else "paragraph"

    @staticmethod
    def _validate_parent_type(parent_type: str) -> str:
        allowed = {"page_id", "database_id"}
        return parent_type if parent_type in allowed else "page_id"

    # ── Tool implementations ──────────────────────────────────────────────────

    async def _create_page(self, args: dict[str, Any]) -> dict[str, Any]:
        import aiohttp

        parent_id = str(args.get("parent_id") or "").strip()
        title = str(args.get("title") or "").strip()
        content = str(args.get("content", "") or "")
        parent_type = self._validate_parent_type(str(args.get("parent_type", "page_id") or "page_id"))

        if not parent_id:
            return {"success": False, "data": None, "error": "parent_id is required"}
        if not title:
            return {"success": False, "data": None, "error": "title is required"}

        children = []
        if content:
            children.append(
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"type": "text", "text": {"content": content[:2000]}}]
                    },
                }
            )

        payload: dict[str, Any] = {
            "parent": {parent_type: parent_id},
            "properties": {
                "title": {
                    "title": [{"type": "text", "text": {"content": title[:255]}}]
                }
            },
        }
        if children:
            payload["children"] = children

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{_NOTION_BASE}/pages",
                json=payload,
                headers=self._headers(),
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                data = await resp.json()
                if resp.status not in (200, 201):
                    return {
                        "success": False,
                        "data": None,
                        "error": data.get("message", str(resp.status)),
                    }
                return {
                    "success": True,
                    "data": {
                        "page_id": data["id"],
                        "url": data.get("url"),
                    },
                    "error": None,
                }

    async def _query_database(self, args: dict[str, Any]) -> dict[str, Any]:
        import aiohttp

        database_id = str(args.get("database_id") or "").strip()
        if not database_id:
            return {"success": False, "data": None, "error": "database_id is required"}

        page_size = min(50, int(args.get("page_size", 10) or 10))
        filter_prop = str(args.get("filter_property", "") or "")
        filter_val = str(args.get("filter_value", "") or "")

        payload: dict[str, Any] = {"page_size": page_size}
        if filter_prop and filter_val:
            payload["filter"] = {
                "property": filter_prop,
                "rich_text": {"contains": filter_val},
            }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{_NOTION_BASE}/databases/{database_id}/query",
                json=payload,
                headers=self._headers(),
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                data = await resp.json()
                if resp.status != 200:
                    return {
                        "success": False,
                        "data": None,
                        "error": data.get("message", str(resp.status)),
                    }

        results = [
            {
                "id": page["id"],
                "url": page.get("url"),
                "created_time": page.get("created_time"),
            }
            for page in data.get("results", [])
        ]
        return {"success": True, "data": {"results": results, "has_more": data.get("has_more", False)}, "error": None}

    async def _append_block(self, args: dict[str, Any]) -> dict[str, Any]:
        import aiohttp

        page_id = str(args.get("page_id") or "").strip()
        text = str(args.get("text") or "").strip()
        block_type = self._validate_block_type(str(args.get("block_type", "paragraph") or "paragraph"))

        if not page_id:
            return {"success": False, "data": None, "error": "page_id is required"}
        if not text:
            return {"success": False, "data": None, "error": "text is required"}

        block = {
            "object": "block",
            "type": block_type,
            block_type: {
                "rich_text": [{"type": "text", "text": {"content": text[:2000]}}]
            },
        }

        async with aiohttp.ClientSession() as session:
            async with session.patch(
                f"{_NOTION_BASE}/blocks/{page_id}/children",
                json={"children": [block]},
                headers=self._headers(),
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                data = await resp.json()
                if resp.status not in (200, 201):
                    return {
                        "success": False,
                        "data": None,
                        "error": data.get("message", str(resp.status)),
                    }
                return {
                    "success": True,
                    "data": {"block_id": data.get("results", [{}])[0].get("id", "")},
                    "error": None,
                }

    async def _get_page(self, args: dict[str, Any]) -> dict[str, Any]:
        import aiohttp

        page_id = str(args.get("page_id") or "").strip()
        if not page_id:
            return {"success": False, "data": None, "error": "page_id is required"}

        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{_NOTION_BASE}/pages/{page_id}",
                headers=self._headers(),
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                data = await resp.json()
                if resp.status != 200:
                    return {
                        "success": False,
                        "data": None,
                        "error": data.get("message", str(resp.status)),
                    }
            # Fetch children blocks
            async with session.get(
                f"{_NOTION_BASE}/blocks/{page_id}/children",
                headers=self._headers(),
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp2:
                blocks_data = await resp2.json()

        blocks = [
            {
                "type": b.get("type"),
                "id": b.get("id"),
            }
            for b in blocks_data.get("results", [])[:20]
        ]
        return {
            "success": True,
            "data": {
                "id": data["id"],
                "url": data.get("url"),
                "created_time": data.get("created_time"),
                "last_edited_time": data.get("last_edited_time"),
                "blocks": blocks,
            },
            "error": None,
        }


__all__ = ["NotionIntegration"]




# --- FILE: integrations/clients/spotify.py ---

"""Spotify integration via Spotify Web API (async aiohttp, OAuth PKCE refresh).

Required env vars:
    SPOTIFY_CLIENT_ID
    SPOTIFY_CLIENT_SECRET
    SPOTIFY_REFRESH_TOKEN  — From OAuth Authorization Code flow

Rules:
- Token refreshed on every call (no in-memory caching to avoid stale state)
- Fail gracefully if no active playback device
- Irreversible actions (play, create_playlist) gated as confirm-risk
"""

# internal import removed: from __future__ import annotations

import os
from typing import Any

# internal import removed: from integrations.base import BaseIntegration

integrations_clients_spotify__TOKEN_URL = "https://accounts.spotify.com/api/token"
_SPOTIFY_BASE = "https://api.spotify.com/v1"


class SpotifyIntegration(BaseIntegration):
    """Spotify Web API integration — playback control, search, playlists."""

    name = "spotify"
    description = "Control Spotify playback, search music, and manage playlists"
    required_config: list[str] = [
        "SPOTIFY_CLIENT_ID",
        "SPOTIFY_CLIENT_SECRET",
        "SPOTIFY_REFRESH_TOKEN",
    ]

    def is_available(self) -> bool:
        try:
            import aiohttp  # noqa: F401
        except ImportError:
            self.unavailable_reason = "aiohttp not installed"
            return False
        if not all(bool(os.environ.get(k)) for k in self.required_config):
            missing = [k for k in self.required_config if not os.environ.get(k)]
            self.unavailable_reason = f"Missing env vars: {missing}"
            return False
        return True

    def get_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "play_track",
                "description": "Play a Spotify track by URI or search query on the active device",
                "risk": "confirm",
                "args": {
                    "track_uri": {
                        "type": "string",
                        "description": "Spotify track URI (spotify:track:...) — preferred",
                        "default": "",
                    },
                    "query": {
                        "type": "string",
                        "description": "If track_uri is empty, search by this query and play first result",
                        "default": "",
                    },
                    "device_id": {
                        "type": "string",
                        "description": "Optional Spotify device ID; uses active device if omitted",
                        "default": "",
                    },
                },
                "required_args": [],
            },
            {
                "name": "pause",
                "description": "Pause the current Spotify playback",
                "risk": "low",
                "args": {
                    "device_id": {"type": "string", "default": ""},
                },
                "required_args": [],
            },
            {
                "name": "search_track",
                "description": "Search Spotify for tracks matching a query",
                "risk": "low",
                "args": {
                    "query": {"type": "string", "description": "Search query (artist, track, album)"},
                    "limit": {"type": "integer", "default": 5},
                },
                "required_args": ["query"],
            },
            {
                "name": "get_current_track",
                "description": "Get the currently playing track on Spotify",
                "risk": "low",
                "args": {},
                "required_args": [],
            },
            {
                "name": "create_playlist",
                "description": "Create a new Spotify playlist for the current user",
                "risk": "confirm",
                "args": {
                    "name": {"type": "string", "description": "Playlist name"},
                    "description": {"type": "string", "default": ""},
                    "public": {"type": "boolean", "default": False},
                },
                "required_args": ["name"],
            },
        ]

    async def execute(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        args = args or {}
        try:
            token = await self._refresh_access_token()
            if tool_name == "play_track":
                return await self._play_track(token, args)
            if tool_name == "pause":
                return await self._pause(token, str(args.get("device_id", "") or ""))
            if tool_name == "search_track":
                return await self._search_track(
                    token,
                    query=str(args.get("query") or ""),
                    limit=min(50, int(args.get("limit", 5) or 5)),
                )
            if tool_name == "get_current_track":
                return await self._get_current_track(token)
            if tool_name == "create_playlist":
                return await self._create_playlist(token, args)
            return {"success": False, "data": None, "error": f"Unknown tool: {tool_name}"}
        except Exception as exc:  # noqa: BLE001
            return {"success": False, "data": None, "error": str(exc)}

    # ── OAuth ─────────────────────────────────────────────────────────────────

    async def _refresh_access_token(self) -> str:
        import aiohttp
        import base64

        creds = base64.b64encode(
            f"{os.environ['SPOTIFY_CLIENT_ID']}:{os.environ['SPOTIFY_CLIENT_SECRET']}".encode()
        ).decode()
        headers = {"Authorization": f"Basic {creds}", "Content-Type": "application/x-www-form-urlencoded"}
        payload = {
            "grant_type": "refresh_token",
            "refresh_token": os.environ["SPOTIFY_REFRESH_TOKEN"],
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                integrations_clients_spotify__TOKEN_URL, data=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                data = await resp.json()
                if "access_token" not in data:
                    raise RuntimeError(f"Token refresh failed: {data.get('error', 'unknown')}")
                return str(data["access_token"])

    # ── Tool implementations ──────────────────────────────────────────────────

    async def _search_track(self, token: str, query: str, limit: int = 5) -> dict[str, Any]:
        import aiohttp

        if not query.strip():
            return {"success": False, "data": None, "error": "query is required"}

        headers = {"Authorization": f"Bearer {token}"}
        params: dict[str, str | int] = {"q": query, "type": "track", "limit": limit}

        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{_SPOTIFY_BASE}/search", headers=headers, params=params, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                data = await resp.json()
                if resp.status != 200:
                    return {
                        "success": False,
                        "data": None,
                        "error": data.get("error", {}).get("message", str(resp.status)),
                    }

        tracks = [
            {
                "uri": t["uri"],
                "name": t["name"],
                "artist": ", ".join(a["name"] for a in t.get("artists", [])),
                "album": t.get("album", {}).get("name", ""),
                "duration_ms": t.get("duration_ms", 0),
            }
            for t in data.get("tracks", {}).get("items", [])
        ]
        return {"success": True, "data": {"tracks": tracks}, "error": None}

    async def _play_track(self, token: str, args: dict[str, Any]) -> dict[str, Any]:
        import aiohttp

        track_uri = str(args.get("track_uri", "") or "").strip()
        query = str(args.get("query", "") or "").strip()
        device_id = str(args.get("device_id", "") or "").strip()

        # If no URI provided, search first
        if not track_uri and query:
            search_result = await self._search_track(token, query, limit=1)
            if not search_result["success"] or not search_result["data"]["tracks"]:
                return {"success": False, "data": None, "error": f"No tracks found for: {query!r}"}
            track_uri = search_result["data"]["tracks"][0]["uri"]

        if not track_uri:
            return {"success": False, "data": None, "error": "Either track_uri or query is required"}

        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        payload = {"uris": [track_uri]}
        params = {}
        if device_id:
            params["device_id"] = device_id

        async with aiohttp.ClientSession() as session:
            async with session.put(
                f"{_SPOTIFY_BASE}/me/player/play",
                json=payload,
                headers=headers,
                params=params,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status == 204:
                    return {"success": True, "data": {"playing": track_uri}, "error": None}
                if resp.status == 404:
                    return {
                        "success": False,
                        "data": None,
                        "error": "No active Spotify device found. Open Spotify on a device first.",
                    }
                body = await resp.json()
                return {
                    "success": False,
                    "data": None,
                    "error": body.get("error", {}).get("message", str(resp.status)),
                }

    async def _pause(self, token: str, device_id: str = "") -> dict[str, Any]:
        import aiohttp

        headers = {"Authorization": f"Bearer {token}"}
        params = {}
        if device_id:
            params["device_id"] = device_id

        async with aiohttp.ClientSession() as session:
            async with session.put(
                f"{_SPOTIFY_BASE}/me/player/pause",
                headers=headers,
                params=params,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status == 204:
                    return {"success": True, "data": {"paused": True}, "error": None}
                if resp.status == 404:
                    return {"success": False, "data": None, "error": "No active device to pause."}
                body = await resp.json()
                return {
                    "success": False,
                    "data": None,
                    "error": body.get("error", {}).get("message", str(resp.status)),
                }

    async def _get_current_track(self, token: str) -> dict[str, Any]:
        import aiohttp

        headers = {"Authorization": f"Bearer {token}"}

        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{_SPOTIFY_BASE}/me/player/currently-playing",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status == 204:
                    return {"success": True, "data": {"playing": False, "track": None}, "error": None}
                if resp.status != 200:
                    return {"success": False, "data": None, "error": f"HTTP {resp.status}"}
                data = await resp.json()

        item = data.get("item") or {}
        return {
            "success": True,
            "data": {
                "playing": data.get("is_playing", False),
                "track": {
                    "uri": item.get("uri"),
                    "name": item.get("name"),
                    "artist": ", ".join(a["name"] for a in item.get("artists", [])),
                    "album": item.get("album", {}).get("name"),
                    "progress_ms": data.get("progress_ms"),
                    "duration_ms": item.get("duration_ms"),
                }
                if item
                else None,
            },
            "error": None,
        }

    async def _create_playlist(self, token: str, args: dict[str, Any]) -> dict[str, Any]:
        import aiohttp

        name = str(args.get("name") or "").strip()
        if not name:
            return {"success": False, "data": None, "error": "name is required"}

        description = str(args.get("description", "") or "")
        public = bool(args.get("public", False))

        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

        # Get current user ID first
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{_SPOTIFY_BASE}/me", headers=headers, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status != 200:
                    return {"success": False, "data": None, "error": "Could not retrieve Spotify user ID"}
                user_data = await resp.json()
                user_id = user_data["id"]

            payload = {"name": name[:100], "description": description[:300], "public": public}
            async with session.post(
                f"{_SPOTIFY_BASE}/users/{user_id}/playlists",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                data = await resp.json()
                if resp.status not in (200, 201):
                    return {
                        "success": False,
                        "data": None,
                        "error": data.get("error", {}).get("message", str(resp.status)),
                    }
                return {
                    "success": True,
                    "data": {
                        "playlist_id": data["id"],
                        "name": data["name"],
                        "url": data.get("external_urls", {}).get("spotify"),
                    },
                    "error": None,
                }


__all__ = ["SpotifyIntegration"]




# --- FILE: integrations/clients/telegram.py ---

"""Telegram integration via python-telegram-bot async Bot API.

Required env vars:
    TELEGRAM_BOT_TOKEN  — Bot token from @BotFather
    TELEGRAM_CHAT_ID    — Target chat ID (your personal or group chat)
"""

# internal import removed: from __future__ import annotations

import os
from typing import Any

# internal import removed: from integrations.base import BaseIntegration


class TelegramIntegration(BaseIntegration):
    """Send messages and receive updates via a Telegram bot."""

    name = "telegram"
    description = "Send and receive Telegram messages via a bot"
    required_config: list[str] = ["TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"]

    def is_available(self) -> bool:
        try:
            import telegram  # noqa: F401
        except ImportError:
            self.unavailable_reason = "python-telegram-bot not installed"
            return False
        if not all(bool(os.environ.get(k)) for k in self.required_config):
            missing = [k for k in self.required_config if not os.environ.get(k)]
            self.unavailable_reason = f"Missing env vars: {missing}"
            return False
        return True

    def get_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "send_telegram",
                "description": "Send a Telegram message to the configured chat",
                "risk": "confirm",
                "args": {
                    "message": {
                        "type": "string",
                        "description": "The text message to send",
                    },
                    "parse_mode": {
                        "type": "string",
                        "description": "Optional: HTML or Markdown",
                        "default": "HTML",
                    },
                },
                "required_args": ["message"],
            },
            {
                "name": "get_updates",
                "description": "Retrieve the latest incoming messages for the bot",
                "risk": "low",
                "args": {
                    "limit": {
                        "type": "integer",
                        "description": "Number of updates to fetch (max 100)",
                        "default": 10,
                    },
                },
                "required_args": [],
            },
        ]

    async def execute(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        args = args or {}
        try:
            if tool_name == "send_telegram":
                return await self._send_telegram(
                    message=str(args.get("message") or ""),
                    parse_mode=str(args.get("parse_mode", "HTML") or "HTML"),
                )
            if tool_name == "get_updates":
                limit = max(1, min(100, int(args.get("limit", 10) or 10)))
                return await self._get_updates(limit=limit)
            return {"success": False, "data": None, "error": f"Unknown tool: {tool_name}"}
        except Exception as exc:  # noqa: BLE001
            return {"success": False, "data": None, "error": str(exc)}

    async def _send_telegram(self, message: str, parse_mode: str = "HTML") -> dict[str, Any]:
        if not message.strip():
            return {"success": False, "data": None, "error": "message is required"}

        from telegram import Bot

        token = os.environ["TELEGRAM_BOT_TOKEN"]
        chat_id = os.environ["TELEGRAM_CHAT_ID"]

        bot = Bot(token=token)
        async with bot:
            sent = await bot.send_message(
                chat_id=chat_id,
                text=message,
                parse_mode=parse_mode,
            )
        return {
            "success": True,
            "data": {"message_id": sent.message_id, "chat_id": str(chat_id)},
            "error": None,
        }

    async def _get_updates(self, limit: int = 10) -> dict[str, Any]:
        from telegram import Bot

        token = os.environ["TELEGRAM_BOT_TOKEN"]
        bot = Bot(token=token)
        async with bot:
            updates = await bot.get_updates(limit=limit)

        messages = []
        for update in updates:
            if update.message:
                messages.append(
                    {
                        "update_id": update.update_id,
                        "from": update.message.from_user.username if update.message.from_user else None,
                        "text": update.message.text or "",
                        "date": str(update.message.date),
                    }
                )
        return {"success": True, "data": {"updates": messages}, "error": None}


__all__ = ["TelegramIntegration"]




# --- FILE: integrations/clients/template.py ---

"""Template integration showing the new plugin contract."""

# internal import removed: from __future__ import annotations

from typing import Any

# internal import removed: from integrations.base import BaseIntegration


class TemplateIntegration(BaseIntegration):
    name = "template"
    description = "Reference template for future integrations"
    required_config: list[str] = []

    def is_available(self) -> bool:
        # Keep template disabled by default so it never registers as a real plugin.
        return False

    def get_tools(self) -> list[dict]:
        return []

    async def execute(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        del tool_name, args
        return {"success": False, "data": None, "error": "Template integration is not executable"}


__all__ = ["TemplateIntegration"]




# --- FILE: integrations/clients/weather.py ---

"""Weather integration backed by Open-Meteo public APIs."""

# internal import removed: from __future__ import annotations

import asyncio
import json
import logging
import urllib.parse
import urllib.request
from typing import Any

# internal import removed: from integrations.base import BaseIntegration

logger = logging.getLogger(__name__)


class WeatherIntegration(BaseIntegration):
    name = "weather"
    description = "Fetch current weather by city"

    def __init__(self, config: Any = None) -> None:
        super().__init__(config=config)

    def is_available(self) -> bool:
        # No API key required for Open-Meteo.
        self.unavailable_reason = ""
        return True

    def get_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "get_current_weather",
                "description": "Get current weather data for a city.",
                "risk": "LOW",
                "args": {
                    "city": {
                        "type": "string",
                        "description": "City name, for example Delhi or New York.",
                    }
                },
                "required_args": ["city"],
            }
        ]

    async def execute(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        if tool_name != "get_current_weather":
            return {"success": False, "data": None, "error": f"Unknown tool '{tool_name}'"}

        city = str((args or {}).get("city") or "").strip()
        if not city:
            return {"success": False, "data": None, "error": "city is required"}

        loop = asyncio.get_running_loop()
        try:
            data = await loop.run_in_executor(None, self._fetch_weather, city)
            return {"success": True, "data": data, "error": None}
        except Exception as exc:  # noqa: BLE001
            logger.warning("Weather request failed for city '%s': %s", city, exc)
            return {"success": False, "data": None, "error": str(exc)}

    def _fetch_weather(self, city: str) -> dict[str, Any]:
        geo_url = (
            "https://geocoding-api.open-meteo.com/v1/search?"
            + urllib.parse.urlencode({"name": city, "count": 1})
        )
        with urllib.request.urlopen(geo_url, timeout=10) as response:
            geo_data = json.loads(response.read().decode("utf-8"))
        results = geo_data.get("results") or []
        if not results:
            raise ValueError(f"No geocode result for city '{city}'")

        first = results[0]
        lat = first["latitude"]
        lon = first["longitude"]

        weather_url = (
            "https://api.open-meteo.com/v1/forecast?"
            + urllib.parse.urlencode(
                {
                    "latitude": lat,
                    "longitude": lon,
                    "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
                }
            )
        )
        with urllib.request.urlopen(weather_url, timeout=10) as response:
            weather_data = json.loads(response.read().decode("utf-8"))
        current = weather_data.get("current", {})

        return {
            "city": first.get("name", city),
            "country": first.get("country", ""),
            "temperature_c": current.get("temperature_2m"),
            "humidity": current.get("relative_humidity_2m"),
            "wind_speed_kmh": current.get("wind_speed_10m"),
        }


__all__ = ["WeatherIntegration"]




# --- FILE: integrations/clients/whatsapp.py ---

"""WhatsApp integration via Twilio."""

# internal import removed: from __future__ import annotations

import asyncio
import os
from typing import Any

# internal import removed: from integrations.base import BaseIntegration


class WhatsAppIntegration(BaseIntegration):
    name = "whatsapp"
    description = "Send WhatsApp messages via Twilio"
    required_config: list[str] = ["TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_WHATSAPP_FROM"]

    def is_available(self) -> bool:
        try:
            import twilio  # noqa: F401
            return all(bool(os.environ.get(key)) for key in self.required_config)
        except ImportError:
            return False

    def get_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "send_whatsapp",
                "description": "Send a WhatsApp message",
                "risk": "confirm",
                "args": {
                    "to": {
                        "type": "string",
                        "description": "Recipient phone number with country code",
                    },
                    "message": {"type": "string", "description": "Message body"},
                },
                "required_args": ["to", "message"],
            }
        ]

    async def execute(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        if tool_name != "send_whatsapp":
            return {"success": False, "data": None, "error": f"Unknown tool: {tool_name}"}

        args = args or {}
        loop = asyncio.get_running_loop()
        try:
            data = await loop.run_in_executor(
                None,
                lambda: self._send_whatsapp(
                    to=str(args.get("to") or ""),
                    message=str(args.get("message") or ""),
                ),
            )
            return {"success": True, "data": data, "error": None}
        except Exception as exc:  # noqa: BLE001
            return {"success": False, "data": None, "error": str(exc)}

    def _send_whatsapp(self, to: str, message: str) -> dict[str, Any]:
        if not to.strip():
            raise ValueError("to is required")
        if not message:
            raise ValueError("message is required")

        from twilio.rest import Client

        client = Client(os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"])
        msg = client.messages.create(
            from_=os.environ["TWILIO_WHATSAPP_FROM"],
            to=f"whatsapp:{to}",
            body=message,
        )
        return {"sid": str(msg.sid)}


__all__ = ["WhatsAppIntegration"]




# --- FILE: integrations/loader.py ---

"""Auto-loader for integration clients under integrations/clients."""

# internal import removed: from __future__ import annotations

import importlib
import inspect
import logging
from pathlib import Path
from typing import Any

# internal import removed: from integrations.base import BaseIntegration

logger = logging.getLogger(__name__)


class IntegrationLoader:
    def load_all(self, config: Any, registry: Any) -> dict[str, list[str]]:
        del config  # Kept for callsite compatibility; integrations use env-only config.

        loaded: list[str] = []
        skipped: list[str] = []

        clients_dir = Path(__file__).parent / "clients"
        if not clients_dir.exists():
            return {"loaded": [], "skipped": ["clients/ dir not found"]}

        for py_file in sorted(clients_dir.glob("*.py")):
            if py_file.name.startswith("_"):
                continue

            module_name = f"integrations.clients.{py_file.stem}"
            try:
                module = importlib.import_module(module_name)
            except Exception as exc:  # noqa: BLE001
                skipped.append(f"{py_file.stem} (import error: {exc})")
                logger.warning("Integration import failed %s: %s", py_file.stem, exc)
                continue

            for _, cls in inspect.getmembers(module, inspect.isclass):
                if cls is BaseIntegration or not issubclass(cls, BaseIntegration):
                    continue
                if cls.__module__ != module_name:
                    continue

                try:
                    instance = cls()
                except Exception as exc:  # noqa: BLE001
                    skipped.append(f"{cls.__name__} (init error: {exc})")
                    logger.warning("Integration init failed %s: %s", cls.__name__, exc)
                    continue

                try:
                    if bool(instance.is_available()):
                        registry.register(instance)
                        loaded.append(instance.name or cls.__name__)
                        logger.info("Integration loaded: %s", instance.name or cls.__name__)
                    else:
                        skipped.append(f"{cls.__name__} (not available)")
                        logger.debug("Integration skipped: %s", cls.__name__)
                except Exception as exc:  # noqa: BLE001
                    skipped.append(f"{cls.__name__} (availability/register error: {exc})")
                    logger.warning("Integration registration failed %s: %s", cls.__name__, exc)

        return {"loaded": loaded, "skipped": skipped}


def load_all(config: Any, registry: Any) -> dict[str, list[str]]:
    """Backward-compatible function wrapper for older callsites."""
    return IntegrationLoader().load_all(config=config, registry=registry)


__all__ = ["IntegrationLoader", "load_all"]



############################################################
# AGENTS
############################################################


# --- FILE: core/agent/__init__.py ---





# --- FILE: core/autonomy/autonomy_governor.py ---

"""
AutonomyGovernor — enforces permission levels for tool execution dynamically.
Conforms to Rule 3.1 by avoiding hardcoded lists of tool names.
"""

import logging
import threading
from enum import IntEnum
from typing import Any

logger = logging.getLogger("Jarvis.AutonomyGovernor")


class AutonomyLevel(IntEnum):
    CHAT_ONLY = 0
    SUGGEST_ONLY = 1
    READ_ONLY = 2
    WRITE_WITH_CONFIRM = 3
    AUTONOMOUS = 4


class AutonomyGovernor:
    def __init__(self, level: int = 1, registry: Any = None):
        self.level = AutonomyLevel(level)
        self.registry = registry
        self.read_only_tools: set[str] = set()
        self.write_tools: set[str] = set()
        self._cache_is_write: dict[str, bool] = {}
        self._cache_is_known: dict[str, bool] = {}
        self._lock = threading.Lock()
        logger.info(f"Autonomy level set to: LEVEL_{self.level} ({self.level.name})")

    def register_read_only_tool(self, tool_name: str) -> None:
        """Dynamically register a tool as read-only."""
        name_clean = tool_name.strip().lower()
        with self._lock:
            self.read_only_tools.add(name_clean)
            self.write_tools.discard(name_clean)
            self._cache_is_write.pop(name_clean, None)
            self._cache_is_known.pop(name_clean, None)
        logger.debug("Dynamically registered read-only tool: %s", tool_name)

    def register_write_tool(self, tool_name: str) -> None:
        """Dynamically register a tool as a write tool."""
        name_clean = tool_name.strip().lower()
        with self._lock:
            self.write_tools.add(name_clean)
            self.read_only_tools.discard(name_clean)
            self._cache_is_write.pop(name_clean, None)
            self._cache_is_known.pop(name_clean, None)
        logger.debug("Dynamically registered write tool: %s", tool_name)

    def _is_known_tool(self, tool_name: str) -> bool:
        name_clean = tool_name.strip().lower()
        is_known = self._cache_is_known.get(name_clean)
        if is_known is not None:
            return is_known
            
        with self._lock:
            is_known = self._cache_is_known.get(name_clean)
            if is_known is not None:
                return is_known
                
            is_known = False
            if name_clean in self.read_only_tools or name_clean in self.write_tools:
                is_known = True
            elif self.registry and self.registry.get(name_clean) is not None:
                is_known = True
                
            if len(self._cache_is_known) > 1000:
                self._cache_is_known.clear()
            self._cache_is_known[name_clean] = is_known
            return is_known

    def _is_write_tool(self, tool_name: str) -> bool:
        name_clean = tool_name.strip().lower()
        is_write = self._cache_is_write.get(name_clean)
        if is_write is not None:
            return is_write
            
        with self._lock:
            is_write = self._cache_is_write.get(name_clean)
            if is_write is not None:
                return is_write
                
            if name_clean in self.write_tools:
                if len(self._cache_is_write) > 1000:
                    self._cache_is_write.clear()
                self._cache_is_write[name_clean] = True
                return True
            if name_clean in self.read_only_tools:
                if len(self._cache_is_write) > 1000:
                    self._cache_is_write.clear()
                self._cache_is_write[name_clean] = False
                return False

            if self.registry:
                cap = self.registry.get(name_clean)
                if cap:
                    if hasattr(cap, "is_write_operation"):
                        val = cap.is_write_operation
                        res = val() if callable(val) else bool(val)
                        if len(self._cache_is_write) > 1000:
                            self._cache_is_write.clear()
                        self._cache_is_write[name_clean] = res
                        return res
                    if hasattr(cap, "is_write"):
                        res = bool(cap.is_write)
                        if len(self._cache_is_write) > 1000:
                            self._cache_is_write.clear()
                        self._cache_is_write[name_clean] = res
                        return res

            # Fallback safe keyword-based check to avoid hardcoding tool name strings
            write_keywords = {
                "write", "delete", "remove", "unlink", "launch", "execute", 
                "run", "click", "type", "press", "move", "drag", "scroll", 
                "send", "add", "create", "mark", "clear", "play", "toggle", 
                "turn_on", "turn_off", "set_thermostat", "call_service", 
                "double_click", "right_click", "focus_window", "clipboard_set",
                "clipboard_paste", "hotkey"
            }
            res = any(kw in name_clean for kw in write_keywords)
            if len(self._cache_is_write) > 1000:
                self._cache_is_write.clear()
            self._cache_is_write[name_clean] = res
            return res

    def can_execute(self, tool_name: str) -> tuple[bool, str]:
        """
        Returns (allowed: bool, reason: str).
        """
        if not self._is_known_tool(tool_name):
            return False, f"Unknown tool '{tool_name}' is blocked by default. Add it to WRITE_TOOLS or READ_ONLY_TOOLS."

        if self.level == AutonomyLevel.CHAT_ONLY:
            return False, "Autonomy LEVEL_0: tool execution is disabled."

        if self.level == AutonomyLevel.SUGGEST_ONLY:
            return False, f"Autonomy LEVEL_1: would call '{tool_name}' but only suggesting actions."

        is_write = self._is_write_tool(tool_name)

        if not is_write:
            return True, f"Read-only tool '{tool_name}' approved at LEVEL_{self.level}."

        if self.level >= AutonomyLevel.AUTONOMOUS:
            return True, f"Write tool '{tool_name}' approved at LEVEL_4 (fully autonomous)."

        if self.level >= AutonomyLevel.WRITE_WITH_CONFIRM:
            return True, f"Write tool '{tool_name}' approved at LEVEL_3 (confirmation required separately)."
        
        return False, f"Write tool '{tool_name}' blocked at LEVEL_{self.level} (need LEVEL_3)."

    def requires_confirmation(self, tool_name: str) -> bool:
        """Write tools at LEVEL_3 always need explicit user confirmation."""
        if self.level >= AutonomyLevel.AUTONOMOUS:
            return False
        return self.level == AutonomyLevel.WRITE_WITH_CONFIRM and self._is_write_tool(tool_name)

    def escalate(self, new_level: int) -> bool:
        """Temporarily escalate autonomy (user must consent upstream)."""
        if new_level > AutonomyLevel.AUTONOMOUS:
            logger.warning("Escalation above LEVEL_4 is not permitted.")
            return False
        old = self.level
        self.level = AutonomyLevel(new_level)
        logger.info(f"Autonomy escalated: {old.name} -> {self.level.name}")
        return True

    def describe(self) -> str:
        descriptions = {
            0: "Chat only — no tool execution.",
            1: "Suggest only — describes actions but never runs them.",
            2: "Read-only — can inspect files, web, screen, and status automatically.",
            3: "Write with confirmation — can change files, apps, and desktop state after your approval.",
            4: "Fully autonomous — can run any allowed tool without confirmation.",
        }
        return f"LEVEL_{self.level}: {descriptions.get(self.level, 'Unknown')}"




# --- FILE: core/agent/agent_loop.py ---

"""Agent loop engine: plan -> risk -> confirm -> execute -> reflect."""

# internal import removed: from __future__ import annotations

import asyncio
import inspect
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    import httpx
except Exception:  # noqa: BLE001
    httpx = None  # type: ignore[assignment]

# internal import removed: from core.state_machine import State as AgentState, StateMachine
AgentState = State
# internal import removed: from core.context.context import TaskExecutionContext
# internal import removed: from core.autonomy.autonomy_governor import AutonomyGovernor
# internal import removed: from core.autonomy.risk_evaluator import RiskEvaluator
# internal import removed: from core.planner.planner import TaskPlanner
# internal import removed: from core.metrics.confidence import ConfidenceModel
# internal import removed: from core.registry.registry import ToolObservation, CapabilityRegistry

logger = logging.getLogger("Jarvis.AgentLoop")

_DEFAULT_MAX_ITERATIONS = 10

REFLECT_SYSTEM_PROMPT = (
    "You are Jarvis, an expert AI assistant. Review the executed plan and observations.\n"
    "If any tool failed: state the root cause first, then the fix.\n"
    "If successful: summarize concisely what was accomplished.\n"
    "Be direct and technical. No filler phrases. Address the user in second person."
)


def _truncate_obs(text: str, max_chars: int = 800) -> str:
    """Truncate long observations to keep both leading and trailing context."""
    text = text or ""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    omitted = len(text) - max_chars
    return text[:half] + f"\n...[{omitted} chars omitted]...\n" + text[-half:]


def _truncate_observation(text: str, max_chars: int = 800) -> str:
    return _truncate_obs(text, max_chars=max_chars)


@dataclass
class ExecutionTrace:
    goal: str
    iterations: int = 0
    plan: Optional[dict[str, Any]] = None
    observations: list[dict[str, Any]] = field(default_factory=list)
    risk_scores: list[dict[str, Any]] = field(default_factory=list)
    think_blocks: list[str] = field(default_factory=list)
    reflection: Optional[str] = None
    final_response: str = ""
    success: bool = False
    stop_reason: str = ""
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None

    def close(self, success: bool, reason: str) -> None:
        self.success = success
        self.stop_reason = reason
        self.ended_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "iterations": self.iterations,
            "plan": self.plan,
            "observations": self.observations,
            "risk_scores": self.risk_scores,
            "think_blocks": self.think_blocks,
            "reflection": self.reflection,
            "final_response": self.final_response,
            "success": self.success,
            "stop_reason": self.stop_reason,
            "duration_seconds": round((self.ended_at or time.time()) - self.started_at, 3),
        }


class AgentLoopEngine:
    def __init__(
        self,
        state_machine: StateMachine | None = None,
        task_planner: TaskPlanner | None = None,
        tool_router: CapabilityRegistry | None = None,
        risk_evaluator: RiskEvaluator | None = None,
        autonomy_governor: AutonomyGovernor | None = None,
        model: str = "mistral",
        ollama_url: str = "http://localhost:11434",
        max_iterations: int = _DEFAULT_MAX_ITERATIONS,
        llm: Any = None,  # Optional[LLMClientV2] — avoids import cycle
        container: Any = None,
    ):
        self.container = container
        self.sm = state_machine or (container.resolve("state_machine") if container and container.has("state_machine") else None)
        self.planner = task_planner or (container.resolve("task_planner") if container and container.has("task_planner") else None)
        self.router = tool_router or (container.resolve("tool_router") if container and container.has("tool_router") else None)
        self.risk = risk_evaluator or (container.resolve("risk_evaluator") if container and container.has("risk_evaluator") else None)
        self.gov = autonomy_governor or (container.resolve("autonomy_governor") if container and container.has("autonomy_governor") else None)
        self.model = model or "deepseek-r1:8b"
        self.ollama_url = ollama_url
        self.max_iterations = max(1, int(max_iterations or _DEFAULT_MAX_ITERATIONS))
        self.llm = llm or (container.resolve("llm") if container and container.has("llm") else None)
        self.confidence = ConfidenceModel()
        self._interrupt = asyncio.Event()
        self._run_lock = asyncio.Lock()

    def request_interrupt(self) -> None:
        self._interrupt.set()

    def _check_interrupt(self) -> bool:
        return self._interrupt.is_set()

    async def run(
        self,
        goal: str,
        context: TaskExecutionContext,
        confirm_callback=None,
    ) -> ExecutionTrace:
        async with context:
            async with self._run_lock:
                if self._check_interrupt():
                    return self._stop(ExecutionTrace(goal=goal), "user_interrupt", context.state_machine)

                self._interrupt.clear()

                sm = context.state_machine

            trace = ExecutionTrace(goal=goal)
            logger.info("Agent loop start: %s", goal)

            try:
                self._ensure_thinking_state(sm)
                sm.transition(AgentState.PLANNING)

                context_str = context.get("context_block", "")
                plan = await self._build_plan(goal, context_str)
                if not plan:
                    trace.final_response = "I couldn't generate a plan for that goal."
                    return self._stop(trace, "planning_failed", sm)

                trace.plan = plan
                intent_score = getattr(plan, "confidence", None)
                if intent_score is None and isinstance(plan, dict):
                    intent_score = plan.get("confidence", plan.get("intent_confidence", 0.5))
                try:
                    intent_score_value = float(intent_score) if intent_score is not None else 0.5
                except (TypeError, ValueError):
                    intent_score_value = 0.5
                self.confidence.update("intent_clarity", intent_score_value)

                if plan.get("clarification_needed"):
                    trace.final_response = str(
                        plan.get("clarification_prompt")
                        or plan.get("summary")
                        or "I need clarification before I can continue."
                    )
                    return self._stop(trace, "clarification_needed", sm)

                sm.transition(AgentState.RISK_EVALUATION)

                if self.risk is None:
                    raise RuntimeError("risk_evaluator is required but not provided.")
                plan_risk = self.risk.evaluate_plan(plan)
                trace.risk_scores.append(
                    {
                        "scope": "plan",
                        "level": plan_risk.level.label(),
                        "blocking": list(plan_risk.blocking_actions),
                        "confirm": list(plan_risk.confirm_actions),
                        "high": list(plan_risk.high_risk_actions),
                    }
                )

                if plan_risk.is_blocked:
                    trace.final_response = (
                        "I cannot execute that safely because the plan contains blocked actions: "
                        + ", ".join(plan_risk.blocking_actions)
                    )
                    sm.transition(AgentState.CANCELLED)
                    return self._stop(trace, "risk_threshold_exceeded", sm)

                from core.autonomy.autonomy_governor import AutonomyLevel
                is_autonomous = self.gov is not None and self.gov.level >= AutonomyLevel.AUTONOMOUS

                if plan_risk.requires_confirmation and not is_autonomous:
                    sm.transition(AgentState.AWAITING_CONFIRMATION)
                    approved = await self._ask_confirmation(
                        "This request includes high-impact actions. Continue? [y/N]: ",
                        confirm_callback,
                        context,
                    )
                    if not approved:
                        sm.transition(AgentState.CANCELLED)
                        return self._stop(trace, "user_interrupt", sm)
                    sm.transition(AgentState.APPROVED)
                else:
                    sm.transition(AgentState.APPROVED)

                # ── DAG Execution Engine integration (Session 8 Target Architecture) ──
                logger.info("Starting DAG Executor for goal: %s", goal, extra={"metadata": {"goal": goal}})
                
                if self.container and self.container.has("dag_executor"):
                    executor = self.container.resolve("dag_executor", tool_router=self.router, risk_evaluator=self.risk, autonomy_governor=self.gov)
                else:
                    raise RuntimeError("dag_executor not found in container")
                
                sm.transition(AgentState.EXECUTING)

                try:
                    # Enforce 5 minute task-level timeout (Part 5)
                    async with asyncio.timeout(300):
                        res = await executor.execute(plan, context)
                except asyncio.TimeoutError:
                    logger.error("Task execution timed out.", extra={"metadata": {"timeout_s": 300}})
                    res = {"status": "failure", "error": "Task execution timed out after 300s."}
                except Exception as exc:
                    logger.error("Execution engine failure: %s", exc, extra={"metadata": {"error": str(exc)}})
                    res = {"status": "failure", "error": str(exc)}

                # Map execution results back into trace observations
                observations: list[ToolObservation] = []
                for sid, step_res in res.get("results", {}).items():
                    obs = ToolObservation(
                        tool_name=step_res.get("tool_name", ""),
                        arguments=step_res.get("arguments", {}),
                        execution_status=step_res.get("execution_status", "failure"),
                        output_summary=step_res.get("output_summary", ""),
                        error_message=step_res.get("error_message"),
                        duration_seconds=step_res.get("duration_seconds", 0.0),
                    )
                    observations.append(obs)

                    obs_dict = obs.to_dict()
                    obs_dict["step_id"] = sid
                    obs_dict["output_summary"] = _truncate_obs(str(obs_dict.get("output_summary", "")))
                    if obs_dict.get("error_message"):
                        obs_dict["error_message"] = _truncate_obs(str(obs_dict["error_message"]))
                    trace.observations.append(obs_dict)

                    tool_success = 1.0 if obs.execution_status == "success" else 0.0
                    self.confidence.update("tool_reliability", tool_success)

                if res.get("status") == "success":
                    sm.transition(AgentState.REFLECTING)
                    response = await self._reflect(goal, plan, observations, trace)
                    trace.reflection = response
                    trace.final_response = response
                    trace.close(True, "goal_completed")
                    sm.transition(AgentState.SPEAKING)
                    sm.transition(AgentState.COMPLETED)
                    sm.transition(AgentState.IDLE)
                else:
                    trace.final_response = res.get("error") or "Task execution failed."
                    return self._stop(trace, "unrecoverable_tool_failure", sm)

                logger.info("Agent loop complete: success=%s", trace.success)
                return trace

            except asyncio.CancelledError:
                logger.warning("Agent loop cancelled via asyncio.CancelledError")
                if sm.state not in {AgentState.ABORTED, AgentState.ERROR, AgentState.SHUTDOWN}:
                    if sm.can_transition(AgentState.ABORTED):
                        sm.transition(AgentState.ABORTED)
                    else:
                        sm.force_idle()
                raise
            except Exception as e:
                import traceback
                logger.error("Agent loop crashed: %s\n%s", e, traceback.format_exc())
                if sm.state not in {AgentState.ERROR, AgentState.ABORTED, AgentState.SHUTDOWN}:
                    if sm.can_transition(AgentState.ERROR):
                        sm.transition(AgentState.ERROR)
                    else:
                        sm.force_idle()
                        sm.transition(AgentState.ERROR)
                trace.final_response = f"Internal error during execution: {e}"
                return self._stop(trace, "internal_error", sm)
            finally:
                if sm.state not in {AgentState.IDLE, AgentState.SHUTDOWN, AgentState.ERROR, AgentState.ABORTED}:
                    try:
                        sm.force_idle()
                    except Exception:
                        pass

    def _ensure_thinking_state(self, sm: StateMachine) -> None:
        if sm.state == AgentState.THINKING:
            return
        if sm.state == AgentState.IDLE:
            sm.transition(AgentState.THINKING)
            return
        sm.force_idle()
        sm.transition(AgentState.THINKING)

    async def _build_plan(self, goal: str, context: str) -> dict[str, Any]:
        plan_fn = getattr(self.planner, "plan", None)
        if plan_fn is None:
            return {}

        try:
            if inspect.iscoroutinefunction(plan_fn):
                result = await plan_fn(goal, context)
            else:
                result = await asyncio.to_thread(plan_fn, goal, context)
        except Exception as exc:  # noqa: BLE001
            logger.error("Planner execution failed: %s", exc, exc_info=True)
            return {}

        if isinstance(result, dict):
            return result
        return {}

    def _normalize_steps(self, plan: dict[str, Any]) -> list[dict[str, Any]]:
        steps = plan.get("steps", []) if isinstance(plan, dict) else []
        if not isinstance(steps, list):
            return []

        normalized: list[dict[str, Any]] = []
        for idx, step in enumerate(steps, start=1):
            if not isinstance(step, dict):
                continue
            params = step.get("params")
            if not params:
                params = step.get("parameters") or step.get("args") or {}
            if not isinstance(params, dict):
                params = {}
            normalized.append(
                {
                    "id": int(step.get("id", idx)),
                    "action": str(step.get("action", "")).strip(),
                    "description": str(step.get("description", "")).strip(),
                    "params": params,
                }
            )
        return normalized

    async def _ask_confirmation(self, prompt: str, confirm_callback, context: TaskExecutionContext) -> bool:
        if context.get("approval_called"):
            logger.warning("Approval already handled. Returning idempotent result: %s", context.get("approval_result"))
            return bool(context.get("approval_result"))

        approved = False
        if confirm_callback is None:
            import sys
            from core.autonomy.autonomy_governor import AutonomyLevel
            
            is_headless = not sys.stdin.isatty()
            is_autonomous = self.gov is not None and self.gov.level >= AutonomyLevel.AUTONOMOUS
            
            if is_headless or is_autonomous:
                logger.warning("Bypassing manual confirmation due to headless environment or LEVEL_4 autonomy.")
                approved = True
            else:
                answer = await asyncio.to_thread(input, prompt)
                approved = str(answer).strip().lower() in {"y", "yes"}
        else:
            try:
                result = confirm_callback(prompt)
                if inspect.isawaitable(result):
                    result = await result
                approved = bool(result)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Confirmation callback failed: %s", exc)
                approved = False

        context.set("approval_called", True)
        context.set("approval_result", approved)
        return approved

    async def _reflect(
        self,
        goal: str,
        plan: dict[str, Any],
        observations: list[ToolObservation],
        trace: ExecutionTrace,
    ) -> str:
        if observations:
            obs_lines = []
            for obs in observations:
                obs_text = _truncate_obs(obs.output_summary or obs.error_message or "")
                obs_lines.append(f"- {obs.tool_name}: {obs_text}")
            obs_text = "\n".join(obs_lines)
        else:
            obs_text = "No tool observations."

        user_prompt = (
            f"Goal:\n{goal}\n\n"
            f"Plan:\n{self._plan_summary(plan)}\n\n"
            f"Tool observations:\n{obs_text}\n"
        )

        # ── Prefer LLMClientV2 if injected ────────────────────────────────────
        if self.llm is not None and hasattr(self.llm, "complete"):
            try:
                result = await self.llm.complete(
                    user_prompt,
                    system=REFLECT_SYSTEM_PROMPT,
                    temperature=0.2,
                    task_type="reflection",
                )
                cleaned = re.sub(r"<think>.*?</think>", "", result or "", flags=re.DOTALL).strip()
                if cleaned:
                    return cleaned
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLMClientV2 reflection failed: %s", exc)
            # Fall through to httpx if LLMClientV2 returned empty

        # ── Fallback: direct httpx Ollama call ────────────────────────────────
        if httpx is None:
            return self._fallback_reflection(plan, observations)

        payload = {
            "model": self.model, # self.model now just acts as fallback if llm is missing
            "messages": [
                {"role": "system", "content": REFLECT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(f"{self.ollama_url}/api/chat", json=payload)
                response.raise_for_status()
                data = response.json()
                raw = str(data.get("message", {}).get("content", "")).strip()

                think_matches = re.findall(r"<think>(.*?)</think>", raw, re.DOTALL)
                trace.think_blocks = [block.strip() for block in think_matches if block.strip()]

                cleaned_response = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
                return cleaned_response or self._fallback_reflection(plan, observations)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Reflection model call failed: %s", exc)
            return self._fallback_reflection(plan, observations)

    def _plan_summary(self, plan: dict[str, Any]) -> str:
        summary = str(plan.get("summary", "")).strip()
        steps = self._normalize_steps(plan)
        lines = [summary] if summary else []
        for step in steps:
            action = step["action"] or "(no action)"
            desc = step["description"] or ""
            lines.append(f"{step['id']}. {action} - {desc}")
        return "\n".join(lines) if lines else "No plan summary available."

    def _fallback_reflection(self, plan: dict[str, Any], observations: list[ToolObservation]) -> str:
        if any(obs.execution_status != "success" for obs in observations):
            failed = [obs.tool_name for obs in observations if obs.execution_status != "success"]
            return (
                "You hit a tool failure. Root cause: one or more tool calls returned an error. "
                f"Failed tools: {', '.join(failed)}. "
                "Fix the failing tool inputs or environment and run the task again."
            )

        if observations:
            tool_list = ", ".join(obs.tool_name for obs in observations)
            return f"You completed the requested task successfully. Tools used: {tool_list}."

        summary = str(plan.get("summary", "")).strip()
        return summary or "You completed the task successfully."

    def _stop(self, trace: ExecutionTrace, reason: str, sm: StateMachine) -> ExecutionTrace:
        trace.close(False, reason)
        if not trace.final_response:
            defaults = {
                "user_interrupt": "Understood. I stopped the task.",
                "planning_failed": "I couldn't create a workable plan.",
                "clarification_needed": "I need clarification before proceeding.",
                "risk_threshold_exceeded": "I can't continue because the requested action is blocked by safety rules.",
                "iteration_limit_reached": "I reached the maximum number of iterations for this task.",
                "unrecoverable_tool_failure": "A required tool failed, so I stopped execution.",
            }
            trace.final_response = defaults.get(reason, "Task stopped.")

        sm.force_idle()
        return trace


__all__ = ["AgentLoopEngine", "ExecutionTrace", "_truncate_observation"]




# --- FILE: core/autonomy/goal_manager.py ---

"""
core/autonomy/goal_manager.py

Owns the lifecycle of long-lived agent goals.
A Goal is a high-level desired outcome that may span multiple Missions.

Responsibilities:
- Create / update / complete / cancel goals
- Prioritise active goals
- Query which goal the agent should work on next
- Persist goal state (via snapshot / restore)
"""

# internal import removed: from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


def core_autonomy_goal_manager__utcnow() -> datetime:
    return datetime.now(timezone.utc)


class GoalStatus(str, Enum):
    PENDING   = "pending"    # created, not yet started
    ACTIVE    = "active"     # currently being pursued
    PAUSED    = "paused"     # temporarily on hold
    COMPLETED = "completed"  # achieved
    FAILED    = "failed"     # could not be achieved
    CANCELLED = "cancelled"  # explicitly abandoned


@dataclass
class Goal:
    """A single long-lived agent objective."""

    goal_id: str
    description: str
    priority: int = 5              # 1 (highest) – 10 (lowest)
    status: GoalStatus = GoalStatus.PENDING
    parent_goal_id: Optional[str] = None   # for sub-goals
    metadata: dict = field(default_factory=dict)

    created_at: datetime = field(default_factory=core_autonomy_goal_manager__utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    deadline: Optional[datetime] = None

    outcome: Optional[str] = None   # human-readable result

    def start(self) -> None:
        if self.status != GoalStatus.PENDING:
            raise ValueError(f"Cannot start goal in status '{self.status}'")
        self.status = GoalStatus.ACTIVE
        self.started_at = core_autonomy_goal_manager__utcnow()

    def complete(self, outcome: str = "") -> None:
        self.status = GoalStatus.COMPLETED
        self.completed_at = core_autonomy_goal_manager__utcnow()
        self.outcome = outcome

    def fail(self, reason: str = "") -> None:
        self.status = GoalStatus.FAILED
        self.completed_at = core_autonomy_goal_manager__utcnow()
        self.outcome = reason

    def cancel(self, reason: str = "") -> None:
        self.status = GoalStatus.CANCELLED
        self.completed_at = core_autonomy_goal_manager__utcnow()
        self.outcome = reason

    def pause(self) -> None:
        if self.status == GoalStatus.ACTIVE:
            self.status = GoalStatus.PAUSED

    def resume(self) -> None:
        if self.status == GoalStatus.PAUSED:
            self.status = GoalStatus.ACTIVE

    @property
    def is_terminal(self) -> bool:
        return self.status in (GoalStatus.COMPLETED, GoalStatus.FAILED, GoalStatus.CANCELLED)

    def to_dict(self) -> dict:
        return {
            "goal_id": self.goal_id,
            "description": self.description,
            "priority": self.priority,
            "status": self.status.value,
            "parent_goal_id": self.parent_goal_id,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "outcome": self.outcome,
        }


class GoalManager:
    """
    Registry and lifecycle manager for all agent goals.

    Usage:
        gm = GoalManager()
        gid = gm.create_goal("Summarise all emails from today", priority=2)
        gm.start_goal(gid)
        ...
        gm.complete_goal(gid, outcome="12 emails summarised")
    """

    def __init__(self) -> None:
        self._goals: dict[str, Goal] = {}
        self._lock = threading.RLock()

    # ── CRUD ─────────────────────────────────────────────────────────────

    def create_goal(
        self,
        description: str,
        priority: int = 5,
        parent_goal_id: Optional[str] = None,
        deadline: Optional[datetime] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        with self._lock:
            goal_id = str(uuid.uuid4())
            self._goals[goal_id] = Goal(
                goal_id=goal_id,
                description=description,
                priority=priority,
                parent_goal_id=parent_goal_id,
                deadline=deadline,
                metadata=metadata or {},
            )
            return goal_id

    def get_goal(self, goal_id: str) -> Goal:
        with self._lock:
            if goal_id not in self._goals:
                raise KeyError(f"Unknown goal: {goal_id}")
            return self._goals[goal_id]

    def start_goal(self, goal_id: str) -> None:
        with self._lock:
            self.get_goal(goal_id).start()

    def complete_goal(self, goal_id: str, outcome: str = "") -> None:
        with self._lock:
            self.get_goal(goal_id).complete(outcome)

    def fail_goal(self, goal_id: str, reason: str = "") -> None:
        with self._lock:
            self.get_goal(goal_id).fail(reason)

    def cancel_goal(self, goal_id: str, reason: str = "") -> None:
        with self._lock:
            self.get_goal(goal_id).cancel(reason)

    def pause_goal(self, goal_id: str) -> None:
        with self._lock:
            self.get_goal(goal_id).pause()

    def resume_goal(self, goal_id: str) -> None:
        with self._lock:
            self.get_goal(goal_id).resume()

    def update_goal(
        self,
        goal_id: str,
        description: Optional[str] = None,
        priority: Optional[int] = None,
        deadline: Optional[datetime] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        with self._lock:
            goal = self.get_goal(goal_id)
            if description is not None:
                goal.description = description
            if priority is not None:
                goal.priority = priority
            if deadline is not None:
                goal.deadline = deadline
            if metadata is not None:
                goal.metadata.update(metadata)

    def remove_goal(self, goal_id: str) -> None:
        with self._lock:
            if goal_id in self._goals:
                del self._goals[goal_id]

    # ── Queries ──────────────────────────────────────────────────────────

    def next_goal(self) -> Optional[Goal]:
        """Return the highest-priority pending or paused goal."""
        with self._lock:
            candidates = [
                g for g in self._goals.values()
                if g.status in (GoalStatus.PENDING, GoalStatus.PAUSED)
            ]
            if not candidates:
                return None
            return min(candidates, key=lambda g: (g.priority, g.created_at))

    def active_goals(self) -> list[Goal]:
        with self._lock:
            return [g for g in self._goals.values() if g.status == GoalStatus.ACTIVE]

    def all_goals(self) -> list[Goal]:
        with self._lock:
            return list(self._goals.values())

    def get_goals_by_status(self, status: GoalStatus) -> list[Goal]:
        with self._lock:
            return [g for g in self._goals.values() if g.status == status]

    def get_subgoals(self, parent_goal_id: str) -> list[Goal]:
        with self._lock:
            return [g for g in self._goals.values() if g.parent_goal_id == parent_goal_id]

    # ── Persistence ──────────────────────────────────────────────────────

    def snapshot(self) -> list[dict]:
        with self._lock:
            return [g.to_dict() for g in self._goals.values()]

    def restore(self, data: list[dict]) -> None:
        """Reload goals from a persisted snapshot (e.g. after restart)."""
        with self._lock:
            for d in data:
                goal = Goal(
                    goal_id=d["goal_id"],
                    description=d["description"],
                    priority=d["priority"],
                    status=GoalStatus(d["status"]),
                    parent_goal_id=d.get("parent_goal_id"),
                    metadata=d.get("metadata", {}),
                    created_at=datetime.fromisoformat(d["created_at"]),
                    outcome=d.get("outcome"),
                )
                if d.get("started_at"):
                    goal.started_at = datetime.fromisoformat(d["started_at"])
                if d.get("completed_at"):
                    goal.completed_at = datetime.fromisoformat(d["completed_at"])
                if d.get("deadline"):
                    goal.deadline = datetime.fromisoformat(d["deadline"])
                self._goals[goal.goal_id] = goal





# --- FILE: core/autonomy/__init__.py ---





# --- FILE: core/autonomy/scheduler.py ---

"""
core/autonomy/scheduler.py

Manages delayed and retry execution of Missions.

Responsibilities:
- Queue missions for execution at a future time
- Implement exponential back-off for retries
- Expose the next due mission (pull model — no background threads)
- Persist schedule across restarts

Design note:
  This is a *pull-based* scheduler.  The caller (main loop / dispatcher)
  asks `scheduler.due()` on each tick.  There are no background threads,
  no asyncio tasks, no hidden loops — exactly as the spec requires.
"""

# internal import removed: from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional


def core_autonomy_scheduler__utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ScheduleStatus(str, Enum):
    WAITING   = "waiting"    # not yet due
    DUE       = "due"        # ready to run
    RUNNING   = "running"    # currently executing
    COMPLETED = "completed"  # finished (success or final failure)
    CANCELLED = "cancelled"


@dataclass
class ScheduledMission:
    """Entry in the scheduler queue."""

    entry_id: str
    mission_id: str                     # refers to a Mission object (owned externally)
    goal_id: str

    run_at: datetime                    # when to execute
    status: ScheduleStatus = ScheduleStatus.WAITING

    # Retry book-keeping
    attempt_number: int = 1
    max_attempts: int = 3
    base_delay_seconds: float = 30.0   # first retry delay
    backoff_factor: float = 2.0        # multiply delay each attempt

    # Optional label for humans / LLMs
    description: str = ""

    created_at: datetime = field(default_factory=core_autonomy_scheduler__utcnow)
    last_run_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @property
    def is_due(self) -> bool:
        return self.status == ScheduleStatus.WAITING and core_autonomy_scheduler__utcnow() >= self.run_at

    @property
    def next_retry_delay(self) -> float:
        """Seconds to wait before the next retry attempt."""
        return self.base_delay_seconds * (self.backoff_factor ** (self.attempt_number - 1))

    def mark_completed(self) -> None:
        self.status = ScheduleStatus.COMPLETED
        self.completed_at = core_autonomy_scheduler__utcnow()

    def mark_cancelled(self) -> None:
        self.status = ScheduleStatus.CANCELLED
        self.completed_at = core_autonomy_scheduler__utcnow()

    def schedule_retry(self) -> bool:
        """
        Advance the attempt counter and set a new run_at.
        Returns False if max_attempts is exhausted (caller should cancel).
        """
        if self.attempt_number >= self.max_attempts:
            return False
        delay = self.next_retry_delay
        self.attempt_number += 1
        self.run_at = core_autonomy_scheduler__utcnow() + timedelta(seconds=delay)
        self.status = ScheduleStatus.WAITING
        return True

    def to_dict(self) -> dict:
        return {
            "entry_id": self.entry_id,
            "mission_id": self.mission_id,
            "goal_id": self.goal_id,
            "run_at": self.run_at.isoformat(),
            "status": self.status.value,
            "attempt_number": self.attempt_number,
            "max_attempts": self.max_attempts,
            "base_delay_seconds": self.base_delay_seconds,
            "backoff_factor": self.backoff_factor,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "last_run_at": self.last_run_at.isoformat() if self.last_run_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class Scheduler:
    """
    Pull-based mission scheduler with exponential back-off.

    Usage:
        scheduler = Scheduler()
        scheduler.enqueue(mission_id="abc", goal_id="xyz", delay_seconds=0)

        # In your main loop:
        for entry in scheduler.due():
            entry.mark_running()
            run_mission(entry.mission_id)
            entry.mark_completed()
    """

    def __init__(self) -> None:
        self._queue: dict[str, ScheduledMission] = {}
        self._lock = threading.RLock()

    # ── Enqueue ──────────────────────────────────────────────────────────

    def enqueue(
        self,
        mission_id: str,
        goal_id: str,
        delay_seconds: float = 0.0,
        max_attempts: int = 3,
        base_delay_seconds: float = 30.0,
        backoff_factor: float = 2.0,
        description: str = "",
    ) -> ScheduledMission:
        with self._lock:
            entry = ScheduledMission(
                entry_id=str(uuid.uuid4()),
                mission_id=mission_id,
                goal_id=goal_id,
                run_at=core_autonomy_scheduler__utcnow() + timedelta(seconds=delay_seconds),
                max_attempts=max_attempts,
                base_delay_seconds=base_delay_seconds,
                backoff_factor=backoff_factor,
                description=description,
            )
            self._queue[entry.entry_id] = entry
            return entry

    # ── Query ────────────────────────────────────────────────────────────

    def due(self) -> list[ScheduledMission]:
        """Return all entries that are currently due, sorted by run_at."""
        with self._lock:
            entries = [e for e in self._queue.values() if e.is_due]
            return sorted(entries, key=lambda e: e.run_at)

    def get(self, entry_id: str) -> ScheduledMission:
        with self._lock:
            if entry_id not in self._queue:
                raise KeyError(f"No scheduled entry: {entry_id}")
            return self._queue[entry_id]

    def cancel(self, entry_id: str) -> None:
        with self._lock:
            if entry_id in self._queue:
                self._queue[entry_id].mark_cancelled()

    def pending(self) -> list[ScheduledMission]:
        with self._lock:
            return [e for e in self._queue.values() if e.status == ScheduleStatus.WAITING]

    # ── Serialisation ────────────────────────────────────────────────────

    def snapshot(self) -> list[dict]:
        with self._lock:
            return [e.to_dict() for e in self._queue.values()]

    def restore(self, data: list[dict]) -> None:
        with self._lock:
            for d in data:
                entry = ScheduledMission(
                    entry_id=d["entry_id"],
                    mission_id=d["mission_id"],
                    goal_id=d["goal_id"],
                    run_at=datetime.fromisoformat(d["run_at"]),
                    status=ScheduleStatus(d["status"]),
                    attempt_number=d["attempt_number"],
                    max_attempts=d["max_attempts"],
                    base_delay_seconds=d["base_delay_seconds"],
                    backoff_factor=d["backoff_factor"],
                    description=d.get("description", ""),
                    created_at=datetime.fromisoformat(d["created_at"]),
                    last_run_at=datetime.fromisoformat(d["last_run_at"]) if d.get("last_run_at") else None,
                    completed_at=datetime.fromisoformat(d["completed_at"]) if d.get("completed_at") else None,
                )
                self._queue[entry.entry_id] = entry




############################################################
# ORCHESTRATION
############################################################


# --- FILE: audit/__init__.py ---

"""Audit helpers."""

# internal import removed: from .audit_logger import scrub_secrets

__all__ = ["scrub_secrets"]




# --- FILE: core/__init__.py ---





# --- FILE: core/planner/planner.py ---

"""Asynchronous task planner to generate execution plans."""

# internal import removed: from __future__ import annotations

import json
import logging
import re
import inspect
from typing import Any

# internal import removed: from core.autonomy.risk_evaluator import core_autonomy_risk_evaluator_RiskLevel, RiskEvaluator

# SYSTEM_TOOL_SCHEMA removed in favor of CapabilityRegistry dynamic discovery

_GUI_TOOL_NAMES = {
    "click",
    "double_click",
    "right_click",
    "click_text_on_screen",
    "click_screen_target",
    "double_click_screen_target",
    "right_click_screen_target",
    "move_mouse",
    "scroll",
    "drag",
    "type_text",
    "press_key",
    "hotkey",
    "focus_window",
    "clipboard_get",
    "clipboard_set",
    "clipboard_paste",
}


def _strip_planner_artifacts(raw: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.DOTALL).strip()
    cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", cleaned)
    cleaned = re.sub(r"\n?```$", "", cleaned)
    return cleaned.strip()

logger = logging.getLogger("Jarvis.Planner")


class TaskPlanner:
    def __init__(self, config=None, llm=None, registry=None) -> None:
        self.config = config
        self.risk_evaluator = RiskEvaluator(config)
        self.llm = llm
        self.registry = registry

    def _tool_schema(self) -> dict[str, list[dict[str, Any]]]:
        tools = []
        if self.registry:
            for name in self.registry.registered_tools():
                cap = self.registry.get(name)
                desc = getattr(cap, "description", "") or f"Execute {name}"
                schema: dict[str, Any] = {"name": name, "description": desc}
                
                if hasattr(cap, "handler") and cap.handler:
                    try:
                        sig = inspect.signature(cap.handler)
                        params = {}
                        for param_name, param in sig.parameters.items():
                            if param_name == "context":
                                continue
                            param_info: dict[str, Any] = {"type": "string"}
                            if param.annotation != inspect.Parameter.empty:
                                if hasattr(param.annotation, "__name__"):
                                    param_info["type"] = param.annotation.__name__
                                else:
                                    param_info["type"] = str(param.annotation)
                            if param.default != inspect.Parameter.empty:
                                if isinstance(param.default, (str, int, float, bool, type(None))):
                                    param_info["default"] = param.default
                                else:
                                    param_info["default"] = str(param.default)
                            else:
                                param_info["required"] = True
                            params[param_name] = param_info
                        schema["parameters"] = params
                    except Exception:
                        pass
                tools.append(schema)

        allow_gui = False
        try:
            if self.config:
                allow_gui = self.config.getboolean(
                    "execution",
                    "allow_gui_automation",
                    fallback=False,
                )
        except Exception:
            allow_gui = False

        if not allow_gui:
            tools = [tool for tool in tools if tool["name"] not in _GUI_TOOL_NAMES]

        return {"tools": tools}

    async def _call_ollama(self, prompt: str) -> str:
        if not self.llm or not hasattr(self.llm, "complete"):
            return ""
        try:
            res = self.llm.complete(prompt, task_type="tool_picker")
            if inspect.isawaitable(res):
                return str(await res)
            return str(res)
        except Exception as exc:
            logger.error("LLM completion failed: %s", exc)
            return ""

    async def plan(self, user_input: str, context: str = "") -> dict[str, Any]:
        text = str(user_input or "").strip()
        logger.info("Starting task planning", extra={"metadata": {"intent_length": len(text), "context_length": len(context)}})
        raw = await self._call_ollama(self._build_prompt(text, context))

        if raw:
            logger.info("Raw LLM output: %s", raw)
            parsed = self._parse_llm_plan(raw)
            if parsed is not None:
                logger.info("Successfully parsed plan from LLM output", extra={"metadata": {"confidence": parsed.get("confidence")}})
                return self._enrich_plan(text, parsed)
            logger.warning("Failed to parse LLM plan, falling back to clarification")
            return self._clarification_plan(text)

        logger.warning("Empty LLM response, falling back to basic plan")
        return self._enrich_plan(text, self._fallback_plan(text))

    def _build_prompt(self, user_input: str, context: str) -> str:
        schema_format = {
            "intent": "user request",
            "summary": "overall plan summary",
            "confidence": 0.9,
            "steps": [
                {
                    "id": 1,
                    "action": "tool_name",
                    "description": "why we call this tool",
                    "parameters": {"<argument_name>": "<argument_value>"}
                }
            ],
            "clarification_needed": False,
            "clarification_prompt": ""
        }
        
        example_json = {
            "intent": "read a file",
            "summary": "I will use read_file",
            "confidence": 0.99,
            "steps": [
                {
                    "id": 1,
                    "action": "read_file",
                    "description": "Read the config file",
                    "parameters": {"path": "/etc/config.json"}
                }
            ],
            "clarification_needed": False,
            "clarification_prompt": ""
        }

        return (
            "You are a task planner. Create a step-by-step action plan using the available tools to satisfy the user request.\n"
            f"User request: {user_input}\n"
            f"Context: {context}\n"
            f"Available tools: {json.dumps(self._tool_schema())}\n\n"
            "You MUST return a valid JSON object matching the following structure exactly:\n"
            f"{json.dumps(schema_format, indent=2)}\n\n"
            "CRITICAL: For EVERY tool step in 'steps', you MUST include a 'parameters' dictionary containing the required arguments. "
            "The keys in 'parameters' MUST exactly match the argument names shown in the tool's schema.\n\n"
            f"Example Output:\n{json.dumps(example_json, indent=2)}\n\n"
            "Return ONLY the strict JSON object. No explanations, no markdown block, no extra text."
        )

    def _parse_llm_plan(self, raw: str) -> dict[str, Any] | None:
        cleaned = _strip_planner_artifacts(raw)
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            # Attempt basic repair for missing commas (a common issue with smaller models)
            import re
            cleaned = re.sub(r'"\s*\n\s*"', '",\n"', cleaned)
            try:
                payload = json.loads(cleaned)
            except json.JSONDecodeError:
                return None
        if not isinstance(payload, dict):
            return None
        return payload

    def _fallback_plan(self, text: str) -> dict[str, Any]:
        return self._clarification_plan(text)

    def _clarification_plan(self, text: str) -> dict[str, Any]:
        return {
            "intent": text,
            "summary": "Need clarification before taking action.",
            "confidence": 0.2,
            "steps": [],
            "clarification_needed": True,
            "clarification_prompt": "Please clarify what you want me to do.",
        }

    def _enrich_plan(self, text: str, plan: dict[str, Any]) -> dict[str, Any]:
        steps_list = self._normalize_steps(plan.get("steps", []))
        normalized: dict[str, Any] = {
            "intent": str(plan.get("intent", text)),
            "summary": str(plan.get("summary", "")).strip(),
            "confidence": float(plan.get("confidence", 0.0) or 0.0),
            "steps": steps_list,
            "clarification_needed": bool(plan.get("clarification_needed", False)),
            "clarification_prompt": str(plan.get("clarification_prompt", "") or ""),
        }

        tools_required = [
            step["action"] for step in steps_list if step.get("action")
        ]
        risk = self.risk_evaluator.evaluate(tools_required)
        risk_label = risk.level.label().lower()
        if risk.level >= core_autonomy_risk_evaluator_RiskLevel.CRITICAL:
            risk_label = "critical"
        elif risk.level >= core_autonomy_risk_evaluator_RiskLevel.HIGH:
            risk_label = "high"
        elif risk.level >= core_autonomy_risk_evaluator_RiskLevel.MEDIUM:
            risk_label = "medium"
        else:
            risk_label = "low"

        normalized["tools_required"] = tools_required
        normalized["risk_level"] = risk_label
        normalized["confirmation_required"] = risk.requires_confirmation
        logger.debug("Plan enriched with risk evaluation", extra={"metadata": {"risk_level": risk_label, "tools_count": len(tools_required)}})
        return normalized

    def _normalize_steps(self, steps: Any) -> list[dict[str, Any]]:
        if not isinstance(steps, list):
            return []
        normalized: list[dict[str, Any]] = []
        step_list: list[Any] = steps
        for idx, step in enumerate(step_list, start=1):
            if not isinstance(step, dict):
                continue
            params = step.get("params", {})
            normalized.append(
                {
                    "id": int(step.get("id", idx)),
                    "action": str(step.get("action", "")).strip(),
                    "description": str(step.get("description", "")).strip(),
                    "params": params if isinstance(params, dict) else {},
                }
            )
        return normalized




# --- FILE: core/agentic/__init__.py ---

import warnings

warnings.warn(
    "The 'core.agentic' module is deprecated and will be removed in a future version. "
    "Please use 'core.autonomy' instead.",
    DeprecationWarning,
    stacklevel=2
)




# --- FILE: core/agentic/goal_manager.py ---

# internal import removed: from core.autonomy.goal_manager import *  # noqa: F403
import warnings

warnings.warn(
    "The 'core.agentic.goal_manager' module is deprecated and will be removed in a future version. "
    "Please use 'core.autonomy.goal_manager' instead.",
    DeprecationWarning,
    stacklevel=2
)




# --- FILE: core/automation/scan_pipeline.py ---

"""Async scan pipeline primitives for live automation ingestion."""

# internal import removed: from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

SUMMARY_KEYS = (
    "commands_processed",
    "files_ingested",
    "chunks_ingested",
    "failed_files",
    "skipped_files",
    "scanned_files",
)


def blank_scan_summary() -> dict[str, int]:
    return {key: 0 for key in SUMMARY_KEYS}


@dataclass(frozen=True)
class ScanBatch:
    name: str
    candidates: tuple[Path, ...]
    mark_seen: bool
    process: Callable[[Path], Awaitable[dict[str, int]]]
    on_preexisting: Callable[[Path], None] | None = None
    on_error: Callable[[Path, Exception], None] | None = None


ReadinessCheck = Callable[[Path, bool], tuple[bool, str]]


class ScanPipeline:
    def __init__(self, batches: list[ScanBatch] | tuple[ScanBatch, ...]) -> None:
        self._batches = tuple(batches)

    async def run(self, readiness: ReadinessCheck) -> dict[str, int]:
        summary = blank_scan_summary()
        summary["scanned_files"] = sum(len(batch.candidates) for batch in self._batches)

        for batch in self._batches:
            for path in batch.candidates:
                ready, reason = readiness(path, batch.mark_seen)
                if not ready:
                    summary["skipped_files"] += 1
                    if reason == "preexisting" and batch.on_preexisting is not None:
                        batch.on_preexisting(path)
                    continue

                try:
                    delta = await batch.process(path)
                except Exception as exc:  # noqa: BLE001
                    summary["failed_files"] += 1
                    if batch.on_error is not None:
                        batch.on_error(path, exc)
                    continue

                for key, value in delta.items():
                    if key in summary:
                        summary[key] += int(value)

        return summary


__all__ = [
    "ReadinessCheck",
    "ScanBatch",
    "ScanPipeline",
    "blank_scan_summary",
]




# --- FILE: core/automation/scan_rules.py ---

"""Routing rules for live automation scan targets."""

# internal import removed: from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ScanRouteKind = Literal["command", "ingest"]


@dataclass(frozen=True)
class ScanRoute:
    name: str
    kind: ScanRouteKind
    folder: Path
    allowed_extensions: set[str] | None
    mark_seen: bool
    source: str = ""
    move_after: bool = False
    move_to_failed: bool = False
    failure_label: str = "Ingestion"


def build_scan_routes(
    *,
    commands_dir: Path,
    rag_dir: Path,
    screenshots_dir: Path,
    recordings_dir: Path,
    command_extensions: set[str],
    image_extensions: set[str],
    video_extensions: set[str],
    watch_screenshots: bool,
    watch_recordings: bool,
) -> tuple[ScanRoute, ...]:
    routes: list[ScanRoute] = [
        ScanRoute(
            name="commands",
            kind="command",
            folder=commands_dir,
            allowed_extensions=command_extensions,
            mark_seen=False,
            move_to_failed=True,
            failure_label="Command ingestion",
        ),
        ScanRoute(
            name="rag",
            kind="ingest",
            folder=rag_dir,
            allowed_extensions=None,
            mark_seen=False,
            source="drop_rag",
            move_after=True,
            move_to_failed=True,
            failure_label="RAG ingestion",
        ),
    ]

    if watch_screenshots:
        routes.append(
            ScanRoute(
                name="screenshots",
                kind="ingest",
                folder=screenshots_dir,
                allowed_extensions=image_extensions,
                mark_seen=True,
                source="screenshot",
                move_after=False,
                move_to_failed=False,
                failure_label="Screenshot ingestion",
            )
        )

    if watch_recordings:
        routes.append(
            ScanRoute(
                name="recordings",
                kind="ingest",
                folder=recordings_dir,
                allowed_extensions=video_extensions,
                mark_seen=True,
                source="recording",
                move_after=False,
                move_to_failed=False,
                failure_label="Recording ingestion",
            )
        )

    return tuple(routes)


__all__ = ["ScanRoute", "ScanRouteKind", "build_scan_routes"]




# --- FILE: core/runtime/paths.py ---

# internal import removed: from __future__ import annotations

import os
from pathlib import Path

# Path to the project root folder (two levels up from core/runtime/paths.py)
core_runtime_paths_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(path_str: str | Path) -> Path:
    """
    Resolve a path relative to the project root directory if it is relative,
    otherwise return it as an absolute path.
    """
    raw_path = Path(os.path.expandvars(str(path_str))).expanduser()
    if raw_path.is_absolute():
        return raw_path.resolve(strict=False)
    return (core_runtime_paths_PROJECT_ROOT / raw_path).resolve(strict=False)




# --- FILE: core/automation/live_automation.py ---

"""Always-on local automation for command inbox and RAG ingestion."""

# internal import removed: from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import re
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

# internal import removed: from core.automation.scan_pipeline import ScanBatch, ScanPipeline
# internal import removed: from core.automation.scan_rules import ScanRoute, build_scan_routes
# internal import removed: from core.runtime.paths import _resolve_path

logger = logging.getLogger(__name__)

CommandHandler = Callable[[str], Awaitable[str]]

core_automation_live_automation__TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".rst",
    ".json",
    ".yaml",
    ".yml",
    ".csv",
    ".tsv",
    ".py",
    ".js",
    ".ts",
    ".html",
    ".css",
    ".ini",
    ".log",
}
core_automation_live_automation__IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"}
core_automation_live_automation__VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
_COMMAND_EXTENSIONS = {".txt", ".md", ".task", ".cmd"}

_DEFAULT_DROP_ROOT = "workspace/jarvis_dropbox"
_DEFAULT_SCREENSHOT_DIR = "outputs/screenshots"
_DEFAULT_RECORDING_DIR = "outputs/screen_recordings"


def _cfg_bool(config: Any, section: str, key: str, fallback: bool) -> bool:
    try:
        return bool(config.getboolean(section, key, fallback=fallback))
    except Exception:
        return fallback


def _cfg_float(config: Any, section: str, key: str, fallback: float) -> float:
    try:
        return float(config.get(section, key, fallback=str(fallback)))
    except Exception:
        return fallback


def _cfg_int(config: Any, section: str, key: str, fallback: int) -> int:
    try:
        return int(config.get(section, key, fallback=str(fallback)))
    except Exception:
        return fallback


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def core_automation_live_automation__normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def core_automation_live_automation__truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


@dataclass
class AutomationStats:
    started_at: str = ""
    last_scan_at: str = ""
    last_error: str = ""
    scanned_files: int = 0
    ingested_files: int = 0
    ingested_chunks: int = 0
    commands_executed: int = 0
    failed_files: int = 0
    skipped_files: int = 0
    live_screen_updates: int = 0


class LiveAutomationEngine:
    """Poll-based automation engine for command inbox and RAG ingestion."""

    def __init__(
        self,
        *,
        config: Any,
        memory: Any,
        llm: Any | None = None,
        command_handler: CommandHandler | None = None,
        desktop_observer: Any | None = None,
        notifier: Any | None = None,
        dag_executor: Any | None = None,
    ) -> None:
        self.config = config
        self.memory = memory
        self.llm = llm
        self.command_handler = command_handler
        self.desktop_observer = desktop_observer
        self.notifier = notifier
        self.dag_executor = dag_executor

        self.enabled = _cfg_bool(config, "automation", "enabled", True)
        self.auto_execute_commands = _cfg_bool(
            config,
            "automation",
            "auto_execute_commands",
            True,
        )
        self.watch_screenshots = _cfg_bool(
            config,
            "automation",
            "watch_screenshots",
            True,
        )
        self.watch_recordings = _cfg_bool(
            config,
            "automation",
            "watch_recordings",
            True,
        )
        self.live_screen_enabled = _cfg_bool(
            config,
            "automation",
            "live_screen_enabled",
            True,
        )
        self.ingest_existing_on_start = _cfg_bool(
            config,
            "automation",
            "ingest_existing_on_start",
            False,
        )

        self.poll_interval_seconds = max(
            0.5,
            _cfg_float(config, "automation", "poll_interval_seconds", 3.0),
        )
        self.min_file_age_seconds = max(
            0.0,
            _cfg_float(config, "automation", "min_file_age_seconds", 2.0),
        )
        self.live_screen_interval_seconds = max(
            5.0,
            _cfg_float(config, "automation", "live_screen_interval_seconds", 20.0),
        )
        self.video_frame_interval_seconds = max(
            0.5,
            _cfg_float(config, "automation", "video_frame_interval_seconds", 2.0),
        )
        self.max_video_samples = max(
            1,
            _cfg_int(config, "automation", "max_video_samples", 20),
        )
        self.max_text_chars_per_item = max(
            500,
            _cfg_int(config, "automation", "max_text_chars_per_item", 12000),
        )
        self.chunk_size_chars = max(
            300,
            _cfg_int(config, "automation", "chunk_size_chars", 1200),
        )
        self.chunk_overlap_chars = max(
            0,
            _cfg_int(config, "automation", "chunk_overlap_chars", 120),
        )
        self.max_seen_fingerprints = max(
            200,
            _cfg_int(config, "automation", "max_seen_fingerprints", 10000),
        )

        raw_drop_root = str(
            config.get("automation", "drop_root", fallback=_DEFAULT_DROP_ROOT)
        ).strip()
        self.drop_root = _resolve_path(raw_drop_root or _DEFAULT_DROP_ROOT)
        self.commands_dir = _resolve_path(
            str(
                config.get(
                    "automation",
                    "commands_folder",
                    fallback=str(self.drop_root / "commands"),
                )
            )
        )
        self.rag_dir = _resolve_path(
            str(
                config.get(
                    "automation",
                    "rag_folder",
                    fallback=str(self.drop_root / "rag"),
                )
            )
        )
        self.processed_dir = _resolve_path(
            str(
                config.get(
                    "automation",
                    "processed_folder",
                    fallback=str(self.drop_root / "processed"),
                )
            )
        )
        self.failed_dir = _resolve_path(
            str(
                config.get(
                    "automation",
                    "failed_folder",
                    fallback=str(self.drop_root / "failed"),
                )
            )
        )
        self.screenshots_dir = _resolve_path(
            str(
                config.get(
                    "automation",
                    "screenshots_folder",
                    fallback=_DEFAULT_SCREENSHOT_DIR,
                )
            )
        )
        self.recordings_dir = _resolve_path(
            str(
                config.get(
                    "automation",
                    "recordings_folder",
                    fallback=_DEFAULT_RECORDING_DIR,
                )
            )
        )
        self.log_file = _resolve_path(
            str(
                config.get(
                    "automation",
                    "ingest_log_file",
                    fallback="runtime/automation_ingest.jsonl",
                )
            )
        )
        self.state_file = _resolve_path(
            str(
                config.get(
                    "automation",
                    "state_file",
                    fallback="runtime/automation_state.json",
                )
            )
        )

        self._running = False
        self._task: asyncio.Task | None = None
        self._startup_ts = time.time()
        self._fingerprints: set[str] = set()
        self._fingerprints_order: list[str] = []
        self._last_live_screen_hash = ""
        self._last_live_screen_at = 0.0
        self._stats = AutomationStats()
        self._load_state()

    async def start(self) -> None:
        if not self.enabled:
            logger.info("Live automation is disabled by config.")
            return
        if self._running:
            return
        self._ensure_directories()
        self._running = True
        self._stats.started_at = _iso_now()
        self._task = asyncio.create_task(
            self._run_loop(),
            name="jarvis-live-automation",
        )
        logger.info("Live automation started.")

    async def stop(self) -> None:
        self._running = False
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._task = None
        self._save_state()
        logger.info("Live automation stopped.")

    async def enable(self) -> dict[str, Any]:
        self.enabled = True
        await self.start()
        return self.status()

    async def disable(self) -> dict[str, Any]:
        self.enabled = False
        await self.stop()
        return self.status()

    async def force_scan(self) -> dict[str, Any]:
        self._ensure_directories()
        return await self.scan_once()

    async def scan_once(self) -> dict[str, Any]:
        batches = await self._build_scan_batches()
        pipeline = ScanPipeline(batches)
        summary = await pipeline.run(self._scan_readiness)
        self._apply_scan_summary(summary)
        self._save_state()
        return summary

    async def _build_scan_batches(self) -> list[ScanBatch]:
        routes = build_scan_routes(
            commands_dir=self.commands_dir,
            rag_dir=self.rag_dir,
            screenshots_dir=self.screenshots_dir,
            recordings_dir=self.recordings_dir,
            command_extensions=_COMMAND_EXTENSIONS,
            image_extensions=core_automation_live_automation__IMAGE_EXTENSIONS,
            video_extensions=core_automation_live_automation__VIDEO_EXTENSIONS,
            watch_screenshots=self.watch_screenshots,
            watch_recordings=self.watch_recordings,
        )

        batches: list[ScanBatch] = []
        for route in routes:
            candidates = tuple(await self._iter_files(route.folder, route.allowed_extensions))
            if route.kind == "command":
                batches.append(
                    self._build_command_scan_batch(route, candidates)
                )
                continue
            batches.append(self._build_ingest_scan_batch(route, candidates))
        return batches

    def _build_command_scan_batch(
        self,
        route: ScanRoute,
        candidates: tuple[Path, ...],
    ) -> ScanBatch:
        async def _process(path: Path) -> dict[str, int]:
            await self._process_command_file(path)
            return {"commands_processed": 1}

        return ScanBatch(
            name=route.name,
            candidates=candidates,
            mark_seen=route.mark_seen,
            process=_process,
            on_preexisting=self._remember_file,
            on_error=lambda path, exc: self._handle_scan_failure(route, path, exc),
        )

    def _build_ingest_scan_batch(
        self,
        route: ScanRoute,
        candidates: tuple[Path, ...],
    ) -> ScanBatch:
        async def _process(path: Path) -> dict[str, int]:
            chunks = await self._ingest_file(
                path,
                source=route.source,
                move_after=route.move_after,
            )
            return {
                "files_ingested": 1,
                "chunks_ingested": chunks,
            }

        return ScanBatch(
            name=route.name,
            candidates=candidates,
            mark_seen=route.mark_seen,
            process=_process,
            on_preexisting=self._remember_file,
            on_error=lambda path, exc: self._handle_scan_failure(route, path, exc),
        )

    def _scan_readiness(self, path: Path, mark_seen: bool) -> tuple[bool, str]:
        return self._file_ready(path, mark_seen=mark_seen)

    def _handle_scan_failure(
        self,
        route: ScanRoute,
        path: Path,
        exc: Exception,
    ) -> None:
        logger.warning("%s failed for %s: %s", route.failure_label, path, exc)
        self._stats.last_error = str(exc)
        if route.move_to_failed:
            self._move_to_failed(path, error=str(exc))

    def _apply_scan_summary(self, summary: dict[str, int]) -> None:
        self._stats.scanned_files += int(summary.get("scanned_files", 0))
        self._stats.ingested_files += int(summary.get("files_ingested", 0))
        self._stats.ingested_chunks += int(summary.get("chunks_ingested", 0))
        self._stats.failed_files += int(summary.get("failed_files", 0))
        self._stats.skipped_files += int(summary.get("skipped_files", 0))
        self._stats.last_scan_at = _iso_now()

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "running": self._running,
            "auto_execute_commands": self.auto_execute_commands,
            "drop_root": str(self.drop_root),
            "commands_dir": str(self.commands_dir),
            "rag_dir": str(self.rag_dir),
            "screenshots_dir": str(self.screenshots_dir),
            "recordings_dir": str(self.recordings_dir),
            "processed_dir": str(self.processed_dir),
            "failed_dir": str(self.failed_dir),
            "stats": asdict(self._stats),
        }

    def status_line(self) -> str:
        state = "running" if self._running else "stopped"
        return (
            f"Automation {state} (enabled={self.enabled}) | "
            f"commands={self._stats.commands_executed} | "
            f"ingested_files={self._stats.ingested_files} | "
            f"ingested_chunks={self._stats.ingested_chunks} | "
            f"live_screen_updates={self._stats.live_screen_updates}"
        )

    async def search_rag(self, query: str, top_k: int = 5) -> str:
        query = core_automation_live_automation__normalize_text(query)
        if not query:
            return "Provide a query after 'rag search'."

        try:
            recalled = await self.memory.recall_all(query, top_k=max(top_k, 10))
        except Exception as exc:  # noqa: BLE001
            return f"RAG search failed: {exc}"

        episodes = recalled.get("episodes", []) if isinstance(recalled, dict) else []
        matches: list[tuple[float, str]] = []
        for item in episodes:
            if not isinstance(item, dict):
                continue
            category = str(item.get("category", "") or "").lower()
            event = str(item.get("event") or item.get("document") or "")
            if "rag" not in category and "[RAG Source]" not in event:
                continue
            try:
                score = float(item.get("score", 0.0) or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            matches.append((score, event))

        if not matches:
            return f"No RAG matches found for '{query}'."

        matches.sort(key=lambda row: row[0], reverse=True)
        lines: list[str] = []
        for index, (score, event) in enumerate(matches[: max(1, top_k)], start=1):
            source = self._extract_metadata_value(event, "source")
            content = self._extract_metadata_value(event, "content")
            snippet = core_automation_live_automation__truncate(core_automation_live_automation__normalize_text(content or event), 180)
            if source:
                lines.append(f"{index}. [{source}] {snippet} (score={score:.2f})")
            else:
                lines.append(f"{index}. {snippet} (score={score:.2f})")
        return "RAG matches:\n" + "\n".join(lines)

    async def _run_loop(self) -> None:
        try:
            while self._running:
                try:
                    await self.scan_once()
                    await self._poll_live_screen()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Live automation loop error: %s", exc, exc_info=True)
                    self._stats.last_error = str(exc)
                await asyncio.sleep(self.poll_interval_seconds)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.critical("Fatal error in live automation loop: %s", exc, exc_info=True)
            self._stats.last_error = str(exc)
            await asyncio.sleep(self.poll_interval_seconds * 2)

    async def _process_command_file(self, path: Path) -> None:
        command_text = await self._read_text_file(path)
        command_text = self._extract_command(command_text)
        if not command_text:
            raise ValueError("Command file is empty.")

        if not self.auto_execute_commands:
            raise RuntimeError("Auto command execution is disabled by config.")

        if self.command_handler is None and self.dag_executor is None:
            raise RuntimeError("No command handler or DAG executor is attached.")

        if self.dag_executor is not None and command_text.strip().startswith("{"):
            try:
                plan = json.loads(command_text)
                if "steps" in plan:
                    from core.context.context import TaskExecutionContext
                    ctx = TaskExecutionContext(
                        task_id=path.stem,
                        trace_id=hashlib.md5(path.name.encode()).hexdigest()[:8],
                    )
                    result = await self.dag_executor.execute(plan, context=ctx)
                    response = json.dumps(result)
                else:
                    response = await self.command_handler(command_text) if self.command_handler else "No handler"
            except Exception as e:
                logger.warning("Failed to execute plan via DAGExecutor: %s", e)
                response = await self.command_handler(command_text) if self.command_handler else f"Failed: {e}"
        else:
            response = await self.command_handler(command_text) if self.command_handler else "No handler"
        self._stats.commands_executed += 1
        await self._append_log(
            {
                "timestamp": _iso_now(),
                "type": "command",
                "path": str(path),
                "command": command_text,
                "response": str(response or ""),
            }
        )

        await self._store_rag_text(
            source="command_result",
            path=path,
            text=f"Command: {command_text}\nResult: {response}",
        )

        processed_path = self._relocate(path, self.processed_dir / "commands")
        result_file = processed_path.with_suffix(processed_path.suffix + ".result.txt")
        result_file.parent.mkdir(parents=True, exist_ok=True)
        def _write():
            result_file.write_text(
                f"Command: {command_text}\n\nResult:\n{response}\n",
                encoding="utf-8",
            )
        await asyncio.to_thread(_write)

        if self.notifier is not None:
            notify = getattr(self.notifier, "notify", None)
            if callable(notify):
                notify(f"Jarvis command executed from inbox: {command_text}")

    async def _ingest_file(self, path: Path, *, source: str, move_after: bool) -> int:
        text = await asyncio.to_thread(self._extract_text_payload, path)
        if not text:
            text = f"File ingested with no extractable text: {path.name}"

        chunks = await self._store_rag_text(source=source, path=path, text=text)

        await self._append_log(
            {
                "timestamp": _iso_now(),
                "type": "rag_ingest",
                "path": str(path),
                "source": source,
                "chars": len(text),
                "chunks": chunks,
            }
        )

        if move_after:
            self._relocate(path, self.processed_dir / "rag")

        return chunks

    def _extract_text_payload(self, path: Path) -> str:
        from core.automation.payload_extractor import PayloadExtractor
        extractor = PayloadExtractor(
            self.max_text_chars_per_item,
            self.video_frame_interval_seconds,
            self.max_video_samples
        )
        return extractor.extract_text_payload(path)

    # OCR methods have been extracted to core.automation.payload_extractor

    async def _poll_live_screen(self) -> None:
        if not self.live_screen_enabled:
            return
        if self.desktop_observer is None:
            return
        now = time.time()
        if now - self._last_live_screen_at < self.live_screen_interval_seconds:
            return
        self._last_live_screen_at = now

        observe = getattr(self.desktop_observer, "observe", None)
        if not callable(observe):
            return
        try:
            observation = await observe(label="live_automation")
        except Exception as exc:  # noqa: BLE001
            logger.debug("Live screen observation failed: %s", exc)
            return

        ocr_text = core_automation_live_automation__normalize_text(str(getattr(observation, "ocr_text", "") or ""))
        if not ocr_text:
            return

        digest = hashlib.sha256(ocr_text.encode("utf-8", errors="replace")).hexdigest()
        if digest == self._last_live_screen_hash:
            return
        self._last_live_screen_hash = digest

        screenshot_path = str(getattr(observation, "screenshot_path", "") or "")
        text = f"Live screen OCR: {ocr_text}"
        if screenshot_path:
            text += f"\nScreenshot: {screenshot_path}"

        await self._store_rag_text(
            source="live_screen",
            path=Path(screenshot_path) if screenshot_path else self.screenshots_dir,
            text=text,
        )
        self._stats.live_screen_updates += 1

    async def _store_rag_text(self, *, source: str, path: Path, text: str) -> int:
        from core.automation.rag_ingester import RagIngester
        ingester = RagIngester(
            self.chunk_size_chars,
            self.chunk_overlap_chars,
            self.memory,
            self._stats
        )
        return await ingester.store_rag_text(source=source, path=path, text=text)

    def _file_ready(self, path: Path, *, mark_seen: bool) -> tuple[bool, str]:
        if not path.exists() or not path.is_file():
            return False, "not_file"
        if path.name.startswith("."):
            return False, "hidden"

        stat = path.stat()
        age = time.time() - float(stat.st_mtime)
        if age < self.min_file_age_seconds:
            return False, "too_new"

        if (not self.ingest_existing_on_start) and (stat.st_mtime < self._startup_ts):
            return False, "preexisting"

        if mark_seen:
            fingerprint = self._fingerprint(path, stat)
            if fingerprint in self._fingerprints:
                return False, "seen"
            self._remember_fingerprint(fingerprint)
        return True, "ready"

    async def _iter_files(self, folder: Path, allowed_extensions: set[str] | None) -> list[Path]:
        if not folder.exists() or not folder.is_dir():
            return []
        files: list[Path] = []
        count = 0
        try:
            for item in folder.iterdir():
                count += 1
                if count % 100 == 0:
                    await asyncio.sleep(0.001)
                
                if count > 2000:  # batch limit
                    break

                if not item.is_file():
                    continue
                if allowed_extensions is not None and item.suffix.lower() not in allowed_extensions:
                    continue
                files.append(item)
        except OSError:
            pass
            
        files.sort(key=lambda p: p.stat().st_mtime)
        return files[:500]

    @staticmethod
    def _extract_command(raw_text: str) -> str:
        text = str(raw_text or "").strip()
        if not text:
            return ""
        first_line = text.splitlines()[0].strip()
        lowered = first_line.lower()
        prefixes = ("command:", "cmd:", "task:")
        for prefix in prefixes:
            if lowered.startswith(prefix):
                return text[len(prefix) :].strip()
        return text

    @staticmethod
    async def _read_text_file(path: Path, max_bytes: int = 2_000_000) -> str:
        def _read() -> bytes:
            return path.read_bytes()[: max(1, max_bytes)]
        data: bytes = await asyncio.to_thread(_read)
        return data.decode("utf-8", errors="replace")

    def _move_to_failed(self, path: Path, *, error: str) -> None:
        destination = self._relocate(path, self.failed_dir)
        error_file = destination.with_suffix(destination.suffix + ".error.txt")
        error_file.parent.mkdir(parents=True, exist_ok=True)
        def _write():
            error_file.write_text(error + "\n", encoding="utf-8")
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(asyncio.to_thread(_write))
        except RuntimeError:
            _write()

    def _relocate(self, source: Path, destination_dir: Path) -> Path:
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = self._unique_path(destination_dir / source.name)
        shutil.move(str(source), str(destination))
        return destination

    @staticmethod
    def _unique_path(path: Path) -> Path:
        if not path.exists():
            return path
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        counter = 1
        while True:
            candidate = path.with_name(f"{path.stem}_{stamp}_{counter}{path.suffix}")
            if not candidate.exists():
                return candidate
            counter += 1

    def _fingerprint(self, path: Path, stat: Any | None = None) -> str:
        if stat is None:
            stat = path.stat()
        raw = f"{path.resolve()}::{int(stat.st_mtime)}::{int(stat.st_size)}"
        return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()

    def _remember_file(self, path: Path) -> None:
        try:
            self._remember_fingerprint(self._fingerprint(path))
        except Exception:
            return

    def _remember_fingerprint(self, fingerprint: str) -> None:
        if fingerprint in self._fingerprints:
            return
        self._fingerprints.add(fingerprint)
        self._fingerprints_order.append(fingerprint)
        while len(self._fingerprints_order) > self.max_seen_fingerprints:
            oldest = self._fingerprints_order.pop(0)
            self._fingerprints.discard(oldest)

    def _ensure_directories(self) -> None:
        for folder in (
            self.drop_root,
            self.commands_dir,
            self.rag_dir,
            self.processed_dir,
            self.failed_dir,
            self.screenshots_dir,
            self.recordings_dir,
            self.log_file.parent,
            self.state_file.parent,
        ):
            folder.mkdir(parents=True, exist_ok=True)

    async def _append_log(self, payload: dict[str, Any]) -> None:
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        def _write():
            with self.log_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
        await asyncio.to_thread(_write)

    def _load_state(self) -> None:
        if not self.state_file.exists():
            return
        try:
            raw = json.loads(self.state_file.read_text(encoding="utf-8"))
            seen = raw.get("seen_fingerprints", [])
            if isinstance(seen, list):
                for item in seen[-self.max_seen_fingerprints :]:
                    if isinstance(item, str):
                        self._remember_fingerprint(item)
            stats = raw.get("stats", {})
            if isinstance(stats, dict):
                self._stats = AutomationStats(**{**asdict(self._stats), **stats})
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not load automation state: %s", exc)

    def _save_state(self) -> None:
        def _write():
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "saved_at": _iso_now(),
                "seen_fingerprints": self._fingerprints_order[-self.max_seen_fingerprints :],
                "stats": asdict(self._stats),
            }
            self.state_file.write_text(
                json.dumps(payload, indent=2, ensure_ascii=True),
                encoding="utf-8",
            )
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(asyncio.to_thread(_write))
        except RuntimeError:
            _write()

    @staticmethod
    def _extract_metadata_value(block: str, key: str) -> str:
        pattern = rf"^{re.escape(key)}=(.*)$"
        match = re.search(pattern, str(block or ""), flags=re.MULTILINE)
        if not match:
            return ""
        return str(match.group(1) or "").strip()


__all__ = ["LiveAutomationEngine", "AutomationStats"]




# --- FILE: core/automation/__init__.py ---

"""Automation primitives for always-on Jarvis workflows."""

# internal import removed: from core.automation.live_automation import LiveAutomationEngine

__all__ = ["LiveAutomationEngine"]




# --- FILE: core/automation/payload_extractor.py ---

from pathlib import Path
import re


core_automation_payload_extractor__TEXT_EXTENSIONS = {
    ".txt", ".md", ".rst", ".json", ".yaml", ".yml",
    ".csv", ".tsv", ".py", ".js", ".ts", ".html",
    ".css", ".ini", ".log",
}
core_automation_payload_extractor__IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"}
core_automation_payload_extractor__VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def core_automation_payload_extractor__normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def core_automation_payload_extractor__truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


def read_text_file(path: Path, max_bytes: int = 2_000_000) -> str:
    data = path.read_bytes()[: max(1, max_bytes)]
    return data.decode("utf-8", errors="replace")


class PayloadExtractor:
    def __init__(self, max_text_chars_per_item: int, video_frame_interval_seconds: float, max_video_samples: int):
        self.max_text_chars_per_item = max_text_chars_per_item
        self.video_frame_interval_seconds = video_frame_interval_seconds
        self.max_video_samples = max_video_samples

    def extract_text_payload(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix in core_automation_payload_extractor__TEXT_EXTENSIONS:
            raw = read_text_file(path)
            return core_automation_payload_extractor__truncate(raw, self.max_text_chars_per_item)
        if suffix in core_automation_payload_extractor__IMAGE_EXTENSIONS:
            text = self.extract_text_from_image(path)
            return core_automation_payload_extractor__truncate(text, self.max_text_chars_per_item)
        if suffix in core_automation_payload_extractor__VIDEO_EXTENSIONS:
            text = self.extract_text_from_video(path)
            return core_automation_payload_extractor__truncate(text, self.max_text_chars_per_item)
        return f"Unsupported file type for direct parsing: {path.name}"

    def extract_text_from_image(self, path: Path) -> str:
        try:
            from PIL import Image
            import pytesseract
        except Exception as exc:  # noqa: BLE001
            return f"OCR dependency missing for image '{path.name}': {exc}"

        try:
            with Image.open(path) as image:
                raw = pytesseract.image_to_string(image)
        except Exception as exc:  # noqa: BLE001
            return f"Image OCR failed for '{path.name}': {exc}"

        text = core_automation_payload_extractor__normalize_text(raw)
        if not text:
            return f"No OCR text found in image '{path.name}'."
        return text

    def extract_text_from_video(self, path: Path) -> str:
        try:
            import cv2
            from PIL import Image
            import pytesseract
        except Exception as exc:  # noqa: BLE001
            return f"Video OCR dependency missing for '{path.name}': {exc}"

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return f"Could not open video '{path.name}'."

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            fps = 1.0
        sample_every_frames = max(1, int(round(fps * self.video_frame_interval_seconds)))

        frame_index = 0
        captured = 0
        snippets: list[str] = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        
        try:
            while captured < self.max_video_samples:
                if total_frames > 0 and frame_index >= total_frames:
                    break
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = cap.read()
                if not ok:
                    break
                    
                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(rgb)
                    raw = pytesseract.image_to_string(image)
                    text = core_automation_payload_extractor__normalize_text(raw)
                    if text:
                        second = frame_index / max(fps, 1.0)
                        snippets.append(f"[t={second:.1f}s] {text}")
                        captured += 1
                except Exception:
                    pass
                frame_index += sample_every_frames
        finally:
            cap.release()

        if not snippets:
            return f"No OCR text found in video '{path.name}'."
        return "\n".join(snippets)




# --- FILE: core/automation/rag_ingester.py ---

from pathlib import Path
from typing import Any
import logging

logger = logging.getLogger(__name__)

class RagIngester:
    def __init__(self, chunk_size_chars: int, chunk_overlap_chars: int, memory: Any, stats: Any):
        self.chunk_size_chars = chunk_size_chars
        self.chunk_overlap_chars = chunk_overlap_chars
        self.memory = memory
        self.stats = stats

    async def store_rag_text(self, *, source: str, path: Path, text: str) -> int:
        clean = str(text or "").strip()
        if not clean:
            return 0

        chunks = self.chunk_text(clean)
        total = len(chunks)
        payloads = []
        for index, chunk in enumerate(chunks, start=1):
            payload = (
                "[RAG Source]\n"
                f"source={source}\n"
                f"path={path}\n"
                f"chunk={index}/{total}\n"
                f"content={chunk}"
            )
            payloads.append(payload)

        stored = 0
        try:
            # Assuming store_episodes_batch accepts a list of payloads and category
            await self.memory.store_episodes_batch(payloads, category="rag")
            stored = len(payloads)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to store RAG chunks: %s", exc)
            self.stats.last_error = str(exc)

        return stored

    def chunk_text(self, text: str) -> list[str]:
        size = self.chunk_size_chars
        overlap = min(self.chunk_overlap_chars, max(0, size - 20))
        if len(text) <= size:
            return [text]

        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(len(text), start + size)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(text):
                break
            start = max(start + 1, end - overlap)
        return chunks




# --- FILE: core/base_controller.py ---

import abc
import asyncio
import logging
from typing import Any, List

logger = logging.getLogger(__name__)

class BaseController(abc.ABC):
    """
    Abstract Base Class for Controllers.
    Enforces startup and shutdown methods for subclasses, and provides
    implementations using asyncio.TaskGroup to synchronize the 
    startup and shutdown of child modules (injectable subsystems).
    """

    def __init__(self) -> None:
        self._subsystems: List[Any] = []

    def register_subsystem(self, subsystem: Any) -> None:
        """Registers a child module/subsystem to be synchronized."""
        self._subsystems.append(subsystem)

    @abc.abstractmethod
    async def startup(self) -> None:
        """
        Starts up the controller.
        Subclasses MUST override this method.
        To synchronize the startup of registered subsystems, call `await super().startup()`.
        """
        if self._subsystems:
            logger.info(f"{self.__class__.__name__}: Starting {len(self._subsystems)} subsystems concurrently...")
            async with asyncio.TaskGroup() as tg:
                for subsystem in self._subsystems:
                    if hasattr(subsystem, "startup") and callable(subsystem.startup):
                        tg.create_task(subsystem.startup())
            logger.info(f"{self.__class__.__name__}: Subsystem startup complete.")

    @abc.abstractmethod
    async def shutdown(self) -> None:
        """
        Shuts down the controller.
        Subclasses MUST override this method.
        To synchronize the shutdown of registered subsystems, call `await super().shutdown()`.
        """
        if self._subsystems:
            logger.info(f"{self.__class__.__name__}: Shutting down {len(self._subsystems)} subsystems concurrently...")
            async with asyncio.TaskGroup() as tg:
                for subsystem in self._subsystems:
                    if hasattr(subsystem, "shutdown") and callable(subsystem.shutdown):
                        tg.create_task(subsystem.shutdown())
            logger.info(f"{self.__class__.__name__}: Subsystem shutdown complete.")




# --- FILE: core/controller/__init__.py ---

"""Controller intents and services."""




# --- FILE: core/controller/automation_manager.py ---

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

class AutomationManager:
    def __init__(
        self,
        config: Any,
        memory: Any,
        llm: Any,
        notifier: Any,
        desktop_observer: Any,
        container: Any,
        command_handler: Callable[[str], Any]
    ) -> None:
        self.config = config
        self.live_automation = None
        
        if hasattr(config, "has_section") and config.has_section("automation") and config.getboolean("automation", "enabled", fallback=False):
            try:
                from core.automation.live_automation import LiveAutomationEngine
                
                async def _handler(cmd: str) -> str:
                    return str(await command_handler(cmd))

                dag_executor = None
                if container is not None and hasattr(container, "has") and container.has("dag_executor"):
                    dag_executor = container.resolve("dag_executor")

                self.live_automation = LiveAutomationEngine(
                    config=config,
                    memory=memory,
                    llm=llm,
                    command_handler=_handler,
                    desktop_observer=desktop_observer,
                    notifier=notifier,
                    dag_executor=dag_executor,
                )
            except Exception as exc:
                logger.warning("Failed to initialize LiveAutomationEngine: %s", exc, exc_info=True)

    async def startup(self) -> None:
        if self.live_automation is not None:
            await self.live_automation.start()

    async def shutdown(self) -> None:
        if self.live_automation is not None:
            await self.live_automation.stop()




# --- FILE: core/controller/complexity_scorer.py ---

"""Heuristics-based complexity scorer for Adaptive Intelligence Routing."""

# internal import removed: from __future__ import annotations

import re
import logging
from typing import Any

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword sets (unchanged categories)
# ---------------------------------------------------------------------------
_REFLEX_KEYWORDS = {"weather", "time", "date", "status", "hello", "hi", "ping"}

_DEEP_KEYWORDS = {
    "architecture", "debug", "refactor", "complex", "system design",
    "explain how", "why is this failing", "optimize",
}

_AGENTIC_KEYWORDS = {
    "create", "write", "plan", "workflow", "automate", "search", "find",
    "organize", "download", "fetch", "open", "launch", "start", "close",
    "type", "click", "do", "execute", "run", "make",
}

_CONDITIONAL_WORDS = {"if", "when", "unless", "assuming", "provided", "suppose"}
_TECHNICAL_TERMS = {
    "api", "async", "await", "class", "function", "method", "endpoint",
    "database", "schema", "deploy", "container", "docker", "kubernetes",
    "pipeline", "microservice", "oauth", "jwt", "websocket", "regex",
}

# Token-estimation multipliers per category
_TOKEN_MULTIPLIERS: dict[str, float] = {
    "Reflex": 1.5,
    "Chat": 4.0,
    "Agentic": 6.0,
    "Deep_Reasoning": 10.0,
}


# ---------------------------------------------------------------------------
# Structural / vocabulary helpers
# ---------------------------------------------------------------------------
def _structural_signals(text: str) -> dict[str, Any]:
    """Extract structural signals from the raw input."""
    words = text.split()
    word_count = len(words)
    sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
    question_marks = text.count("?")
    has_code = "```" in text or bool(re.search(r'`[^`]+`', text))
    has_vision = bool(re.search(r'\.(png|jpg|jpeg|gif|bmp|webp|svg)\b', text, re.I))

    # Multi-part detection: numbered lists, bullets, conjunctions
    multi_part = bool(re.search(r'(\d+[.)]\s)|([•\-\*]\s)|(,\s*and\s)|\balso\b|\bthen\b', text))

    # Technical term density (fraction of words that are technical)
    lower_words = [w.strip(".,;:!?\"'()[]{}") for w in words]
    tech_count = sum(1 for w in lower_words if w in _TECHNICAL_TERMS)
    tech_density = tech_count / max(word_count, 1)

    # Conditional word count
    cond_count = sum(1 for w in lower_words if w in _CONDITIONAL_WORDS)

    return {
        "word_count": word_count,
        "sentence_count": len(sentences),
        "question_marks": question_marks,
        "has_code": has_code,
        "has_vision": has_vision,
        "multi_part": multi_part,
        "tech_density": tech_density,
        "conditional_count": cond_count,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def classify_request(user_input: str) -> dict[str, Any]:
    """Classify the complexity and type of request to determine routing.

    Classes: Reflex, Chat, Agentic, Deep_Reasoning.

    Returns a dict with routing metadata *and* enriched signals:
    ``class``, ``complexity``, ``route``, ``skip_planner``,
    ``estimated_tokens``, ``needs_reasoning``, ``needs_tools``,
    ``needs_vision``, ``context_weight``.
    """
    text = user_input.lower().strip()
    sig = _structural_signals(text)

    # --- 1. Base classification (keyword matching) ------------------------
    if text in _REFLEX_KEYWORDS or any(
        text.startswith(f"what is the {k}") or text.startswith(f"whats the {k}")
        or text.startswith("what time")
        for k in _REFLEX_KEYWORDS
    ):
        cls, base_cx, route, skip = "Reflex", 0.1, "direct", True
    elif any(k in text for k in _DEEP_KEYWORDS):
        cls, base_cx, route, skip = "Deep_Reasoning", 0.9, "premium", False
    elif any(k in text for k in _AGENTIC_KEYWORDS) or text in {"do it", "go"}:
        cls, base_cx, route, skip = "Agentic", 0.6, "planner", False
    else:
        cls, base_cx, route, skip = "Chat", 0.4, "mid-tier", True

    # --- 2. Complexity modifiers ------------------------------------------
    cx = base_cx
    if sig["word_count"] > 200:
        cx += 0.2
    if sig["multi_part"]:
        cx += 0.15
    if sig["has_code"]:
        cx += 0.1
    # Extra nudge for heavy conditional / technical language
    if sig["tech_density"] > 0.15:
        cx += 0.05
    if sig["conditional_count"] >= 2:
        cx += 0.05
    cx = max(0.0, min(cx, 1.0))

    # --- 3. Derived signals -----------------------------------------------
    estimated_tokens = int(sig["word_count"] * _TOKEN_MULTIPLIERS.get(cls, 4.0))
    needs_reasoning = cls == "Deep_Reasoning" or cx >= 0.75
    needs_tools = cls == "Agentic"
    context_weight = round(min(1.0, 0.2 + sig["tech_density"] + 0.1 * sig["sentence_count"]), 2)

    return {
        "class": cls,
        "complexity": round(cx, 2),
        "route": route,
        "skip_planner": skip,
        "estimated_tokens": estimated_tokens,
        "needs_reasoning": needs_reasoning,
        "needs_tools": needs_tools,
        "needs_vision": sig["has_vision"],
        "context_weight": context_weight,
    }


__all__ = ["classify_request"]




# --- FILE: core/controller/goal_runner.py ---

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

class GoalRunner:
    """Handles goal persistence, checking due goals, and notifications."""
    
    def __init__(
        self,
        goal_manager,
        scheduler,
        notifier,
        voice_layer,
        goals_file: Path,
        goal_check_interval_seconds: int,
        dashboard_update_cb: Callable
    ):
        self.goal_manager = goal_manager
        self.scheduler = scheduler
        self.notifier = notifier
        self.voice_layer = voice_layer
        self.goals_file = Path(goals_file)
        self.goal_check_interval_seconds = goal_check_interval_seconds
        self.dashboard_update = dashboard_update_cb

    async def load_goal_state(self) -> None:
        if not self.goals_file.exists():
            return
        try:
            def _read():
                return self.goals_file.read_text(encoding="utf-8")
            content = await asyncio.to_thread(_read)
            data = json.loads(content)
            goals = data.get("goals", [])
            schedule = data.get("schedule", [])
            if isinstance(goals, list):
                self.goal_manager.restore(goals)
            if isinstance(schedule, list):
                self.scheduler.restore(schedule)
        except Exception as exc:
            logger.warning("Failed to load goals: %s", exc, exc_info=True)

    async def persist_goal_state(self) -> None:
        try:
            self.goals_file.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "goals": self.goal_manager.snapshot(),
                "schedule": self.scheduler.snapshot(),
            }
            def _write():
                self.goals_file.write_text(
                    json.dumps(payload, indent=2),
                    encoding="utf-8",
                )
            await asyncio.to_thread(_write)
        except Exception as exc:
            logger.warning("Failed to persist goals: %s", exc, exc_info=True)

    async def speak_via_voice_layer(self, text: str) -> None:
        if self.voice_layer is None:
            return
        voice_loop = getattr(self.voice_layer, "_loop", None)
        tts = getattr(voice_loop, "tts", None)
        speak = getattr(tts, "speak", None)
        if speak is None:
            return
        result = speak(text)
        if asyncio.iscoroutine(result):
            await result

    async def check_due_goals(self) -> None:
        backoff = 1.0
        while True:
            try:
                await asyncio.sleep(self.goal_check_interval_seconds)
                due_items = self.scheduler.due()
                for item in due_items:
                    msg = f"Due: {item.description or item.goal_id}"
                    self.notifier.notify(msg)
                    item.mark_completed()
                    try:
                        await self.speak_via_voice_layer(msg)
                    except Exception as e:
                        logger.warning("Failed to speak due goal via voice layer: %s", e, exc_info=True)
                if due_items:
                    await self.persist_goal_state()
                self.dashboard_update(active_goals=len(self.goal_manager.active_goals()))
                backoff = 1.0
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Goal check loop error: %s", e, exc_info=True)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)




# --- FILE: core/controller/request_rules.py ---

"""Reusable request-classification rules for controller routing."""

# internal import removed: from __future__ import annotations

DESKTOP_CONTROL_KEYWORDS = (
    "mouse",
    "cursor",
    "desktop",
    "screen",
    "keyboard",
    "hotkey",
    "click",
    "scroll",
    "drag",
    "clipboard",
)

AGENTIC_KEYWORDS = (
    "search",
    "look up",
    "find",
    "check",
    "scrape",
    "get",
    "download",
    "fetch",
    "read",
    "analyze",
    "create",
    "make",
    "write",
    "run",
    "execute",
    "automate",
    "browse",
    "internet",
    "online",
    "web",
    "website",
    "latest",
    "current",
    "today",
    "live",
    "news",
    "price",
    "weather",
    "stats",
    "score",
    "runs",
    "toss",
    "ipl",
    "match",
    "mouse",
    "cursor",
    "desktop",
    "screen",
    "keyboard",
    "hotkey",
    "click",
    "scroll",
    "drag",
    "clipboard",
)

# Phrases that unambiguously indicate the user wants a live web search.
# When any of these are detected Jarvis skips the LLM planner entirely
# and calls web_search directly, then synthesises the answer.
WEB_SEARCH_EXPLICIT_PHRASES = (
    "search the web",
    "search the internet",
    "search online",
    "browse the web",
    "browse the internet",
    "browse online",
    "look it up online",
    "look up online",
    "google it",
    "google for",
    "find online",
    "find on the internet",
    "find on the web",
    "search for",
    "search web for",
    "web search",
    "internet search",
)

LIVE_WEB_HINTS = (
    "internet",
    "online",
    "web",
    "website",
    "latest",
    "current",
    "today",
    "live",
    "news",
    "price",
    "weather",
    "score",
    "stats",
    "runs",
    "toss",
    "ipl",
    "match",
)

LIVE_WEB_REQUEST_MARKERS = (
    "search",
    "browse",
    "find",
    "check",
    "look up",
    "google",
    "get",
    "give me",
    "tell me",
    "update",
    "use internet",
    "what is",
    "who is",
    "when is",
)

ACTIVE_WINDOW_PHRASES = (
    "which app is active",
    "what app is active",
    "what window is active",
    "which window is active",
    "tell me the active window",
    "get the active window",
)


def looks_like_desktop_control_request(lowered: str) -> bool:
    return any(keyword in lowered for keyword in DESKTOP_CONTROL_KEYWORDS)


def is_explicit_web_search(lowered: str) -> bool:
    """Return True when the user unambiguously asks for a live web search."""
    return any(phrase in lowered for phrase in WEB_SEARCH_EXPLICIT_PHRASES)


def should_force_web_search(lowered: str) -> bool:
    if is_explicit_web_search(lowered):
        return True
    if not any(hint in lowered for hint in LIVE_WEB_HINTS):
        return False
    return any(marker in lowered for marker in LIVE_WEB_REQUEST_MARKERS)


def is_active_window_request(lowered: str) -> bool:
    if any(phrase in lowered for phrase in ACTIVE_WINDOW_PHRASES):
        return True
    if "watch the screen" in lowered and "app" in lowered:
        return True
    if "screen" in lowered and "active" in lowered and (
        "app" in lowered or "window" in lowered
    ):
        return True
    return False


def is_preference_relevant(key: str, query: str) -> bool:
    """Determine if a retrieved preference key is relevant to the user query."""
    import re
    def clean(s: str) -> str:
        return re.sub(r'[^a-z0-9\s]', '', s.lower()).strip()
    
    clean_key = clean(key)
    clean_query = clean(query)
    
    if not clean_key or not clean_query:
        return False
        
    # 1. Direct substring match
    if clean_key in clean_query:
        return True
        
    # 2. Key contains query
    if clean_query in clean_key:
        return True
        
    # 3. Word overlap: check if all significant words of key are in query
    stop_words = {"the", "a", "an", "in", "on", "at", "for", "to", "of", "and", "or", "is", "are"}
    key_words = [w for w in clean_key.split() if w not in stop_words]
    if key_words and all(w in clean_query for w in key_words):
        return True
        
    return False


__all__ = [
    "ACTIVE_WINDOW_PHRASES",
    "AGENTIC_KEYWORDS",
    "DESKTOP_CONTROL_KEYWORDS",
    "LIVE_WEB_HINTS",
    "LIVE_WEB_REQUEST_MARKERS",
    "WEB_SEARCH_EXPLICIT_PHRASES",
    "is_active_window_request",
    "is_explicit_web_search",
    "is_preference_relevant",
    "looks_like_desktop_control_request",
    "should_force_web_search",
]





# --- FILE: core/controller/web_search.py ---

"""
Web search fast-path controller logic for Jarvis.
Handles explicit web searches directly, bypassing the full planner,
and synthesizes natural language responses.
"""

# internal import removed: from __future__ import annotations

import logging
import re
from typing import Any

# internal import removed: from core.controller.request_rules import is_explicit_web_search, should_force_web_search

logger = logging.getLogger(__name__)


async def handle_web_search(
    user_input: str,
    trace_id: str,
    memory: Any,
    llm: Any,
    model_router: Any,
    profile: Any,
) -> str:
    """Perform a live web search, synthesize a natural language response, and fall back if needed."""
    try:
        from core.tools.web_tools import web_search as _web_search
        from core.tools.web_tools import _basic_query_cleanup

        query = _basic_query_cleanup(user_input)
        if not query:
            query = user_input.strip()

        logger.info("Executing explicit web search: %r", query, extra={"trace_id": trace_id})
        raw_results = await _web_search(query, max_results=5)
    except Exception as exc:
        logger.warning("Web search tool failed: %s", exc, extra={"trace_id": trace_id})
        return await _dispatch_llm_fallback(user_input, trace_id, memory, llm, model_router, profile)

    if not raw_results or raw_results.startswith("Web search is disabled"):
        logger.info("Web search disabled or empty, falling back to LLM", extra={"trace_id": trace_id})
        return await _dispatch_llm_fallback(user_input, trace_id, memory, llm, model_router, profile)

    if raw_results.startswith("Search failed"):
        logger.warning("Web search returned search failure: %s", raw_results, extra={"trace_id": trace_id})
        return raw_results

    # Synthesis Prompt
    synthesis_prompt = (
        f"The user asked: {user_input}\n\n"
        f"Here are the live web search results:\n{raw_results}\n\n"
        "Please give a concise, helpful reply based only on these results. "
        "Cite URLs where relevant. Do not invent information not present in the results."
    )

    if memory:
        try:
            await memory.build_context_block(user_input)
        except Exception as exc:
            logger.warning("Context building failed: %s", exc, extra={"trace_id": trace_id})

    selected_model = model_router.get_best_available("web_search_summary") if model_router else None
    if selected_model and llm:
        llm.model = selected_model

    messages = [{"role": "user", "content": synthesis_prompt}]
    profile_summary = profile.get_communication_style() if profile else ""

    try:
        if llm:
            response = await llm.chat_async(
                messages,
                query_for_memory=user_input,
                profile_summary=profile_summary,
                trace_id=trace_id,
                task_type="web_search_summary",
            )
            if response and str(response) != "LLM Offline.":
                return str(response)
    except Exception as exc:
        logger.warning("Web search synthesis LLM call failed: %s", exc, extra={"trace_id": trace_id})

    return _format_raw_fallback(raw_results)


async def _dispatch_llm_fallback(
    user_input: str,
    trace_id: str,
    memory: Any,
    llm: Any,
    model_router: Any,
    profile: Any,
) -> str:
    """Clean fallback to raw LLM completion when search tool fails or is disabled."""
    if memory:
        try:
            await memory.build_context_block(user_input)
        except Exception as exc:
            logger.warning("Context building failed during fallback: %s", exc, extra={"trace_id": trace_id})

    selected_model = model_router.get_best_available("chat") if model_router else None
    if selected_model and llm:
        llm.model = selected_model

    messages = [{"role": "user", "content": user_input}]
    profile_summary = profile.get_communication_style() if profile else ""

    try:
        if llm:
            response = await llm.chat_async(
                messages,
                query_for_memory=user_input,
                profile_summary=profile_summary,
                trace_id=trace_id,
            )
            if response and str(response) != "LLM Offline.":
                return str(response)
    except Exception as exc:
        logger.warning("Fallback LLM dispatch failed: %s", exc, extra={"trace_id": trace_id})

    # Recovery preference recall from memory database
    if memory:
        try:
            from core.controller.request_rules import is_preference_relevant
            prefs = await memory.recall_preferences(user_input, top_k=5)
            for pref in prefs:
                val = pref.get("value")
                key = pref.get("key")
                if val and key and is_preference_relevant(key, user_input):
                    return f"Offline fallback from memory: {val}"
        except Exception as exc:
            logger.warning("Memory preference recall failed: %s", exc, extra={"trace_id": trace_id})

    return "I don't know while offline."


def _format_raw_fallback(raw_results: str) -> str:
    """Parse and format raw search results nicely when LLM synthesis is not available."""
    lines = raw_results.splitlines()
    formatted_lines = []
    current_title = None
    current_num = None

    for line in lines:
        line_stripped = line.strip()
        match = re.match(r"^(\d+)\.\s+(.*)$", line_stripped)
        if match:
            if current_title is not None:
                formatted_lines.append(f"{current_num}. {current_title}")
            current_num = match.group(1)
            current_title = match.group(2)
            continue

        if line_stripped.startswith("URL:") and current_title is not None:
            url = line_stripped[4:].strip()
            formatted_lines.append(f"{current_num}. {current_title} ({url})")
            current_title = None
            current_num = None
            continue

        if current_title is not None:
            formatted_lines.append(f"{current_num}. {current_title}")
            current_title = None
            current_num = None

        formatted_lines.append(line)

    if current_title is not None:
        formatted_lines.append(f"{current_num}. {current_title}")

    return "\n".join(formatted_lines)


__all__ = [
    "handle_web_search",
    "is_explicit_web_search",
    "should_force_web_search",
]




# --- FILE: core/controller/intent_handlers.py ---

import uuid
import logging
from typing import TYPE_CHECKING
# internal import removed: from core.controller.request_rules import is_active_window_request, is_explicit_web_search
# internal import removed: from core.desktop.shortcuts import handle_desktop_command, plan_desktop_command
# internal import removed: from core.controller.web_search import handle_web_search

if TYPE_CHECKING:
    from core.controller_v2 import JarvisControllerV2

logger = logging.getLogger(__name__)


def register_intent_routes(ctx: "JarvisControllerV2") -> None:
    async def handle_status(lowered: str, user_input: str, ctx: "JarvisControllerV2") -> str | None:
        return f"Session: {ctx.session_id} | Memory Mode: {ctx.memory.mode}"
    ctx.intent_router.register(lambda _l, _u, c: _l == "status", handle_status)

    async def handle_help(lowered: str, user_input: str, ctx: "JarvisControllerV2") -> str | None:
        return "Commands: status, help, exit, remember <fact>, what's <query>, open <app>, search <query> in <browser>"
    ctx.intent_router.register(lambda _l, _u, c: _l == "help", handle_help)

    async def handle_automation(lowered: str, user_input: str, ctx: "JarvisControllerV2") -> str | None:
        am = getattr(ctx, "automation_manager", None)
        la = getattr(am, "live_automation", None) if am else None
        if la is None:
            return None
        if lowered == "automation status":
            status_info = la.status()
            return f"{la.status_line()}\nDrop Root: {status_info.get('drop_root')}\nCommands Dir: {status_info.get('commands_dir')}\nRAG Dir: {status_info.get('rag_dir')}"
        elif lowered == "automation scan":
            scan_res = await la.force_scan()
            return f"Scan completed: commands={scan_res.get('commands_processed', 0)} files={scan_res.get('files_ingested', 0)} chunks={scan_res.get('chunks_ingested', 0)}"
        elif lowered.startswith("rag search "):
            query = user_input[len("rag search "):].strip()
            res = await la.search_rag(query)
            return str(res) if res is not None else None
        return None
    ctx.intent_router.register(lambda _l, _u, c: (getattr(getattr(c, "automation_manager", None), "live_automation", None) is not None) and (_l in ("automation status", "automation scan") or _l.startswith("rag search ")), handle_automation)

    async def handle_goal(lowered: str, user_input: str, ctx: "JarvisControllerV2") -> str | None:
        return await ctx._handle_goal_intent(lowered, user_input)
    # Always run, returns None if not matched inside
    ctx.intent_router.register(lambda _l, _u, c: True, handle_goal)

    async def handle_pref(lowered: str, user_input: str, ctx: "JarvisControllerV2") -> str | None:
        return await ctx._handle_preference_intent(lowered, user_input)
    ctx.intent_router.register(lambda _l, _u, c: True, handle_pref)

    async def handle_active_window(lowered: str, user_input: str, ctx: "JarvisControllerV2") -> str | None:
        if is_active_window_request(lowered):
            obs = await ctx.desktop_observer.observe()
            title = obs.active_window.get("title", "")
            return f"The active window is: {title}"
        return None
    ctx.intent_router.register(lambda _l, _u, c: True, handle_active_window)

    async def handle_desktop_plan(lowered: str, user_input: str, ctx: "JarvisControllerV2") -> str | None:
        desktop_plan = plan_desktop_command(user_input)
        if desktop_plan is not None:
            if not ctx._app_launch_enabled:
                return ctx._app_launch_disabled_message()
            return await handle_desktop_command(user_input)
        return None
    ctx.intent_router.register(lambda _l, _u, c: True, handle_desktop_plan)

    async def handle_desktop_disabled(lowered: str, user_input: str, ctx: "JarvisControllerV2") -> str | None:
        if ctx._looks_like_desktop_control_request(lowered) and not ctx._gui_automation_enabled:
            return ctx._desktop_control_disabled_message()
        return None
    ctx.intent_router.register(lambda _l, _u, c: True, handle_desktop_disabled)

    async def handle_explicit_web(lowered: str, user_input: str, ctx: "JarvisControllerV2") -> str | None:
        if is_explicit_web_search(lowered):
            ctx._dashboard_update(state="EXECUTE", last_input=user_input)

            web_response = await handle_web_search(
                user_input=user_input, 
                trace_id=uuid.uuid4().hex[:8], 
                memory=ctx.memory, 
                llm=ctx.llm, 
                model_router=ctx.model_router, 
                profile=ctx.profile
            )
            if web_response:
                await ctx.memory.store_conversation(user_input, web_response, ctx.session_id)
                return web_response
        return None
    ctx.intent_router.register(lambda _l, _u, c: True, handle_explicit_web)

    async def handle_agentic(lowered: str, user_input: str, ctx: "JarvisControllerV2") -> str | None:
        classification = getattr(ctx, "current_classification", {})
        if not classification.get("skip_planner", False) and classification.get("route") in ("planner", "premium", "full stack"):
            ctx._dashboard_update(state="PLANNING", last_input=user_input)
            
            # Let task_planner also know about the complexity
            plan = await ctx.task_planner.plan(user_input)
            if plan and plan.get("tools_required"):
                ctx._dashboard_update(state="EXECUTE", last_input=user_input)
                
                task_sm = ctx.container.resolve("state_machine") if ctx.container else None
                if not task_sm:
                    raise RuntimeError("state_machine not found in container")
                
                def _update_dash_state(_old, new):
                    ctx._dashboard_update(state=new.value)
                task_sm.add_listener(_update_dash_state)
                
                try:
                    context = ctx.container.resolve(
                        "task_execution_context",
                        task_id=uuid.uuid4().hex[:8],
                        trace_id=uuid.uuid4().hex[:8],
                        state_machine=task_sm,
                    )
                    
                    # Only build massive context if it's high complexity
                    if classification.get("complexity", 0.5) > 0.5:
                        context_block = await ctx.memory.build_context_block(user_input)
                        context.set("context_block", context_block)
                    else:
                        context.set("context_block", "")
                    
                    trace = await ctx.agent_loop.run(
                        goal=user_input,
                        context=context,
                    )
                    return str(trace.final_response)
                finally:
                    if hasattr(task_sm, "remove_listener"):
                        task_sm.remove_listener(_update_dash_state)
        return None
    ctx.intent_router.register(lambda _l, _u, c: True, handle_agentic)




# --- FILE: core/controller/intent_router.py ---

# internal import removed: from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

@dataclass
class IntentRoute:
    condition: Callable[[str, str, Any], bool]
    handler: Callable[[str, str, Any], Awaitable[str | None]]

class IntentRouter:
    def __init__(self):
        self._routes: list[IntentRoute] = []

    def register(
        self,
        condition: Callable[[str, str, Any], bool],
        handler: Callable[[str, str, Any], Awaitable[str | None]],
    ) -> None:
        self._routes.append(IntentRoute(condition, handler))

    async def route(self, lowered: str, user_input: str, context: Any) -> str | None:
        for route in self._routes:
            if route.condition(lowered, user_input, context):
                result = await route.handler(lowered, user_input, context)
                if result is not None:
                    return result
        return None




# --- FILE: core/controller/intents.py ---

# internal import removed: from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_GOAL_CREATE_KEYWORDS = (
    "remind me",
    "set goal",
    "schedule",
    "don't forget",
    "remember to",
)
_GOAL_STRIP_KEYWORDS = (
    "remind me to",
    "set goal",
    "schedule",
    "don't forget to",
    "remember to",
)
_GOAL_LIST_KEYWORDS = ("what are my goals", "show goals", "list goals", "my goals")


@dataclass(frozen=True)
class GoalIntentResult:
    response: str
    mutated: bool = False


def parse_time_delay_with_parsedatetime(text: str) -> float:
    try:
        import parsedatetime
        import time
        from datetime import datetime
        
        cal = parsedatetime.Calendar()
        now_dt = datetime.now()
        time_struct, parse_status = cal.parse(text, now_dt.timetuple())
        if parse_status > 0:
            target_epoch = time.mktime(time_struct)
            now_epoch = time.time()
            delay = target_epoch - now_epoch
            if delay < -60:
                return 0.0
            return max(0.0, delay)
    except Exception as exc:
        logger.warning("Failed parsing relative time with parsedatetime: %s", exc)
    return 0.0


def handle_goal_intent(
    text: str,
    user_input: str,
    *,
    goal_manager: Any,
    scheduler: Any,
) -> GoalIntentResult | None:
    if any(keyword in text for keyword in _GOAL_CREATE_KEYWORDS):
        description = user_input
        for keyword in _GOAL_STRIP_KEYWORDS:
            description = re.sub(
                re.escape(keyword),
                "",
                description,
                flags=re.IGNORECASE,
            ).strip()
        description = description.strip(" .?!")
        if description:
            # Predict priority and delay using parsedatetime and keyword fallback
            pred_priority = 5
            if any(w in user_input.lower() for w in ("urgent", "asap", "high priority", "important")):
                pred_priority = 9

            pdt_delay = parse_time_delay_with_parsedatetime(user_input)
            regex_delay = extract_goal_delay_seconds(user_input)
            delay_seconds = pdt_delay if pdt_delay > 0.0 else regex_delay

            goal_id = goal_manager.create_goal(
                description=description,
                priority=pred_priority,
            )
            try:
                goal_manager.start_goal(goal_id)
            except (ValueError, KeyError) as exc:
                logger.warning("Failed to start goal %r: %s", goal_id, exc)
            scheduler.enqueue(
                mission_id=goal_id,
                goal_id=goal_id,
                delay_seconds=delay_seconds,
                description=description,
            )
            
            # Nicely format response mentioning the predicted parameters
            if delay_seconds > 0.0:
                response = f"Goal set: {description} (priority: {pred_priority}, scheduled in {int(delay_seconds)}s)"
            else:
                response = f"Goal set: {description} (priority: {pred_priority}, scheduled immediately)"
            
            return GoalIntentResult(
                response=response,
                mutated=True,
            )

    if any(keyword in text for keyword in _GOAL_LIST_KEYWORDS):
        goals = goal_manager.active_goals()
        if not goals:
            return GoalIntentResult(response="No active goals.")
        lines = [f"- [{goal.priority}] {goal.description}" for goal in goals]
        return GoalIntentResult(response="Active goals:\n" + "\n".join(lines))

    return None


async def handle_preference_intent(
    text: str,
    user_input: str,
    *,
    memory: Any,
) -> str | None:
    if text.startswith("remember i like "):
        value = user_input[16:].strip()
        if value:
            await memory.store_preference(f"likes_{value[:12]}", value)
            return f"I will remember you like {value}."

    if text.startswith("my name is "):
        value = user_input[11:].strip()
        if value:
            await memory.store_preference("name", value)
            return f"I will remember your name is {value}."

    if text.startswith("i prefer "):
        value = user_input[9:].strip()
        if value:
            await memory.store_preference(f"prefer_{value[:12]}", value)
            return f"I will remember you prefer {value}."

    if text.startswith("i work in "):
        value = user_input[10:].strip()
        if value:
            await memory.store_preference("work", value)
            return f"I will remember you work in {value}."

    return None


def extract_goal_delay_seconds(user_input: str) -> float:
    lowered = user_input.lower()
    if "tomorrow" in lowered:
        return 24 * 60 * 60
    match = re.search(
        r"\bin\s+(\d+)\s+(minute|minutes|hour|hours|day|days)\b",
        lowered,
    )
    if not match:
        return 0.0
    value = int(match.group(1))
    unit = match.group(2)
    if unit.startswith("minute"):
        return float(value * 60)
    if unit.startswith("hour"):
        return float(value * 60 * 60)
    if unit.startswith("day"):
        return float(value * 24 * 60 * 60)
    return 0.0


__all__ = [
    "GoalIntentResult",
    "extract_goal_delay_seconds",
    "handle_goal_intent",
    "handle_preference_intent",
]




# --- FILE: core/controller/llm_dispatcher.py ---

"""Handles routing the input to the appropriate LLM task type and generating the response."""

# internal import removed: from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class LLMDispatcher:
    """Routes classified requests to the appropriate LLM model via the adaptive router."""

    def __init__(self, llm, model_router, memory, profile):
        self.llm = llm
        self.model_router = model_router
        self.memory = memory
        self.profile = profile

    async def dispatch(self, text: str, classification: dict[str, Any], session_id: str, trace_id: str) -> str:
        complexity = classification.get("complexity", 0.5)

        profile_summary = ""

        # Selective context injection
        if complexity > 0.2:
            await self.memory.build_context_block(text)
            profile_summary = (
                self.profile.get_communication_style() if self.profile else ""
            )

        # Map class route to task_type
        task_type = classification.get("route", "chat")
        if task_type == "direct":
            task_type = "reflex"
        elif task_type == "premium":
            task_type = "deep_reasoning"
        elif task_type == "planner":
            task_type = "planning"
        elif task_type == "mid-tier":
            task_type = "chat"

        selected_model = self.model_router.get_best_available(task_type)
        self.llm.model = selected_model

        messages = [{"role": "user", "content": text}]

        logger.info("Dispatching to LLM: %r (task_type=%s)", selected_model, task_type, extra={"trace_id": trace_id})

        # Pass full classification through to LLMClientV2 for adaptive routing
        response = await self.llm.chat_async(
            messages,
            query_for_memory=text if complexity > 0.2 else "",
            profile_summary=profile_summary,
            trace_id=trace_id,
        )

        if not response or response == "LLM Offline.":
            from core.controller.request_rules import is_preference_relevant
            prefs = await self.memory.recall_preferences(text, top_k=5)
            for pref in prefs:
                val = pref.get("value")
                key = pref.get("key")
                if val and key and is_preference_relevant(key, text):
                    return f"Offline fallback from memory: {val}"
            return "I don't know while offline."

        await self.memory.store_conversation(text, response, session_id)
        return str(response)




# --- FILE: core/controller/llm_orchestrator.py ---

import asyncio
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

class LLMOrchestrator:
    def __init__(self, llm_dispatcher: Any) -> None:
        self.llm_dispatcher = llm_dispatcher
        self._inflight_llm_calls = 0

    async def startup(self) -> None:
        pass

    async def shutdown(self) -> None:
        # Wait for inflight calls to complete
        while self._inflight_llm_calls > 0:
            await asyncio.sleep(0.1)

    async def dispatch(self, text: str, classification: Dict[str, Any], session_id: str, trace_id: str) -> str:
        self._inflight_llm_calls += 1
        try:
            return str(await self.llm_dispatcher.dispatch(text, classification, session_id, trace_id))
        finally:
            self._inflight_llm_calls -= 1




# --- FILE: core/controller/memory_subsystem.py ---

import asyncio
import logging
from typing import Any, List

logger = logging.getLogger(__name__)

class MemorySubsystem:
    def __init__(self, memory: Any, profile: Any, synthesizer: Any, config: Any) -> None:
        self.memory = memory
        self.profile = profile
        self.synthesizer = synthesizer
        self.config = config
        
        self._synthesis_tasks: set[asyncio.Task] = set()
        self._conversation_buffer: List[str] = []
        self._runtime_loop: asyncio.AbstractEventLoop | None = None

    async def startup(self) -> None:
        self._runtime_loop = asyncio.get_running_loop()
        index_path = ""
        if hasattr(self.config, "get") and callable(self.config.get):
            # Assuming config is a configparser.ConfigParser
            try:
                index_path = self.config.get("memory", "index_path", fallback="")
            except Exception:
                pass

        self.memory_status = await self.memory.initialize(index_path=index_path)

    async def shutdown(self) -> None:
        # Cancel and wait for synthesis tasks
        for task in list(self._synthesis_tasks):
            task.cancel()
        if self._synthesis_tasks:
            try:
                await asyncio.gather(*self._synthesis_tasks, return_exceptions=True)
            except Exception as e:
                logger.warning("Error gathering synthesis tasks during shutdown: %s", e, exc_info=True)

        if self.memory is not None:
            try:
                await self.memory.close()
            except Exception:
                logger.warning("Memory cleanup error during shutdown", exc_info=True)

    def update_profile(self, user_input: str, response: str) -> None:
        try:
            self.profile.update_from_conversation(user_input, response)
            self._conversation_buffer.append(f"User: {user_input}\nJarvis: {response}")

            if self.synthesizer.should_run(self.profile):
                self._schedule_synthesis(self._conversation_buffer[-20:])
                self._conversation_buffer.clear()
            elif len(self._conversation_buffer) > 50:
                self._conversation_buffer = self._conversation_buffer[-50:]
        except Exception as exc:
            logger.warning("Profile update/synthesis scheduling failed: %s", exc, exc_info=True)

    def _schedule_synthesis(self, conversations: List[str]) -> None:
        coro = self.synthesizer.synthesize(conversations, self.profile)
        
        def _track(task: asyncio.Task) -> None:
            self._synthesis_tasks.add(task)
            task.add_done_callback(self._synthesis_tasks.discard)

        try:
            task = asyncio.create_task(coro)
            _track(task)
            return
        except RuntimeError:
            pass

        if self._runtime_loop is not None and self._runtime_loop.is_running():
            def _create_and_track() -> None:
                t = asyncio.create_task(coro)
                _track(t)
            self._runtime_loop.call_soon_threadsafe(_create_and_track)
            return

        coro.close()
        logger.warning("No running loop available; skipped profile synthesis task.")




# --- FILE: core/proactive/background_monitor.py ---

import asyncio
import logging

logger = logging.getLogger(__name__)


class BackgroundMonitor:
    def __init__(self, notifier, config=None):
        self.notifier = notifier
        self.cpu_threshold = 90
        self.ram_threshold = 90
        if config and config.has_section("proactive"):
            self.cpu_threshold = config.getint("proactive", "cpu_alert_threshold", fallback=90)
            self.ram_threshold = config.getint("proactive", "ram_alert_threshold", fallback=90)
        self._tasks: list = []
        self._running = False

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._tasks.append(asyncio.create_task(self._monitor_resources()))

    async def stop(self) -> None:
        self._running = False
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    async def _monitor_resources(self) -> None:
        while self._running:
            await asyncio.sleep(60)
            try:
                import psutil

                cpu = await asyncio.to_thread(psutil.cpu_percent, interval=1)
                ram_mem = await asyncio.to_thread(psutil.virtual_memory)
                ram = ram_mem.percent
                if cpu > self.cpu_threshold:
                    self.notifier.notify(f"\u26a0\ufe0f CPU at {cpu:.0f}%", level="warn")
                if ram > self.ram_threshold:
                    self.notifier.notify(f"\u26a0\ufe0f RAM at {ram:.0f}%", level="warn")
            except ImportError:
                pass
            except Exception as exc:
                logger.warning("Resource monitor error: %s", exc, exc_info=True)




# --- FILE: core/proactive/notifier.py ---

import logging
import time

logger = logging.getLogger(__name__)


class NotificationManager:
    def notify(self, message: str, level: str = "info", voice_layer=None) -> None:
        ts = time.strftime("%H:%M")
        print(f"\n[{ts}][JARVIS/{level.upper()}] {message}")
        try:
            from plyer import notification

            notification.notify(title="Jarvis", message=message[:256], timeout=5)
        except Exception:
            pass  # plyer optional

        if voice_layer is not None:
            try:
                import asyncio
                import inspect

                loop = asyncio.get_event_loop()
                if loop.is_running():
                    res = voice_layer.speak(message)
                    if inspect.iscoroutine(res):
                        loop.create_task(res)
            except Exception:
                pass

    def schedule_reminder(self, message: str, in_seconds: int) -> None:
        import asyncio

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._delayed_notify(message, in_seconds))
        except RuntimeError:
            pass  # no running loop - skip

    async def _delayed_notify(self, message: str, delay: int) -> None:
        import asyncio

        await asyncio.sleep(delay)
        self.notify(message)




# --- FILE: core/runtime/event_bus.py ---

"""Lightweight pub/sub event bus for decoupled component communications."""

# internal import removed: from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Union

logger = logging.getLogger("Jarvis.EventBus")

EventCallback = Union[Callable[[Any], None], Callable[[Any], Awaitable[None]]]


@dataclass(frozen=True)
class EventRecord:
    """Replayable event envelope stored by the local event bus."""

    event_id: str
    event_type: str
    payload: Any
    source: str = "runtime"
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "payload": self.payload,
            "source": self.source,
            "created_at": self.created_at,
        }


class EventBus:
    """
    Publish/Subscribe Event Bus allowing loose coupling between modules.
    """

    def __init__(self, *, history_limit: int = 500) -> None:
        self._listeners: dict[str, list[EventCallback]] = {}
        self._history: deque[EventRecord] = deque(maxlen=max(0, int(history_limit)))
        self._lock = threading.RLock()
        self._main_loop: asyncio.AbstractEventLoop | None = None
        try:
            self._main_loop = asyncio.get_running_loop()
        except RuntimeError:
            pass

    def _try_capture_loop(self) -> None:
        if self._main_loop is None or self._main_loop.is_closed():
            try:
                self._main_loop = asyncio.get_running_loop()
            except RuntimeError:
                pass

    def subscribe(
        self,
        event_type: str,
        callback: EventCallback,
        *,
        replay_history: bool = False,
    ) -> None:
        """Register a callback for a specific event type."""
        event_key = event_type.strip().lower()
        self._try_capture_loop()
        with self._lock:
            if event_key not in self._listeners:
                self._listeners[event_key] = []
            if callback not in self._listeners[event_key]:
                self._listeners[event_key].append(callback)
                logger.debug("Subscribed callback to event '%s'", event_key)
            records_to_replay = self.replay(event_key) if replay_history else []

        for record in records_to_replay:
            self._dispatch_callback(callback, record.payload, event_key)

    def unsubscribe(self, event_type: str, callback: EventCallback) -> None:
        """Unregister a callback for a specific event type."""
        event_key = event_type.strip().lower()
        with self._lock:
            if event_key in self._listeners:
                try:
                    self._listeners[event_key].remove(callback)
                    logger.debug("Unsubscribed callback from event '%s'", event_key)
                except ValueError:
                    pass

    def publish(self, event_type: str, data: Any, *, source: str = "runtime") -> EventRecord:
        """
        Publish an event to all registered subscribers.
        Dispatches both synchronous and asynchronous callbacks safely.
        """
        self._try_capture_loop()
        with self._lock:
            record = self._record(event_type, data, source=source)
            callbacks = self._callbacks_for(record.event_type)

        for callback in callbacks:
            self._dispatch_callback(callback, data, record.event_type)
        return record

    async def publish_async(self, event_type: str, data: Any, *, source: str = "runtime") -> EventRecord:
        """
        Asynchronously publish an event to all registered subscribers.
        Awaits any coroutine callbacks.
        """
        self._try_capture_loop()
        with self._lock:
            record = self._record(event_type, data, source=source)
            callbacks = self._callbacks_for(record.event_type)

        tasks = []
        for callback in callbacks:
            try:
                res = callback(data)
                if asyncio.iscoroutine(res):
                    tasks.append(res)
            except Exception as e:
                logger.error("Error in async subscriber callback for event '%s': %s", record.event_type, e, exc_info=True)

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.error("Error in async subscriber execution for event '%s': %s", record.event_type, result, exc_info=True)
        return record

    def replay(self, event_type: str | None = None, *, limit: int | None = None) -> list[EventRecord]:
        """Return recent events, optionally filtered by type."""
        event_key = event_type.strip().lower() if event_type else None
        with self._lock:
            items = [
                record
                for record in self._history
                if event_key in (None, "*") or record.event_type == event_key
            ]
            if limit is not None:
                return items[-max(0, int(limit)) :]
            return items

    def clear_history(self) -> None:
        with self._lock:
            self._history.clear()

    def _record(self, event_type: str, payload: Any, *, source: str) -> EventRecord:
        event_key = event_type.strip().lower()
        record = EventRecord(
            event_id=uuid.uuid4().hex,
            event_type=event_key,
            payload=payload,
            source=source,
        )
        if self._history.maxlen != 0:
            self._history.append(record)
        return record

    def _callbacks_for(self, event_key: str) -> list[EventCallback]:
        callbacks: list[EventCallback] = []
        callbacks.extend(self._listeners.get(event_key, []))
        callbacks.extend(self._listeners.get("*", []))
        return list(callbacks)

    def _dispatch_callback(self, callback: EventCallback, data: Any, event_key: str) -> None:
        try:
            res = callback(data)
            if asyncio.iscoroutine(res):
                self._try_capture_loop()
                
                try:
                    current_loop = asyncio.get_running_loop()
                    if current_loop is self._main_loop and not current_loop.is_closed():
                        current_loop.create_task(res)
                        return
                except RuntimeError:
                    pass

                if self._main_loop is not None and not self._main_loop.is_closed():
                    try:
                        asyncio.run_coroutine_threadsafe(res, self._main_loop)
                    except RuntimeError as e:
                        logger.error("Failed to run coroutine thread-safely on captured loop: %s", e)
                        res.close()
                else:
                    logger.error("No running/open event loop available to schedule async callback for event '%s'", event_key, exc_info=True)
                    res.close()
        except Exception as e:
            logger.error("Error in subscriber callback for event '%s': %s", event_key, e, exc_info=True)


__all__ = ["EventBus", "EventCallback", "EventRecord"]




# --- FILE: core/synthesis.py ---

"""Profile synthesis helpers."""

# internal import removed: from __future__ import annotations

import inspect
import json
import re
from typing import Any


def _strip_wrappers(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
    cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", cleaned)
    cleaned = re.sub(r"\n?```$", "", cleaned)
    return cleaned.strip()


class ProfileSynthesizer:
    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def should_run(self, profile: Any) -> bool:
        try:
            count = int(getattr(profile, "interaction_count", 0))
            return count >= 20 and count % 20 == 0
        except Exception:
            return False

    async def synthesize(
        self,
        conversations: list[str],
        profile: Any,
    ) -> dict[str, Any]:
        prompt = (
            "Update the user profile from the conversation snippets below.\n"
            "Return strict JSON only, where each key maps to "
            '{"value": ..., "confidence": 0.0-1.0}.\n\n'
            + "\n".join(conversations[-20:])
        )

        complete = getattr(self.llm, "complete", None)
        if complete is None:
            return {"error": "llm_unavailable", "updated_fields": []}

        try:
            raw = complete(prompt)
            if inspect.isawaitable(raw):
                raw = await raw
            payload = json.loads(_strip_wrappers(str(raw)))
        except json.JSONDecodeError:
            return {"error": "invalid_json", "updated_fields": []}
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc), "updated_fields": []}

        if not isinstance(payload, dict):
            return {"error": "invalid_payload", "updated_fields": []}

        try:
            updated_fields = list(profile.apply_delta(payload))
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc), "updated_fields": []}

        return {"updated_fields": updated_fields, "delta": payload}


__all__ = ["ProfileSynthesizer"]




# --- FILE: core/controller/services.py ---

# internal import removed: from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# internal import removed: from core.agent.agent_loop import AgentLoopEngine
# internal import removed: from core.state_machine import StateMachine
# internal import removed: from core.autonomy.scheduler import Scheduler
# internal import removed: from core.autonomy.autonomy_governor import AutonomyGovernor
# internal import removed: from core.autonomy.goal_manager import GoalManager
# internal import removed: from core.autonomy.risk_evaluator import RiskEvaluator
# internal import removed: from core.desktop.actions import DesktopActionExecutor
# internal import removed: from core.desktop.observation import DesktopObserver
# internal import removed: from core.llm.client import LLMClientV2
# internal import removed: from core.llm.model_router import ModelRouter
# internal import removed: from core.llm.telemetry import RoutingTelemetry
# internal import removed: from core.planner.planner import TaskPlanner
# internal import removed: from core.llm.defaults import core_llm_defaults_DEFAULT_MODEL
# internal import removed: from core.memory.hybrid_memory import HybridMemory
# internal import removed: from core.profile import UserProfileEngine
# internal import removed: from core.proactive.background_monitor import BackgroundMonitor
# internal import removed: from core.proactive.notifier import NotificationManager
# internal import removed: from core.synthesis import ProfileSynthesizer
# internal import removed: from core.runtime.paths import _resolve_path
# internal import removed: from core.tools.builtin_tools import register_all_tools
# internal import removed: from core.registry.registry import CapabilityRegistry
# internal import removed: from core.runtime.event_bus import EventBus


@dataclass(frozen=True)
class ControllerSettings:
    db_path: str
    chroma_path: str
    model_name: str
    embedding_model: str
    base_url: str
    enable_context_titles: bool
    goal_check_interval_seconds: int
    goals_file: Path


@dataclass(frozen=True)
class ControllerServices:
    memory: HybridMemory
    model_router: ModelRouter
    profile: UserProfileEngine
    llm: LLMClientV2
    synthesizer: ProfileSynthesizer
    state_machine: StateMachine
    task_planner: TaskPlanner
    tool_router: CapabilityRegistry
    risk_evaluator: RiskEvaluator
    autonomy_governor: AutonomyGovernor
    agent_loop: AgentLoopEngine
    goal_manager: GoalManager
    scheduler: Scheduler
    notifier: NotificationManager
    monitor: BackgroundMonitor
    desktop_executor: DesktopActionExecutor = None  # type: ignore[assignment]
    desktop_observer: DesktopObserver = None  # type: ignore[assignment]
    desktop_bridge: Any = None
    event_bus: EventBus = None  # type: ignore[assignment]
    container: Any = None


def build_controller_services(
    config: configparser.ConfigParser,
    *,
    container: Any = None,
    db_path: str = "memory/memory.db",
    chroma_path: str = "data/chroma",
    model_name: str = core_llm_defaults_DEFAULT_MODEL,
    embedding_model: str = "all-MiniLM-L6-v2",
    memory_cls=HybridMemory,
    model_router_cls=ModelRouter,
    profile_cls=UserProfileEngine,
    llm_cls=LLMClientV2,
    synthesizer_cls=ProfileSynthesizer,
    state_machine_cls=StateMachine,
    task_planner_cls=TaskPlanner,
    tool_router_cls=CapabilityRegistry,
    risk_evaluator_cls=RiskEvaluator,
    autonomy_governor_cls=AutonomyGovernor,
    agent_loop_cls=AgentLoopEngine,
    goal_manager_cls=GoalManager,
    scheduler_cls=Scheduler,
    notifier_cls=NotificationManager,
    monitor_cls=BackgroundMonitor,
    register_tools=register_all_tools,
) -> tuple[ControllerSettings, ControllerServices]:
    from core.config import JarvisConfig
    if not isinstance(config, JarvisConfig):
        jc = JarvisConfig()
        jc.read_dict(config)
        config = jc

    resolved_db_path = str(
        _resolve_path(
            config.get_str(
                "memory",
                "db_path",
                fallback=config.get_str("memory", "sqlite_file", fallback=db_path),
            )
        )
    )
    resolved_chroma_path = str(
        _resolve_path(
            config.get_str(
                "memory",
                "chroma_path",
                fallback=config.get_str("memory", "chroma_dir", fallback=chroma_path),
            )
        )
    )
    resolved_model_name = config.get_str(
        "models",
        "chat_model",
        fallback=config.get_str(
            "llm",
            "model",
            fallback=config.get_str("ollama", "planner_model", fallback=model_name),
        ),
    )
    resolved_embedding_model = config.get_str(
        "memory",
        "embedding_model",
        fallback=embedding_model,
    )
    base_url = config.get_str("ollama", "base_url", fallback="http://localhost:11434")
    enable_context_titles = config.get_bool(
        "memory",
        "llm_context_titles",
        fallback=True,
    )
    goal_check_interval_seconds = max(
        1,
        config.get_int("proactive", "goal_check_interval_minutes", fallback=5),
    ) * 60
    goals_file = _resolve_path(
        config.get_str("memory", "goals_file", fallback="memory/goals.json")
    )

    settings = ControllerSettings(
        db_path=resolved_db_path,
        chroma_path=resolved_chroma_path,
        model_name=resolved_model_name,
        embedding_model=resolved_embedding_model,
        base_url=base_url,
        enable_context_titles=enable_context_titles,
        goal_check_interval_seconds=goal_check_interval_seconds,
        goals_file=goals_file,
    )

    if container is None:
        from core.runtime.container import ServiceContainer
        container = ServiceContainer()

    # 0. Register EventBus
    if not container.has("event_bus"):
        container.register("event_bus", lambda: EventBus())

    # 1. Register Memory
    if not container.has("memory"):
        container.register(
            "memory",
            lambda: memory_cls(
                settings.db_path,
                chroma_path=settings.chroma_path,
                model_name=settings.embedding_model,
            )
        )

    # 2. Register Model Router
    if not container.has("model_router"):
        container.register("model_router", lambda: model_router_cls(config=config))

    # 3. Register User Profile
    if not container.has("profile"):
        container.register("profile", lambda: profile_cls())

    # 4. Register LLM Client
    if not container.has("llm"):
        container.register(
            "llm",
            lambda: llm_cls(
                hybrid_memory=container.resolve("memory"),
                model=container.resolve("model_router").route("chat"),
                profile=container.resolve("profile"),
                base_url=settings.base_url,
            )
        )

    # 5. Register Profile Synthesizer
    if not container.has("synthesizer"):
        container.register("synthesizer", lambda: synthesizer_cls(container.resolve("llm")))

    # 6. Register State Machine
    if not container.has("state_machine"):
        def make_state_machine():
            import inspect
            sig = inspect.signature(state_machine_cls)
            if "event_bus" in sig.parameters:
                return state_machine_cls(event_bus=container.resolve("event_bus"))
            else:
                inst = state_machine_cls()
                try:
                    inst.event_bus = container.resolve("event_bus")
                except AttributeError:
                    pass
                return inst
        container.register("state_machine", make_state_machine)

    # 7. Register Task Planner
    if not container.has("task_planner"):
        def make_planner():
            try:
                return task_planner_cls(config, llm=container.resolve("llm"), registry=container.resolve("tool_router"))
            except TypeError:
                return task_planner_cls(config)
        container.register("task_planner", make_planner)

    # 8. Register Tool Router
    if not container.has("tool_router"):
        container.register("tool_router", lambda: tool_router_cls())

    # 9. Register Risk Evaluator
    if not container.has("risk_evaluator"):
        container.register("risk_evaluator", lambda: risk_evaluator_cls(config))

    # 10. Register Autonomy Governor
    if not container.has("autonomy_governor"):
        container.register("autonomy_governor", lambda: autonomy_governor_cls(level=config.get_int("autonomy", "level", fallback=3)))

    # 11. Register Desktop Executor
    if not container.has("desktop_executor"):
        container.register("desktop_executor", lambda: DesktopActionExecutor(risk_evaluator=container.resolve("risk_evaluator")))

    # 12. Register Desktop Observer
    if not container.has("desktop_observer"):
        container.register("desktop_observer", lambda: DesktopObserver())

    # 13. Register Desktop Bridge
    if not container.has("desktop_bridge"):
        container.register(
            "desktop_bridge",
            lambda: None
        )

    # 13a. Register TaskExecutionContext
    if not container.has("task_execution_context"):
        from core.context.context import TaskExecutionContext
        container.register("task_execution_context", TaskExecutionContext, is_singleton=False)

    # 13b. Register DAGExecutor
    if not container.has("dag_executor"):
        from core.executor.engine import DAGExecutor
        container.register("dag_executor", lambda: DAGExecutor(tool_router=container.resolve("tool_router")), is_singleton=False)

    # 14. Register Agent Loop Engine
    if not container.has("agent_loop"):
        container.register(
            "agent_loop",
            lambda: agent_loop_cls(
                model=settings.model_name,
                ollama_url=settings.base_url,
                container=container,
            )
        )

    # 15. Register Goal Manager
    if not container.has("goal_manager"):
        container.register("goal_manager", lambda: goal_manager_cls())

    # 16. Register Scheduler
    if not container.has("scheduler"):
        container.register("scheduler", lambda: scheduler_cls())

    # 17. Register Notifier
    if not container.has("notifier"):
        container.register("notifier", lambda: notifier_cls())

    # 18. Register Background Monitor
    if not container.has("monitor"):
        container.register("monitor", lambda: monitor_cls(container.resolve("notifier"), config))

    # Resolve instances and wire them up
    memory = container.resolve(
        "memory",
        db_path=settings.db_path,
        chroma_path=settings.chroma_path,
        model_name=settings.embedding_model,
    )
    model_router = container.resolve("model_router", config=config)
    try:
        model_router.refresh_available_models(base_url=settings.base_url)
    except Exception as e:
        import logging
        logging.getLogger("Jarvis.Services").warning("Failed to refresh models: %s", e, exc_info=True)

    profile = container.resolve("profile")
    llm = container.resolve(
        "llm",
        hybrid_memory=memory,
        model=model_router.route("chat"),
        profile=profile,
        base_url=settings.base_url,
    )
    llm.set_router(model_router)

    # Wire telemetry if enabled
    telemetry_enabled = True
    try:
        telemetry_enabled = str(
            config.get("routing", "telemetry_enabled", fallback="true")
        ).lower() in ("true", "1", "yes")
    except Exception:
        pass

    if telemetry_enabled:
        if not container.has("telemetry"):
            container.register("telemetry", lambda: RoutingTelemetry())
        telemetry = container.resolve("telemetry")
        model_router.set_telemetry(telemetry)
        llm.set_telemetry(telemetry)

    if hasattr(memory, "set_llm"):
        memory.set_llm(
            llm,
            enable_context_titles=settings.enable_context_titles,
        )
    try:
        setattr(llm, "profile", profile)
    except Exception as e:
        import logging
        logging.getLogger("Jarvis.Services").warning("Failed to set profile on LLM: %s", e, exc_info=True)

    synthesizer = container.resolve("synthesizer", llm=llm)
    state_machine = container.resolve("state_machine")
    task_planner = container.resolve("task_planner", config=config, llm=llm)
    tool_router = container.resolve("tool_router")

    register_tools(
        tool_router,
        llm=llm,
        config=config,
    )

    # Dynamic plugin tool loading
    try:
        plugins_dir = config.get("plugins", "directory", fallback="core/plugins")
        resolved_plugins_dir = _resolve_path(plugins_dir)
        if hasattr(tool_router, "load_plugins"):
            loaded_plugins = tool_router.load_plugins(resolved_plugins_dir)
            if loaded_plugins:
                from core.tools.builtin_tools import logger as tools_logger
                tools_logger.info("Loaded dynamic plugins: %s", loaded_plugins)
    except Exception as e:
        from core.tools.builtin_tools import logger as tools_logger
        tools_logger.warning("Failed to load dynamic plugins: %s", e, exc_info=True)

    risk_evaluator = container.resolve("risk_evaluator", config=config)
    risk_evaluator.registry = tool_router
    autonomy_level = config.get_int("autonomy", "level", fallback=3)
    autonomy_governor = container.resolve("autonomy_governor", level=autonomy_level)
    autonomy_governor.registry = tool_router
    desktop_executor = container.resolve("desktop_executor", risk_evaluator=risk_evaluator)
    desktop_observer = container.resolve("desktop_observer")
    desktop_bridge = container.resolve(
        "desktop_bridge",
        container=container,
    )
    agent_loop = container.resolve(
        "agent_loop",
        model=settings.model_name,
        ollama_url=settings.base_url,
        container=container,
    )
    goal_manager = container.resolve("goal_manager")
    scheduler = container.resolve("scheduler")
    notifier = container.resolve("notifier")
    monitor = container.resolve("monitor", notifier=notifier, config=config)
    event_bus = container.resolve("event_bus")

    return settings, ControllerServices(
        memory=memory,
        model_router=model_router,
        profile=profile,
        llm=llm,
        synthesizer=synthesizer,
        state_machine=state_machine,
        task_planner=task_planner,
        tool_router=tool_router,
        risk_evaluator=risk_evaluator,
        autonomy_governor=autonomy_governor,
        agent_loop=agent_loop,
        goal_manager=goal_manager,
        scheduler=scheduler,
        notifier=notifier,
        monitor=monitor,
        desktop_executor=desktop_executor,
        desktop_observer=desktop_observer,
        desktop_bridge=desktop_bridge,
        event_bus=event_bus,
        container=container,
    )


__all__ = ["ControllerServices", "ControllerSettings", "build_controller_services"]




# --- FILE: core/controller_v2.py ---

"""JarvisControllerV2: memory + LLM orchestration with CLI/voice runtime modes."""

# internal import removed: from __future__ import annotations

import asyncio
import configparser
import logging
import uuid
from typing import Any

# internal import removed: from core.base_controller import BaseController
# internal import removed: from core.controller.intents import (
# internal import removed:     handle_goal_intent,
# internal import removed:     handle_preference_intent,
# internal import removed: )
# internal import removed: from core.controller.intent_router import IntentRouter
# internal import removed: from core.controller.services import build_controller_services
# internal import removed: from core.llm.defaults import core_llm_defaults_DEFAULT_MODEL

# Import extracted facade components
# internal import removed: from core.controller.llm_dispatcher import LLMDispatcher
# internal import removed: from core.controller.goal_runner import GoalRunner

# Import extracted subsystems
# internal import removed: from core.controller.llm_orchestrator import LLMOrchestrator
# internal import removed: from core.controller.memory_subsystem import MemorySubsystem
# internal import removed: from core.controller.automation_manager import AutomationManager

logger = logging.getLogger(__name__)

class JarvisControllerV2(BaseController):
    def __init__(
        self,
        config: configparser.ConfigParser | None = None,
        voice: bool = False,
        db_path: str = "memory/memory.db",
        chroma_path: str = "data/chroma",
        model_name: str = core_llm_defaults_DEFAULT_MODEL,
        embedding_model: str = "all-MiniLM-L6-v2",
        container: Any = None,
        services: Any = None,
        settings: Any = None,
    ) -> None:
        super().__init__()
        
        self.config = (
            config
            if isinstance(config, configparser.ConfigParser)
            else configparser.ConfigParser()
        )
        for section in ["execution", "memory", "automation", "voice"]:
            if not self.config.has_section(section):
                self.config.add_section(section)
        self.voice_enabled = bool(voice)
        self.session_id = uuid.uuid4().hex[:8]
        self._gui_automation_enabled = self.config.getboolean(
            "execution",
            "allow_gui_automation",
            fallback=False,
        )
        self._app_launch_enabled = self.config.getboolean(
            "execution",
            "allow_app_launch",
            fallback=True,
        )

        if services is None or settings is None:
            settings, services = build_controller_services(
                self.config,
                container=container,
                db_path=db_path,
                chroma_path=chroma_path,
                model_name=model_name,
                embedding_model=embedding_model,
            )

        self.memory = services.memory
        self.model_router = services.model_router
        self.profile = services.profile
        self.llm = services.llm
        self.synthesizer = services.synthesizer
        self.state_machine = services.state_machine
        self.task_planner = services.task_planner
        self.tool_router = services.tool_router
        self.risk_evaluator = services.risk_evaluator
        self.autonomy_governor = services.autonomy_governor
        self.agent_loop = services.agent_loop
        self.goal_manager = services.goal_manager
        self.scheduler = services.scheduler
        self.notifier = services.notifier
        self.monitor = services.monitor
        self.desktop_executor = services.desktop_executor
        self.desktop_observer = services.desktop_observer
        self.desktop_bridge = services.desktop_bridge
        self.container = services.container

        self._runtime_loop: asyncio.AbstractEventLoop | None = None
        self._on_state_update = lambda **_: None
        self.exchanges = 0
        self._state_lock = asyncio.Lock()
        self.voice_loop = None

        self._voice_layer = None
        if voice:
            try:
                from core.voice.voice_layer import VoiceLayer

                self._voice_layer = VoiceLayer(
                    controller=self,
                    config=self.config,
                )
                logger.info("VoiceLayer initialized")
            except ImportError as exc:
                logger.warning("Voice unavailable: %s", exc, exc_info=True)
            except Exception as exc:  # noqa: BLE001
                logger.warning("VoiceLayer init failed: %s", exc, exc_info=True)

        # Initialize facades
        self.llm_dispatcher = LLMDispatcher(
            llm=self.llm,
            model_router=self.model_router,
            memory=self.memory,
            profile=self.profile
        )
        
        self.goal_runner = GoalRunner(
            goal_manager=self.goal_manager,
            scheduler=self.scheduler,
            notifier=self.notifier,
            voice_layer=self._voice_layer,
            goals_file=settings.goals_file,
            goal_check_interval_seconds=settings.goal_check_interval_seconds,
            dashboard_update_cb=self._dashboard_update
        )
        self._goal_check_task: asyncio.Task | None = None

        # Initialize Subsystems
        self.llm_orchestrator = LLMOrchestrator(self.llm_dispatcher)
        self.memory_subsystem = MemorySubsystem(self.memory, self.profile, self.synthesizer, self.config)
        
        async def cmd_handler(cmd: str) -> str:
            return await self.process(cmd)
            
        self.automation_manager = AutomationManager(
            config=self.config,
            memory=self.memory,
            llm=self.llm,
            notifier=self.notifier,
            desktop_observer=self.desktop_observer,
            container=self.container,
            command_handler=cmd_handler
        )

        self.register_subsystem(self.llm_orchestrator)
        self.register_subsystem(self.memory_subsystem)
        self.register_subsystem(self.automation_manager)

        self.intent_router = IntentRouter()
        self._setup_intent_routes()

    async def initialize(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "memory_mode": getattr(self.memory_subsystem, "memory_status", {}).get("mode"),
            "memory_init": getattr(self.memory_subsystem, "memory_status", {}),
        }

    async def _handle_goal_intent(self, text: str, user_input: str) -> str | None:
        result = handle_goal_intent(
            text,
            user_input,
            goal_manager=self.goal_manager,
            scheduler=self.scheduler,
        )
        if result is None:
            return None
        if result.mutated:
            await self.goal_runner.persist_goal_state()
            self._dashboard_update(active_goals=len(self.goal_manager.active_goals()))
        return result.response

    async def _handle_preference_intent(
        self,
        text: str,
        user_input: str,
    ) -> str | None:
        return await handle_preference_intent(
            text,
            user_input,
            memory=self.memory,
        )

    async def _dispatch_llm(self, text: str, classification: dict, trace_id: str) -> str:
        return await self.llm_orchestrator.dispatch(text, classification, self.session_id, trace_id)

    def _looks_like_desktop_control_request(self, lowered: str) -> bool:
        return any(k in lowered for k in ["click", "desktop", "mouse", "keyboard", "screen", "type"])

    def _desktop_control_disabled_message(self) -> str:
        return (
            "Desktop control is disabled. Set [execution] allow_gui_automation = true "
            "in config/jarvis.ini, install the desktop extras with "
            "'pip install -r requirements/desktop.txt', then restart Jarvis."
        )

    def _app_launch_disabled_message(self) -> str:
        return (
            "Application launch is disabled. Set [execution] allow_app_launch = true "
            "in config/jarvis.ini, then restart Jarvis."
        )

    def _setup_intent_routes(self) -> None:
        from core.controller.intent_handlers import register_intent_routes
        register_intent_routes(self)

    async def process(self, user_input: str, trace_id: str | None = None) -> str:
        if not trace_id:
            trace_id = uuid.uuid4().hex[:8]
        logger.info("Controller process started", extra={"trace_id": trace_id})

        self._dashboard_update(state="THINKING", last_input=user_input)
        
        async with self._state_lock:
            self.exchanges += 1
            text = (user_input or "").strip()
            if len(text) > 4000:
                text = text[:4000]
            
        lowered = text.lower()
        
        try:
            from core.controller.complexity_scorer import classify_request
            classification = classify_request(text)
        except ImportError:
            classification = {"class": "Chat", "complexity": 0.4, "skip_planner": False, "route": "chat"}

        async def _respond(response: str) -> str:
            async with self._state_lock:
                self.memory_subsystem.update_profile(user_input, response)

            self._dashboard_update(
                state="IDLE",
                last_response=response,
                active_goals=len(self.goal_manager.active_goals()),
            )
            return response

        routed_response = await self.intent_router.route(lowered, text, self)
        if routed_response is not None:
            return await _respond(routed_response)

        response = await self._dispatch_llm(text, classification, trace_id=trace_id)
        return await _respond(response)

    def session_summary(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "exchanges": self.exchanges,
        }

    async def startup(self) -> None:
        self._runtime_loop = asyncio.get_running_loop()
        await super().startup()
        await self.monitor.start()
        await self.goal_runner.load_goal_state()
        if self._goal_check_task is None or self._goal_check_task.done():
            self._goal_check_task = asyncio.create_task(
                self.goal_runner.check_due_goals(),
                name="jarvis_goal_due_checker",
            )

    async def start(self) -> None:
        """Alias for startup() to maintain backward compatibility."""
        await self.startup()

    async def run_cli(self) -> None:
        if self._voice_layer is not None:
            logger.info("Starting in voice mode...")
            self._dashboard_update(state="IDLE")
            await self._voice_layer.start()
            task = getattr(self._voice_layer, "_task", None)
            if task is not None:
                await task
            return

        print(f"Jarvis V2 ready (session {self.session_id}). Type 'exit' to quit.")

        loop = asyncio.get_running_loop()
        self._runtime_loop = loop

        while True:
            try:
                user_input = await loop.run_in_executor(None, input, "You: ")
            except EOFError:
                break

            text = user_input.strip()
            if not text:
                continue
            if text.lower() in {"exit", "quit"}:
                break

            print(f"DEBUG: Before process(text='{text}')", flush=True)
            try:
                response = await self.process(text)
                print(f"DEBUG: After process, response='{response}'", flush=True)
                print(f"Jarvis: {response}", flush=True)
            except Exception as e:
                print(f"Error processing command: {e}")

    async def shutdown(self) -> None:
        await self.monitor.stop()

        if self._goal_check_task is not None and not self._goal_check_task.done():
            self._goal_check_task.cancel()
            try:
                await self._goal_check_task
            except asyncio.CancelledError:
                pass

        await self.goal_runner.persist_goal_state()

        if self._voice_layer is not None:
            try:
                await self._voice_layer.stop()
            except Exception:
                logger.warning("VoiceLayer stop error", exc_info=True)

        if self.voice_loop is not None:
            try:
                await self.voice_loop.stop()
            except Exception:
                logger.warning("voice_loop stop error", exc_info=True)

        await super().shutdown()

    def _dashboard_update(self, **kwargs: Any) -> None:
        try:
            self._on_state_update(**kwargs)
        except Exception as e:
            logger.warning("Failed to update dashboard state: %s", e, exc_info=True)

Controller = JarvisControllerV2

__all__ = ["JarvisControllerV2", "Controller"]




# --- FILE: core/execution/__init__.py ---

# internal import removed: from .dispatcher import DispatchPipeline

__all__ = ["DispatchPipeline"]




# --- FILE: core/executor/dag.py ---

"""
core/executor/dag.py
────────────────────
Dependency parsing and topological sorting for DAG execution plans.
"""

# internal import removed: from __future__ import annotations

from typing import Any, Dict, List, Set


class DependencyGraphError(ValueError):
    """Raised when there is an issue with the dependency graph (e.g. cycle)."""


class PlanDAG:
    def __init__(self, steps: List[Dict[str, Any]]):
        self.steps = steps
        # Map step ID (normalized to string) -> step definition
        self.step_map: Dict[str, Dict[str, Any]] = {str(step.get("id", "")): step for step in steps if step.get("id")}
        self.adj_list: Dict[str, Set[str]] = {str(step_id): set() for step_id in self.step_map}
        self.in_degree: Dict[str, int] = {str(step_id): 0 for step_id in self.step_map}

        # Build graph: parent -> child edge. A parent must complete before child executes.
        for step_id, step in self.step_map.items():
            depends_on = step.get("depends_on", [])
            if isinstance(depends_on, str):
                depends_on = [depends_on]

            for dep in depends_on:
                dep_str = str(dep)
                # Ensure the dependency exists in our steps
                if dep_str in self.step_map:
                    self.adj_list[dep_str].add(step_id)
                    self.in_degree[step_id] += 1
                else:
                    # Ignore missing dependencies to prevent graph breakage
                    pass

    def topological_sort(self) -> List[str]:
        """Perform Kahn's algorithm for topological sorting and cycle detection."""
        in_deg = self.in_degree.copy()
        queue = [node for node, deg in in_deg.items() if deg == 0]
        sorted_nodes = []

        while queue:
            node = queue.pop(0)
            sorted_nodes.append(node)
            for neighbor in self.adj_list[node]:
                in_deg[neighbor] -= 1
                if in_deg[neighbor] == 0:
                    queue.append(neighbor)

        if len(sorted_nodes) != len(self.step_map):
            raise DependencyGraphError("Circular dependency detected in execution plan.")

        return sorted_nodes




# --- FILE: core/executor/engine.py ---

"""
core/executor/engine.py
───────────────────────
Asynchronous DAG execution engine with LIFO rollback, retry semantics, and timeouts.
"""

# internal import removed: from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any, Callable, Dict, Set

# internal import removed: from core.context.context import TaskExecutionContext
# internal import removed: from core.executor.dag import PlanDAG

logger = logging.getLogger("Jarvis.Executor.Engine")


class DAGExecutor:
    """Executes planned task steps concurrently conforming to dependency constraints."""

    def __init__(self, tool_router: Any, risk_evaluator: Any = None, autonomy_governor: Any = None):
        self.router = tool_router
        self.risk = risk_evaluator
        self.gov = autonomy_governor

    async def execute(self, plan: Dict[str, Any], context: TaskExecutionContext) -> Dict[str, Any]:
        rollbacks: Dict[str, Callable[[], Any]] = {}

        def register_rollback(step_id: str, step_def: Dict[str, Any]):
            rollback_def = step_def.get("rollback")
            if rollback_def:
                rb_action = rollback_def.get("action")
                rb_params = rollback_def.get("params", {})
                async def rollback_callback():
                    logger.info("Rolling back step %s via %s", step_id, rb_action, extra={"metadata": {"step_id": step_id, "action": rb_action}})
                    sig_rb = inspect.signature(self.router.execute)
                    if "context" in sig_rb.parameters:
                        await self.router.execute(rb_action, rb_params, context=context)
                    else:
                        await self.router.execute(rb_action, rb_params)
                rollbacks[step_id] = rollback_callback

        steps = plan.get("steps", [])
        if not steps:
            return {"status": "success", "message": "No steps to execute."}

        # Helper to publish events
        def publish_event(event_type: str, data: Any):
            eb = getattr(context, "event_bus", None)
            if eb:
                eb.publish(event_type, data)

        publish_event("task_started", {"trace_id": context.trace_id, "task_id": context.task_id, "plan": plan})

        dag = PlanDAG(steps)
        try:
            topo_order = dag.topological_sort()
        except Exception as e:
            logger.error("Topological sort failed: %s", e, extra={"metadata": {"error": str(e)}})
            publish_event("task_finished", {"trace_id": context.trace_id, "task_id": context.task_id, "status": "failed", "error": str(e)})
            return {"status": "failure", "error": str(e)}

        # Track execution states of steps
        step_states = {step_id: "pending" for step_id in dag.step_map}
        
        # Load or initialize step_results in context.variables for replay capability
        if "_step_results" not in context.variables:
            context.variables["_step_results"] = {}
        step_results = context.variables["_step_results"]
        
        completed_steps: Set[str] = set()
        
        # In replay mode, check if step is already completed
        replay_active = context.get("_replay_active", False)
        if replay_active:
            for sid, res in step_results.items():
                if isinstance(res, dict) and res.get("success", True) is not False:
                    step_states[sid] = "success"
                    completed_steps.add(sid)
                    logger.info("Replay: loaded step %s status as completed from snapshot", sid, extra={"metadata": {"step_id": sid}})
                    if sid in dag.step_map:
                        register_rollback(sid, dag.step_map[sid])

        cond = asyncio.Condition()
        # Build incoming dependency tracking dictionary
        dependencies: Dict[str, Set[str]] = {step_id: set() for step_id in dag.step_map}
        for step_id, step in dag.step_map.items():
            depends_on = step.get("depends_on", [])
            if isinstance(depends_on, str):
                depends_on = [depends_on]
            for dep in depends_on:
                dep_str = str(dep)
                if dep_str in dag.step_map:
                    dependencies[step_id].add(dep_str)

        async def run_step(step_id: str) -> Dict[str, Any]:
            step = dag.step_map[step_id]
            action = step.get("action") or step.get("tool")
            params = step.get("params", {})
            retry_count = int(step.get("retry_count", 0))

            # Replay mocking check
            if replay_active and step_id in step_results:
                prior_res = step_results[step_id]
                if isinstance(prior_res, dict) and prior_res.get("success", True) is not False:
                    logger.info("Replay: mocking execution of step %s with prior successful result", step_id, extra={"metadata": {"step_id": step_id}})
                    return prior_res

            if self.gov and action:
                allowed, reason = self.gov.can_execute(action)
                if not allowed:
                    raise RuntimeError(f"Autonomy Governor blocked '{action}': {reason}")

            backoff = 1.0
            for attempt in range(retry_count + 1):
                try:
                    logger.info(
                        "Step %s: executing '%s' (attempt %d)", step_id, action, attempt + 1,
                        extra={"metadata": {"step_id": step_id, "action": action, "attempt": attempt + 1}}
                    )
                    # Snapshot at start of step
                    await context.save_snapshot(step_id=step_id, metadata={"status": "running", "action": action, "attempt": attempt + 1})
                    publish_event("step_executing", {"trace_id": context.trace_id, "task_id": context.task_id, "step_id": step_id, "action": action})

                    sig = inspect.signature(self.router.execute)
                    if "context" in sig.parameters:
                        observation = await self.router.execute(action, params, context=context)
                    else:
                        observation = await self.router.execute(action, params)

                    if observation.execution_status == "success":
                        logger.info("Step %s succeeded.", step_id, extra={"metadata": {"step_id": step_id}})
                        res_dict: Dict[str, Any] = dict(observation.to_dict())

                        # Snapshot at end of step (success)
                        await context.save_snapshot(step_id=step_id, metadata={"status": "success", "action": action})
                        publish_event("step_completed", {"trace_id": context.trace_id, "task_id": context.task_id, "step_id": step_id, "status": "success", "result": res_dict})

                        # Set up step rollback callback if defined in step schema
                        register_rollback(step_id, step)

                        return res_dict
                    else:
                        raise RuntimeError(observation.error_message or "Tool execution failed")
                except Exception as exc:
                    if attempt < retry_count:
                        logger.warning(
                            "Step %s failed: %s. Retrying in %ss...", step_id, exc, backoff,
                            extra={"metadata": {"step_id": step_id, "error": str(exc), "retry_backoff": backoff}}
                        )
                        await asyncio.sleep(backoff)
                        backoff *= 2.0
                    else:
                        # Snapshot at end of step (failure)
                        await context.save_snapshot(step_id=step_id, metadata={"status": "failed", "action": action, "error": str(exc)})
                        publish_event("step_completed", {"trace_id": context.trace_id, "task_id": context.task_id, "step_id": step_id, "status": "failed", "error": str(exc)})
                        raise exc
            raise RuntimeError(f"Step {step_id} failed to execute.")

        running_tasks: set[asyncio.Task] = set()

        async def run_step_task(sid: str) -> None:
            try:
                res = await run_step(sid)
                async with cond:
                    step_states[sid] = "success"
                    step_results[sid] = res
                    completed_steps.add(sid)
            except asyncio.CancelledError:
                async with cond:
                    step_states[sid] = "cancelled"
                    step_results[sid] = {"success": False, "error": "Cancelled"}
                    logger.info("Step %s cancelled.", sid, extra={"metadata": {"step_id": sid}})
                raise
            except Exception as e:
                async with cond:
                    step_states[sid] = "failed"
                    step_results[sid] = {"success": False, "error": str(e)}
                    logger.error("Step %s permanently failed: %s", sid, e, extra={"metadata": {"step_id": sid, "error": str(e)}})
            except BaseException as e:
                async with cond:
                    step_states[sid] = "failed"
                    step_results[sid] = {"success": False, "error": f"BaseException: {type(e).__name__}"}
                    logger.error("Step %s aborted due to critical error: %s", sid, e, extra={"metadata": {"step_id": sid, "error": str(e)}})
                raise
            finally:
                async with cond:
                    cond.notify_all()

        async def scheduler_loop():
            while True:
                ready_to_run = []
                async with cond:
                    if len(completed_steps) == len(dag.step_map):
                        break

                    # Halt everything if any step failed
                    if any(state == "failed" for state in step_states.values()):
                        break

                    for step_id in dag.step_map:
                        if step_states[step_id] == "pending":
                            deps = dependencies[step_id]
                            if deps.issubset(completed_steps):
                                step_states[step_id] = "running"
                                ready_to_run.append(step_id)

                    if not ready_to_run:
                        if any(state == "running" for state in step_states.values()):
                            await cond.wait()
                            continue
                        else:
                            break

                for sid in ready_to_run:
                    task = asyncio.create_task(run_step_task(sid))
                    running_tasks.add(task)
                    task.add_done_callback(running_tasks.discard)

        try:
            await scheduler_loop()
        finally:
            if running_tasks:
                # Cancel remaining tasks before waiting to prevent infinite hangs on failure
                tasks_to_await = list(running_tasks)
                for t in tasks_to_await:
                    if not t.done():
                        t.cancel()
                await asyncio.gather(*tasks_to_await, return_exceptions=True)

        failed_steps = [sid for sid, state in step_states.items() if state == "failed"]
        if failed_steps:
            logger.error(
                "DAG execution halted at failed steps %s. Rolling back in reverse topological order...", failed_steps,
                extra={"metadata": {"failed_steps": failed_steps}}
            )
            for sid in reversed(topo_order):
                if sid in rollbacks:
                    try:
                        await rollbacks[sid]()
                    except Exception as e:
                        logger.error("Rollback callback failure for step %s: %s", sid, e, extra={"metadata": {"step_id": sid, "error": str(e)}})
            
            publish_event("task_finished", {"trace_id": context.trace_id, "task_id": context.task_id, "status": "failed", "failed_steps": failed_steps})
            return {"status": "failure", "failed_steps": failed_steps, "results": step_results}

        publish_event("task_finished", {"trace_id": context.trace_id, "task_id": context.task_id, "status": "success"})
        return {"status": "success", "results": step_results}




# --- FILE: core/execution/dispatcher.py ---

"""
core/execution/dispatcher.py
"""

import logging
from typing import Any, Dict

# internal import removed: from core.executor.engine import DAGExecutor

logger = logging.getLogger("Jarvis.Execution.Dispatcher")

class DispatchPipeline:
    """
    High-level wrapper around DAGExecutor.
    Enforces a hardcoded max_recursion_depth to prevent unbounded execution loops.
    """
    
    def __init__(self, executor: DAGExecutor):
        self.executor = executor
        self.max_recursion_depth = 5
        
    async def execute(self, plan: Dict[str, Any], context: Any, current_depth: int = 0) -> Dict[str, Any]:
        """
        Executes a plan via DAGExecutor, checking recursion depth.
        """
        if current_depth > self.max_recursion_depth:
            err_msg = f"DispatchPipeline max recursion depth of {self.max_recursion_depth} breached."
            logger.critical(err_msg)
            raise RecursionError(err_msg)
            
        return await self.executor.execute(plan, context)




# --- FILE: core/executor/__init__.py ---

"""DAG-based async executor."""




# --- FILE: core/introspection/__init__.py ---

"""Runtime health exports."""

# internal import removed: from .health import (
# internal import removed:     HealthCheck,
# internal import removed:     HealthReport,
# internal import removed:     HealthStatus,
# internal import removed:     run_lightweight_health_check,
# internal import removed:     run_startup_health_check,
# internal import removed: )

__all__ = [
    "HealthCheck",
    "HealthReport",
    "HealthStatus",
    "run_lightweight_health_check",
    "run_startup_health_check",
]




# --- FILE: core/ops/production.py ---

# internal import removed: from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


PUBLIC_HOSTS = {"0.0.0.0", "::", "[::]"}
DANGEROUS_ENV_FLAGS = {
    "allow_gui_automation": "JARVIS_ENABLE_GUI_AUTOMATION",
    "allow_shell_execution": "JARVIS_ENABLE_SHELL",
    "hardware_enabled": "JARVIS_ENABLE_HARDWARE",
}


@dataclass
class ProductionCheck:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _get(config: Any, section: str, key: str, fallback: str = "") -> str:
    try:
        return str(config.get(section, key, fallback=fallback))
    except Exception:
        return fallback


def _get_bool(config: Any, section: str, key: str, fallback: bool = False) -> bool:
    try:
        return bool(config.getboolean(section, key, fallback=fallback))
    except Exception:
        return fallback


def is_production(config: Any) -> bool:
    env = os.environ.get("JARVIS_ENV") or _get(config, "general", "environment", "")
    return env.strip().lower() == "production"


def validate_production_config(config: Any, *, dashboard_enabled: bool = False) -> ProductionCheck:
    result = ProductionCheck()
    prod = is_production(config)

    secret = os.environ.get("JARVIS_SECRET_KEY", "")
    if prod and (not secret or secret in {"jarvis", "development-only-secret"}):
        result.errors.append("JARVIS_SECRET_KEY must be set to a strong non-default value in production.")

    admin_user = os.environ.get("JARVIS_ADMIN_USER", "")
    admin_password = os.environ.get("JARVIS_ADMIN_PASSWORD", "")
    if prod and (not admin_user or not admin_password):
        result.errors.append("JARVIS_ADMIN_USER and JARVIS_ADMIN_PASSWORD are required for first production bootstrap.")
    if prod and admin_password and len(admin_password) < 12:
        result.errors.append("JARVIS_ADMIN_PASSWORD must be at least 12 characters.")

    if dashboard_enabled:
        host = os.environ.get("JARVIS_DASHBOARD_HOST") or _get(config, "dashboard", "host", "127.0.0.1")
        public_bind = host.strip() in PUBLIC_HOSTS
        require_https = os.environ.get("JARVIS_REQUIRE_HTTPS", "true").lower() != "false"
        proxy_ack = os.environ.get("JARVIS_PUBLIC_DASHBOARD_ACK", "").lower() == "true"
        if prod and public_bind and require_https and not proxy_ack:
            result.errors.append(
                "Public dashboard binding in production requires HTTPS/reverse-proxy acknowledgement via JARVIS_PUBLIC_DASHBOARD_ACK=true."
            )

    risky_enabled = {
        "allow_gui_automation": _get_bool(config, "execution", "allow_gui_automation", False),
        "allow_shell_execution": _get_bool(config, "execution", "allow_shell_execution", False),
        "hardware_enabled": _get_bool(config, "hardware", "enabled", False),
    }
    if prod:
        for key, enabled in risky_enabled.items():
            flag = DANGEROUS_ENV_FLAGS[key]
            if enabled and os.environ.get(flag, "").lower() != "true":
                result.errors.append(f"{key} is enabled but {flag}=true was not set.")

    provider_order = os.environ.get("JARVIS_MODEL_PROVIDER_ORDER")
    if provider_order is None:
        provider_order = _get(
            config,
            "models",
            "provider_order",
            "gemini,openai,groq,anthropic,ollama",
        )
    providers = [item.strip() for item in provider_order.split(",") if item.strip()]
    if prod and not providers:
        result.errors.append("At least one model provider must be configured.")
    if prod and "ollama" not in providers:
        result.warnings.append("Ollama is not in JARVIS_MODEL_PROVIDER_ORDER; local fallback is disabled.")

    return result


__all__ = ["ProductionCheck", "is_production", "validate_production_config"]




# --- FILE: core/introspection/health.py ---

"""Startup and lightweight runtime health checks."""

# internal import removed: from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from urllib.request import urlopen

# internal import removed: from core.ops.production import is_production, validate_production_config
# internal import removed: from core.runtime.paths import _resolve_path


class HealthStatus(str, Enum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True)
class HealthCheck:
    name: str
    status: HealthStatus
    message: str


@dataclass
class HealthReport:
    checks: list[HealthCheck] = field(default_factory=list)

    @property
    def has_failures(self) -> bool:
        return any(check.status == HealthStatus.FAIL for check in self.checks)

    @property
    def is_healthy(self) -> bool:
        return not self.has_failures

    @property
    def ollama_reachable(self) -> bool:
        check = next((item for item in self.checks if item.name == "ollama"), None)
        return bool(check and check.status == HealthStatus.OK)

    def summary(self) -> str:
        return "\n".join(
            f"{check.name}: {check.status.value.upper()} - {check.message}"
            for check in self.checks
        )


def _config_get(config, section: str, option: str, fallback: str = "") -> str:
    try:
        return str(config.get(section, option, fallback=fallback))
    except Exception:
        return fallback


def _config_get_bool(
    config,
    section: str,
    option: str,
    fallback: bool = False,
) -> bool:
    try:
        return bool(config.getboolean(section, option, fallback=fallback))
    except Exception:
        return fallback


def _module_available(import_name: str) -> bool:
    return importlib.util.find_spec(import_name) is not None


def _path_ready(path: Path, *, expect_file: bool) -> tuple[HealthStatus, str]:
    if not str(path):
        return HealthStatus.FAIL, "not configured"
    resolved_path = _resolve_path(path)
    target = resolved_path.parent if expect_file else resolved_path
    try:
        exists = target.exists()
        writable = os.access(target if exists else target.parent, os.W_OK)
    except OSError as exc:
        return HealthStatus.FAIL, f"{resolved_path} ({exc})"

    if exists and writable:
        return HealthStatus.OK, str(resolved_path)
    if exists:
        return HealthStatus.WARN, f"{resolved_path} (exists but may not be writable)"
    return HealthStatus.WARN, f"{resolved_path} (path will be created at runtime)"


def _collect_config_checks(config) -> list[HealthCheck]:
    checks: list[HealthCheck] = []

    environment = _config_get(
        config,
        "general",
        "environment",
        os.environ.get("JARVIS_ENV", "development"),
    )
    checks.append(
        HealthCheck(
            name="runtime_environment",
            status=HealthStatus.OK,
            message=environment or "development",
        )
    )

    sqlite_path = _config_get(config, "memory", "sqlite_file", "")
    if sqlite_path:
        sqlite_status, sqlite_message = _path_ready(Path(sqlite_path), expect_file=True)
    else:
        sqlite_status, sqlite_message = HealthStatus.FAIL, "memory.sqlite_file is not configured"
    checks.append(
        HealthCheck(
            name="memory_sqlite_config",
            status=sqlite_status,
            message=sqlite_message,
        )
    )

    app_log_path = _config_get(config, "logging", "app_file", "")
    if app_log_path:
        app_status, app_message = _path_ready(Path(app_log_path), expect_file=True)
    else:
        app_status, app_message = HealthStatus.WARN, "logging.app_file is not configured"
    checks.append(
        HealthCheck(
            name="logging_app_file",
            status=app_status,
            message=app_message,
        )
    )

    audit_path = _config_get(config, "logging", "audit_file", "")
    if audit_path:
        audit_status, audit_message = _path_ready(Path(audit_path), expect_file=True)
    else:
        audit_status, audit_message = HealthStatus.WARN, "logging.audit_file is not configured"
    checks.append(
        HealthCheck(
            name="logging_audit_file",
            status=audit_status,
            message=audit_message,
        )
    )

    raw_safe_dirs = _config_get(config, "execution", "safe_directories", "")
    safe_dirs = [Path(item.strip()) for item in raw_safe_dirs.split(",") if item.strip()]
    if not safe_dirs:
        checks.append(
            HealthCheck(
                name="execution_safe_directories",
                status=HealthStatus.WARN,
                message="No execution.safe_directories configured",
            )
        )
    else:
        ready = []
        pending = []
        for path in safe_dirs:
            status, _ = _path_ready(path, expect_file=False)
            if status == HealthStatus.OK:
                ready.append(str(path))
            else:
                pending.append(str(path))
        if pending:
            message = f"ready={len(ready)} pending={len(pending)}"
            status = HealthStatus.WARN
        else:
            message = ", ".join(str(path) for path in safe_dirs)
            status = HealthStatus.OK
        checks.append(
            HealthCheck(
                name="execution_safe_directories",
                status=status,
                message=message,
            )
        )

    if _config_get_bool(config, "automation", "enabled", True):
        drop_root = _config_get(config, "automation", "drop_root", "workspace/jarvis_dropbox")
        drop_status, drop_message = _path_ready(Path(drop_root), expect_file=False)
        checks.append(
            HealthCheck(
                name="automation_drop_root",
                status=drop_status,
                message=drop_message,
            )
        )

        watch_recordings = _config_get_bool(config, "automation", "watch_recordings", True)
        if watch_recordings:
            has_cv2 = _module_available("cv2")
            checks.append(
                HealthCheck(
                    name="automation_video_ocr_dependency",
                    status=HealthStatus.OK if has_cv2 else HealthStatus.WARN,
                    message="opencv available" if has_cv2 else "opencv (cv2) not installed; video OCR will be limited",
                )
            )

        uses_ocr = _config_get_bool(config, "automation", "watch_screenshots", True) or _config_get_bool(
            config,
            "automation",
            "live_screen_enabled",
            True,
        )
        if uses_ocr:
            def _check_tesseract() -> tuple[HealthStatus, str]:
                if not _module_available("pytesseract"):
                    return HealthStatus.WARN, "pytesseract not installed; screenshot OCR will be limited"

                import sys
                import subprocess

                # Configure Tesseract path dynamically
                if getattr(sys, "frozen", False):
                    base_dir = getattr(sys, "_MEIPASS", "")
                    tesseract_dir = os.path.join(base_dir, "bin", "tesseract")
                    cmd = os.path.join(tesseract_dir, "tesseract.exe")
                else:
                    cmd = str(os.environ.get("TESSERACT_CMD") or "")
                    if not cmd:
                        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                        local_bundled = os.path.join(project_root, "bin", "tesseract", "tesseract.exe")
                        if os.path.exists(local_bundled):
                            cmd = local_bundled
                        else:
                            cmd = "tesseract"

                try:
                    res = subprocess.run([cmd, "--version"], capture_output=True, text=True, timeout=5)
                    if res.returncode == 0:
                        version_line = res.stdout.splitlines()[0] if res.stdout else "unknown version"
                        return HealthStatus.OK, f"pytesseract available, tesseract executable responsive: {version_line}"
                    else:
                        return HealthStatus.FAIL, f"tesseract --version failed with return code {res.returncode}"
                except Exception as exc:
                    return HealthStatus.FAIL, f"tesseract executable not found or not executable at '{cmd}': {exc}"

            tess_status, tess_msg = _check_tesseract()
            checks.append(
                HealthCheck(
                    name="automation_ocr_dependency",
                    status=tess_status,
                    message=tess_msg,
                )
            )

    control_file = _config_get(config, "dashboard", "control_file", "")
    if control_file:
        control_status, control_message = _path_ready(Path(control_file), expect_file=True)
        checks.append(
            HealthCheck(
                name="dashboard_control_file",
                status=control_status,
                message=control_message,
            )
        )

    if _config_get_bool(config, "voice", "enabled", False):
        required_voice_modules = [
            "sounddevice",
            "speech_recognition",
            "pvporcupine",
            "pvrecorder",
        ]
        missing = [
            module_name for module_name in required_voice_modules if not _module_available(module_name)
        ]
        checks.append(
            HealthCheck(
                name="voice_dependencies",
                status=HealthStatus.FAIL if missing else HealthStatus.OK,
                message="missing: " + ", ".join(missing) if missing else "voice dependencies available",
            )
        )

    if _config_get_bool(config, "execution", "allow_gui_automation", False):
        required_desktop_modules = [
            "pyautogui",
            "pygetwindow",
            "pytesseract",
            "PIL",
        ]
        missing = [
            module_name for module_name in required_desktop_modules if not _module_available(module_name)
        ]
        checks.append(
            HealthCheck(
                name="desktop_dependencies",
                status=HealthStatus.FAIL if missing else HealthStatus.OK,
                message="missing: " + ", ".join(missing) if missing else "desktop dependencies available",
            )
        )

    if _config_get_bool(config, "execution", "allow_web_search", True) and _config_get_bool(
        config,
        "web_search",
        "enabled",
        True,
    ):
        ready_providers: list[str] = []
        issues: list[str] = []
        tavily_api_key = (
            os.environ.get("TAVILY_API_KEY")
            or _config_get(config, "web_search", "tavily_api_key", "")
        ).strip()

        if _module_available("ddgs"):
            ready_providers.append("ddgs")
        else:
            issues.append("ddgs package missing")

        if tavily_api_key:
            if _module_available("requests"):
                ready_providers.append("tavily")
            else:
                issues.append("requests package missing for Tavily")

        if ready_providers:
            checks.append(
                HealthCheck(
                    name="web_search_dependencies",
                    status=HealthStatus.OK,
                    message="ready providers: " + ", ".join(ready_providers),
                )
            )
        else:
            checks.append(
                HealthCheck(
                    name="web_search_dependencies",
                    status=HealthStatus.WARN,
                    message="no web-search provider ready" + (
                        f" ({'; '.join(issues)})" if issues else ""
                    ),
                )
            )

    production_check = validate_production_config(config)
    if production_check.errors:
        checks.append(
            HealthCheck(
                name="production_guardrails",
                status=HealthStatus.FAIL,
                message="; ".join(production_check.errors),
            )
        )
    elif production_check.warnings:
        checks.append(
            HealthCheck(
                name="production_guardrails",
                status=HealthStatus.WARN,
                message="; ".join(production_check.warnings),
            )
        )
    else:
        message = (
            "production guardrails passed"
            if is_production(config)
            else "not running in production mode"
        )
        checks.append(
            HealthCheck(
                name="production_guardrails",
                status=HealthStatus.OK,
                message=message,
            )
        )

    return checks


def _ollama_check(base_url: str) -> HealthCheck:
    reachable = False
    try:
        with urlopen(f"{base_url}/api/tags", timeout=5):
            reachable = True
    except Exception:
        reachable = False
    return HealthCheck(
        name="ollama",
        status=HealthStatus.OK if reachable else HealthStatus.WARN,
        message=f"{base_url} reachable={reachable}",
    )


def run_startup_health_check(controller, verbose: bool = False) -> HealthReport:
    del verbose
    checks: list[HealthCheck] = []

    config = getattr(controller, "config", None)
    if config is not None:
        checks.extend(_collect_config_checks(config))

    raw_db_path = getattr(getattr(controller, "memory", None), "db_path", "")
    db_path = _resolve_path(raw_db_path) if str(raw_db_path) else None
    exists = bool(db_path and db_path.exists())
    checks.append(
        HealthCheck(
            name="memory_sqlite",
            status=HealthStatus.OK if exists else HealthStatus.FAIL,
            message=str(db_path) if db_path is not None else "controller memory path missing",
        )
    )

    # Preflight Import dependency validation
    try:
        from core.runtime.import_validator import StartupValidator
        proj_root = Path(__file__).resolve().parents[2]
        validator = StartupValidator(proj_root)
        preflight = validator.run_preflight_checks()
        if preflight["status"] == "GREEN":
            checks.append(
                HealthCheck(
                    name="import_dependency_health",
                    status=HealthStatus.OK,
                    message="all critical submodules loaded successfully",
                )
            )
        else:
            failed_names = [f["module"] for f in preflight["failed"]]
            checks.append(
                HealthCheck(
                    name="import_dependency_health",
                    status=HealthStatus.FAIL,
                    message=f"failed modules: {', '.join(failed_names)}",
                )
            )
    except Exception as exc:
        checks.append(
            HealthCheck(
                name="import_dependency_health",
                status=HealthStatus.WARN,
                message=f"failed to run import preflight validation: {exc}",
            )
        )

    base_url = getattr(getattr(controller, "llm", None), "base_url", "http://localhost:11434")
    checks.append(_ollama_check(str(base_url)))
    return HealthReport(checks=checks)


def run_lightweight_health_check(config) -> HealthReport:
    checks = _collect_config_checks(config)

    # Lightweight import dependency validation
    try:
        from core.runtime.import_validator import StartupValidator
        proj_root = Path(__file__).resolve().parents[2]
        validator = StartupValidator(proj_root)
        preflight = validator.run_preflight_checks()
        if preflight["status"] == "GREEN":
            checks.append(
                HealthCheck(
                    name="import_dependency_health",
                    status=HealthStatus.OK,
                    message="all critical submodules loaded successfully",
                )
            )
        else:
            failed_names = [f["module"] for f in preflight["failed"]]
            checks.append(
                HealthCheck(
                    name="import_dependency_health",
                    status=HealthStatus.FAIL,
                    message=f"failed modules: {', '.join(failed_names)}",
                )
            )
    except Exception as exc:
        checks.append(
            HealthCheck(
                name="import_dependency_health",
                status=HealthStatus.WARN,
                message=f"failed to run import preflight validation: {exc}",
            )
        )

    base_url = _config_get(config, "ollama", "base_url", "http://localhost:11434")
    checks.append(_ollama_check(base_url))
    return HealthReport(checks=checks)




# --- FILE: core/logging/__init__.py ---

"""Logging package exports."""

# internal import removed: from . import logger

__all__ = ["logger"]




# --- FILE: core/ops/__init__.py ---

"""Operational helpers for production deployment."""




# --- FILE: core/permission_matrix.py ---

"""Compatibility permission matrix built on top of the risk config."""

# internal import removed: from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PermissionResult:
    blocked_actions: list[str] = field(default_factory=list)
    confirmation_actions: list[str] = field(default_factory=list)

    @property
    def has_blocked(self) -> bool:
        return bool(self.blocked_actions)

    @property
    def needs_confirmation(self) -> bool:
        return bool(self.confirmation_actions)


class PermissionMatrix:
    def __init__(self, config=None) -> None:
        self.config = config

    def evaluate(self, actions: list[str]) -> PermissionResult:
        blocked = self._parse_csv("risk", "blocked_actions")
        if not blocked:
            blocked = self._parse_csv("risk", "critical_actions")
        if not blocked:
            blocked = self._parse_csv("risk", "forbidden_actions")

        confirmation = self._parse_csv("risk", "user_confirmed_actions")
        if not confirmation:
            confirmation = self._parse_csv("risk", "high_risk_actions")

        normalized = [str(action).strip().lower() for action in actions if str(action).strip()]
        blocked_actions = [action for action in normalized if action in blocked]
        confirmation_actions = [
            action for action in normalized if action in confirmation and action not in blocked_actions
        ]
        return PermissionResult(
            blocked_actions=blocked_actions,
            confirmation_actions=confirmation_actions,
        )

    def _parse_csv(self, section: str, key: str) -> set[str]:
        if self.config is None:
            return set()
        try:
            raw = self.config.get(section, key, fallback="")
        except Exception:
            return set()
        return {item.strip().lower() for item in raw.split(",") if item.strip()}


__all__ = ["PermissionMatrix", "PermissionResult"]




# --- FILE: core/planner/__init__.py ---

"""Plan generation layer."""




# --- FILE: core/proactive/__init__.py ---






# --- FILE: core/runtime/bootstrap.py ---

# internal import removed: from __future__ import annotations

import argparse
import asyncio
import configparser
import contextlib
import dataclasses
import faulthandler
import io
import json
import logging
import math
import os
import signal
import sys
import threading
from pathlib import Path
from typing import Any

# internal import removed: from core.ops.production import validate_production_config
# internal import removed: from core.runtime.paths import core_runtime_paths_PROJECT_ROOT, _resolve_path
core_runtime_bootstrap_DEFAULT_CONFIG_PATH = "config/jarvis.ini"
DEFAULT_DASHBOARD_HOST = "127.0.0.1"
DEFAULT_DASHBOARD_PORT = 7070
DEFAULT_SHUTDOWN_TIMEOUT_S = 15.0


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        # Load config/settings.env first as default/fallback
        settings_env = core_runtime_paths_PROJECT_ROOT / "config" / "settings.env"
        if settings_env.exists():
            load_dotenv(dotenv_path=settings_env)

        # Then load root .env to override, if it exists
        root_env = core_runtime_paths_PROJECT_ROOT / ".env"
        if root_env.exists():
            load_dotenv(dotenv_path=root_env, override=True)
    except Exception:
        return


def _enable_fault_diagnostics() -> None:
    try:
        faulthandler.enable(all_threads=True)
    except Exception:
        return


def _configure_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                continue


_load_dotenv()
_enable_fault_diagnostics()
_configure_stdio()


def _uprint(msg: str, *, file=None) -> None:
    """Print safely even on Windows consoles with non-UTF encodings."""
    target = file or sys.stdout
    try:
        print(msg, file=target)
    except UnicodeEncodeError:
        raw = getattr(target, "buffer", None)
        if raw is not None:
            raw.write((msg + "\n").encode("utf-8", errors="replace"))
        else:
            fallback = msg.encode("ascii", errors="replace").decode("ascii")
            print(fallback, file=target)


_bootstrap = logging.getLogger("jarvis.bootstrap")
if not _bootstrap.handlers:
    _bootstrap.addHandler(logging.StreamHandler(sys.stderr))
_bootstrap_level = (
    logging.DEBUG
    if os.environ.get("JARVIS_LOG_LEVEL", "").upper() == "DEBUG"
    else logging.INFO
)
_bootstrap.setLevel(_bootstrap_level)
_bootstrap.propagate = False


class ExitCode:
    OK = 0
    GENERIC_ERROR = 1
    CONFIG_ERROR = 2
    AUDIT_FAILED = 3
    STARTUP_ERROR = 4


@dataclasses.dataclass
class StartupValidation:
    errors: list[str] = dataclasses.field(default_factory=list)
    warnings: list[str] = dataclasses.field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors


# _resolve_path is imported from core.runtime.paths


def _ensure_section(config: configparser.ConfigParser, section: str) -> None:
    if not config.has_section(section):
        config.add_section(section)


def core_runtime_bootstrap_load_config(config_path: str) -> configparser.ConfigParser:
    """
    Load INI config from an absolute path or relative to core_runtime_paths_PROJECT_ROOT.
    Raises SystemExit(CONFIG_ERROR) if the file is missing in production.
    """
    from core.config import core_runtime_bootstrap_load_config as _load_config
    try:
        config = _load_config(config_path)
        _bootstrap.debug("Config loaded from %s", config_path)
        return config
    except SystemExit:
        raise
    except Exception as e:
        _bootstrap.critical("Failed to load configuration from %s: %s", config_path, e)
        sys.exit(ExitCode.CONFIG_ERROR)


def apply_cli_overrides(
    config: configparser.ConfigParser,
    args: argparse.Namespace,
) -> None:
    """Merge CLI arguments into config without clobbering unrelated keys."""
    if getattr(args, "log_level", None):
        _ensure_section(config, "logging")
        config["logging"]["level"] = str(args.log_level)

    if getattr(args, "session_name", None):
        _ensure_section(config, "general")
        config["general"]["session_name"] = str(args.session_name)

    if getattr(args, "voice", False):
        _ensure_section(config, "voice")
        config["voice"]["enabled"] = "true"

    dashboard_host = getattr(args, "dashboard_host", None)
    if dashboard_host:
        _ensure_section(config, "dashboard")
        config["dashboard"]["host"] = str(dashboard_host)

    dashboard_port = getattr(args, "dashboard_port", None)
    if dashboard_port is not None:
        _ensure_section(config, "dashboard")
        config["dashboard"]["port"] = str(int(dashboard_port))

    if getattr(args, "headless", False):
        _ensure_section(config, "autonomy")
        config["autonomy"]["level"] = "4"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    dashboard_port_default = os.environ.get("JARVIS_DASHBOARD_PORT")

    parser = argparse.ArgumentParser(
        description="Jarvis local runtime",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--voice", action="store_true", help="Start Jarvis in voice mode")
    parser.add_argument("--gui", action="store_true", help="Start the dashboard server")
    parser.add_argument("--dashboard", action="store_true", help="Alias for --gui")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Do not start CLI or voice loop; keep services alive until shutdown",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify audit log integrity and exit",
    )
    parser.add_argument(
        "--health-check",
        action="store_true",
        help="Run startup health diagnostics and exit",
    )
    parser.add_argument(
        "--strict-health",
        action="store_true",
        help="Fail startup if health diagnostics report failures",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print configured model routing and discovered availability",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the effective config after CLI overrides",
    )
    parser.add_argument(
        "--config",
        default=os.environ.get("JARVIS_CONFIG", core_runtime_bootstrap_DEFAULT_CONFIG_PATH),
        help="Config file path (also reads JARVIS_CONFIG)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=os.environ.get("JARVIS_LOG_LEVEL"),
        help="Override log level (also reads JARVIS_LOG_LEVEL)",
    )
    parser.add_argument("--session-name", help="Optional session name for this run")
    parser.add_argument(
        "--dashboard-host",
        default=os.environ.get("JARVIS_DASHBOARD_HOST"),
        help="Dashboard bind host",
    )
    parser.add_argument(
        "--dashboard-port",
        type=int,
        default=int(dashboard_port_default) if dashboard_port_default else None,
        help="Dashboard bind port",
    )
    parser.add_argument(
        "--shutdown-timeout",
        type=float,
        default=float(
            os.environ.get(
                "JARVIS_SHUTDOWN_TIMEOUT_S",
                str(DEFAULT_SHUTDOWN_TIMEOUT_S),
            )
        ),
        help="Graceful shutdown timeout in seconds",
    )
    parser.add_argument(
        "--replay",
        help="Path to an execution trace snapshot file (.json) to replay",
    )
    return parser.parse_args(argv)


class _ShutdownCoordinator:
    """Signal-aware shutdown gate."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._event = asyncio.Event()

    def request_shutdown(self, signame: str = "manual") -> None:
        _bootstrap.info("Shutdown requested via %s", signame)
        self._loop.call_soon_threadsafe(self._event.set)

    def install_signal_handlers(self) -> None:
        if sys.platform == "win32":
            supported_signals = [signal.SIGINT]
            sigbreak = getattr(signal, "SIGBREAK", None)
            if sigbreak is not None:
                supported_signals.append(sigbreak)
            for sig in supported_signals:
                try:
                    signal.signal(
                        sig,
                        lambda *_, _signame=getattr(sig, "name", str(sig)): self.request_shutdown(_signame),
                    )
                except (OSError, RuntimeError, ValueError) as exc:
                    _bootstrap.warning(
                        "Unable to install shutdown handler for %s: %s",
                        getattr(sig, "name", sig),
                        exc,
                    )
            return

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                self._loop.add_signal_handler(sig, self.request_shutdown, getattr(sig, "name", str(sig)))
            except (NotImplementedError, RuntimeError, ValueError) as exc:
                _bootstrap.warning(
                    "Unable to install shutdown handler for %s: %s",
                    getattr(sig, "name", str(sig)),
                    exc,
                )

    async def wait(self) -> None:
        await self._event.wait()


def _install_process_exception_hooks(log: logging.Logger) -> None:
    def _sys_hook(exc_type, exc_value, exc_tb) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        log.critical(
            "Unhandled process exception",
            exc_info=(exc_type, exc_value, exc_tb),
        )

    sys.excepthook = _sys_hook

    if hasattr(threading, "excepthook"):
        def _thread_hook(args) -> None:
            if issubclass(args.exc_type, KeyboardInterrupt):
                return
            log.critical(
                "Unhandled thread exception in %s",
                getattr(args.thread, "name", "unknown"),
                exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
            )

        threading.excepthook = _thread_hook


def _install_loop_exception_handler(
    loop: asyncio.AbstractEventLoop,
    log: logging.Logger,
) -> None:
    def _handler(_loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
        exc = context.get("exception")
        message = context.get("message", "Unhandled event loop exception")
        if exc is None:
            log.error("%s | context=%s", message, context)
            return
        log.error("%s", message, exc_info=exc)

    loop.set_exception_handler(_handler)


def _prepare_runtime_environment(config: configparser.ConfigParser) -> None:
    environment = config.get(
        "general",
        "environment",
        fallback=os.environ.get("JARVIS_ENV", "development"),
    )
    os.environ.setdefault("JARVIS_ENV", str(environment))


def _prepare_runtime_paths(config: configparser.ConfigParser) -> None:
    entries = [
        ("logging", "log_dir", False),
        ("logging", "app_file", True),
        ("logging", "audit_file", True),
        ("logging", "trace_dir", False),
        ("memory", "data_dir", False),
        ("memory", "sqlite_file", True),
        ("memory", "db_path", True),
        ("memory", "chroma_dir", False),
        ("memory", "chroma_path", False),
        ("dashboard", "control_file", True),
        ("plugins", "manifest_directory", False),
        ("ai_os", "workflow_catalog_dir", False),
        ("automation", "drop_root", False),
        ("automation", "commands_folder", False),
        ("automation", "rag_folder", False),
        ("automation", "processed_folder", False),
        ("automation", "failed_folder", False),
        ("automation", "screenshots_folder", False),
        ("automation", "recordings_folder", False),
        ("automation", "ingest_log_file", True),
        ("automation", "state_file", True),
    ]

    for section, key, is_file in entries:
        if not config.has_option(section, key):
            continue
        raw_value = config.get(section, key, fallback="").strip()
        if not raw_value:
            continue
        try:
            path = _resolve_path(raw_value)
            target = path.parent if is_file else path
            target.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            _bootstrap.warning("Failed to create path for [%s] %s: %s", section, key, e)

    raw_safe_dirs = config.get("execution", "safe_directories", fallback="")
    for raw_dir in raw_safe_dirs.split(","):
        value = raw_dir.strip()
        if not value:
            continue
        try:
            _resolve_path(value).mkdir(parents=True, exist_ok=True)
        except OSError as e:
            _bootstrap.warning("Failed to create safe directory %s: %s", value, e)


def _resolve_voice_enabled(
    config: configparser.ConfigParser,
    args: argparse.Namespace,
) -> bool:
    cli_value = bool(getattr(args, "voice", False))
    try:
        config_value = config.getboolean("voice", "enabled", fallback=False)
    except (configparser.Error, ValueError):
        config_value = False
    return cli_value or config_value


def _resolve_dashboard_binding(
    config: configparser.ConfigParser,
    args: argparse.Namespace,
) -> tuple[str, int]:
    host = str(
        getattr(args, "dashboard_host", None)
        or config.get("dashboard", "host", fallback=DEFAULT_DASHBOARD_HOST)
        or DEFAULT_DASHBOARD_HOST
    )
    port_arg = getattr(args, "dashboard_port", None)
    if port_arg is not None:
        return host, int(port_arg)
    try:
        port = config.getint("dashboard", "port", fallback=DEFAULT_DASHBOARD_PORT)
    except (configparser.Error, TypeError, ValueError):
        port = DEFAULT_DASHBOARD_PORT
    return host, port


def _resolve_runtime_mode(
    *,
    voice_enabled: bool,
    dashboard_enabled: bool,
    headless: bool,
) -> str:
    if headless and dashboard_enabled:
        return "headless+dashboard"
    if headless:
        return "headless"
    if voice_enabled and dashboard_enabled:
        return "voice+dashboard"
    if voice_enabled:
        return "voice"
    if dashboard_enabled:
        return "cli+dashboard"
    return "cli"


def _validate_startup_settings(
    config: configparser.ConfigParser,
    args: argparse.Namespace,
    *,
    voice_enabled: bool,
    dashboard_enabled: bool,
    headless: bool,
    shutdown_timeout: float,
) -> StartupValidation:
    result = StartupValidation()

    if getattr(args, "verify", False) and getattr(args, "health_check", False):
        result.errors.append("Choose either --verify or --health-check, not both.")

    if not math.isfinite(shutdown_timeout) or shutdown_timeout <= 0:
        result.errors.append("--shutdown-timeout must be greater than 0 seconds.")

    if dashboard_enabled:
        host = str(
            getattr(args, "dashboard_host", None)
            or config.get("dashboard", "host", fallback=DEFAULT_DASHBOARD_HOST)
            or ""
        ).strip()
        if not host:
            result.errors.append("Dashboard host cannot be empty when dashboard mode is enabled.")

        port = getattr(args, "dashboard_port", None)
        if port is None:
            try:
                port = config.getint("dashboard", "port", fallback=DEFAULT_DASHBOARD_PORT)
            except (configparser.Error, TypeError, ValueError):
                port = None
        if port is None or not 1 <= int(port) <= 65535:
            result.errors.append("Dashboard port must be between 1 and 65535.")

    raw_safe_dirs = config.get("execution", "safe_directories", fallback="")
    safe_dirs = [item.strip() for item in raw_safe_dirs.split(",") if item.strip()]
    if not safe_dirs:
        result.warnings.append(
            "No execution.safe_directories are configured; file operations may be blocked more often."
        )

    if headless and voice_enabled:
        result.warnings.append(
            "Headless mode disables the interactive voice loop even if voice mode is enabled."
        )

    production_check = validate_production_config(
        config,
        dashboard_enabled=dashboard_enabled,
    )
    result.errors.extend(production_check.errors)
    result.warnings.extend(production_check.warnings)

    return result


def _redact_key(key: str, value: str) -> str:
    lowered = key.lower()
    if any(
        token in lowered
        for token in ("secret", "token", "password", "api_key", "access_key")
    ):
        return "***REDACTED***"
    return value


def _config_snapshot(config: configparser.ConfigParser) -> dict[str, dict[str, str]]:
    snapshot: dict[str, dict[str, str]] = {}
    for section in config.sections():
        snapshot[section] = {
            key: _redact_key(key, value)
            for key, value in config.items(section)
        }
    return snapshot


def _print_config_snapshot(
    config: configparser.ConfigParser,
    config_path: Path,
) -> None:
    payload = {
        "config_path": str(config_path),
        "project_root": str(core_runtime_paths_PROJECT_ROOT),
        "sections": _config_snapshot(config),
    }
    _uprint(json.dumps(payload, indent=2, sort_keys=True))


def _build_model_inventory(
    config: configparser.ConfigParser,
) -> dict[str, Any]:
    from core.llm.model_router import ModelRouter

    router = ModelRouter(config=config)
    inventory: dict[str, Any] = {}

    for task_type in (
        "intent_classification",
        "memory_summarization",
        "tool_selection",
        "planning",
        "chat",
        "vision",
        "synthesis",
        "fallback",
    ):
        primary = router.route(task_type)
        entry: dict[str, Any] = {
            "primary": primary,
            "primary_available": router.is_available(primary),
        }
        try:
            entry["best_available"] = router.get_best_available(task_type)
        except Exception as exc:
            entry["best_available"] = None
            entry["error"] = str(exc)
        inventory[task_type] = entry

    inventory["discovered"] = router.list_available()
    return inventory


def _print_model_inventory(config: configparser.ConfigParser) -> None:
    inventory = _build_model_inventory(config)
    _uprint(json.dumps(inventory, indent=2, sort_keys=True))


def _should_exit_after_info(args: argparse.Namespace) -> bool:
    info_requested = bool(
        getattr(args, "print_config", False)
        or getattr(args, "list_models", False)
    )
    runtime_requested = bool(
        getattr(args, "voice", False)
        or getattr(args, "gui", False)
        or getattr(args, "dashboard", False)
        or getattr(args, "headless", False)
        or getattr(args, "verify", False)
        or getattr(args, "health_check", False)
    )
    return info_requested and not runtime_requested


def _safe_audit(
    logger_mod: Any,
    event_type: str,
    payload: dict[str, Any],
    log: logging.Logger,
) -> None:
    audit_fn = getattr(logger_mod, "audit", None)
    if not callable(audit_fn):
        return
    try:
        audit_fn(event_type, payload)
    except Exception:
        log.debug("Audit event '%s' failed", event_type, exc_info=True)


def _load_logger_module():
    logger_mod = sys.modules.get("core.logging.logger")
    if logger_mod is not None:
        return logger_mod

    try:
        from core.logging import logger as logger_mod
        return logger_mod
    except Exception as e:
        _bootstrap.error("Failed to load logger module: %s", e)
        return None


def _load_controller_class():
    controller_mod = sys.modules.get("core.controller_v2")
    if controller_mod is not None:
        return controller_mod.JarvisControllerV2

    try:
        from core.controller_v2 import JarvisControllerV2
        return JarvisControllerV2
    except Exception as e:
        _bootstrap.critical("Failed to load JarvisControllerV2: %s", e)
        sys.exit(ExitCode.STARTUP_ERROR)


def _load_integrations(
    controller: Any,
    config: configparser.ConfigParser,
    log: logging.Logger,
) -> dict[str, list[str]]:
    try:
        from integrations.loader import IntegrationLoader
        from integrations.registry import integration_registry

        loader = IntegrationLoader()
        result = loader.load_all(config=config, registry=integration_registry)

        # Dynamically register loaded integration safety rules & risk profiles
        gov = getattr(controller, "autonomy_governor", None)
        risk_eval = getattr(controller, "risk_evaluator", None)
        integration_registry.register_safety_rules(gov, risk_eval)

        tool_router = getattr(controller, "tool_router", None)
        if tool_router is not None:
            for tool_dict in integration_registry.get_tools():
                tool_name = tool_dict.get("name")
                if tool_name and not tool_router.get(tool_name):
                    def make_handler(t_name: str):
                        async def _integration_handler(**kwargs):
                            return await integration_registry.execute(t_name, kwargs)
                        _integration_handler.__doc__ = f"Integration tool: {t_name}"
                        return _integration_handler
                    tool_router.register(tool_name, make_handler(tool_name))

        setattr(controller, "integration_loader", loader)
        setattr(controller, "integration_registry", integration_registry)
        setattr(controller, "_integration_result", result)
        log.info(
            "Integrations loaded=%d skipped=%d",
            len(result.get("loaded", [])),
            len(result.get("skipped", [])),
        )
        return result
    except Exception:
        log.exception("Integration bootstrap failed")
        result = {"loaded": [], "skipped": ["bootstrap failed"]}
        setattr(controller, "_integration_result", result)
        return result


def _run_startup_health_check(controller: Any | None, *, verbose: bool) -> Any:
    from core.introspection.health import HealthReport, HealthCheck, HealthStatus, run_startup_health_check

    try:
        if verbose:
            return run_startup_health_check(controller)

        sink = io.StringIO()
        with contextlib.redirect_stdout(sink):
            report = run_startup_health_check(controller)
        if report is None:
            return HealthReport()
        return report
    except Exception as exc:
        _bootstrap.error("Startup health check failed: %s", exc, exc_info=True)
        return HealthReport(checks=[
            HealthCheck(
                name="startup_health_check_execution",
                status=HealthStatus.FAIL,
                message=f"Exception during check: {exc}",
            )
        ])


async def _cancel_task(task: asyncio.Task[Any]) -> None:
    if task.done():
        if task.cancelled():
            return
        with contextlib.suppress(Exception):
            task.exception()
        return

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await task


__all__ = [
    "core_runtime_bootstrap_DEFAULT_CONFIG_PATH",
    "DEFAULT_DASHBOARD_HOST",
    "DEFAULT_DASHBOARD_PORT",
    "DEFAULT_SHUTDOWN_TIMEOUT_S",
    "ExitCode",
    "core_runtime_paths_PROJECT_ROOT",
    "_ShutdownCoordinator",
    "_bootstrap",
    "_build_model_inventory",
    "_cancel_task",
    "_config_snapshot",
    "_install_loop_exception_handler",
    "_install_process_exception_hooks",
    "_load_controller_class",
    "_load_integrations",
    "_load_logger_module",
    "_prepare_runtime_environment",
    "_prepare_runtime_paths",
    "_print_config_snapshot",
    "_print_model_inventory",
    "_resolve_dashboard_binding",
    "_resolve_runtime_mode",
    "_resolve_path",
    "_resolve_voice_enabled",
    "_run_startup_health_check",
    "_safe_audit",
    "_should_exit_after_info",
    "_uprint",
    "_validate_startup_settings",
    "apply_cli_overrides",
    "core_runtime_bootstrap_load_config",
    "parse_args",
    "StartupValidation",
]




# --- FILE: core/runtime/__init__.py ---

# internal import removed: from core.runtime.bootstrap import (
# internal import removed:     core_runtime_bootstrap_DEFAULT_CONFIG_PATH,
# internal import removed:     DEFAULT_DASHBOARD_HOST,
# internal import removed:     DEFAULT_DASHBOARD_PORT,
# internal import removed:     DEFAULT_SHUTDOWN_TIMEOUT_S,
# internal import removed:     ExitCode,
# internal import removed:     PROJECT_ROOT,
# internal import removed:     _ShutdownCoordinator,
# internal import removed:     _bootstrap,
# internal import removed:     apply_cli_overrides,
# internal import removed:     core_runtime_bootstrap_load_config,
# internal import removed:     parse_args,
# internal import removed: )


def __getattr__(name: str):
    if name == "async_run":
        from core.runtime.entrypoint import async_run

        return async_run
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "core_runtime_bootstrap_DEFAULT_CONFIG_PATH",
    "DEFAULT_DASHBOARD_HOST",
    "DEFAULT_DASHBOARD_PORT",
    "DEFAULT_SHUTDOWN_TIMEOUT_S",
    "ExitCode",
    "PROJECT_ROOT",
    "_ShutdownCoordinator",
    "_bootstrap",
    "apply_cli_overrides",
    "async_run",
    "core_runtime_bootstrap_load_config",
    "parse_args",
]




# --- FILE: core/runtime/container.py ---

"""
Dependency Injection (DI) Service Container for Project Jarvis.
Provides clean decoupling of services and allows dynamic overrides/registrations.
"""

# internal import removed: from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Tuple, Type, Union

logger = logging.getLogger("Jarvis.Container")


class ServiceContainer:
    """
    Lightweight, thread-safe Dependency Injection (DI) Container.
    Enables components to be decoupled and custom subclasses/mock implementations
    to be dynamically registered and resolved.
    """

    def __init__(self) -> None:
        self._providers: Dict[str, Tuple[Union[Type, Callable], bool]] = {}
        self._instances: Dict[str, Any] = {}

    def register(self, name: str, factory_or_class: Union[Type, Callable], is_singleton: bool = True) -> None:
        """Register a class or a factory function for a service."""
        key = name.strip().lower()
        self._providers[key] = (factory_or_class, is_singleton)
        logger.debug("Registered service '%s' (singleton=%s)", key, is_singleton)
        # Invalidate any cached instance if re-registered
        if key in self._instances:
            del self._instances[key]

    def register_instance(self, name: str, instance: Any) -> None:
        """Register a pre-constructed instance of a service."""
        key = name.strip().lower()
        self._instances[key] = instance
        logger.debug("Registered instance for service '%s'", key)

    def has(self, name: str) -> bool:
        """Check if a service is registered in the container."""
        key = name.strip().lower()
        return key in self._instances or key in self._providers

    def resolve(self, name: str, **kwargs) -> Any:
        """
        Resolve a service instance. If registered as a singleton, the cached instance
        is returned; otherwise, a fresh instance is constructed. Any keyword arguments
        passed will be forwarded to the factory/constructor.
        """
        key = name.strip().lower()
        if key in self._instances:
            return self._instances[key]

        if key not in self._providers:
            raise ValueError(f"Service '{name}' is not registered in the container.")

        factory_or_class, is_singleton = self._providers[key]

        try:
            import inspect
            if isinstance(factory_or_class, type):
                try:
                    sig = inspect.signature(factory_or_class)
                    has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
                    if has_var_keyword:
                        valid_kwargs = kwargs
                    else:
                        valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
                except Exception:
                    valid_kwargs = kwargs
                instance = factory_or_class(**valid_kwargs)
            elif callable(factory_or_class):
                try:
                    sig = inspect.signature(factory_or_class)
                    has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
                    if has_var_keyword:
                        valid_kwargs = kwargs
                    else:
                        valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
                except Exception:
                    valid_kwargs = {}
                instance = factory_or_class(**valid_kwargs)
            else:
                instance = factory_or_class
        except Exception as e:
            logger.exception("Failed to resolve service '%s'", name)
            raise RuntimeError(f"Failed to instantiate service '{name}': {e}") from e

        if is_singleton:
            self._instances[key] = instance

        return instance

    def reset(self) -> None:
        """Clears all registered providers and cached instances."""
        self._providers.clear()
        self._instances.clear()
        logger.debug("Container reset.")




# --- FILE: core/runtime/dashboard_runtime.py ---

# internal import removed: from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
from typing import Any


class DashboardRuntime:
    def __init__(
        self,
        host: str,
        port: int,
        log: logging.Logger,
    ) -> None:
        self.host = host
        self.port = port
        self.log = log
        self._server: Any = None
        self._thread: threading.Thread | None = None
        self._thread_error: BaseException | None = None

    async def start(self, controller: Any, health_report: Any | None = None) -> None:
        if self._thread and self._thread.is_alive():
            return

        import uvicorn
        from dashboard.server import app as dashboard_app
        from dashboard.server import set_controller, update_state

        set_controller(controller)
        config = uvicorn.Config(
            dashboard_app,
            host=self.host,
            port=self.port,
            log_level="warning",
            access_log=False,
        )
        self._server = uvicorn.Server(config)

        def _serve() -> None:
            try:
                self._server.run()
            except BaseException as exc:
                self._thread_error = exc

        self._thread_error = None
        self._thread = threading.Thread(
            target=_serve,
            name="jarvis-dashboard",
            daemon=True,
        )
        self._thread.start()

        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if self._thread_error is not None:
                raise RuntimeError(
                    "Dashboard thread crashed during startup"
                ) from self._thread_error
            if getattr(self._server, "started", False):
                break
            if self._thread and not self._thread.is_alive():
                raise RuntimeError("Dashboard server exited before reporting ready")
            await asyncio.sleep(0.1)

        if not getattr(self._server, "started", False):
            self.log.warning("Dashboard startup was not confirmed within 10 seconds")

        llm_obj = getattr(controller, "llm", None)
        model_name = getattr(llm_obj, "model", getattr(llm_obj, "model_name", "unknown"))
        active_goals = 0
        goal_manager = getattr(controller, "goal_manager", None)
        if goal_manager is not None and hasattr(goal_manager, "active_goals"):
            with contextlib.suppress(Exception):
                active_goals = len(goal_manager.active_goals())

        update_state(
            session_id=str(getattr(controller, "session_id", "jarvis")),
            model=str(model_name),
            state="IDLE",
            active_goals=active_goals,
            ollama_online=bool(getattr(health_report, "ollama_reachable", False)),
        )
        self.log.info("Dashboard listening on http://%s:%s", self.host, self.port)

    async def stop(self, timeout: float = 5.0) -> None:
        if self._server is not None:
            try:
                self._server.should_exit = True
                self._server.force_exit = True
                
                # Wake up the uvicorn event loop if it is idle
                import socket
                host = "127.0.0.1" if self.host == "0.0.0.0" else self.host
                with contextlib.suppress(Exception):
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(0.5)
                    sock.connect((host, self.port))
                    sock.close()
            except Exception:
                pass

        if self._thread and self._thread.is_alive():
            await asyncio.to_thread(self._thread.join, timeout)
            if self._thread.is_alive():
                self.log.warning(
                    "Dashboard thread did not stop within %.1f seconds",
                    timeout,
                )

        with contextlib.suppress(Exception):
            from dashboard.server import update_state

            update_state(state="OFFLINE")


__all__ = ["DashboardRuntime"]




# --- FILE: core/runtime/entrypoint.py ---

# internal import removed: from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import sys
from typing import Any

# internal import removed: from core.introspection.health import HealthStatus, run_lightweight_health_check
# internal import removed: from core.runtime.paths import _resolve_path
# internal import removed: from core.runtime.bootstrap import (
# internal import removed:     core_runtime_bootstrap_DEFAULT_CONFIG_PATH,
# internal import removed:     DEFAULT_SHUTDOWN_TIMEOUT_S,
# internal import removed:     ExitCode,
# internal import removed:     _cancel_task,
# internal import removed:     _install_loop_exception_handler,
# internal import removed:     _install_process_exception_hooks,
# internal import removed:     _load_controller_class,
# internal import removed:     _load_integrations,
# internal import removed:     _load_logger_module,
# internal import removed:     _prepare_runtime_environment,
# internal import removed:     _prepare_runtime_paths,
# internal import removed:     _print_config_snapshot,
# internal import removed:     _print_model_inventory,
# internal import removed:     _resolve_dashboard_binding,
# internal import removed:     _resolve_runtime_mode,
# internal import removed:     _resolve_voice_enabled,
# internal import removed:     _run_startup_health_check,
# internal import removed:     _safe_audit,
# internal import removed:     _should_exit_after_info,
# internal import removed:     _uprint,
# internal import removed:     _validate_startup_settings,
# internal import removed:     apply_cli_overrides,
# internal import removed:     core_runtime_bootstrap_load_config,
# internal import removed: )
# internal import removed: from core.runtime.bootstrap import _ShutdownCoordinator as DefaultShutdownCoordinator
DefaultShutdownCoordinator = _ShutdownCoordinator
# internal import removed: from core.runtime.dashboard_runtime import DashboardRuntime


def _log_health_report(log: logging.Logger, report: Any, *, prefix: str) -> None:
    for check in getattr(report, "checks", []):
        status = getattr(check, "status", HealthStatus.OK)
        message = f"{prefix} health {check.name}: {status.value.upper()} - {check.message}"
        if status == HealthStatus.FAIL:
            log.error(message)
        elif status == HealthStatus.WARN:
            log.warning(message)
        else:
            log.info(message)


async def _run_runtime_loop(
    controller: Any,
    shutdown: Any,
    *,
    headless: bool,
    log: logging.Logger,
) -> int:
    run_cli = getattr(controller, "run_cli", None)

    if headless:
        log.info("Running in headless mode; waiting for shutdown signal")
        await shutdown.wait()
        return ExitCode.OK

    if not callable(run_cli):
        log.warning("Controller has no run_cli(); waiting for shutdown signal")
        await shutdown.wait()
        return ExitCode.OK

    cli_task = asyncio.create_task(run_cli(), name="jarvis-cli")
    shutdown_task = asyncio.create_task(shutdown.wait(), name="jarvis-shutdown")

    done, pending = await asyncio.wait(
        {cli_task, shutdown_task},
        return_when=asyncio.FIRST_COMPLETED,
    )

    for task in pending:
        await _cancel_task(task)

    if cli_task in done and not cli_task.cancelled():
        exc = cli_task.exception()
        if exc is not None:
            raise exc

    return ExitCode.OK


async def async_run(
    args,
    *,
    shutdown_cls: type = DefaultShutdownCoordinator,
) -> int:
    """
    Core coroutine. Returns an integer exit code.
    Never calls sys.exit() directly.
    """
    config_path = _resolve_path(getattr(args, "config", core_runtime_bootstrap_DEFAULT_CONFIG_PATH))
    config = core_runtime_bootstrap_load_config(str(config_path))
    apply_cli_overrides(config, args)
    _prepare_runtime_environment(config)
    _prepare_runtime_paths(config)

    try:
        logger_mod = _load_logger_module()

        logger_mod.setup(config)
        log = logger_mod.get()
    except Exception as exc:
        from core.runtime.bootstrap import _bootstrap

        _bootstrap.critical("Failed to initialize logging subsystem: %s", exc)
        return ExitCode.STARTUP_ERROR

    _install_process_exception_hooks(log)

    if getattr(args, "print_config", False):
        _print_config_snapshot(config, config_path)

    if getattr(args, "list_models", False):
        try:
            _print_model_inventory(config)
        except Exception:
            log.exception("Failed to inspect model inventory")
            if _should_exit_after_info(args):
                return ExitCode.STARTUP_ERROR

    if _should_exit_after_info(args):
        return ExitCode.OK

    if getattr(args, "verify", False):
        try:
            ok, count, err = logger_mod.verify_audit()
            if ok:
                _uprint(f"[OK] Audit OK - {count} entries verified")
                log.info("Audit verification passed (%d entries)", count)
                return ExitCode.OK
            _uprint(f"[FAIL] Audit FAILED - {err}", file=sys.stderr)
            log.error("Audit verification failed: %s", err)
            return ExitCode.AUDIT_FAILED
        except Exception:
            log.exception("Unexpected error during audit verification")
            _uprint("[ERROR] Audit verification crashed", file=sys.stderr)
            return ExitCode.GENERIC_ERROR

    voice_enabled = _resolve_voice_enabled(config, args)
    dashboard_enabled = bool(
        getattr(args, "gui", False) or getattr(args, "dashboard", False)
    )
    headless = bool(getattr(args, "headless", False))
    runtime_mode = _resolve_runtime_mode(
        voice_enabled=voice_enabled,
        dashboard_enabled=dashboard_enabled,
        headless=headless,
    )
    shutdown_timeout = float(
        getattr(args, "shutdown_timeout", DEFAULT_SHUTDOWN_TIMEOUT_S)
    )
    version = config.get("general", "version", fallback="unknown")
    environment = config.get(
        "general",
        "environment",
        fallback=os.environ.get("JARVIS_ENV", "development"),
    )
    validation = _validate_startup_settings(
        config,
        args,
        voice_enabled=voice_enabled,
        dashboard_enabled=dashboard_enabled,
        headless=headless,
        shutdown_timeout=shutdown_timeout,
    )
    for warning in validation.warnings:
        log.warning("Startup validation: %s", warning)
    if not validation.is_valid:
        for error in validation.errors:
            log.error("Startup validation failed: %s", error)
            _uprint(f"[ERROR] {error}", file=sys.stderr)
        return ExitCode.CONFIG_ERROR

    loop = asyncio.get_running_loop()
    _install_loop_exception_handler(loop, log)
    shutdown = shutdown_cls(loop)
    shutdown.install_signal_handlers()

    controller = None
    dashboard: DashboardRuntime | None = None
    health_report: Any | None = None
    exit_code = ExitCode.OK
    phase = "startup"

    if headless and voice_enabled:
        log.warning(
            "Voice mode requested together with headless mode; headless mode wins"
        )

    log.info(
        "Starting Jarvis version=%s env=%s mode=%s voice=%s headless=%s dashboard=%s config=%s",
        version,
        environment,
        runtime_mode,
        voice_enabled,
        headless,
        dashboard_enabled,
        config_path,
    )

    if getattr(args, "health_check", False):
        light_report = run_lightweight_health_check(config)
        log.info(
            "Lightweight health check complete: is_healthy=%s",
            light_report.is_healthy,
        )
        _log_health_report(log, light_report, prefix="Lightweight")
        _uprint(light_report.summary())
        has_failures = bool(getattr(light_report, "has_failures", False))
        if has_failures:
            if bool(getattr(args, "strict_health", False)):
                log.error("Health check failed in strict mode")
                return ExitCode.STARTUP_ERROR
            log.warning("Health check reported failures")
        return ExitCode.OK

    preflight_report = run_lightweight_health_check(config)
    _log_health_report(log, preflight_report, prefix="Preflight")
    if bool(getattr(args, "strict_health", False)) and bool(
        getattr(preflight_report, "has_failures", False)
    ):
        log.error("Preflight health check failed in strict mode")
        return ExitCode.STARTUP_ERROR

    try:
        controller_cls = _load_controller_class()
        controller = controller_cls(config=config, voice=voice_enabled)
        _load_integrations(controller, config, log)

        _safe_audit(
            logger_mod,
            "startup",
            {
                "config": str(config_path),
                "environment": environment,
                "voice": voice_enabled,
                "headless": headless,
                "dashboard": dashboard_enabled,
                "mode": runtime_mode,
            },
            log,
        )

        await controller.start()

        if shutdown._event.is_set():
            return ExitCode.OK

        verbose_health = not headless
        health_report = _run_startup_health_check(controller, verbose=verbose_health)
        if getattr(args, "strict_health", False) and bool(
            getattr(health_report, "has_failures", False)
        ):
            log.error(
                "Startup health check reported failures and strict mode is enabled"
            )
            return ExitCode.STARTUP_ERROR

        if dashboard_enabled:
            host, port = _resolve_dashboard_binding(config, args)
            dashboard = DashboardRuntime(
                host=host,
                port=port,
                log=log,
            )
            await dashboard.start(controller, health_report=health_report)
            _uprint(f"Dashboard: http://{host}:{port}")

        if shutdown._event.is_set():
            return ExitCode.OK

        phase = "runtime"
        exit_code = await _run_runtime_loop(
            controller,
            shutdown,
            headless=headless,
            log=log,
        )

    except asyncio.CancelledError:
        request_shutdown = getattr(shutdown, "request_shutdown", None)
        if callable(request_shutdown):
            with contextlib.suppress(Exception):
                request_shutdown("cancelled")
        log.info("Main task cancelled during %s", phase)
        exit_code = ExitCode.OK
    except Exception:
        if phase == "startup":
            log.critical("Startup failure", exc_info=True)
            exit_code = ExitCode.STARTUP_ERROR
        else:
            log.critical("Unhandled runtime failure", exc_info=True)
            exit_code = ExitCode.GENERIC_ERROR
    finally:
        if dashboard is not None:
            await dashboard.stop(timeout=min(5.0, shutdown_timeout))

        if controller is not None:
            try:
                await asyncio.wait_for(
                    controller.shutdown(),
                    timeout=shutdown_timeout,
                )
                log.info("Controller shut down cleanly")
            except asyncio.TimeoutError:
                log.error(
                    "Controller shutdown timed out after %.1f seconds",
                    shutdown_timeout,
                )
                exit_code = ExitCode.GENERIC_ERROR
            except asyncio.CancelledError:
                log.info("Controller shutdown cancelled")
            except Exception:
                log.exception("Error during controller shutdown")
                exit_code = ExitCode.GENERIC_ERROR

        if controller is not None:
            summary_fn = getattr(controller, "session_summary", None)
            session_summary: dict[str, Any] = {}
            if callable(summary_fn):
                try:
                    session_summary = summary_fn() or {}
                except Exception:
                    log.debug("Failed to collect session summary", exc_info=True)
            payload = {
                "exit_code": exit_code,
                "phase": phase,
                "session_id": getattr(controller, "session_id", None),
                "summary": session_summary,
            }
        else:
            payload = {
                "exit_code": exit_code,
                "phase": phase,
            }
        _safe_audit(logger_mod, "shutdown", payload, log)

    return exit_code


__all__ = ["async_run"]




# --- FILE: core/runtime/import_validator.py ---

"""
Import Validator, Safe Import Utility, Dependency Scanner, and Runtime Protection Wrapper.
Prevents unhandled ModuleNotFoundErrors and circular dependency failures from crashing the runtime.
"""

# internal import removed: from __future__ import annotations

import ast
import importlib
import importlib.util
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar, cast

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


# =====================================================================
# 1. Safe Import Utility & Fallback Wrapper
# =====================================================================

class FallbackMock:
    """Mock object that acts as a fallback for missing modules, logging warnings when accessed."""

    def __init__(self, name: str, reason: str = "") -> None:
        self.__name = name
        self.__reason = reason

    def __getattr__(self, item: str) -> Any:
        logger.warning(
            "Accessing attribute %r on missing/mock module %r (Reason: %s)",
            item, self.__name, self.__reason
        )
        return FallbackMock(f"{self.__name}.{item}", self.__reason)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        logger.warning(
            "Calling missing/mock module/function %r (Reason: %s)",
            self.__name, self.__reason
        )
        return None


def safe_import(module_name: str, fallback_obj: Any = None) -> Any:
    """
    Attempt to import a module. Returns the imported module,
    or a fallback mock object if the module is missing or fails to load.
    """
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        logger.warning(
            "Safe import failed for module %r, utilizing fallback/mock. Error: %s",
            module_name, exc
        )
        if fallback_obj is not None:
            return fallback_obj
        return FallbackMock(module_name, str(exc))


# =====================================================================
# 2. Runtime Protection Wrapper / Crash Boundary
# =====================================================================

def protect_runtime(fallback_value: Any = None) -> Callable[[F], F]:
    """
    Decorator to wrap sync or async functions in a runtime safety boundary.
    Catches all exceptions, logs them, and returns a fallback value instead of crashing.
    """
    def decorator(func: F) -> F:
        import asyncio

        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    return await func(*args, **kwargs)
                except Exception as exc:
                    logger.error(
                        "Runtime crash prevented in async function %r: %s",
                        func.__name__, exc, exc_info=True
                    )
                    return fallback_value
            return cast(F, async_wrapper)
        else:
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    logger.error(
                        "Runtime crash prevented in sync function %r: %s",
                        func.__name__, exc, exc_info=True
                    )
                    return fallback_value
            return cast(F, sync_wrapper)

    return decorator


# =====================================================================
# 3. Missing Module Detector & Dependency Scanner (AST Based)
# =====================================================================

@dataclass
class ImportDiagnostic:
    file_path: Path
    line_number: int
    raw_import_string: str
    target_module: str
    is_relative: bool
    status: str  # 'OK', 'FAIL', 'CIRCULAR'
    error_message: str = ""


class DependencyScanner:
    """Scans codebase python files, extracts imports via AST, and validates they resolve."""

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir.resolve()

    def scan_project(self) -> list[ImportDiagnostic]:
        root_dir_str = str(self.root_dir)
        if root_dir_str not in sys.path:
            sys.path.insert(0, root_dir_str)
        diagnostics: list[ImportDiagnostic] = []
        for root, _, files in os.walk(self.root_dir):
            # Ignore environment and git directories
            if any(folder in root for folder in ("jarvis_env", ".git", "__pycache__", ".venv", "venv", ".mypy_cache")):
                continue

            for file in files:
                if file.endswith(".py"):
                    file_path = Path(root) / file
                    diagnostics.extend(self._scan_file(file_path))

        return diagnostics

    def _scan_file(self, file_path: Path) -> list[ImportDiagnostic]:
        file_diagnostics: list[ImportDiagnostic] = []
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(content, filename=str(file_path))
        except Exception as exc:
            logger.error("Failed to parse AST for %s: %s", file_path, exc)
            return file_diagnostics

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    diag = self._validate_import(file_path, node.lineno, alias.name, f"import {alias.name}", False)
                    file_diagnostics.append(diag)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    is_relative = node.level > 0
                    diag = self._validate_import(
                        file_path,
                        node.lineno,
                        node.module,
                        f"from {'.' * node.level}{node.module} import ...",
                        is_relative,
                        level=node.level
                    )
                    file_diagnostics.append(diag)

        return file_diagnostics

    def _validate_import(
        self,
        file_path: Path,
        lineno: int,
        module_name: str,
        raw_str: str,
        is_relative: bool,
        level: int = 0
    ) -> ImportDiagnostic:
        # Determine fully qualified module name
        fq_name = module_name
        if is_relative:
            # Resolve relative import to package root path
            rel_package = file_path.parent
            for _ in range(level - 1):
                rel_package = rel_package.parent
            
            # Match package folder structure to import paths
            parts = []
            curr = rel_package
            while curr != self.root_dir and curr != curr.parent:
                parts.append(curr.name)
                curr = curr.parent
            parts.reverse()
            fq_name = ".".join(parts + [module_name]) if parts else module_name

        try:
            # Attempt static resolution or spec checking to avoid executing module at top level
            spec = None
            # Check sys.modules cache
            if fq_name in sys.modules:
                spec = getattr(sys.modules[fq_name], "__spec__", None)
            
            if spec is None:
                spec = importlib.util.find_spec(fq_name)

            if spec is None:
                # Try finding as nested file/package in sys.path
                raise ModuleNotFoundError(f"No spec found for module {fq_name}")

            return ImportDiagnostic(
                file_path=file_path,
                line_number=lineno,
                raw_import_string=raw_str,
                target_module=fq_name,
                is_relative=is_relative,
                status="OK"
            )
        except Exception as exc:
            return ImportDiagnostic(
                file_path=file_path,
                line_number=lineno,
                raw_import_string=raw_str,
                target_module=fq_name,
                is_relative=is_relative,
                status="FAIL",
                error_message=str(exc)
            )


# =====================================================================
# 4. Startup Validator & Import Health Checker
# =====================================================================

class StartupValidator:
    """Verifies that all core controllers, tools, and memory submodules can resolve imports."""

    CRITICAL_MODULES = [
        "core.controller_v2",
        "core.controller.intents",
        "core.controller.intent_router",
        "core.controller.services",
        "core.controller.request_rules",
        "core.controller.web_search",
        "core.tools.builtin_tools",
        "core.tools.web_tools",
        "core.memory.hybrid_memory",
        "core.memory.semantic_memory",
        "core.runtime.paths",
        "core.runtime.bootstrap",
    ]

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir

    def run_preflight_checks(self) -> dict[str, Any]:
        """Perform preflight checks on critical imports and returns a summary health report."""
        root_dir_str = str(self.root_dir)
        if root_dir_str not in sys.path:
            sys.path.insert(0, root_dir_str)
        report: dict[str, Any] = {
            "status": "GREEN",
            "passed": [],
            "failed": [],
        }

        for module in self.CRITICAL_MODULES:
            try:
                importlib.util.find_spec(module)
                # Test loading
                importlib.import_module(module)
                report["passed"].append(module)
            except Exception as exc:
                logger.error("Preflight validation failed for %r: %s", module, exc)
                report["failed"].append({"module": module, "error": str(exc)})
                report["status"] = "RED"

        return report

    def generate_dependency_graph(self) -> dict[str, list[str]]:
        """Static AST scan to map modules to their dependencies."""
        scanner = DependencyScanner(self.root_dir)
        diags = scanner.scan_project()

        graph: dict[str, list[str]] = {}
        for diag in diags:
            # Map source relative path to target imports
            rel_src = str(diag.file_path.relative_to(self.root_dir)).replace(os.sep, ".")
            if rel_src.endswith(".py"):
                rel_src = rel_src[:-3]
            
            if rel_src not in graph:
                graph[rel_src] = []
            
            if diag.target_module not in graph[rel_src]:
                graph[rel_src].append(diag.target_module)

        return graph


def run_diagnostics(root_dir: Path) -> None:
    """Runs a complete diagnostics scan and prints it out."""
    root_dir_str = str(root_dir)
    if root_dir_str not in sys.path:
        sys.path.insert(0, root_dir_str)
    print("====================================================")
    print("Python Runtime Recovery - Preflight Diagnostics Scan")
    print("====================================================")

    validator = StartupValidator(root_dir)
    preflight = validator.run_preflight_checks()

    print(f"\nCritical Modules Preflight: {preflight['status']}")
    print(f"Passed: {len(preflight['passed'])} modules")
    if preflight["failed"]:
        print(f"Failed: {len(preflight['failed'])} modules")
        for fail in preflight["failed"]:
            print(f"  - {fail['module']}: {fail['error']}")

    print("\nAST Dependency Scanner Report:")
    scanner = DependencyScanner(root_dir)
    diags = scanner.scan_project()
    failed_imports = [d for d in diags if d.status == "FAIL"]

    if not failed_imports:
        print("  [OK] No missing dependencies detected via static analysis.")
    else:
        print(f"  [WARN] Found {len(failed_imports)} broken imports in project files:")
        for diag in failed_imports:
            rel_path = diag.file_path.relative_to(root_dir)
            print(f"    - {rel_path}:{diag.line_number} -> Failed to resolve: {diag.target_module} ({diag.error_message})")

    print("====================================================")


# main block removed: if __name__ == "__main__":
# main block removed:     # Resolve project root relative to this file
# main block removed:     proj_root = Path(__file__).resolve().parents[2]
# main block removed:     run_diagnostics(proj_root)




# --- FILE: core/security/auth.py ---

# internal import removed: from __future__ import annotations

import contextlib
import base64
import hashlib
import hmac
import os
import secrets
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import bcrypt
except Exception:  # pragma: no cover - optional optimized hasher
    bcrypt = None  # type: ignore[assignment]


SESSION_COOKIE = "jarvis_session"
CSRF_COOKIE = "jarvis_csrf"
SESSION_TTL_S = 60 * 60 * 12
PBKDF2_ROUNDS = 260_000


@dataclass(frozen=True)
class AuthUser:
    username: str
    is_admin: bool = True


class AuthManager:
    def __init__(self, db_path: str | Path, secret_key: str | None = None) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.secret_key = secret_key or os.environ.get("JARVIS_SECRET_KEY", "")
        if not self.secret_key:
            self.secret_key = "development-only-secret"
            import logging
            logging.getLogger("Jarvis.Security").warning(
                "Using fallback secret key 'development-only-secret'. "
                "For security-sensitive environments, please set the JARVIS_SECRET_KEY environment variable!"
            )
        self._init_db()

    @contextlib.contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password_hash TEXT NOT NULL,
                    is_admin INTEGER NOT NULL DEFAULT 1,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS api_tokens (
                    token_hash TEXT PRIMARY KEY,
                    label TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    last_used_at REAL
                )
                """
            )

    def bootstrap_admin_from_env(self) -> bool:
        username = os.environ.get("JARVIS_ADMIN_USER", "").strip()
        password = os.environ.get("JARVIS_ADMIN_PASSWORD", "")
        if not username or not password:
            return False
        if self.user_count() > 0:
            return False
        self.create_user(username, password, is_admin=True)
        return True

    def user_count(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM users").fetchone()[0])

    def create_user(self, username: str, password: str, *, is_admin: bool = True) -> None:
        username = username.strip()
        if not username:
            raise ValueError("username is required")
        if "|" in username:
            raise ValueError("Username cannot contain the pipe character '|'")
        if any(c.isspace() for c in username):
            raise ValueError("Username cannot contain spaces or whitespace")
        if len(password) < 12:
            raise ValueError("password must be at least 12 characters")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO users(username, password_hash, is_admin, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (username, self.hash_password(password), int(is_admin), time.time()),
            )

    def authenticate(self, username: str, password: str) -> AuthUser | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT username, password_hash, is_admin FROM users WHERE username = ?",
                (username.strip(),),
            ).fetchone()
        if row is None or not self.verify_password(password, str(row["password_hash"])):
            return None
        return AuthUser(username=str(row["username"]), is_admin=bool(row["is_admin"]))

    def create_api_token(self, label: str = "automation") -> str:
        token = secrets.token_urlsafe(32)
        digest = self._token_hash(token)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO api_tokens(token_hash, label, created_at) VALUES (?, ?, ?)",
                (digest, label, time.time()),
            )
        return token

    def verify_api_token(self, token: str) -> bool:
        if not token:
            return False
        digest = self._token_hash(token)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT token_hash FROM api_tokens WHERE token_hash = ?",
                (digest,),
            ).fetchone()
            if row is None:
                return False
            conn.execute(
                "UPDATE api_tokens SET last_used_at = ? WHERE token_hash = ?",
                (time.time(), digest),
            )
        return True

    def sign_session(self, user: AuthUser) -> str:
        expires = int(time.time() + SESSION_TTL_S)
        nonce = secrets.token_urlsafe(12)
        payload = f"{user.username}|{int(user.is_admin)}|{expires}|{nonce}"
        sig = self._sign(payload)
        raw = f"{payload}|{sig}".encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii")

    def verify_session(self, token: str) -> AuthUser | None:
        try:
            raw = base64.urlsafe_b64decode(token.encode("ascii")).decode("utf-8")
            username, is_admin, expires, nonce, sig = raw.split("|", 4)
        except Exception:
            return None
        del nonce
        payload = "|".join([username, is_admin, expires, raw.split("|", 4)[3]])
        if not hmac.compare_digest(sig, self._sign(payload)):
            return None
        if int(expires) < int(time.time()):
            return None
        return AuthUser(username=username, is_admin=bool(int(is_admin)))

    def make_csrf_token(self, session_token: str) -> str:
        nonce = secrets.token_urlsafe(18)
        sig = self._sign(f"{session_token}|{nonce}")
        return f"{nonce}.{sig}"

    def verify_csrf_token(self, session_token: str, csrf_token: str) -> bool:
        if "." not in csrf_token:
            return False
        nonce, sig = csrf_token.split(".", 1)
        expected = self._sign(f"{session_token}|{nonce}")
        return hmac.compare_digest(sig, expected)

    def hash_password(self, password: str) -> str:
        if bcrypt is not None:
            hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12))
            return "bcrypt$" + hashed.decode("ascii")
        salt = secrets.token_bytes(16)
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ROUNDS)
        return "pbkdf2_sha256${}${}${}".format(
            PBKDF2_ROUNDS,
            base64.b64encode(salt).decode("ascii"),
            base64.b64encode(digest).decode("ascii"),
        )

    def verify_password(self, password: str, password_hash: str) -> bool:
        if password_hash.startswith("bcrypt$") and bcrypt is not None:
            return bool(
                bcrypt.checkpw(
                    password.encode("utf-8"),
                    password_hash.removeprefix("bcrypt$").encode("ascii"),
                )
            )
        if password_hash.startswith("pbkdf2_sha256$"):
            _, rounds, salt_b64, digest_b64 = password_hash.split("$", 3)
            salt = base64.b64decode(salt_b64.encode("ascii"))
            expected = base64.b64decode(digest_b64.encode("ascii"))
            actual = hashlib.pbkdf2_hmac(
                "sha256",
                password.encode("utf-8"),
                salt,
                int(rounds),
            )
            return hmac.compare_digest(actual, expected)
        return False

    def _sign(self, payload: str) -> str:
        return hmac.new(
            self.secret_key.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _token_hash(self, token: str) -> str:
        return hmac.new(
            self.secret_key.encode("utf-8"),
            token.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()


def auth_db_from_config(config: Any) -> Path:
    try:
        raw = config.get("security", "auth_db", fallback="data/auth.db")
    except Exception:
        raw = "data/auth.db"
    path = Path(raw)
    if path.is_absolute():
        return path
    from core.runtime.bootstrap import PROJECT_ROOT

    return PROJECT_ROOT / path


__all__ = ["AuthManager", "AuthUser", "SESSION_COOKIE", "CSRF_COOKIE", "auth_db_from_config"]




# --- FILE: core/security/__init__.py ---

"""Security helpers for Jarvis production surfaces."""

# internal import removed: from core.security.auth import AuthManager

__all__ = ["AuthManager"]



############################################################
# API
############################################################

############################################################
# MAIN ENTRYPOINT
############################################################


# --- FILE: main.py ---

r"""
Production-ready Jarvis entry point.

Usage (cross-platform):
  python main.py
  python main.py --voice
  python main.py --gui
  python main.py --headless --gui
  python main.py --health-check
  python main.py --verify

Windows PowerShell convenience:
  .\Start.ps1
  .\Start.ps1 --voice
  .\Start.ps1 --gui
  .\Start.ps1 --headless --gui
  .\Start.ps1 --health-check
  .\Start.ps1 --verify
"""

# internal import removed: from __future__ import annotations

import asyncio
import signal
import sys
import traceback
import argparse
from collections.abc import Callable, Awaitable

# internal import removed: from core.runtime.bootstrap import (
# internal import removed:     ExitCode,
# internal import removed:     PROJECT_ROOT,
# internal import removed:     _ShutdownCoordinator,
# internal import removed:     _bootstrap,
# internal import removed:     _uprint,
# internal import removed:     apply_cli_overrides,
# internal import removed:     core_runtime_bootstrap_load_config,
# internal import removed:     parse_args,
# internal import removed: )
# internal import removed: from core.runtime.entrypoint import async_run

ArgsParser = Callable[[list[str] | None], argparse.Namespace]
AsyncEntry = Callable[[argparse.Namespace], Awaitable[int]]


async def async_main(args: argparse.Namespace) -> int:
    return await async_run(args, shutdown_cls=_ShutdownCoordinator)


def run_entrypoint(
    *,
    parse_args_fn: ArgsParser = parse_args,
    async_entry_fn: AsyncEntry = async_main,
    argv: list[str] | None = None,
    interrupted_message: str = "Interrupted - goodbye.",
) -> int:
    args = parse_args_fn(argv)

    try:
        return asyncio.run(async_entry_fn(args))  # type: ignore[arg-type]
    except KeyboardInterrupt:
        _uprint(interrupted_message, file=sys.stderr)
        return ExitCode.OK
    except Exception:
        _bootstrap.critical(
            "Unhandled top-level exception:\n%s",
            "".join(traceback.format_exception(*sys.exc_info())),
        )
        return ExitCode.GENERIC_ERROR


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(
        run_entrypoint(
            parse_args_fn=parse_args,
            async_entry_fn=async_main,
            argv=argv,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "ExitCode",
    "PROJECT_ROOT",
    "_ShutdownCoordinator",
    "_bootstrap",
    "_uprint",
    "apply_cli_overrides",
    "async_main",
    "core_runtime_bootstrap_load_config",
    "main",
    "parse_args",
    "run_entrypoint",
    "signal",
    "sys",
]


