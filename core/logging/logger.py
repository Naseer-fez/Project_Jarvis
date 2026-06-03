"""Application logging and append-only audit log support."""

from __future__ import annotations

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

            trace_id = getattr(record, "trace_id", None)
            task_id = getattr(record, "task_id", None)
            if not trace_id and not task_id:
                return redact_sensitive_data(super().format(record))

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

            if "message" not in envelope["metadata"]:
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
        self.queue.put(dummy_record)
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


__all__ = ["AuditLog", "_audit", "audit", "get", "get_logger", "setup", "verify_audit", "flush"]
