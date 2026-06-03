from __future__ import annotations

import json
import logging
from pathlib import Path
from configparser import ConfigParser

from core.logging import logger as jarvis_logger


def test_logging_setup_and_levels(tmp_path: Path):
    app_log = tmp_path / "app.log"
    audit_log = tmp_path / "audit.jsonl"
    
    cfg = ConfigParser()
    cfg.add_section("logging")
    cfg.set("logging", "level", "INFO")
    cfg.set("logging", "app_file", str(app_log))
    cfg.set("logging", "audit_file", str(audit_log))
    
    jarvis_logger.setup(cfg)
    
    # Get logger Jarvis.Test
    log = logging.getLogger("Jarvis.Test")
    log.info("This is an info message")
    log.debug("This is a debug message")
    
    # Read app_log and verify
    assert app_log.exists()
    content = app_log.read_text(encoding="utf-8")
    assert "This is an info message" in content
    assert "This is a debug message" not in content


def test_logging_gaps_with_non_jarvis_loggers(tmp_path: Path):
    app_log = tmp_path / "app_gap.log"
    audit_log = tmp_path / "audit_gap.jsonl"
    
    cfg = ConfigParser()
    cfg.add_section("logging")
    cfg.set("logging", "level", "INFO")
    cfg.set("logging", "app_file", str(app_log))
    cfg.set("logging", "audit_file", str(audit_log))
    
    jarvis_logger.setup(cfg)
    
    # Get a logger mimicking core.llm.client
    log = logging.getLogger("core.llm.client")
    log.info("LLM client initialized")
    
    # Verify that it is written to the log file
    assert app_log.exists()
    content = app_log.read_text(encoding="utf-8")
    assert "LLM client initialized" in content


def test_json_formatter_with_ids_and_exceptions(tmp_path: Path):
    app_log = tmp_path / "app_json.log"
    audit_log = tmp_path / "audit_json.jsonl"
    
    cfg = ConfigParser()
    cfg.add_section("logging")
    cfg.set("logging", "level", "INFO")
    cfg.set("logging", "app_file", str(app_log))
    cfg.set("logging", "audit_file", str(audit_log))
    
    jarvis_logger.setup(cfg)
    
    log = logging.getLogger("Jarvis.Task")
    
    # Log with extra attributes (trace_id and task_id)
    log.info("Starting subtask", extra={"trace_id": "trace-101", "task_id": "task-202"})
    
    # Log exception with extra attributes
    try:
        raise ValueError("Something went wrong in the task")
    except ValueError:
        log.exception("Task error occurred", extra={"trace_id": "trace-101", "task_id": "task-202"})
        
    assert app_log.exists()
    lines = app_log.read_text(encoding="utf-8").splitlines()
    
    # The first line should be JSON
    envelope1 = json.loads(lines[0])
    assert envelope1["trace_id"] == "trace-101"
    assert envelope1["task_id"] == "task-202"
    assert envelope1["metadata"]["message"] == "Starting subtask"
    
    # The second line should be JSON containing the exception details and stack trace
    envelope2 = json.loads(lines[1])
    assert envelope2["trace_id"] == "trace-101"
    assert envelope2["task_id"] == "task-202"
    assert envelope2["metadata"]["message"] == "Task error occurred"
    assert "stack_trace" in envelope2
    assert "ValueError: Something went wrong" in envelope2["stack_trace"]


def test_audit_log_hashing_and_utf8(tmp_path: Path):
    from core.logging.logger import AuditLog
    audit_file = tmp_path / "test_audit.jsonl"
    
    # Initialize AuditLog
    audit = AuditLog(str(audit_file))
    
    # Write event 1 (with unicode)
    hash1 = audit.write("event_one", {"msg": "Hello World", "unicode": "🚀-漢-字"})
    assert hash1 is not None
    assert len(hash1) == 64
    
    # Write event 2 (with nested dict & unicode)
    hash2 = audit.write("event_two", {"nested": {"unicode_key": "🌟", "val": 42}})
    assert hash2 is not None
    assert len(hash2) == 64
    
    # Stop the worker thread to flush queues
    audit.stop()
    
    # Check that file exists and contents are written in UTF-8
    assert audit_file.exists()
    content = audit_file.read_text(encoding="utf-8")
    assert "🚀-漢-字" in content
    assert "🌟" in content
    
    # Check newline is LF (\n) and not translated to CRLF
    # Since we read using read_bytes(), we can look for CRLF bytes.
    raw_bytes = audit_file.read_bytes()
    assert b"\r\n" not in raw_bytes
    assert b"\n" in raw_bytes
    
    # Verify the audit log file
    # We must instantiate a new AuditLog to verify, or run verify() directly
    verifier = AuditLog(str(audit_file))
    ok, count, err = verifier.verify()
    assert ok is True
    assert count == 2
    assert err == ""
    verifier.stop()
    
    # Induce a hash mismatch corruption
    # We modify the hash of the second line in the file
    lines = content.splitlines()
    corrupted_data = json.loads(lines[1])
    corrupted_data["hash"] = "a" * 64
    lines[1] = json.dumps(corrupted_data, ensure_ascii=False)
    audit_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    
    corrupt_verifier = AuditLog(str(audit_file))
    ok, count, err = corrupt_verifier.verify()
    assert ok is False
    assert "Hash chain mismatch" in err
    corrupt_verifier.stop()

    # Induce corruption in the payload structure
    # We modify the payload of the first line
    corrupted_data_2 = json.loads(lines[0])
    corrupted_data_2["payload"]["msg"] = "Modified msg"
    lines[0] = json.dumps(corrupted_data_2, ensure_ascii=False)
    audit_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    corrupt_verifier_2 = AuditLog(str(audit_file))
    ok, count, err = corrupt_verifier_2.verify()
    assert ok is False
    assert "Hash chain mismatch" in err
    corrupt_verifier_2.stop()


def test_console_and_formatter_utf8():
    import sys
    from core.logging.logger import JSONFormatter
    import logging
    
    # Ensure stdout/stderr reconfigure doesn't raise exception
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
                
    # Verify JSONFormatter handles non-ASCII characters correctly
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="Unicode message 🚀 漢 字",
        args=(),
        exc_info=None
    )
    # Set trace_id to trigger JSON envelope
    record.trace_id = "test-trace"
    record.task_id = "test-task"
    
    formatted_str = formatter.format(record)
    assert "Unicode message 🚀 漢 字" in formatted_str
    
    # Try to load as JSON to verify structure
    envelope = json.loads(formatted_str)
    assert envelope["trace_id"] == "test-trace"
    assert envelope["task_id"] == "test-task"
    assert envelope["metadata"]["message"] == "Unicode message 🚀 漢 字"
