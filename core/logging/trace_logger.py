"""
Structured trace logger for decision/action/error observability.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any


class TraceLogger:
    def __init__(self, output_dir: str | Path = "outputs/Jarvis-Session", session_id: str | None = None) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id or f"session_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        self._lock = threading.Lock()

    def _append(self, filename: str, data: dict[str, Any]) -> None:
        payload = {
            "session_id": self.session_id,
            "timestamp": time.time(),
            "data": data,
        }
        with self._lock:
            with (self.output_dir / filename).open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def decision(self, event: str, **data: Any) -> None:
        self._append("decision_trace.jsonl", {"event": event, **data})

    def action(self, action: str, status: str, **data: Any) -> None:
        self._append("action_trace.jsonl", {"action": action, "status": status, **data})

    def error(self, source: str, message: str, **data: Any) -> None:
        self._append("error_trace.jsonl", {"source": source, "message": message, **data})

    def transcription(self, text: str, is_final: bool, confidence: float | None = None) -> None:
        payload = {"text": text, "is_final": bool(is_final)}
        if confidence is not None:
            payload["confidence"] = float(confidence)
        self._append("voice_transcription.jsonl", payload)

    def state(self, state: str, source: str = "controller") -> None:
        self._append("state_trace.jsonl", {"state": state, "source": source})

