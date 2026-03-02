"""
AuditLogger — append-only session artifact writer.
Logs: plans, execution traces, tool observations, risk scores, reflections.
All writes are append-only JSON files in outputs/.

Session 8 additions:
  - Log rotation: files >50 MB are gzipped before the next write.
  - Secret scrubbing: all text content is scrubbed before being written.
"""

import gzip
import json
import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("Jarvis.AuditLogger")

OUTPUTS_DIR = Path("./outputs/Jarvis-Session/")

# ── Secret scrubber ───────────────────────────────────────────────────────────

_SECRET_PATTERNS = [
    r'(?i)(password|passwd|pwd|token|secret|sid|api_key|apikey)\s*[=:]\s*\S+',
    r'[A-Za-z0-9]{32,}',   # long random strings (tokens)
]


def scrub_secrets(text: str) -> str:
    """Replace secrets and long random tokens with [REDACTED]."""
    for pattern in _SECRET_PATTERNS:
        text = re.sub(pattern, "[REDACTED]", text)
    return text


# ── Log rotation helper ───────────────────────────────────────────────────────

_MAX_LOG_BYTES = 50 * 1024 * 1024   # 50 MB


def _rotate_if_needed(log_path: Path) -> None:
    """If *log_path* exists and exceeds the size limit, gzip it and remove the original."""
    if log_path.exists() and log_path.stat().st_size > _MAX_LOG_BYTES:
        ts = int(time.time())
        gz_path = Path(f"{log_path}.{ts}.gz")
        with open(log_path, "rb") as f_in:
            with gzip.open(gz_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(log_path)
        logger.info("Rotated audit log %s → %s", log_path.name, gz_path.name)


class AuditLogger:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.dir = OUTPUTS_DIR
        self.dir.mkdir(parents=True, exist_ok=True)
        self._session_start = time.time()
        logger.info(f"Audit logger initialized: {self.dir}")

    def _append(self, filename: str, data: Any):
        path = self.dir / filename
        _rotate_if_needed(path)
        # Scrub secrets from string representation of data
        raw = json.dumps(data)
        scrubbed_raw = scrub_secrets(raw)
        try:
            scrubbed_data = json.loads(scrubbed_raw)
        except json.JSONDecodeError:
            scrubbed_data = {"_raw": scrubbed_raw}
        entry = {"session_id": self.session_id, "timestamp": time.time(), "data": scrubbed_data}
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def log_plan(self, plan_dict: dict):
        self._append("agent_plan.jsonl", plan_dict)

    def log_trace(self, trace_dict: dict):
        self._append("execution_trace.jsonl", trace_dict)

    def log_observation(self, obs_dict: dict):
        self._append("tool_observations.jsonl", obs_dict)

    def log_risk(self, risk_dict: dict):
        self._append("risk_assessment_log.jsonl", risk_dict)

    def log_voice_interaction(self, role: str, content: str):
        path = self.dir / "voice_interaction_log.txt"
        _rotate_if_needed(path)
        scrubbed = scrub_secrets(content)
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] {role.upper()}: {scrubbed}\n")

    def log_autonomy_decision(self, tool: str, allowed: bool, reason: str):
        self._append("autonomy_decisions.jsonl", {"tool": tool, "allowed": allowed, "reason": reason})

    def log_reflection(self, reflection: str):
        self._append("final_reflection.jsonl", {"reflection": reflection})

    def log_memory_snapshot(self, snapshot: dict):
        path = self.dir / "memory_snapshot.json"
        _rotate_if_needed(path)
        raw = json.dumps(snapshot, indent=2)
        scrubbed = scrub_secrets(raw)
        try:
            scrubbed_snapshot = json.loads(scrubbed)
        except json.JSONDecodeError:
            scrubbed_snapshot = {"_raw": scrubbed}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(scrubbed_snapshot, f, indent=2)
