"""
AuditLogger — append-only session artifact writer.
Logs: plans, execution traces, tool observations, risk scores, reflections.
All writes are append-only JSON files in outputs/.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("Jarvis.AuditLogger")

OUTPUTS_DIR = Path("./outputs/Jarvis-Session/")


class AuditLogger:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.dir = OUTPUTS_DIR
        self.dir.mkdir(parents=True, exist_ok=True)
        self._session_start = time.time()
        logger.info(f"Audit logger initialized: {self.dir}")

    def _append(self, filename: str, data: Any):
        path = self.dir / filename
        entry = {"session_id": self.session_id, "timestamp": time.time(), "data": data}
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
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] {role.upper()}: {content}\n")

    def log_autonomy_decision(self, tool: str, allowed: bool, reason: str):
        self._append("autonomy_decisions.jsonl", {"tool": tool, "allowed": allowed, "reason": reason})

    def log_reflection(self, reflection: str):
        self._append("final_reflection.jsonl", {"reflection": reflection})

    def log_memory_snapshot(self, snapshot: dict):
        path = self.dir / "memory_snapshot.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2)
