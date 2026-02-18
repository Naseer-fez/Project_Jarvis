"""
core/audit_bridge.py
═════════════════════
Writes dashboard-compatible JSONL log files from Jarvis V5 events.

Files written to outputs/Jarvis-Session/:
  - execution_trace.jsonl
  - tool_observations.jsonl
  - risk_assessment_log.jsonl
  - agent_plan.jsonl
"""

import json
from datetime import datetime, timezone
from pathlib import Path

OUTPUT_DIR = Path("outputs/Jarvis-Session")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append(filename: str, session_id: str, data: dict):
    record = {
        "session_id": session_id,
        "timestamp": _now(),
        "data": data,
    }
    with open(OUTPUT_DIR / filename, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


class AuditBridge:
    def __init__(self, session_id: str):
        self.session_id = session_id
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def log_trace(self, goal: str, plan: dict, success: bool,
                  final_response: str, duration_seconds: float, stop_reason: str):
        _append("execution_trace.jsonl", self.session_id, {
            "goal": goal,
            "plan": plan,
            "success": success,
            "final_response": final_response,
            "duration_seconds": round(duration_seconds, 2),
            "stop_reason": stop_reason,
        })

    def log_tool(self, tool_name: str, execution_status: str,
                 duration_seconds: float, detail: str = ""):
        _append("tool_observations.jsonl", self.session_id, {
            "tool_name": tool_name,
            "execution_status": execution_status,
            "duration_seconds": round(duration_seconds, 3),
            "detail": detail,
        })

    def log_risk(self, tool: str, composite_score: float,
                 level: str, allowed: bool):
        _append("risk_assessment_log.jsonl", self.session_id, {
            "tool": tool,
            "composite_score": round(composite_score, 3),
            "level": level,
            "allowed": allowed,
        })

    def log_plan(self, goal: str, plan: dict):
        _append("agent_plan.jsonl", self.session_id, {
            "goal": goal,
            "feasible": plan.get("feasible", True),
            "steps": plan.get("steps", []),
            "reasoning": plan.get("reasoning", ""),
        })