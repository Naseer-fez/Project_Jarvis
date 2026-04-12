"""Lightweight planner with a stable schema for legacy and runtime paths."""

from __future__ import annotations

import copy
import json
import re
from typing import Any

from core.autonomy.risk_evaluator import RiskLevel, RiskEvaluator

SYSTEM_TOOL_SCHEMA = {
    "tools": [
        {"name": "list_directory", "description": "List files in a safe directory."},
        {"name": "read_file", "description": "Read a text file."},
        {"name": "write_file", "description": "Write a text file."},
        {"name": "delete_file", "description": "Delete a text file."},
        {"name": "launch_application", "description": "Launch a desktop app."},
        {"name": "execute_shell", "description": "Run a shell command."},
        {"name": "capture_screen", "description": "Capture the current screen."},
        {"name": "capture_region", "description": "Capture a screen region."},
        {"name": "find_text_on_screen", "description": "Find text on screen."},
        {"name": "describe_screen", "description": "Describe the current screen."},
        {"name": "get_active_window", "description": "Get the active window."},
        {"name": "click", "description": "Click on screen coordinates."},
        {"name": "double_click", "description": "Double-click on screen coordinates."},
        {"name": "right_click", "description": "Right-click on screen coordinates."},
        {"name": "type_text", "description": "Type text into the active window."},
        {"name": "hotkey", "description": "Send a keyboard shortcut."},
        {"name": "web_search", "description": "Search the web."},
        {"name": "web_scrape", "description": "Read a web page."},
        {"name": "memory_write", "description": "Store a fact in memory."},
        {"name": "memory_read", "description": "Recall stored facts."},
        {"name": "speak", "description": "Speak a response."},
        {"name": "display", "description": "Display a response."},
    ]
}

_GUI_TOOL_NAMES = {"click", "double_click", "right_click", "type_text", "hotkey"}


def _strip_planner_artifacts(raw: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.DOTALL).strip()
    cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", cleaned)
    cleaned = re.sub(r"\n?```$", "", cleaned)
    return cleaned.strip()


class TaskPlanner:
    def __init__(self, config=None) -> None:
        self.config = config
        self.risk_evaluator = RiskEvaluator(config)

    def _tool_schema(self) -> dict[str, list[dict[str, str]]]:
        schema = copy.deepcopy(SYSTEM_TOOL_SCHEMA)
        allow_gui = False
        try:
            allow_gui = self.config.getboolean(
                "execution",
                "allow_gui_automation",
                fallback=False,
            )
        except Exception:
            allow_gui = False

        if not allow_gui:
            schema["tools"] = [
                tool for tool in schema["tools"] if tool["name"] not in _GUI_TOOL_NAMES
            ]
        return schema

    def _call_ollama(self, prompt: str) -> str:
        del prompt
        return ""

    def plan(self, user_input: str, context: str = "") -> dict[str, Any]:
        text = str(user_input or "").strip()
        raw = self._call_ollama(self._build_prompt(text, context))

        if raw:
            parsed = self._parse_llm_plan(raw)
            if parsed is not None:
                return self._enrich_plan(text, parsed)
            return self._clarification_plan(text)

        return self._enrich_plan(text, self._fallback_plan(text))

    def _build_prompt(self, user_input: str, context: str) -> str:
        return (
            "Return strict JSON for a safe action plan.\n"
            f"User request: {user_input}\n"
            f"Context: {context}\n"
            f"Available tools: {json.dumps(self._tool_schema())}"
        )

    def _parse_llm_plan(self, raw: str) -> dict[str, Any] | None:
        try:
            payload = json.loads(_strip_planner_artifacts(raw))
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    def _fallback_plan(self, text: str) -> dict[str, Any]:
        lowered = text.lower()

        if lowered.startswith("remember ") and " is " in lowered:
            key, value = text[9:].split(" is ", 1)
            return {
                "intent": text,
                "summary": f"Store the fact that {key.strip()} is {value.strip()}.",
                "confidence": 0.9,
                "steps": [
                    {
                        "id": 1,
                        "action": "memory_write",
                        "description": "Store a user fact.",
                        "params": {"key": key.strip(), "value": value.strip()},
                    }
                ],
                "clarification_needed": False,
                "clarification_prompt": "",
            }

        if any(word in lowered for word in ("search", "look up", "weather", "latest")):
            return {
                "intent": text,
                "summary": "Search for the requested information.",
                "confidence": 0.6,
                "steps": [
                    {
                        "id": 1,
                        "action": "web_search",
                        "description": "Search the web for the answer.",
                        "params": {"query": text},
                    }
                ],
                "clarification_needed": False,
                "clarification_prompt": "",
            }

        if "write" in lowered and "note" in lowered:
            return {
                "intent": text,
                "summary": "Write a note to a file.",
                "confidence": 0.7,
                "steps": [
                    {
                        "id": 1,
                        "action": "file_write",
                        "description": "Write the requested note.",
                        "params": {"path": "workspace/notes.txt", "content": text},
                    }
                ],
                "clarification_needed": False,
                "clarification_prompt": "",
            }

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
        normalized = {
            "intent": str(plan.get("intent", text)),
            "summary": str(plan.get("summary", "")).strip(),
            "confidence": float(plan.get("confidence", 0.0) or 0.0),
            "steps": self._normalize_steps(plan.get("steps", [])),
            "clarification_needed": bool(plan.get("clarification_needed", False)),
            "clarification_prompt": str(plan.get("clarification_prompt", "") or ""),
        }

        tools_required = [
            step["action"] for step in normalized["steps"] if step.get("action")
        ]
        risk = self.risk_evaluator.evaluate(tools_required)
        risk_label = risk.level.label().lower()
        if risk.level >= RiskLevel.CRITICAL:
            risk_label = "critical"
        elif risk.level >= RiskLevel.HIGH:
            risk_label = "high"
        elif risk.level >= RiskLevel.MEDIUM:
            risk_label = "medium"
        else:
            risk_label = "low"

        normalized["tools_required"] = tools_required
        normalized["risk_level"] = risk_label
        normalized["confirmation_required"] = risk.requires_confirmation
        return normalized

    def _normalize_steps(self, steps: Any) -> list[dict[str, Any]]:
        if not isinstance(steps, list):
            return []
        normalized = []
        for idx, step in enumerate(steps, start=1):
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


__all__ = ["SYSTEM_TOOL_SCHEMA", "TaskPlanner"]
