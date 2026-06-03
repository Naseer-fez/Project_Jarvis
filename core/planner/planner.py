"""Asynchronous task planner to generate execution plans."""

from __future__ import annotations

import json
import logging
import re
import inspect
from typing import Any

from core.autonomy.risk_evaluator import RiskLevel, RiskEvaluator

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

    def _tool_schema(self) -> dict[str, list[dict[str, str]]]:
        tools = []
        if self.registry:
            for name in self.registry.registered_tools():
                cap = self.registry.get(name)
                desc = getattr(cap, "description", "") or f"Execute {name}"
                tools.append({"name": name, "description": desc})

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
                return await res
            return str(res)
        except Exception as exc:
            logger.error("LLM completion failed: %s", exc)
            return ""

    async def plan(self, user_input: str, context: str = "") -> dict[str, Any]:
        text = str(user_input or "").strip()
        raw = await self._call_ollama(self._build_prompt(text, context))

        if raw:
            parsed = self._parse_llm_plan(raw)
            if parsed is not None:
                return self._enrich_plan(text, parsed)
            return self._clarification_plan(text)

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
                    "params": {"param_key": "param_value"}
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
            "You MUST return a valid JSON object matching the following structure:\n"
            f"{json.dumps(schema_format, indent=2)}\n\n"
            "Return ONLY the strict JSON object. No explanations, no markdown block, no extra text."
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

        if "sort" in lowered or "organize" in lowered:
            return {
                "intent": text,
                "summary": "Sort and organize files in the workspace.",
                "confidence": 0.85,
                "steps": [
                    {
                        "id": 1,
                        "action": "sort_files",
                        "description": "Sort files in the workspace directory into categorized folders.",
                        "params": {"directory": "./workspace", "output_dir": "./workspace"},
                    }
                ],
                "clarification_needed": False,
                "clarification_prompt": "",
            }

        if "find" in lowered or "search file" in lowered:
            pattern = "*"
            words = text.split()
            for w in words:
                if "*" in w or "." in w:
                    pattern = w.strip("'\"")
                    break
            return {
                "intent": text,
                "summary": f"Search for files matching '{pattern}' in workspace.",
                "confidence": 0.85,
                "steps": [
                    {
                        "id": 1,
                        "action": "find_files",
                        "description": f"Find files matching pattern {pattern}.",
                        "params": {"pattern": pattern, "directory": "./workspace"},
                    }
                ],
                "clarification_needed": False,
                "clarification_prompt": "",
            }

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

        if any(
            word in lowered
            for word in (
                "search",
                "look up",
                "weather",
                "latest",
                "current",
                "today",
                "live",
                "internet",
                "online",
                "web",
                "news",
                "score",
                "stats",
                "toss",
                "ipl",
                "match",
            )
        ):
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
