"""
core/planning/task_planner.py — LLM planner via Ollama (DeepSeek R1:8b).

Responsibilities:
  - Translate natural-language intent into a structured JSON plan
  - NEVER execute anything — pure planner
  - If the LLM produces invalid JSON or is uncertain, return a safe "unknown" plan
  - Plans are validated against a schema before being returned

Plan schema:
  {
    "intent":  str,           # echo of the user's request
    "summary": str,           # one-sentence human-readable plan summary
    "confidence": float,      # 0.0–1.0
    "steps": [
      {
        "id": int,
        "action": str,        # machine-readable action type
        "description": str,   # human-readable step description
        "params": { ... }     # action-specific parameters
      }
    ],
    "clarification_needed": bool,
    "clarification_prompt": str   # if clarification_needed
  }
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

import urllib.request
import urllib.error

_SYSTEM_PROMPT = """You are Jarvis, a local AI assistant. Your ONLY job is to produce a JSON plan.

RULES:
1. Respond ONLY with valid JSON — no markdown, no explanation, no preamble.
2. If you don't know how to do something, set clarification_needed=true and explain.
3. Never include actions that execute arbitrary shell commands, delete files, or bypass safety controls.
4. Never guess or hallucinate — if uncertain, say so in clarification_prompt.
5. Keep plans concise — 1 to 5 steps maximum.

JSON schema (return exactly this structure):
{
  "intent": "<echo user request>",
  "summary": "<one sentence what you will do>",
  "confidence": 0.0,
  "steps": [
    {
      "id": 1,
      "action": "<action_type>",
      "description": "<human readable>",
      "params": {}
    }
  ],
  "clarification_needed": false,
  "clarification_prompt": ""
}

Valid actions:
memory_read, memory_write, speak, display, status, recall, store_fact, health_check, vision_analyze,
file_read, file_write, system_stats, app_open, web_search, screen_capture, gui_click, gui_type,
gui_hotkey, serial_connect, serial_send, serial_disconnect

Never output these forbidden actions:
shell_exec, file_delete, registry_write, format_disk, wipe_disk
"""

_UNKNOWN_PLAN = {
    "intent": "",
    "summary": "I don't know how to do that safely.",
    "confidence": 0.0,
    "steps": [],
    "clarification_needed": True,
    "clarification_prompt": "I don't have a safe way to accomplish that request. Could you rephrase or give more detail?",
}


def _extract_json(text: str) -> str:
    """Extract the first JSON object from a string (strips markdown fences, thinking tags, etc.)."""
    # Remove DeepSeek <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Remove markdown fences
    text = re.sub(r"```(?:json)?", "", text)
    text = text.strip()

    # Find first { ... } block
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if start == -1:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                return text[start:i + 1]
    return text


def _validate_plan(plan: dict) -> dict:
    """Coerce plan to expected schema. Never raises."""
    return {
        "intent":               str(plan.get("intent", "")),
        "summary":              str(plan.get("summary", "No summary.")),
        "confidence":           float(plan.get("confidence", 0.0)),
        "steps":                plan.get("steps", []) if isinstance(plan.get("steps"), list) else [],
        "clarification_needed": bool(plan.get("clarification_needed", False)),
        "clarification_prompt": str(plan.get("clarification_prompt", "")),
    }


class TaskPlanner:
    def __init__(self, config) -> None:
        self._base_url = config.get("ollama", "base_url", fallback="http://localhost:11434")
        self._model    = config.get("ollama", "planner_model", fallback="deepseek-r1:8b")
        self._timeout  = int(config.get("ollama", "request_timeout_s", fallback="120"))

    def plan(self, intent: str, context: str = "") -> dict[str, Any]:
        """
        Generate a JSON plan for the given intent.
        Always returns a valid plan dict — never raises.
        """
        if not intent.strip():
            p = dict(_UNKNOWN_PLAN)
            p["intent"] = intent
            p["clarification_prompt"] = "Empty request received."
            return p

        prompt = intent
        if context:
            prompt = f"Context:\n{context}\n\nRequest: {intent}"

        try:
            raw = self._call_ollama(prompt)
        except Exception as exc:
            p = dict(_UNKNOWN_PLAN)
            p["intent"] = intent
            p["clarification_prompt"] = f"LLM unavailable: {exc}"
            return p

        try:
            json_str = _extract_json(raw)
            plan = json.loads(json_str)
            plan = _validate_plan(plan)
            plan["intent"] = intent  # always echo the original
            return plan
        except (json.JSONDecodeError, ValueError):
            p = dict(_UNKNOWN_PLAN)
            p["intent"] = intent
            p["clarification_prompt"] = "LLM returned invalid JSON. Please try rephrasing."
            return p

    def _call_ollama(self, prompt: str) -> str:
        """POST to Ollama /api/generate. Returns raw model response string."""
        url = f"{self._base_url}/api/generate"
        payload = json.dumps({
            "model":  self._model,
            "prompt": prompt,
            "system": _SYSTEM_PROMPT,
            "stream": False,
            "options": {
                "temperature": 0.1,   # low temp for deterministic plans
                "num_predict": 1024,
            },
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            body = resp.read().decode("utf-8")

        data = json.loads(body)
        return data.get("response", "")

    def ping(self) -> bool:
        """Check if Ollama is reachable."""
        try:
            url = f"{self._base_url}/api/tags"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5):
                return True
        except Exception:
            return False
