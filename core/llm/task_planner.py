"""LLM task planner with dynamic integration tool prompt injection."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any

import urllib.request

from core.planning.plan_schema import build_unknown_plan, normalize_plan

try:
    from integrations.registry import integration_registry
except ImportError:  # optional integration package wiring
    integration_registry = None  # type: ignore[assignment]


SYSTEM_TOOL_SCHEMA = {
    "tools": [
        {
            "name": "list_directory",
            "description": "List the contents of a directory on the local filesystem.",
            "risk": "low",
            "args": {
                "path": {
                    "type": "string",
                    "description": "Absolute or relative path to the directory.",
                }
            },
            "required_args": ["path"],
        },
        {
            "name": "read_file",
            "description": "Read the text content of a local file.",
            "risk": "low",
            "args": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file.",
                }
            },
            "required_args": ["path"],
        },
        {
            "name": "write_file",
            "description": "Write text content to a local file. Requires confirmation.",
            "risk": "confirm",
            "args": {
                "path": {"type": "string", "description": "Absolute path to the target file."},
                "content": {"type": "string", "description": "Full text content to write."},
                "overwrite": {
                    "type": "boolean",
                    "description": "Set true to overwrite existing files.",
                    "default": False,
                },
            },
            "required_args": ["path", "content"],
        },
        {
            "name": "delete_file",
            "description": "Permanently delete a file. Blocked by default safety policy.",
            "risk": "critical",
            "args": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file to delete.",
                }
            },
            "required_args": ["path"],
        },
        {
            "name": "launch_application",
            "description": "Launch a desktop application or open a file with default program.",
            "risk": "confirm",
            "args": {
                "target": {
                    "type": "string",
                    "description": "Executable path, file path, or URL to open.",
                },
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional CLI arguments.",
                    "default": [],
                },
            },
            "required_args": ["target"],
        },
        {
            "name": "execute_shell",
            "description": "Execute a shell command on the host machine.",
            "risk": "confirm",
            "args": {
                "command": {"type": "string", "description": "The shell command to execute."},
                "working_dir": {
                    "type": "string",
                    "description": "Optional working directory path.",
                    "default": None,
                },
            },
            "required_args": ["command"],
        },
    ]
}


_SYSTEM_PROMPT = """You are Jarvis, a local AI assistant. Your ONLY job is to produce a JSON plan.

RULES:
1. Respond ONLY with valid JSON - no markdown, no explanation, no preamble.
2. If you don't know how to do something, set clarification_needed=true and explain.
3. Never include actions that execute arbitrary shell commands, delete files, or bypass safety controls.
4. Never guess or hallucinate - if uncertain, say so in clarification_prompt.
5. Keep plans concise - 1 to 5 steps maximum.

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
  "tools_required": ["<action_type>"],
  "risk_level": "low",
  "confirmation_required": false,
  "clarification_needed": false,
  "clarification_prompt": ""
}
"""


def _build_tool_schema() -> dict[str, Any]:
    merged = deepcopy(SYSTEM_TOOL_SCHEMA)
    if integration_registry is None:
        return merged

    try:
        integration_tools = integration_registry.get_tools()
    except Exception:
        integration_tools = []

    if integration_tools:
        merged["tools"].extend(integration_tools)
    return merged


def _build_tools_prompt_block() -> str:
    schema = _build_tool_schema()
    return f"""
## Available System Tools
You may instruct Jarvis to execute the following tools by emitting JSON action steps.

{json.dumps(schema, indent=2)}

Rules:
- Only use tool names listed above.
- Populate all required args.
- Prefer read-only tools when possible.
"""


def _extract_json(text: str) -> str:
    """Extract the first JSON object from model output."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"```(?:json)?", "", text)
    text = text.strip()

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
                return text[start : i + 1]
    return text


class TaskPlanner:
    def __init__(self, config) -> None:
        self._base_url = config.get("ollama", "base_url", fallback="http://localhost:11434")
        self._model = config.get("ollama", "planner_model", fallback="deepseek-r1:8b")
        self._timeout = int(config.get("ollama", "request_timeout_s", fallback="120"))

    def plan(self, intent: str, context: str = "") -> dict[str, Any]:
        """Generate a JSON plan for the given intent."""
        if not intent.strip():
            return build_unknown_plan(intent, reason="Empty request received.")

        prompt = intent
        if context:
            prompt = f"Context:\n{context}\n\nRequest: {intent}"

        try:
            raw = self._call_ollama(prompt)
        except Exception as exc:  # noqa: BLE001
            return build_unknown_plan(intent, reason=f"LLM unavailable: {exc}")

        try:
            json_str = _extract_json(raw)
            parsed = json.loads(json_str)
            return normalize_plan(parsed, intent=intent)
        except (json.JSONDecodeError, ValueError):
            return build_unknown_plan(
                intent,
                reason="LLM returned invalid JSON. Please try rephrasing.",
            )

    def _call_ollama(self, prompt: str) -> str:
        """POST to Ollama /api/generate. Returns raw model response string."""
        url = f"{self._base_url}/api/generate"
        payload = json.dumps(
            {
                "model": self._model,
                "prompt": prompt,
                "system": _SYSTEM_PROMPT + _build_tools_prompt_block(),
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 1024,
                },
            }
        ).encode("utf-8")

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
