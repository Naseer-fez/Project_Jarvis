"""Task planner that asks Ollama for strict JSON plans with dynamic tool schema."""

from __future__ import annotations

import json
import logging
import re
import urllib.request
from copy import deepcopy
from typing import Any

from core.planning.plan_schema import build_unknown_plan, normalize_plan

try:
    from integrations.registry import api_registry
except Exception:  # noqa: BLE001
    api_registry = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


SYSTEM_TOOL_SCHEMA: dict[str, Any] = {
    "tools": [
        {
            "name": "list_directory",
            "description": "List filesystem entries in a directory.",
            "risk": "low",
            "args": {
                "path": {
                    "type": "string",
                    "description": "Absolute or relative path to directory.",
                }
            },
            "required_args": ["path"],
        },
        {
            "name": "read_file",
            "description": "Read the text content of a file.",
            "risk": "low",
            "args": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to file.",
                }
            },
            "required_args": ["path"],
        },
        {
            "name": "write_file",
            "description": "Write text content to a local file.",
            "risk": "confirm",
            "args": {
                "path": {"type": "string", "description": "Target file path."},
                "content": {"type": "string", "description": "Full content to write."},
                "overwrite": {
                    "type": "boolean",
                    "description": "Set true to overwrite existing file.",
                    "default": False,
                },
            },
            "required_args": ["path", "content"],
        },
        {
            "name": "delete_file",
            "description": "Permanently delete a file.",
            "risk": "critical",
            "args": {
                "path": {
                    "type": "string",
                    "description": "Path to file to delete.",
                }
            },
            "required_args": ["path"],
        },
        {
            "name": "launch_application",
            "description": "Launch an executable or open a file/url.",
            "risk": "confirm",
            "args": {
                "target": {
                    "type": "string",
                    "description": "Executable path, file path, or URL.",
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
            "description": "Execute a shell command.",
            "risk": "confirm",
            "args": {
                "command": {"type": "string", "description": "Shell command."},
                "working_dir": {
                    "type": "string",
                    "description": "Optional working directory.",
                    "default": None,
                },
            },
            "required_args": ["command"],
        },
        # ── Hardware tools (Session 7) ──────────────────────────────────────
        {
            "name": "send_hardware_command",
            "description": "Send a command to a registered Arduino/hardware device.",
            "risk": "confirm",
            "args": {
                "device_name": {"type": "string", "description": "Registered device name (e.g. 'main_arduino')."},
                "command": {"type": "string", "description": "Command string (e.g. 'LIGHT')."},
                "value": {"type": "string", "description": "Optional value string (e.g. 'ON').", "default": ""},
            },
            "required_args": ["device_name", "command"],
        },
        {
            "name": "read_sensor",
            "description": "Read sensor data from a registered hardware device.",
            "risk": "low",
            "args": {
                "device_name": {"type": "string", "description": "Registered device name."},
                "sensor_type": {"type": "string", "description": "Sensor identifier (e.g. 'TEMPERATURE'). Defaults to 'all'.", "default": "all"},
            },
            "required_args": ["device_name"],
        },
        {
            "name": "list_hardware_devices",
            "description": "List all registered hardware devices and their connection status.",
            "risk": "low",
            "args": {},
            "required_args": [],
        },
        {
            "name": "ping_device",
            "description": "Ping a hardware device to check if its firmware is responsive.",
            "risk": "low",
            "args": {
                "device_name": {"type": "string", "description": "Registered device name."},
            },
            "required_args": ["device_name"],
        },
        # ── Screen tools (Session 7) ────────────────────────────────────────
        {
            "name": "capture_screen",
            "description": "Take a full-screen screenshot and save it to outputs/screenshots/.",
            "risk": "low",
            "args": {},
            "required_args": [],
        },
        {
            "name": "capture_region",
            "description": "Screenshot a specific screen region.",
            "risk": "low",
            "args": {
                "x": {"type": "integer", "description": "Left edge in pixels."},
                "y": {"type": "integer", "description": "Top edge in pixels."},
                "width": {"type": "integer", "description": "Region width in pixels."},
                "height": {"type": "integer", "description": "Region height in pixels."},
            },
            "required_args": ["x", "y", "width", "height"],
        },
        {
            "name": "find_text_on_screen",
            "description": "Search for a text string on screen using OCR and return bounding boxes.",
            "risk": "low",
            "args": {
                "text": {"type": "string", "description": "Text to search for (case-insensitive)."},
            },
            "required_args": ["text"],
        },
        {
            "name": "describe_screen",
            "description": "Describe screen contents via vision LLM or OCR fallback.",
            "risk": "low",
            "args": {},
            "required_args": [],
        },
        # ── GUI control tools (Session 7) ───────────────────────────────────
        {
            "name": "click",
            "description": "Click at the given screen coordinates. Includes mandatory 300ms safety delay.",
            "risk": "confirm",
            "args": {
                "x": {"type": "integer", "description": "X coordinate in pixels."},
                "y": {"type": "integer", "description": "Y coordinate in pixels."},
                "button": {"type": "string", "description": "Mouse button: 'left', 'right', or 'middle'.", "default": "left"},
            },
            "required_args": ["x", "y"],
        },
        {
            "name": "double_click",
            "description": "Double-click at the given screen coordinates.",
            "risk": "confirm",
            "args": {
                "x": {"type": "integer", "description": "X coordinate in pixels."},
                "y": {"type": "integer", "description": "Y coordinate in pixels."},
            },
            "required_args": ["x", "y"],
        },
        {
            "name": "right_click",
            "description": "Right-click at the given screen coordinates.",
            "risk": "confirm",
            "args": {
                "x": {"type": "integer", "description": "X coordinate in pixels."},
                "y": {"type": "integer", "description": "Y coordinate in pixels."},
            },
            "required_args": ["x", "y"],
        },
        {
            "name": "type_text",
            "description": "Type text at the current cursor position. Refuses sensitive keywords.",
            "risk": "confirm",
            "args": {
                "text": {"type": "string", "description": "Text to type."},
                "interval": {"type": "number", "description": "Delay between keystrokes in seconds.", "default": 0.05},
            },
            "required_args": ["text"],
        },
        {
            "name": "hotkey",
            "description": "Press a keyboard shortcut (e.g. Ctrl+C).",
            "risk": "confirm",
            "args": {
                "keys": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of key names to press simultaneously (e.g. ['ctrl', 'c']).",
                },
            },
            "required_args": ["keys"],
        },
        {
            "name": "get_active_window",
            "description": "Return the title and geometry of the currently focused window.",
            "risk": "low",
            "args": {},
            "required_args": [],
        },
        # ── Web Research tools ──────────────────────────────────────────────
        {
            "name": "web_search",
            "description": "Perform a web search using DuckDuckGo to find recent or general information. Returns text summaries.",
            "risk": "low",
            "args": {
                "query": {"type": "string", "description": "The search query."},
                "max_results": {"type": "integer", "default": 5}
            },
            "required_args": ["query"]
        },
        {
            "name": "web_scrape",
            "description": "Fetch and extract readable text from a webpage URL.",
            "risk": "low",
            "args": {
                "url": {"type": "string", "description": "The URL to scrape."},
                "max_chars": {"type": "integer", "default": 8000}
            },
            "required_args": ["url"]
        },
    ]
}

_SYSTEM_PROMPT = """You are Jarvis, a local AI assistant planner. Your only job is to return valid JSON.

Rules:
1. Output valid JSON only. No markdown.
2. Keep plans concise (1-5 steps).
3. Use only tools listed in the tool schema.
4. If uncertain, set clarification_needed=true and explain in clarification_prompt.

Return this structure:
{
  "intent": "<echo user request>",
  "summary": "<one sentence>",
  "confidence": 0.0,
  "steps": [
    {
      "id": 1,
      "action": "<tool_or_action>",
      "description": "<what this step does>",
      "params": {}
    }
  ],
  "tools_required": ["<action>"],
  "risk_level": "low",
  "confirmation_required": false,
  "clarification_needed": false,
  "clarification_prompt": ""
}
"""


def _build_tool_schema() -> dict[str, Any]:
    merged = deepcopy(SYSTEM_TOOL_SCHEMA)
    seen_names = {
        str(tool.get("name", "")).strip()
        for tool in merged["tools"]
        if isinstance(tool, dict) and str(tool.get("name", "")).strip()
    }

    if api_registry is None:
        return merged

    try:
        if hasattr(api_registry, "list_schemas"):
            dynamic_tools = api_registry.list_schemas()  # type: ignore[assignment]
        else:
            dynamic_tools = api_registry.get_tools()  # type: ignore[assignment]
    except Exception:
        dynamic_tools = []

    for tool in dynamic_tools:
        if not isinstance(tool, dict):
            continue
        name = str(tool.get("name", "")).strip()
        if not name:
            continue
        if name in seen_names:
            logger.warning("Skipping dynamic tool schema '%s' because it conflicts with an existing tool", name)
            continue
        merged["tools"].append(dict(tool))
        seen_names.add(name)

    return merged


def _build_tools_prompt_block() -> str:
    schema = _build_tool_schema()
    return (
        "\n## Available System Tools\n"
        "Use only these tools when planning executable steps.\n\n"
        f"{json.dumps(schema, indent=2)}\n\n"
        "Rules:\n"
        "- Populate all required args for any tool call.\n"
        "- Prefer read-only tools where possible.\n"
    )


def _extract_json(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    cleaned = re.sub(r"```(?:json)?", "", cleaned)
    cleaned = cleaned.strip()

    depth = 0
    start = -1
    for idx, char in enumerate(cleaned):
        if char == "{":
            if start == -1:
                start = idx
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start != -1:
                return cleaned[start : idx + 1]
    return cleaned


class TaskPlanner:
    def __init__(self, config) -> None:
        self._base_url = config.get("ollama", "base_url", fallback="http://localhost:11434")
        self._model = config.get("ollama", "planner_model", fallback="deepseek-r1:8b")
        self._timeout = int(config.get("ollama", "request_timeout_s", fallback="120"))

    def plan(self, intent: str, context: str = "") -> dict[str, Any]:
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
            parsed = json.loads(_extract_json(raw))
        except Exception:  # noqa: BLE001
            return build_unknown_plan(intent, reason="Planner returned invalid JSON. Rephrase and try again.")

        return normalize_plan(parsed, intent=intent)

    def _call_ollama(self, prompt: str) -> str:
        url = f"{self._base_url}/api/generate"
        payload = json.dumps(
            {
                "model": self._model,
                "prompt": prompt,
                "system": _SYSTEM_PROMPT + _build_tools_prompt_block(),
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 1024},
            }
        ).encode("utf-8")

        request = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(request, timeout=self._timeout) as response:
            body = response.read().decode("utf-8")

        data = json.loads(body)
        return str(data.get("response", ""))

    def ping(self) -> bool:
        try:
            request = urllib.request.Request(f"{self._base_url}/api/tags", method="GET")
            with urllib.request.urlopen(request, timeout=5):
                return True
        except Exception:
            return False


__all__ = ["TaskPlanner", "SYSTEM_TOOL_SCHEMA"]
