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
    "tools_required": [str],  # deduped action list
    "risk_level": "low|medium|high|critical",
    "confirmation_required": bool,
    "clarification_needed": bool,
    "clarification_prompt": str   # if clarification_needed
  }
"""

from __future__ import annotations

import json
import re
from typing import Any

import urllib.request
import urllib.error

from core.planning.plan_schema import build_unknown_plan, normalize_plan

# ── System tool definitions ────────────────────────────────────────────────────
SYSTEM_TOOL_SCHEMA = {
    "tools": [
        # ── Local Filesystem ─────────────────────────────────────────────
        {
            "name": "list_directory",
            "description": "List the contents of a directory on the local filesystem.",
            "risk": "low",
            "args": {
                "path": {"type": "string", "description": "Absolute or relative path to the directory."}
            },
            "required_args": ["path"]
        },
        {
            "name": "read_file",
            "description": "Read the text content of a local file.",
            "risk": "low",
            "args": {
                "path": {"type": "string", "description": "Absolute path to the file."}
            },
            "required_args": ["path"]
        },
        {
            "name": "write_file",
            "description": "Write text content to a local file. Requires user confirmation.",
            "risk": "HIGH",
            "args": {
                "path":      {"type": "string",  "description": "Absolute path to the target file."},
                "content":   {"type": "string",  "description": "Full text content to write."},
                "overwrite": {"type": "boolean", "description": "Set true to overwrite existing files.", "default": False}
            },
            "required_args": ["path", "content"]
        },
        {
            "name": "delete_file",
            "description": "Permanently delete a file. Requires user confirmation. Use with extreme caution.",
            "risk": "VERY HIGH",
            "args": {
                "path": {"type": "string", "description": "Absolute path to the file to delete."}
            },
            "required_args": ["path"]
        },
        {
            "name": "launch_application",
            "description": "Launch a desktop application or open a file with its default program. Requires user confirmation.",
            "risk": "HIGH",
            "args": {
                "target": {"type": "string", "description": "Executable path, file path, or URL to open."},
                "args":   {"type": "array",  "items": {"type": "string"}, "description": "Optional CLI arguments.", "default": []}
            },
            "required_args": ["target"]
        },
        {
            "name": "execute_shell",
            "description": "Execute a shell command on the host Windows machine and return stdout/stderr. Requires user confirmation.",
            "risk": "HIGH",
            "args": {
                "command":     {"type": "string", "description": "The full shell command to execute."},
                "working_dir": {"type": "string", "description": "Optional working directory path.", "default": None}
            },
            "required_args": ["command"]
        },
        # ── Telegram ─────────────────────────────────────────────
        {
            "name": "send_telegram",
            "description": "Send a Telegram message to the configured chat. Requires confirmation.",
            "risk": "confirm",
            "args": {
                "message": {"type": "string", "description": "Message text (supports HTML)."}
            },
            "required_args": ["message"]
        },
        {
            "name": "get_updates",
            "description": "Fetch the latest incoming Telegram messages for the bot.",
            "risk": "low",
            "args": {
                "limit": {"type": "integer", "default": 10}
            },
            "required_args": []
        },
        # ── Google Calendar ──────────────────────────────────────
        {
            "name": "create_event",
            "description": "Create a Google Calendar event. Requires confirmation.",
            "risk": "confirm",
            "args": {
                "summary":  {"type": "string", "description": "Event title."},
                "start":    {"type": "string", "description": "Start datetime ISO-8601."},
                "end":      {"type": "string", "description": "End datetime ISO-8601."},
                "timezone": {"type": "string", "default": "UTC"}
            },
            "required_args": ["summary", "start", "end"]
        },
        {
            "name": "list_events",
            "description": "List upcoming Google Calendar events.",
            "risk": "low",
            "args": {
                "days_ahead":  {"type": "integer", "default": 7},
                "max_results": {"type": "integer", "default": 10}
            },
            "required_args": []
        },
        {
            "name": "delete_event",
            "description": "Delete a Google Calendar event by ID. Requires confirmation.",
            "risk": "confirm",
            "args": {
                "event_id": {"type": "string", "description": "Google Calendar event ID."}
            },
            "required_args": ["event_id"]
        },
        {
            "name": "find_free_slot",
            "description": "Find the next available free time slot on Google Calendar.",
            "risk": "low",
            "args": {
                "duration_minutes": {"type": "integer", "default": 60},
                "days_ahead":       {"type": "integer", "default": 7}
            },
            "required_args": []
        },
        # ── Gmail ───────────────────────────────────────────────
        {
            "name": "list_unread",
            "description": "List unread Gmail messages.",
            "risk": "low",
            "args": {"max_results": {"type": "integer", "default": 10}},
            "required_args": []
        },
        {
            "name": "send_gmail",
            "description": "Send an email via Gmail. Requires confirmation.",
            "risk": "confirm",
            "args": {
                "to":      {"type": "string", "description": "Recipient email."},
                "subject": {"type": "string"},
                "body":    {"type": "string"}
            },
            "required_args": ["to", "subject", "body"]
        },
        {
            "name": "summarize_unread",
            "description": "Fetch unread Gmail snippets for LLM summarization (content auto-truncated).",
            "risk": "low",
            "args": {"max_results": {"type": "integer", "default": 5}},
            "required_args": []
        },
        {
            "name": "mark_as_read",
            "description": "Mark a Gmail message as read. Requires confirmation.",
            "risk": "confirm",
            "args": {"message_id": {"type": "string"}},
            "required_args": ["message_id"]
        },
        # ── Notion ──────────────────────────────────────────────
        {
            "name": "create_page",
            "description": "Create a Notion page. Requires confirmation.",
            "risk": "confirm",
            "args": {
                "parent_id": {"type": "string"},
                "title":     {"type": "string"},
                "content":   {"type": "string", "default": ""}
            },
            "required_args": ["parent_id", "title"]
        },
        {
            "name": "query_database",
            "description": "Query a Notion database.",
            "risk": "low",
            "args": {
                "database_id": {"type": "string"},
                "page_size":   {"type": "integer", "default": 10}
            },
            "required_args": ["database_id"]
        },
        {
            "name": "append_block",
            "description": "Append a text block to a Notion page. Requires confirmation.",
            "risk": "confirm",
            "args": {
                "page_id": {"type": "string"},
                "text":    {"type": "string"}
            },
            "required_args": ["page_id", "text"]
        },
        {
            "name": "get_page",
            "description": "Get metadata and blocks from a Notion page.",
            "risk": "low",
            "args": {"page_id": {"type": "string"}},
            "required_args": ["page_id"]
        },
        # ── Spotify ─────────────────────────────────────────────
        {
            "name": "play_track",
            "description": "Play a Spotify track (by URI or search query). Requires confirmation.",
            "risk": "confirm",
            "args": {
                "query":     {"type": "string", "description": "Search query if no URI.", "default": ""},
                "track_uri": {"type": "string", "default": ""}
            },
            "required_args": []
        },
        {
            "name": "pause",
            "description": "Pause the current Spotify playback.",
            "risk": "low",
            "args": {},
            "required_args": []
        },
        {
            "name": "search_track",
            "description": "Search Spotify for tracks.",
            "risk": "low",
            "args": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 5}
            },
            "required_args": ["query"]
        },
        {
            "name": "get_current_track",
            "description": "Get what's currently playing on Spotify.",
            "risk": "low",
            "args": {},
            "required_args": []
        },
        {
            "name": "create_playlist",
            "description": "Create a new Spotify playlist. Requires confirmation.",
            "risk": "confirm",
            "args": {
                "name": {"type": "string"},
                "description": {"type": "string", "default": ""}
            },
            "required_args": ["name"]
        },
        {
            "name": "web_search",
            "description": "Perform a web search using DuckDuckGo to find recent or general information.",
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

SYSTEM_TOOLS_PROMPT_BLOCK = f"""
## Available System Tools
You may instruct Jarvis to execute the following tools by emitting a JSON action block.
HIGH and VERY HIGH risk tools will be confirmed with the user before execution.

{json.dumps(SYSTEM_TOOL_SCHEMA, indent=2)}

## Action Output Format
Respond with a JSON object using EXACTLY this schema:
{{
  "tool": "<tool_name>",
  "args": {{ <key-value pairs matching the tool's args> }},
  "rationale": "<one sentence explaining why this action achieves the goal>"
}}

Rules:
- Only emit ONE action per response.
- Never invent tool names not listed above.
- Always populate all required_args.
- For write_file, include the complete intended content in the "content" field.
- Prefer read-only tools when the task can be satisfied without state changes.
"""

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
  "tools_required": ["<action_type>"],
  "risk_level": "low",
  "confirmation_required": false,
  "clarification_needed": false,
  "clarification_prompt": ""
}

Valid actions:
memory_read, memory_write, speak, display, status, recall, store_fact, health_check, vision_analyze,
file_read, file_write, system_stats, app_open, web_search, screen_capture, screen_understand,
vision_click, gui_click, gui_type, gui_hotkey, serial_connect, serial_send, serial_disconnect,
physical_actuate, sensor_read,
send_telegram, get_updates,
create_event, list_events, delete_event, find_free_slot,
list_unread, send_gmail, summarize_unread, mark_as_read,
create_page, query_database, append_block, get_page,
play_track, pause, search_track, get_current_track, create_playlist,
web_search, web_scrape

For multi-step requests (e.g. 'summarize emails AND add to Notion'), emit a workflow plan:
{
  "workflow": [
    {"tool": "<tool_name>", "args": {<args>}},
    {"tool": "<tool_name>", "args": {<args>}}
  ]
}

Never output these forbidden actions:
shell_exec, file_delete, registry_write, format_disk, wipe_disk
"""


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
            return build_unknown_plan(intent, reason="Empty request received.")

        prompt = intent
        if context:
            prompt = f"Context:\n{context}\n\nRequest: {intent}"

        try:
            raw = self._call_ollama(prompt)
        except Exception as exc:
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
        payload = json.dumps({
            "model":  self._model,
            "prompt": prompt,
            "system": _SYSTEM_PROMPT + SYSTEM_TOOLS_PROMPT_BLOCK,
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