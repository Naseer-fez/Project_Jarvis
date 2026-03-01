# ─────────────────────────────────────────────────────────────────────────────
# ADD TO: core/llm/task_planner.py
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_TOOL_SCHEMA = {
    "tools": [
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
        }
    ]
}

import json

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

# IMPORTANT: Ensure SYSTEM_TOOLS_PROMPT_BLOCK is concatenated to your system_prompt
# inside your existing prompt assembly function!
