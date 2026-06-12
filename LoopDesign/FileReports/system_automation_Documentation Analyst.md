# Analysis Report for system_automation.py

## Dependencies
- os
- subprocess
- asyncio
- logging
- dataclasses.dataclass
- dataclasses.field
- pathlib.Path

## Schemas
- ToolResult
- ToolResult attribute: success
- ToolResult attribute: output
- ToolResult attribute: error
- ToolResult attribute: metadata

## API Contracts
- ToolResult.to_reflection_payload(self)
- list_directory(path)
- read_file(path, max_bytes)
- write_file(path, content, overwrite)
- delete_file(path)
- launch_application(target, args, application)

## Configuration Variables
- TOOL_REGISTRY (typed)
- SHELL_TIMEOUT

## Assumptions & Notes
- Module Docstring: core/tools/system_automation.py
Jarvis V3 - System Automation Tools
All tools are synchronous internally; the dispatcher awaits them via asyncio.to_thread.

