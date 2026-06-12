# API Analyst Report: tools\system_automation.py

## Dependencies
- `import os`
- `import subprocess`
- `import asyncio`
- `import logging`
- `from dataclasses import dataclass`
- `from dataclasses import field`
- `from pathlib import Path`

## Configuration Variables
- `SHELL_TIMEOUT` = `10`

## Schemas & API Contracts (Classes)

### Class `ToolResult`
**Fields/Schema:**
  - `success: bool`
  - `output: str`
  - `error: str`
  - `metadata: dict`

**Methods:**
- `def to_reflection_payload(self) -> dict`
  - *Normalised dict consumed by ReflectionEngine.*


## Functions & Endpoints

### `list_directory`
`def list_directory(path: str) -> ToolResult`
> List contents of a directory.

### `read_file`
`def read_file(path: str, max_bytes: int=32768) -> ToolResult`
> Read a text file (capped at max_bytes to protect context window).

### `write_file`
`def write_file(path: str, content: str, overwrite: bool=False) -> ToolResult`
> Write text content to a file. HIGH RISK – requires confirmation.

### `delete_file`
`def delete_file(path: str) -> ToolResult`
> Delete a file. VERY HIGH RISK – requires confirmation.

### `launch_application`
`def launch_application(target: str | None=None, args: list[str] | None=None, application: str | None=None) -> ToolResult`
> Launch a desktop application or open a file with its default handler.
Uses os.startfile on Windows; subprocess on other platforms.

### `execute_shell`
`async def execute_shell(command: str, working_dir: str | None=None) -> ToolResult`
> Execute a shell command and capture stdout/stderr asynchronously.
Hard timeout of SHELL_TIMEOUT seconds – never blocks the event loop.
Uses shell=False (shlex split) to satisfy security policy (no B602).

### `async_list_directory`
`async def async_list_directory(path: str) -> ToolResult`
### `async_read_file`
`async def async_read_file(path: str) -> ToolResult`
### `async_write_file`
`async def async_write_file(path: str, content: str, overwrite: bool=False) -> ToolResult`
### `async_delete_file`
`async def async_delete_file(path: str) -> ToolResult`
### `async_launch_application`
`async def async_launch_application(target: str | None=None, args: list[str] | None=None, application: str | None=None) -> ToolResult`
### `async_execute_shell`
`async def async_execute_shell(command: str, working_dir: str | None=None) -> ToolResult`