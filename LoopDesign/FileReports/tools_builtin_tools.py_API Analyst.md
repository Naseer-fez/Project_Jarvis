# API Analyst Report: tools\builtin_tools.py

## Dependencies
- `import asyncio`
- `import json`
- `import logging`
- `import os`
- `import platform`
- `from pathlib import Path`
- `from core.tools.path_utils import _assert_safe_path`
- `from core.tools.path_utils import _PROJECT_ROOT`

## Configuration Variables
- `_LLM_CLIENT` = `None`
- `_CONFIG` = `None`
- `_LLM_CLIENT` = `llm`
- `_CONFIG` = `config`

## Functions & Endpoints

### `get_time`
`async def get_time() -> str`
> Returns current local time and date.

### `get_system_stats`
`async def get_system_stats() -> str`
> Returns basic system resource usage.

### `list_directory`
`async def list_directory(path: str='./workspace') -> str`
> Lists files in a sandboxed directory.

### `read_file`
`async def read_file(path: str='./workspace/test.txt') -> str`
> Reads a text file from the sandbox.

### `write_file_safe`
`async def write_file_safe(path: str='./workspace/test.txt', content: str='Hello World') -> str`
> Writes content to a file in the sandbox (creates if needed).

### `search_memory`
`async def search_memory(query: str, limit: int=5) -> str`
> Simple keyword search over in-session memory.

### `log_event`
`async def log_event(content: str='', category: str='general') -> str`
> Logs an event to in-session memory and the outputs log file.

### `_fallback_classify_file`
`def _fallback_classify_file(file_path: Path) -> str`
### `sort_files`
`async def sort_files(directory: str='./workspace', output_dir: str='./workspace') -> str`
> Sorts files in a sandboxed directory into subfolders according to their content using LLM classification.

### `find_files`
`async def find_files(pattern: str, directory: str='./workspace') -> str`
> Finds files matching a wildcard pattern in a sandboxed directory (recursive).

### `copy_file`
`async def copy_file(source: str, destination: str) -> str`
> Copies a file from source to destination in the sandbox.

### `move_file`
`async def move_file(source: str, destination: str) -> str`
> Moves a file or directory from source to destination in the sandbox.

### `create_directory`
`async def create_directory(path: str='./workspace') -> str`
> Creates a new directory (and any parent directories) in the sandbox.

### `fast_search`
`async def fast_search(path: str='all', query: str='', content: str='', threads: int=8, case_sensitive: bool=False, no_skip: bool=False, max_results: int=1000) -> str`
> Search files by name pattern and/or by file content (grep) across the PC/drive.
Highly optimized multi-threaded execution.

### `convert_file_format`
`async def convert_file_format(source_path: str, target_format: str, output_path: str | None=None) -> str`
> Convert a file from its current format to target_format (e.g. webp, pdf, html, csv, json, xlsx, mp3, wav, mp4).
Dynamically installs missing libraries on demand.

### `register_all_tools`
`def register_all_tools(router, llm=None, config=None) -> None`
> Register all built-in tools with a CapabilityRegistry instance.
