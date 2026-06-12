# API Analyst Report: tools\fast_search_tool.py

## Dependencies
- `import os`
- `import queue`
- `import threading`
- `import fnmatch`
- `import asyncio`
- `from pathlib import Path`

## Schemas & API Contracts (Classes)

### Class `PythonSearchEngine`
**Methods:**
- `def __init__(self, start_paths, query=None, content_query=None, num_threads=16, case_sensitive=False, no_skip=False, max_results=2000)`
- `def should_skip(self, path)`
- `def is_binary(self, filepath)`
- `def search_file_content(self, filepath)`
- `def worker(self)`
- `def run(self)`


## Functions & Endpoints

### `get_windows_drives`
`def get_windows_drives()`
### `run_fast_search`
`async def run_fast_search(path='all', query='', content='', threads=8, case_sensitive=False, no_skip=False, max_results=1000)`
> Search files by name pattern and/or by file content (grep) using C++ executable.
If the executable is not compiled or fails to run, it falls back to a high-performance Python threaded crawler.
