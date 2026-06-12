# Analysis Report for fast_search_tool.py

## Dependencies
- os
- queue
- threading
- fnmatch
- asyncio
- pathlib.Path

## Schemas
- PythonSearchEngine

## API Contracts
- PythonSearchEngine.__init__(self, start_paths, query, content_query, num_threads, case_sensitive, no_skip, max_results)
- PythonSearchEngine.should_skip(self, path)
- PythonSearchEngine.is_binary(self, filepath)
- PythonSearchEngine.search_file_content(self, filepath)
- PythonSearchEngine.worker(self)
- PythonSearchEngine.run(self)
- get_windows_drives()

## Configuration Variables
None

## Assumptions & Notes
None

