# Analysis Report for builtin_tools.py

## Dependencies
- asyncio
- json
- logging
- os
- platform
- pathlib.Path
- core.tools.path_utils._assert_safe_path
- core.tools.path_utils._PROJECT_ROOT

## Schemas
None

## API Contracts
- _fallback_classify_file(file_path)
- register_all_tools(router, llm, config)

## Configuration Variables
- _memory_store (typed)
- _LLM_CLIENT
- _CONFIG

## Assumptions & Notes
- Module Docstring: Built-in tools for Jarvis.
All tools are async coroutines and sandboxed to allowed directories.

