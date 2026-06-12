# Dependency Analysis Report for tools\fast_search_tool.py

## Library Requirements
- from pathlib import Path
- import asyncio
- import ctypes
- import fnmatch
- import os
- import queue
- import re
- import threading

## Service Dependencies
- asyncio.create_subprocess_exec
- asyncio.get_event_loop
- asyncio.get_running_loop
- asyncio.run

## Hidden Execution Links
- None detected

## Configurations / Assumptions
- Analyzed via AST and regex.
