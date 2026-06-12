# Dependency Analysis Report for runtime\event_bus.py

## Library Requirements
- from __future__ import annotations
- from collections import deque
- from dataclasses import dataclass
- from dataclasses import field
- from typing import Any
- from typing import Awaitable
- from typing import Callable
- from typing import Union
- import asyncio
- import logging
- import threading
- import time
- import uuid

## Service Dependencies
- asyncio.gather
- asyncio.get_running_loop
- asyncio.iscoroutine
- asyncio.run_coroutine_threadsafe

## Hidden Execution Links
- None detected

## Configurations / Assumptions
- Analyzed via AST and regex.
