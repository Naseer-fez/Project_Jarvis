# Dependency Analysis Report for memory\code_indexer_service.py

## Library Requirements
- from core.memory.code_indexer import extract_code_chunks
- from datetime import datetime
- from pathlib import Path
- from typing import Callable
- import asyncio
- import hashlib
- import logging

## Service Dependencies
- asyncio.sleep
- asyncio.to_thread

## Hidden Execution Links
- None detected

## Configurations / Assumptions
- Analyzed via AST and regex.
