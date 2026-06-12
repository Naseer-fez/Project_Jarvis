# Dependency Analysis Report for memory\hybrid_memory.py

## Library Requirements
- from __future__ import annotations
- from core.memory.code_indexer import extract_code_chunks
- from core.memory.code_indexer_service import CodeIndexerService
- from core.memory.context_compressor import ContextCompressor
- from core.memory.retriever import MemoryRetriever
- from core.memory.semantic_memory import SemanticMemory
- from core.memory.sqlite_pool import SQLitePool
- from core.memory.sqlite_storage import SQLiteStorage
- from pathlib import Path
- from typing import Any
- import asyncio
- import contextlib
- import logging

## Service Dependencies
- asyncio.Lock
- asyncio.create_task
- asyncio.gather

## Hidden Execution Links
- None detected

## Configurations / Assumptions
- Analyzed via AST and regex.
