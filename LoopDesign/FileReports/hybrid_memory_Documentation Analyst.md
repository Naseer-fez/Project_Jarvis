# Analysis Report for hybrid_memory.py

## Dependencies
- __future__.annotations
- contextlib
- asyncio
- pathlib.Path
- typing.Any
- logging
- core.memory.semantic_memory.SemanticMemory
- core.memory.sqlite_pool.SQLitePool
- core.memory.sqlite_storage.SQLiteStorage
- core.memory.retriever.MemoryRetriever
- core.memory.code_indexer_service.CodeIndexerService

## Schemas
- HybridMemory

## API Contracts
- _looks_like_config(value)
- HybridMemory.__init__(self, config_or_db_path, chroma_path, model_name)
- HybridMemory._track_background_task(self, task)
- HybridMemory.set_llm(self, llm)
- HybridMemory._query_tokens(query)
- HybridMemory._score_text(text, tokens)
- HybridMemory.stats(self)

## Configuration Variables
None

## Assumptions & Notes
- Module Docstring: Hybrid memory: SQLite for structure plus optional Chroma semantic memory.

