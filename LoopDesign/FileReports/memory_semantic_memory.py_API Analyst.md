# API Analyst Report: memory\semantic_memory.py

## Dependencies
- `import uuid`
- `import asyncio`
- `import logging`
- `from datetime import datetime`
- `from pathlib import Path`
- `from typing import List`
- `from typing import Dict`
- `from typing import Any`

## Configuration Variables
- `DEFAULT_MODEL` = `'all-MiniLM-L6-v2'`
- `DEFAULT_TOP_K` = `5`
- `DEFAULT_THRESHOLD` = `0.3`
- `CHROMA_PATH` = `str(Path(__file__).resolve().parent.parent.parent / 'data' / 'chroma')`
- `COLLECTION_PREFS` = `'jarvis_preferences'`
- `COLLECTION_EPISODES` = `'jarvis_episodes'`
- `COLLECTION_CONVOS` = `'jarvis_conversations'`

## Schemas & API Contracts (Classes)

### Class `SemanticMemory`
> Local vector-based memory using ChromaDB + sentence-transformers.

Usage:
    sm = SemanticMemory()
    sm.store_preference("favorite_drink", "coffee")
    results = sm.recall("What do I like to drink?", top_k=3)

**Methods:**
- `def __init__(self, chroma_path: str=CHROMA_PATH, model_name: str=DEFAULT_MODEL, embedding_manager: Any=None)`
- `async def initialize(self) -> bool`
  - *Lazy initialization — load model and connect to ChromaDB.*
- `async def _ensure_init(self)`
- `async def _embed(self, text: str) -> List[float]`
  - *Generate a normalized embedding vector for the given text.*
- `def _collection(self, name: str)`
- `async def store_preference(self, key: str, value: str) -> str`
  - *Upsert a user preference into the vector store.*
- `async def store_episode(self, event: str, category: str='general') -> str`
  - *Store an episodic memory event.*
- `async def store_episodes_batch(self, events: list[str], category: str='general') -> list[str]`
  - *Store a batch of episodic memory events efficiently.*
- `async def store_conversation_turn(self, user_input: str, assistant_response: str, session_id: str='default') -> str`
  - *Store a conversation exchange as a single vector document.*
- `async def recall_preferences(self, query: str, top_k: int=DEFAULT_TOP_K, threshold: float=DEFAULT_THRESHOLD) -> List[Dict]`
  - *Retrieve the most semantically similar preferences to the query.*
- `async def recall_episodes(self, query: str, top_k: int=DEFAULT_TOP_K, threshold: float=DEFAULT_THRESHOLD) -> List[Dict]`
  - *Retrieve the most relevant episodic memories for the query.*
- `async def recall_conversations(self, query: str, top_k: int=DEFAULT_TOP_K, threshold: float=DEFAULT_THRESHOLD) -> List[Dict]`
  - *Retrieve the most relevant past conversation turns for the query.*
- `async def recall_all(self, query: str, top_k: int=DEFAULT_TOP_K, threshold: float=DEFAULT_THRESHOLD) -> Dict[str, List[Dict]]`
  - *Recall across ALL collections simultaneously.*
- `async def _query_collection(self, collection_name: str, query: str, top_k: int, threshold: float) -> List[Dict]`
  - *Internal: query a ChromaDB collection, filter by threshold, return results.*
- `async def delete_preference(self, key: str) -> bool`
  - *Delete a preference by key. Returns True if deleted.*
- `async def clear_collection(self, collection_name: str) -> bool`
  - *Delete and recreate a collection (full wipe). Use with caution.*
- `async def stats(self) -> Dict[str, Any]`
  - *Return counts for all collections.*
- `def is_ready(self) -> bool`
- `async def close(self) -> None`
  - *Close/stop the ChromaDB client if it has one.*

