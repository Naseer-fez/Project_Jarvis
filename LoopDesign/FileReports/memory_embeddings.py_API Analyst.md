# API Analyst Report: memory\embeddings.py

## Dependencies
- `from __future__ import annotations`
- `import asyncio`
- `import logging`
- `import hashlib`
- `import threading`
- `from typing import Optional`
- `from typing import TYPE_CHECKING`
- `import numpy as np`

## Configuration Variables
- `DEFAULT_MODEL` = `'all-MiniLM-L6-v2'`
- `CACHE_SIZE` = `512`
- `WARM_UP_TEXT` = `'Hello, I am Jarvis.'`

## Schemas & API Contracts (Classes)

### Class `DeterministicMockSentenceTransformer`
> Zero-dependency, offline deterministic bag-of-words embedding model.
Generates word vectors using hash-seeded uniform random projections.
This preserves cosine similarity relations (e.g. sharing words increases similarity),
enabling all offline tests to run cleanly and extremely fast.

**Methods:**
- `def __init__(self, dimension: int=384)`
- `def get_sentence_embedding_dimension(self) -> int`
- `def _get_word_vector(self, word: str) -> np.ndarray`
- `def encode(self, sentences, batch_size=32, **_kwargs)`
- `def _encode_single(self, text: str) -> np.ndarray`


### Class `EmbeddingManager`
> Singleton-style embedding manager.
Loads the model once and provides a clean interface for the rest of Jarvis.

Usage:
    em = EmbeddingManager()
    em.initialize()

    vec = em.embed("I like coffee")
    vecs = em.embed_batch(["coffee", "tea", "water"])
    score = em.similarity("I like coffee", "My favorite drink is coffee")

**Methods:**
- `def __init__(self, model_name: str=DEFAULT_MODEL)`
- `async def initialize(self, warm_up: bool=True) -> bool`
  - *Load the model into memory. Safe to call multiple times.*
- `def is_ready(self) -> bool`
- `async def embed(self, text: str, use_cache: bool=True) -> list[float]`
  - *Generate a normalized embedding for a single text string.*
- `async def embed_batch(self, texts: list[str], batch_size: int=32, show_progress: bool=False) -> list[list[float]]`
  - *Embed a list of texts efficiently in batches.*
- `async def similarity(self, text_a: str, text_b: str) -> float`
  - *Compute cosine similarity between two texts.*
- `async def similarity_batch(self, query: str, candidates: list[str]) -> list[tuple[str, float]]`
  - *Compare one query against many candidates.*
- `async def rank_memories(self, query: str, memory_texts: list[str], top_k: int=5, threshold: float=0.3) -> list[dict]`
  - *Rank a list of memory strings by relevance to a query.*
- @property
- `def dimension(self) -> Optional[int]`
  - *Return the embedding dimension, or None if not initialized.*
- `def info(self) -> dict`
  - *Return model information and usage stats.*
- `def clear_cache(self)`
  - *Clear the embedding cache.*
- `async def preload(self, texts: list[str])`
  - *Pre-warm the cache with a list of texts.*


## Functions & Endpoints

### `get_embedding_manager`
`def get_embedding_manager(model_name: str=DEFAULT_MODEL) -> EmbeddingManager`
> Get (or create) the module-level default EmbeddingManager.
