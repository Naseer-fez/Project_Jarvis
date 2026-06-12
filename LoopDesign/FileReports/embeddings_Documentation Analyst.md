# Analysis Report for embeddings.py

## Dependencies
- __future__.annotations
- asyncio
- logging
- hashlib
- threading
- typing.Optional
- typing.TYPE_CHECKING
- numpy

## Schemas
- DeterministicMockSentenceTransformer
- EmbeddingManager

## API Contracts
- DeterministicMockSentenceTransformer.__init__(self, dimension)
- DeterministicMockSentenceTransformer.get_sentence_embedding_dimension(self)
- DeterministicMockSentenceTransformer._get_word_vector(self, word)
- DeterministicMockSentenceTransformer.encode(self, sentences, batch_size)
- DeterministicMockSentenceTransformer._encode_single(self, text)
- EmbeddingManager.__init__(self, model_name)
- EmbeddingManager.is_ready(self)
- EmbeddingManager.dimension(self)
- EmbeddingManager.info(self)
- EmbeddingManager.clear_cache(self)
- get_embedding_manager(model_name)

## Configuration Variables
- DEFAULT_MODEL
- CACHE_SIZE
- WARM_UP_TEXT
- _default_manager (typed)

## Assumptions & Notes
- Module Docstring: core/embeddings.py
───────────────────
Embedding manager for Jarvis.

Handles:
  - Lazy loading of sentence-transformer model (loads once, stays in memory)
  - Single and batch text embedding
  - Cosine similarity computation
  - Cache for repeated text lookups (avoids re-embedding identical strings)
  - Model health checking and warm-up

Available Models (local, no API):
  Model                     Size    Speed    Quality
  ──────────────────────────────────────────────────
  all-MiniLM-L6-v2         80 MB   Fast     Good      ← Default
  all-MiniLM-L12-v2        120 MB  Medium   Better
  all-mpnet-base-v2         420 MB  Slow     Best
  paraphrase-MiniLM-L6-v2  80 MB   Fast     Good (paraphrase-tuned)

Author: Jarvis Session 4

