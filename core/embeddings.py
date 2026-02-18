"""
core/embeddings.py
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
"""

import logging
import hashlib
from functools import lru_cache
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


# ─── Default Config ───────────────────────────────────────────────────────────

DEFAULT_MODEL    = "all-MiniLM-L6-v2"
CACHE_SIZE       = 512    # LRU cache: max cached embeddings
WARM_UP_TEXT     = "Hello, I am Jarvis."  # Used to pre-warm the model


# ─── EmbeddingManager ─────────────────────────────────────────────────────────

class EmbeddingManager:
    """
    Singleton-style embedding manager.
    Loads the model once and provides a clean interface for the rest of Jarvis.

    Usage:
        em = EmbeddingManager()
        em.initialize()

        vec = em.embed("I like coffee")
        vecs = em.embed_batch(["coffee", "tea", "water"])
        score = em.similarity("I like coffee", "My favorite drink is coffee")
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name  = model_name
        self._model: Optional[SentenceTransformer] = None
        self._initialized = False
        self._embed_count = 0
        self._cache: dict[str, list[float]] = {}

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def initialize(self, warm_up: bool = True) -> bool:
        """
        Load the model into memory. Safe to call multiple times.
        Returns True on success.
        """
        if self._initialized:
            return True
        try:
            logger.info(f"Loading sentence-transformer model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)

            if warm_up:
                logger.debug("Warming up embedding model...")
                _ = self._model.encode(WARM_UP_TEXT, normalize_embeddings=True)

            self._initialized = True
            dim = self._model.get_sentence_embedding_dimension()
            logger.info(f"Embedding model ready. Dimension: {dim}")
            return True

        except Exception as e:
            logger.error(f"Failed to load embedding model '{self.model_name}': {e}")
            return False

    def is_ready(self) -> bool:
        return self._initialized

    # ── Core Embedding ─────────────────────────────────────────────────────────

    def embed(self, text: str, use_cache: bool = True) -> list[float]:
        """
        Generate a normalized embedding for a single text string.
        Caches results to avoid re-embedding identical inputs.

        Args:
            text:      Input text to embed.
            use_cache: Whether to use the in-memory LRU cache.

        Returns:
            List of floats (normalized embedding vector).
        """
        if not self._initialized:
            raise RuntimeError("EmbeddingManager not initialized. Call initialize() first.")

        # Cache key: MD5 hash of (model + text) for safety
        cache_key = hashlib.md5(f"{self.model_name}::{text}".encode()).hexdigest()

        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        vector = self._model.encode(text, normalize_embeddings=True).tolist()
        self._embed_count += 1

        if use_cache:
            # Evict oldest if cache is full
            if len(self._cache) >= CACHE_SIZE:
                oldest = next(iter(self._cache))
                del self._cache[oldest]
            self._cache[cache_key] = vector

        return vector

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> list[list[float]]:
        """
        Embed a list of texts efficiently in batches.
        Returns a list of normalized embedding vectors.
        """
        if not self._initialized:
            raise RuntimeError("EmbeddingManager not initialized.")

        if not texts:
            return []

        vectors = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        self._embed_count += len(texts)
        return vectors.tolist()

    # ── Similarity ─────────────────────────────────────────────────────────────

    def similarity(self, text_a: str, text_b: str) -> float:
        """
        Compute cosine similarity between two texts.
        Returns a float in [0, 1] — higher means more similar.

        Since we use normalized embeddings, cosine similarity = dot product.
        """
        vec_a = np.array(self.embed(text_a))
        vec_b = np.array(self.embed(text_b))
        return float(np.dot(vec_a, vec_b))

    def similarity_batch(
        self,
        query: str,
        candidates: list[str],
    ) -> list[tuple[str, float]]:
        """
        Compare one query against many candidates.
        Returns list of (text, score) tuples sorted by score descending.
        """
        if not candidates:
            return []

        query_vec    = np.array(self.embed(query))
        candidate_vecs = np.array(self.embed_batch(candidates))

        # Dot products (all normalized → cosine similarity)
        scores = (candidate_vecs @ query_vec).tolist()

        pairs = list(zip(candidates, scores))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs

    def rank_memories(
        self,
        query: str,
        memory_texts: list[str],
        top_k: int = 5,
        threshold: float = 0.30,
    ) -> list[dict]:
        """
        Rank a list of memory strings by relevance to a query.
        Returns top_k results above the threshold.

        Each result: {"text": str, "score": float, "rank": int}
        """
        pairs = self.similarity_batch(query, memory_texts)
        results = []
        for rank, (text, score) in enumerate(pairs[:top_k], start=1):
            if score >= threshold:
                results.append({
                    "text":  text,
                    "score": round(score, 4),
                    "rank":  rank,
                })
        return results

    # ── Dimension & Model Info ─────────────────────────────────────────────────

    @property
    def dimension(self) -> Optional[int]:
        """Return the embedding dimension, or None if not initialized."""
        if self._model:
            return self._model.get_sentence_embedding_dimension()
        return None

    def info(self) -> dict:
        """Return model information and usage stats."""
        return {
            "model":         self.model_name,
            "initialized":   self._initialized,
            "dimension":     self.dimension,
            "embed_count":   self._embed_count,
            "cache_size":    len(self._cache),
            "cache_capacity": CACHE_SIZE,
        }

    # ── Cache Management ───────────────────────────────────────────────────────

    def clear_cache(self):
        """Clear the embedding cache."""
        self._cache.clear()
        logger.debug("Embedding cache cleared.")

    def preload(self, texts: list[str]):
        """
        Pre-warm the cache with a list of texts.
        Useful on startup to pre-embed stored preferences.
        """
        if not self._initialized:
            logger.warning("Cannot preload: model not initialized.")
            return
        logger.info(f"Preloading {len(texts)} texts into embedding cache...")
        for text in texts:
            self.embed(text, use_cache=True)
        logger.info("Preload complete.")


# ─── Module-level singleton ────────────────────────────────────────────────────
# Other modules can import this instance directly for convenience.

_default_manager: Optional[EmbeddingManager] = None


def get_embedding_manager(model_name: str = DEFAULT_MODEL) -> EmbeddingManager:
    """
    Get (or create) the module-level default EmbeddingManager.
    Initializes automatically on first call.
    """
    global _default_manager
    if _default_manager is None or _default_manager.model_name != model_name:
        _default_manager = EmbeddingManager(model_name=model_name)
        _default_manager.initialize()
    return _default_manager

