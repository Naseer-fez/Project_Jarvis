"""
core/memory/hybrid_memory.py — Hybrid memory: SQLite facts + Chroma semantic search.

Two complementary stores:
  - SQLite: structured key/value facts with timestamps (deterministic recall)
  - Chroma: vector embeddings for semantic "what do I know about X" queries

Never hallucinates: if nothing is found, returns empty, not fabricated answers.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class Fact:
    key: str
    value: str
    source: str
    created_at: float
    updated_at: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticResult:
    text: str
    distance: float
    metadata: dict[str, Any]


class HybridMemory:
    """Facade over SQLite (facts) + Chroma (semantic recall)."""

    def __init__(self, config) -> None:
        self._lock = threading.RLock()

        data_dir = Path(config.get("memory", "data_dir", fallback="data"))
        data_dir.mkdir(parents=True, exist_ok=True)

        sqlite_path = config.get("memory", "sqlite_file", fallback="data/jarvis_memory.db")
        self._db_path = sqlite_path
        self._init_sqlite()

        # Chroma is optional — degrade gracefully if not installed
        self._chroma = None
        self._collection = None
        chroma_dir = config.get("memory", "chroma_dir", fallback="data/chroma")
        embedding_model = config.get("memory", "embedding_model", fallback="all-MiniLM-L6-v2")
        self._top_k = int(config.get("memory", "semantic_top_k", fallback="5"))

        try:
            import chromadb
            from chromadb.utils import embedding_functions as ef

            self._chroma = chromadb.PersistentClient(path=chroma_dir)
            emb_fn = ef.SentenceTransformerEmbeddingFunction(model_name=embedding_model)
            self._collection = self._chroma.get_or_create_collection(
                name="jarvis_facts",
                embedding_function=emb_fn,
            )
        except ImportError:
            pass  # Chroma not installed — semantic search disabled
        except Exception:
            pass  # Chroma init failed — degrade gracefully

    # ── SQLite setup ──────────────────────────────────────────────────────────

    def _init_sqlite(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS facts (
                    key         TEXT PRIMARY KEY,
                    value       TEXT NOT NULL,
                    source      TEXT NOT NULL DEFAULT 'user',
                    created_at  REAL NOT NULL,
                    updated_at  REAL NOT NULL,
                    metadata    TEXT NOT NULL DEFAULT '{}'
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_updated ON facts(updated_at)")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    # ── Write ─────────────────────────────────────────────────────────────────

    def store_fact(self, key: str, value: str, source: str = "user",
                   metadata: dict | None = None) -> None:
        """Store or update a key/value fact."""
        now = time.time()
        meta = metadata or {}
        with self._lock:
            with self._connect() as conn:
                conn.execute("""
                    INSERT INTO facts (key, value, source, created_at, updated_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        value      = excluded.value,
                        source     = excluded.source,
                        updated_at = excluded.updated_at,
                        metadata   = excluded.metadata
                """, (key, value, source, now, now, json.dumps(meta)))

            # Mirror to Chroma
            if self._collection is not None:
                doc = f"{key}: {value}"
                try:
                    self._collection.upsert(
                        ids=[key],
                        documents=[doc],
                        metadatas=[{"source": source, "key": key, **meta}],
                    )
                except Exception:
                    pass

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_fact(self, key: str) -> Optional[Fact]:
        """Exact key lookup."""
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM facts WHERE key = ?", (key,)
                ).fetchone()
            if row is None:
                return None
            return Fact(
                key=row["key"], value=row["value"], source=row["source"],
                created_at=row["created_at"], updated_at=row["updated_at"],
                metadata=json.loads(row["metadata"]),
            )

    def list_facts(self, limit: int = 20) -> list[Fact]:
        """Most recently updated facts."""
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM facts ORDER BY updated_at DESC LIMIT ?", (limit,)
                ).fetchall()
            return [
                Fact(key=r["key"], value=r["value"], source=r["source"],
                     created_at=r["created_at"], updated_at=r["updated_at"],
                     metadata=json.loads(r["metadata"]))
                for r in rows
            ]

    def semantic_search(self, query: str, top_k: int | None = None) -> list[SemanticResult]:
        """Semantic similarity search via Chroma. Returns [] if unavailable."""
        if self._collection is None:
            return []
        k = top_k or self._top_k
        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=min(k, self._collection.count()),
            )
            out = []
            for i, doc in enumerate(results["documents"][0]):
                dist = results["distances"][0][i]
                meta = results["metadatas"][0][i] if results.get("metadatas") else {}
                out.append(SemanticResult(text=doc, distance=dist, metadata=meta))
            return out
        except Exception:
            return []

    def recall(self, query: str) -> str:
        """
        Best-effort recall for a natural language query.
        Returns a human-readable string or 'I don't know.'
        Never hallucinates.
        """
        # 1. Try exact key match first
        fact = self.get_fact(query)
        if fact:
            return f"{fact.key}: {fact.value}"

        # 2. Try semantic search
        results = self.semantic_search(query, top_k=3)
        if results:
            lines = [f"- {r.text}" for r in results if r.distance < 1.0]
            if lines:
                return "Related facts:\n" + "\n".join(lines)

        return "I don't know."

    def count(self) -> int:
        with self._lock:
            with self._connect() as conn:
                row = conn.execute("SELECT COUNT(*) as n FROM facts").fetchone()
                return row["n"]
