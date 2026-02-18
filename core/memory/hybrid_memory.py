"""
core/memory/hybrid_memory.py
═════════════════════════════
Hybrid memory facade: SQLite (facts) + Chroma (semantic recall).

V1 Rules:
  - ONE write path (this file only)
  - ONE read path (this file only)
  - No memory mutation during planning
  - All memory ops logged
  - Jarvis cannot hallucinate memory — if it's not stored, it returns None
  - No speculative or autonomous memory writes

Architecture:
  SQLite  → structured facts, preferences, session history, system events
  Chroma  → semantic vector search for contextual recall
  Facade  → hybrid_memory.py is the ONLY interface. Never call SQLite/Chroma directly.
"""

import sqlite3
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from core.logger import get_logger, audit

logger = get_logger("memory")

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

SQLITE_PATH = DATA_DIR / "jarvis_memory.db"


# ══════════════════════════════════════════════
# SQLite Layer
# ══════════════════════════════════════════════

def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db():
    """Initialize SQLite schema on first run."""
    with _get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS facts (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                key         TEXT NOT NULL UNIQUE,
                value       TEXT NOT NULL,
                category    TEXT DEFAULT 'general',
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS session_history (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL,
                role        TEXT NOT NULL,
                content     TEXT NOT NULL,
                ts          TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS system_events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type  TEXT NOT NULL,
                detail      TEXT,
                ts          TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_facts_key ON facts(key);
            CREATE INDEX IF NOT EXISTS idx_session_id ON session_history(session_id);
        """)
        logger.debug("SQLite schema initialized")


# ══════════════════════════════════════════════
# Chroma Layer (lazy init — graceful if unavailable)
# ══════════════════════════════════════════════

_chroma_collection = None


def _get_chroma():
    """Lazy-load Chroma. Returns None if Chroma is not installed."""
    global _chroma_collection
    if _chroma_collection is not None:
        return _chroma_collection
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(DATA_DIR / "chroma"))
        _chroma_collection = client.get_or_create_collection(
            name="jarvis_context",
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("Chroma collection ready")
        return _chroma_collection
    except ImportError:
        logger.warning("chromadb not installed. Semantic recall disabled. (pip install chromadb)")
        return None
    except Exception as e:
        logger.error(f"Chroma init failed: {e}. Semantic recall disabled.")
        return None


# ══════════════════════════════════════════════
# Public Facade
# ══════════════════════════════════════════════

class HybridMemory:
    """
    Single access point for all Jarvis memory operations.
    Do not instantiate SQLite or Chroma directly anywhere else.
    """

    def __init__(self):
        _init_db()
        logger.info("HybridMemory initialized")

    # ─── FACTS (SQLite) ───────────────────────────────────

    def write_fact(self, key: str, value: str | dict | list, category: str = "general") -> bool:
        """Store or update a fact. Returns True on success."""
        if isinstance(value, (dict, list)):
            value = json.dumps(value)

        now = datetime.now(timezone.utc).isoformat()
        try:
            with _get_db() as conn:
                conn.execute("""
                    INSERT INTO facts (key, value, category, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        value=excluded.value,
                        category=excluded.category,
                        updated_at=excluded.updated_at
                """, (key, str(value), category, now, now))
            audit(logger, f"MEMORY_WRITE: key={key!r} category={category}", action="memory_write")
            return True
        except Exception as e:
            logger.error(f"write_fact failed: key={key!r} error={e}")
            return False

    def read_fact(self, key: str) -> str | None:
        """Read a fact by key. Returns None if not found."""
        try:
            with _get_db() as conn:
                row = conn.execute("SELECT value FROM facts WHERE key = ?", (key,)).fetchone()
            if row:
                logger.debug(f"MEMORY_READ: key={key!r} found")
                return row["value"]
            logger.debug(f"MEMORY_READ: key={key!r} not found")
            return None
        except Exception as e:
            logger.error(f"read_fact failed: key={key!r} error={e}")
            return None

    def delete_fact(self, key: str) -> bool:
        """Delete a fact. Logged."""
        try:
            with _get_db() as conn:
                conn.execute("DELETE FROM facts WHERE key = ?", (key,))
            audit(logger, f"MEMORY_DELETE: key={key!r}", action="memory_delete")
            return True
        except Exception as e:
            logger.error(f"delete_fact failed: key={key!r} error={e}")
            return False

    def list_facts(self, category: str | None = None) -> list[dict]:
        """List all facts, optionally filtered by category."""
        try:
            with _get_db() as conn:
                if category:
                    rows = conn.execute(
                        "SELECT key, value, category, updated_at FROM facts WHERE category = ?",
                        (category,)
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT key, value, category, updated_at FROM facts"
                    ).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"list_facts failed: {e}")
            return []

    # ─── SESSION HISTORY (SQLite) ─────────────────────────

    def write_session(self, session_id: str, role: str, content: str) -> bool:
        """Append a message to session history."""
        now = datetime.now(timezone.utc).isoformat()
        try:
            with _get_db() as conn:
                conn.execute(
                    "INSERT INTO session_history (session_id, role, content, ts) VALUES (?, ?, ?, ?)",
                    (session_id, role, content, now)
                )
            return True
        except Exception as e:
            logger.error(f"write_session failed: {e}")
            return False

    def read_session(self, session_id: str, limit: int = 20) -> list[dict]:
        """Read the last N messages from a session."""
        try:
            with _get_db() as conn:
                rows = conn.execute("""
                    SELECT role, content, ts FROM session_history
                    WHERE session_id = ?
                    ORDER BY id DESC LIMIT ?
                """, (session_id, limit)).fetchall()
            return list(reversed([dict(r) for r in rows]))
        except Exception as e:
            logger.error(f"read_session failed: {e}")
            return []

    # ─── SYSTEM EVENTS (SQLite) ───────────────────────────

    def log_event(self, event_type: str, detail: str = "") -> bool:
        """Log a system event (state change, error, boot, etc.)."""
        now = datetime.now(timezone.utc).isoformat()
        try:
            with _get_db() as conn:
                conn.execute(
                    "INSERT INTO system_events (event_type, detail, ts) VALUES (?, ?, ?)",
                    (event_type, detail, now)
                )
            return True
        except Exception as e:
            logger.error(f"log_event failed: {e}")
            return False

    # ─── SEMANTIC RECALL (Chroma) ─────────────────────────

    def remember(self, text: str, metadata: dict | None = None, doc_id: str | None = None) -> bool:
        """
        Store text in semantic memory for future recall.
        Falls back gracefully if Chroma is unavailable.
        """
        collection = _get_chroma()
        if collection is None:
            logger.warning("Semantic store unavailable. Falling back to SQLite fact store.")
            key = doc_id or f"semantic_{datetime.now(timezone.utc).timestamp()}"
            return self.write_fact(key, text, category="semantic_fallback")

        try:
            _id = doc_id or f"mem_{datetime.now(timezone.utc).timestamp()}"
            collection.add(
                documents=[text],
                metadatas=[metadata or {}],
                ids=[_id]
            )
            audit(logger, f"SEMANTIC_WRITE: id={_id!r}", action="semantic_write")
            return True
        except Exception as e:
            logger.error(f"remember() failed: {e}")
            return False

    def recall(self, query: str, n_results: int = 3) -> list[dict]:
        """
        Retrieve semantically similar memories.
        Returns list of {document, metadata, distance} dicts.
        Returns [] if Chroma unavailable or no results.
        """
        collection = _get_chroma()
        if collection is None:
            logger.warning("Semantic recall unavailable (Chroma not ready)")
            return []

        try:
            results = collection.query(
                query_texts=[query],
                n_results=min(n_results, collection.count() or 1)
            )
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            recall_results = [
                {"document": d, "metadata": m, "distance": dist}
                for d, m, dist in zip(docs, metas, distances)
            ]
            logger.info(f"SEMANTIC_RECALL: query={query!r} results={len(recall_results)}")
            return recall_results
        except Exception as e:
            logger.error(f"recall() failed: {e}")
            return []

    # ─── DIAGNOSTICS ──────────────────────────────────────

    def status(self) -> dict:
        """Return memory system health status."""
        try:
            with _get_db() as conn:
                fact_count = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
                session_count = conn.execute("SELECT COUNT(*) FROM session_history").fetchone()[0]
        except Exception:
            fact_count = -1
            session_count = -1

        chroma = _get_chroma()
        chroma_count = chroma.count() if chroma else -1

        return {
            "sqlite_facts": fact_count,
            "sqlite_sessions": session_count,
            "chroma_docs": chroma_count,
            "chroma_available": chroma is not None,
        }
