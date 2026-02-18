"""
memory/hybrid_memory.py
════════════════════════
Hybrid memory: SQLite (structured) + Chroma (semantic vector search).
Compatible with JarvisControllerV5.
"""

import sqlite3
import json
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

# ── Chroma lazy import ─────────────────────────────────────
_chroma_client = None
_chroma_collection = None


def _get_chroma(chroma_path: str, embedding_model: str):
    global _chroma_client, _chroma_collection
    if _chroma_collection is not None:
        return _chroma_collection
    try:
        import chromadb
        from chromadb.utils import embedding_functions
        _chroma_client = chromadb.PersistentClient(path=chroma_path)
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        _chroma_collection = _chroma_client.get_or_create_collection(
            name="jarvis_memory", embedding_function=ef,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("Chroma ready")
        return _chroma_collection
    except ImportError:
        logger.warning("chromadb or sentence-transformers not installed. Semantic search disabled.")
        return None
    except Exception as e:
        logger.error(f"Chroma init failed: {e}")
        return None


class HybridMemory:
    def __init__(
        self,
        db_path: str = "memory/memory.db",
        chroma_path: str = "data/chroma",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.db_path = db_path
        self.chroma_path = chroma_path
        self.embedding_model = embedding_model
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    def initialize(self) -> dict:
        self._init_sqlite()
        col = _get_chroma(self.chroma_path, self.embedding_model)
        mode = "hybrid" if col else "sqlite-only"
        logger.info(f"HybridMemory initialized | mode={mode}")
        return {"mode": mode}

    # ── SQLite ─────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_sqlite(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS preferences (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TEXT
                );
                CREATE TABLE IF NOT EXISTS episodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT,
                    category TEXT,
                    created_at TEXT
                );
                CREATE TABLE IF NOT EXISTS facts (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    category TEXT,
                    updated_at TEXT
                );
            """)

    def store_preference(self, key: str, value: str):
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO preferences (key, value, updated_at) VALUES (?,?,?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
            """, (key, str(value), now))
        self._chroma_add(f"preference: {key} = {value}", {"type": "preference", "key": key})
        logger.debug(f"Stored preference: {key}={value}")

    def store_episode(self, content: str, category: str = "general"):
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO episodes (content, category, created_at) VALUES (?,?,?)",
                (content, category, now)
            )
        self._chroma_add(content, {"type": "episode", "category": category})

    def store_fact(self, key: str, value: str, category: str = "general"):
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO facts (key, value, category, updated_at) VALUES (?,?,?,?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
            """, (key, str(value), category, now))
        self._chroma_add(f"{key}: {value}", {"type": "fact", "key": key})

    def get_preference(self, key: str) -> str | None:
        with self._conn() as conn:
            row = conn.execute("SELECT value FROM preferences WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    def get_all_preferences(self) -> dict:
        with self._conn() as conn:
            rows = conn.execute("SELECT key, value FROM preferences").fetchall()
        return {r["key"]: r["value"] for r in rows}

    def get_recent_episodes(self, limit: int = 10) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT content, category, created_at FROM episodes ORDER BY id DESC LIMIT ?",
                (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def build_context_block(self, query: str, n_results: int = 4) -> str:
        """Semantic search → formatted context string."""
        results = self.semantic_search(query, n_results)
        if not results:
            # Fallback: recent episodes
            episodes = self.get_recent_episodes(3)
            if not episodes:
                return ""
            return "\n".join(f"- {e['content']}" for e in episodes)
        return "\n".join(f"- {r['document']}" for r in results)

    # ── Chroma ─────────────────────────────────────────────

    def _chroma_add(self, text: str, metadata: dict):
        col = _get_chroma(self.chroma_path, self.embedding_model)
        if col is None:
            return
        try:
            doc_id = f"doc_{datetime.utcnow().timestamp()}"
            col.add(documents=[text], metadatas=[metadata], ids=[doc_id])
        except Exception as e:
            logger.warning(f"Chroma add failed: {e}")

    def semantic_search(self, query: str, n_results: int = 4) -> list[dict]:
        col = _get_chroma(self.chroma_path, self.embedding_model)
        if col is None:
            return []
        try:
            count = col.count()
            if count == 0:
                return []
            res = col.query(query_texts=[query], n_results=min(n_results, count))
            docs = res.get("documents", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
            dists = res.get("distances", [[]])[0]
            return [{"document": d, "metadata": m, "distance": dist}
                    for d, m, dist in zip(docs, metas, dists)]
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
