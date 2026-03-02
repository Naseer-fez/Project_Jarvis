"""
JARVIS Hybrid Memory - Session 5
SQLite: Structured facts, task history, user preferences
ChromaDB: Semantic embeddings for meaning-based recall
Embeddings: all-MiniLM-L6-v2 (local, offline)
"""

import sqlite3
import json
import logging
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional
import uuid

logger = logging.getLogger("JARVIS.HybridMemory")


class HybridMemory:
    def __init__(self, db_path: str = "jarvis_memory.db"):
        self.db_path = db_path
        self._sqlite_conn: Optional[sqlite3.Connection] = None
        self._chroma_client = None
        self._collection = None
        self._embedder = None
        self._chroma_available = False
        logger.info(f"HybridMemory created. DB: {db_path}")

    async def initialize(self):
        """Initialize both SQLite and ChromaDB."""
        await asyncio.to_thread(self._init_sqlite)
        await asyncio.to_thread(self._init_chroma)
        logger.info("HybridMemory fully initialized.")

    def _init_sqlite(self):
        """Initialize SQLite with all required tables."""
        self._sqlite_conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._sqlite_conn.row_factory = sqlite3.Row
        cur = self._sqlite_conn.cursor()

        cur.executescript("""
            CREATE TABLE IF NOT EXISTS jarvis_facts (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                key         TEXT UNIQUE NOT NULL,
                value       TEXT NOT NULL,
                category    TEXT DEFAULT 'general',
                confidence  REAL DEFAULT 1.0,
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS jarvis_tasks (
                id           TEXT PRIMARY KEY,
                intent       TEXT NOT NULL,
                plan_json    TEXT NOT NULL,
                status       TEXT DEFAULT 'pending',
                risk_score   INTEGER DEFAULT 0,
                started_at   TEXT,
                completed_at TEXT,
                result_json  TEXT
            );

            CREATE TABLE IF NOT EXISTS jarvis_conversations (
                id           TEXT PRIMARY KEY,
                source       TEXT DEFAULT 'cli',
                user_input   TEXT NOT NULL,
                jarvis_reply TEXT,
                transcript   TEXT,
                intent       TEXT,
                timestamp    TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS user_preferences (
                key         TEXT PRIMARY KEY,
                value       TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            );
        """)
        self._sqlite_conn.commit()
        logger.info("SQLite initialized with all tables.")

    def _init_chroma(self):
        """Initialize ChromaDB with sentence-transformers embeddings."""
        try:
            import chromadb
            from chromadb.config import Settings
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer('all-MiniLM-L6-v2')

            self._chroma_client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory="./chroma_db",
                anonymized_telemetry=False
            ))
            self._collection = self._chroma_client.get_or_create_collection(
                name="jarvis_conversations",
                metadata={"hnsw:space": "cosine"}
            )
            self._chroma_available = True
            logger.info("ChromaDB + SentenceTransformer initialized.")
        except ImportError as e:
            logger.warning(f"ChromaDB/SentenceTransformer not available: {e}. Semantic search disabled.")
        except Exception as e:
            logger.warning(f"ChromaDB init failed: {e}. Falling back to SQLite-only.")

    # ─── SQLite Operations ────────────────────────────────────────────────────

    def store_fact(self, key: str, value: str, category: str = "general", confidence: float = 1.0):
        now = datetime.now().isoformat()
        self._sqlite_conn.execute("""
            INSERT INTO jarvis_facts (key, value, category, confidence, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value=excluded.value,
                confidence=excluded.confidence,
                updated_at=excluded.updated_at
        """, (key, value, category, confidence, now, now))
        self._sqlite_conn.commit()

    def get_fact(self, key: str) -> Optional[str]:
        row = self._sqlite_conn.execute(
            "SELECT value FROM jarvis_facts WHERE key=?", (key,)
        ).fetchone()
        return row["value"] if row else None

    def log_task(self, intent: str, plan: dict, risk_score: int = 0) -> str:
        task_id = str(uuid.uuid4())
        self._sqlite_conn.execute("""
            INSERT INTO jarvis_tasks (id, intent, plan_json, risk_score, started_at)
            VALUES (?, ?, ?, ?, ?)
        """, (task_id, intent, json.dumps(plan), risk_score, datetime.now().isoformat()))
        self._sqlite_conn.commit()
        return task_id

    def update_task(self, task_id: str, status: str, result: dict = None):
        self._sqlite_conn.execute("""
            UPDATE jarvis_tasks SET status=?, completed_at=?, result_json=?
            WHERE id=?
        """, (status, datetime.now().isoformat(), json.dumps(result or {}), task_id))
        self._sqlite_conn.commit()

    def log_conversation(self, user_input: str, jarvis_reply: str,
                          source: str = "cli", transcript: str = "", intent: str = "") -> str:
        conv_id = str(uuid.uuid4())
        self._sqlite_conn.execute("""
            INSERT INTO jarvis_conversations
            (id, source, user_input, jarvis_reply, transcript, intent, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (conv_id, source, user_input, jarvis_reply, transcript, intent, datetime.now().isoformat()))
        self._sqlite_conn.commit()

        # Also add to ChromaDB for semantic search
        if self._chroma_available:
            self._add_to_chroma(conv_id, user_input + " " + jarvis_reply, {
                "source": source, "intent": intent, "timestamp": datetime.now().isoformat()
            })

        return conv_id

    # ─── ChromaDB Semantic Search ─────────────────────────────────────────────

    def _add_to_chroma(self, doc_id: str, text: str, metadata: dict):
        try:
            embedding = self._embedder.encode(text).tolist()
            self._collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[text],
                metadatas=[metadata]
            )
        except Exception as e:
            logger.warning(f"ChromaDB add failed: {e}")

    def semantic_search(self, query: str, n_results: int = 5) -> list[dict]:
        """Search memory by meaning, not just keywords."""
        if not self._chroma_available:
            logger.warning("Semantic search unavailable — ChromaDB not initialized.")
            return []
        try:
            embedding = self._embedder.encode(query).tolist()
            results = self._collection.query(
                query_embeddings=[embedding],
                n_results=n_results
            )
            items = []
            for i, doc in enumerate(results.get("documents", [[]])[0]):
                items.append({
                    "text": doc,
                    "distance": results["distances"][0][i],
                    "metadata": results["metadatas"][0][i]
                })
            return items
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []

    def get_recent_conversations(self, limit: int = 10) -> list[dict]:
        rows = self._sqlite_conn.execute("""
            SELECT * FROM jarvis_conversations ORDER BY timestamp DESC LIMIT ?
        """, (limit,)).fetchall()
        return [dict(r) for r in rows]

    def get_preferences(self) -> dict:
        rows = self._sqlite_conn.execute("SELECT key, value FROM user_preferences").fetchall()
        return {r["key"]: r["value"] for r in rows}

    def set_preference(self, key: str, value: str):
        self._sqlite_conn.execute("""
            INSERT INTO user_preferences (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
        """, (key, value, datetime.now().isoformat()))
        self._sqlite_conn.commit()
