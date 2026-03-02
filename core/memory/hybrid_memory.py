"""
Hybrid memory: SQLite (structured) + Chroma (semantic vector search).
"""

from __future__ import annotations

import hashlib
import sqlite3
import logging
from datetime import datetime
from pathlib import Path

from core.memory.semantic_memory import SemanticMemory

logger = logging.getLogger(__name__)


class HybridMemory:
    def __init__(
        self,
        db_path: str = "memory/memory.db",
        chroma_path: str = "data/chroma",
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.db_path = db_path
        self.chroma_path = chroma_path
        self.model_name = model_name
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialize semantic layer
        self.semantic = SemanticMemory(chroma_path=self.chroma_path, model_name=self.model_name)
        self.mode = "sqlite-only"

    def initialize(self, index_path: str = "") -> dict:
        self._init_sqlite()
        semantic_ready = self.semantic.initialize()
        self.mode = "hybrid" if semantic_ready else "sqlite-only"
        result = {"mode": self.mode, "sqlite": True, "semantic": semantic_ready}
        if index_path:
            index_stats = self.index_codebase(index_path)
            result["codebase_index"] = index_stats
            logger.info("Codebase indexed: %s", index_stats)
        return result

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_sqlite(self):
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS preferences (key TEXT PRIMARY KEY, value TEXT, updated_at TEXT);
                CREATE TABLE IF NOT EXISTS episodes (id INTEGER PRIMARY KEY, event TEXT, category TEXT, timestamp TEXT);
                CREATE TABLE IF NOT EXISTS conversations (id INTEGER PRIMARY KEY, user_input TEXT, assistant_response TEXT, session_id TEXT, timestamp TEXT);
            """
            )

    def store_preference(self, key: str, value: str) -> bool:
        now = datetime.now().isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO preferences (key, value, updated_at) VALUES (?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (key, value, now),
            )
        if self.mode == "hybrid":
            self.semantic.store_preference(key, value)
        return True

    def store_episode(self, event: str, category: str = "general") -> bool:
        now = datetime.now().isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO episodes (event, category, timestamp) VALUES (?,?,?)",
                (event, category, now),
            )
        if self.mode == "hybrid":
            self.semantic.store_episode(event, category)
        return True

    def store_conversation(self, user_input: str, assistant_response: str, session_id: str = "default") -> bool:
        now = datetime.now().isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO conversations (user_input, assistant_response, session_id, timestamp) VALUES (?,?,?,?)",
                (user_input, assistant_response, session_id, now),
            )
        if self.mode == "hybrid":
            self.semantic.store_conversation_turn(user_input, assistant_response, session_id)
        return True

    def recall_preferences(self, query: str, top_k: int = 5) -> list:
        if self.mode == "hybrid":
            raw = self.semantic.recall_preferences(query, top_k=top_k, threshold=0.0)
            return [
                {
                    "key": r.get("metadata", {}).get("key", ""),
                    "value": r.get("metadata", {}).get("value", ""),
                    "score": r.get("score", 0.0),
                    "document": r.get("document", ""),
                }
                for r in raw
            ]

        with self._conn() as conn:
            rows = conn.execute("SELECT key, value FROM preferences LIMIT ?", (top_k,)).fetchall()
            return [{"key": r["key"], "value": r["value"], "score": 1.0} for r in rows]

    def recall_all(self, query: str, top_k: int = 5) -> dict:
        if self.mode == "hybrid":
            raw = self.semantic.recall_all(query, top_k=top_k, threshold=0.0)
            return {
                "preferences": [
                    {
                        "key": r.get("metadata", {}).get("key", ""),
                        "value": r.get("metadata", {}).get("value", ""),
                        "score": r.get("score", 0.0),
                        "document": r.get("document", ""),
                    }
                    for r in raw.get("preferences", [])
                ],
                "episodes": [
                    {
                        "event": r.get("metadata", {}).get("event", r.get("document", "")),
                        "category": r.get("metadata", {}).get("category", ""),
                        "timestamp": r.get("metadata", {}).get("timestamp", ""),
                        "score": r.get("score", 0.0),
                        "document": r.get("document", ""),
                    }
                    for r in raw.get("episodes", [])
                ],
                "conversations": [
                    {
                        "user_input": r.get("metadata", {}).get("user_input", ""),
                        "assistant_response": r.get("metadata", {}).get("assistant_response", ""),
                        "timestamp": r.get("metadata", {}).get("timestamp", ""),
                        "score": r.get("score", 0.0),
                        "document": r.get("document", ""),
                    }
                    for r in raw.get("conversations", [])
                ],
            }

        return {"preferences": self.recall_preferences(query, top_k), "episodes": [], "conversations": []}

    def build_context_block(self, query: str, n_results: int = 5) -> str:
        try:
            from core.memory.context_compressor import ContextCompressor

            cc = ContextCompressor(threshold=0.0)
            return cc.compress(query, self.recall_all(query, top_k=n_results))
        except ImportError:
            return ""

    def store_code_file(self, file_path: str, content: str) -> int:
        """
        Parse a Python file with AST and store each function/class as a
        separate semantic chunk. Returns number of chunks stored.
        """
        import ast

        chunks_stored = 0
        try:
            tree = ast.parse(content)
        except SyntaxError:
            # Fall back to storing the whole file as one chunk
            self.store_episode(f"file:{file_path}", category="code")
            return 1

        lines = content.splitlines()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = node.lineno - 1
                end = node.end_lineno
                chunk = "\n".join(lines[start:end])
                chunk_id = f"{file_path}::{node.name}"

                if self.mode == "hybrid":
                    if hasattr(self.semantic, "store_code_chunk"):
                        self.semantic.store_code_chunk(
                            chunk_id,
                            chunk,
                            metadata={
                                "file": file_path,
                                "name": node.name,
                                "type": type(node).__name__,
                                "lines": f"{node.lineno}-{node.end_lineno}",
                            },
                        )
                    else:
                        self.semantic.store_episode(f"{chunk_id}\n{chunk}", category="code")

                chunks_stored += 1

        if chunks_stored == 0:
            self.store_episode(f"file:{file_path}", category="code")
            return 1

        return chunks_stored

    def index_codebase(self, root_path: str) -> dict:
        """
        Hash all .py files in root_path. Store new/changed ones via
        store_code_file().

        Returns {"indexed": N, "skipped": N, "errors": N}
        """
        stats = {"indexed": 0, "skipped": 0, "errors": 0}
        root = Path(root_path)

        for py_file in root.rglob("*.py"):
            if any(p in py_file.parts for p in ("__pycache__", ".git", "venv", ".venv", "node_modules")):
                continue

            try:
                content = py_file.read_text(encoding="utf-8", errors="replace")
                file_hash = hashlib.md5(content.encode()).hexdigest()
                cache_key = f"file_hash:{py_file}"

                with self._conn() as conn:
                    row = conn.execute(
                        "SELECT value FROM preferences WHERE key=?",
                        (cache_key,),
                    ).fetchone()
                    if row and row["value"] == file_hash:
                        stats["skipped"] += 1
                        continue

                    conn.execute(
                        "INSERT INTO preferences (key, value, updated_at) VALUES (?,?,?) "
                        "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                        (cache_key, file_hash, datetime.now().isoformat()),
                    )

                n = self.store_code_file(str(py_file), content)
                stats["indexed"] += n
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to index %s: %s", py_file, exc)
                stats["errors"] += 1

        return stats

    def stats(self) -> dict:
        return {"mode": self.mode, "sqlite": True, "semantic": self.mode == "hybrid"}
