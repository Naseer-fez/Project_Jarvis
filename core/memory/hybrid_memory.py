"""Hybrid memory: SQLite for structure plus optional Chroma semantic memory."""

from __future__ import annotations

import ast
import hashlib
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from core.memory.semantic_memory import SemanticMemory

logger = logging.getLogger(__name__)


def _looks_like_config(value: Any) -> bool:
    return hasattr(value, "get") and hasattr(value, "has_option")


class HybridMemory:
    def __init__(
        self,
        config_or_db_path: Any = "memory/memory.db",
        chroma_path: str = "data/chroma",
        model_name: str = "all-MiniLM-L6-v2",
    ):
        if _looks_like_config(config_or_db_path):
            cfg = config_or_db_path
            db_path = cfg.get("memory", "sqlite_file", fallback=cfg.get("memory", "db_path", fallback="memory/memory.db"))
            chroma_path = cfg.get("memory", "chroma_dir", fallback=cfg.get("memory", "chroma_path", fallback=chroma_path))
            model_name = cfg.get("memory", "embedding_model", fallback=model_name)
        else:
            db_path = str(config_or_db_path)

        self.db_path = db_path
        self.chroma_path = chroma_path
        self.model_name = model_name
        self.mode = "sqlite-only"

        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self.semantic = SemanticMemory(chroma_path=self.chroma_path, model_name=self.model_name)

    def initialize(self, index_path: str = "") -> dict[str, Any]:
        self._init_sqlite()
        semantic_ready = False
        try:
            semantic_ready = bool(self.semantic.initialize())
        except Exception as exc:  # noqa: BLE001
            logger.warning("Semantic memory initialization failed: %s", exc)
            semantic_ready = False

        self.mode = "hybrid" if semantic_ready else "sqlite-only"
        result: dict[str, Any] = {
            "mode": self.mode,
            "sqlite": True,
            "semantic": semantic_ready,
        }

        if index_path:
            result["codebase_index"] = self.index_codebase(index_path)

        return result

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_sqlite(self) -> None:
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS preferences (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TEXT
                );

                CREATE TABLE IF NOT EXISTS episodes (
                    id INTEGER PRIMARY KEY,
                    event TEXT,
                    category TEXT,
                    timestamp TEXT
                );

                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY,
                    user_input TEXT,
                    assistant_response TEXT,
                    session_id TEXT,
                    timestamp TEXT
                );
                """
            )

    def store_preference(self, key: str, value: str) -> bool:
        now = datetime.now().isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO preferences (key, value, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (key, value, now),
            )

        if self.mode == "hybrid":
            try:
                self.semantic.store_preference(key, value)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Semantic preference store failed: %s", exc)
        return True

    def store_episode(self, event: str, category: str = "general") -> bool:
        now = datetime.now().isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO episodes (event, category, timestamp) VALUES (?, ?, ?)",
                (event, category, now),
            )

        if self.mode == "hybrid":
            try:
                self.semantic.store_episode(event, category)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Semantic episode store failed: %s", exc)
        return True

    def store_conversation(self, user_input: str, assistant_response: str, session_id: str = "default") -> bool:
        now = datetime.now().isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO conversations (user_input, assistant_response, session_id, timestamp) VALUES (?, ?, ?, ?)",
                (user_input, assistant_response, session_id, now),
            )

        if self.mode == "hybrid":
            try:
                self.semantic.store_conversation_turn(user_input, assistant_response, session_id)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Semantic conversation store failed: %s", exc)
        return True

    def recall_preferences(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if self.mode == "hybrid":
            try:
                raw = self.semantic.recall_preferences(query, top_k=top_k, threshold=0.0)
                return [
                    {
                        "key": item.get("metadata", {}).get("key", ""),
                        "value": item.get("metadata", {}).get("value", ""),
                        "score": item.get("score", 0.0),
                        "document": item.get("document", ""),
                    }
                    for item in raw
                ]
            except Exception as exc:  # noqa: BLE001
                logger.debug("Semantic preference recall failed: %s", exc)

        with self._conn() as conn:
            rows = conn.execute(
                "SELECT key, value FROM preferences ORDER BY updated_at DESC LIMIT ?",
                (max(1, top_k),),
            ).fetchall()
        return [{"key": row["key"], "value": row["value"], "score": 1.0} for row in rows]

    def recall_all(self, query: str, top_k: int = 5) -> dict[str, list[dict[str, Any]]]:
        if self.mode == "hybrid":
            try:
                raw = self.semantic.recall_all(query, top_k=top_k, threshold=0.0)
                return {
                    "preferences": [
                        {
                            "key": item.get("metadata", {}).get("key", ""),
                            "value": item.get("metadata", {}).get("value", ""),
                            "score": item.get("score", 0.0),
                            "document": item.get("document", ""),
                        }
                        for item in raw.get("preferences", [])
                    ],
                    "episodes": [
                        {
                            "event": item.get("metadata", {}).get("event", item.get("document", "")),
                            "category": item.get("metadata", {}).get("category", ""),
                            "timestamp": item.get("metadata", {}).get("timestamp", ""),
                            "score": item.get("score", 0.0),
                            "document": item.get("document", ""),
                        }
                        for item in raw.get("episodes", [])
                    ],
                    "conversations": [
                        {
                            "user_input": item.get("metadata", {}).get("user_input", ""),
                            "assistant_response": item.get("metadata", {}).get("assistant_response", ""),
                            "timestamp": item.get("metadata", {}).get("timestamp", ""),
                            "score": item.get("score", 0.0),
                            "document": item.get("document", ""),
                        }
                        for item in raw.get("conversations", [])
                    ],
                }
            except Exception as exc:  # noqa: BLE001
                logger.debug("Semantic recall_all failed: %s", exc)

        return {
            "preferences": self.recall_preferences(query, top_k=top_k),
            "episodes": [],
            "conversations": [],
        }

    def build_context_block(self, query: str, n_results: int = 5) -> str:
        try:
            from core.memory.context_compressor import ContextCompressor

            compressor = ContextCompressor(threshold=0.0)
            return compressor.compress(query, self.recall_all(query, top_k=n_results))
        except Exception:  # noqa: BLE001
            return ""

    def recall(self, query: str, top_k: int = 5) -> str:
        """Backward-compatible text recall helper used by legacy paths."""
        hits = self.recall_preferences(query, top_k=top_k)
        if not hits:
            return "I don't know yet. I could not find related facts."

        lines = []
        for item in hits:
            key = item.get("key", "")
            value = item.get("value", "")
            if key or value:
                lines.append(f"{key}: {value}".strip(": "))
        if not lines:
            return "I don't know yet. I could not find related facts."
        return "\n".join(lines)

    def store_code_file(self, file_path: str, content: str) -> int:
        """Parse Python code and store class/function chunks for semantic retrieval."""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            self.store_episode(f"file:{file_path}", category="code")
            return 1

        lines = content.splitlines()
        chunks_stored = 0

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue

            start = max(0, getattr(node, "lineno", 1) - 1)
            end_lineno = getattr(node, "end_lineno", None) or getattr(node, "lineno", 1)
            end = min(len(lines), max(start + 1, end_lineno))
            chunk = "\n".join(lines[start:end]).strip()
            if not chunk:
                continue

            chunk_id = f"{file_path}::{getattr(node, 'name', 'anonymous')}"
            metadata = {
                "file": file_path,
                "name": getattr(node, "name", ""),
                "type": type(node).__name__,
                "lines": f"{start + 1}-{end}",
            }

            if self.mode == "hybrid":
                self._semantic_store_code_chunk(chunk_id, chunk, metadata)

            # Keep a structured breadcrumb in sqlite even when semantic is disabled.
            self.store_episode(f"{chunk_id}\n{chunk[:3000]}", category="code")
            chunks_stored += 1

        if chunks_stored == 0:
            self.store_episode(f"file:{file_path}", category="code")
            return 1

        return chunks_stored

    def _semantic_store_code_chunk(self, chunk_id: str, chunk: str, metadata: dict[str, Any]) -> None:
        try:
            if hasattr(self.semantic, "store_code_chunk"):
                self.semantic.store_code_chunk(chunk_id, chunk, metadata=metadata)
            else:
                self.semantic.store_episode(f"{chunk_id}\n{chunk}", category="code")
        except Exception as exc:  # noqa: BLE001
            logger.debug("Semantic code chunk store failed: %s", exc)

    def index_codebase(self, root_path: str) -> dict[str, int]:
        """Index changed Python files only, based on content hash."""
        stats = {
            "indexed_files": 0,
            "indexed_chunks": 0,
            "skipped_files": 0,
            "errors": 0,
        }

        root = Path(root_path)
        if not root.exists():
            stats["errors"] += 1
            logger.warning("Codebase index path does not exist: %s", root_path)
            return stats

        exclude_dirs = {"__pycache__", ".git", ".venv", "venv", "node_modules", "jarvis_env"}

        for py_file in root.rglob("*.py"):
            if any(part in exclude_dirs for part in py_file.parts):
                continue

            try:
                content = py_file.read_text(encoding="utf-8", errors="replace")
                content_hash = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()
                hash_key = f"code_hash::{py_file.resolve()}"

                with self._conn() as conn:
                    row = conn.execute("SELECT value FROM preferences WHERE key=?", (hash_key,)).fetchone()
                    if row and row["value"] == content_hash:
                        stats["skipped_files"] += 1
                        continue

                    conn.execute(
                        "INSERT INTO preferences (key, value, updated_at) VALUES (?, ?, ?) "
                        "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                        (hash_key, content_hash, datetime.now().isoformat()),
                    )

                chunks = self.store_code_file(str(py_file), content)
                stats["indexed_files"] += 1
                stats["indexed_chunks"] += int(chunks)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to index %s: %s", py_file, exc)
                stats["errors"] += 1

        return stats

    def stats(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "sqlite": True,
            "semantic": self.mode == "hybrid",
        }


__all__ = ["HybridMemory"]
