"""Hybrid memory: SQLite for structure plus optional Chroma semantic memory."""

from __future__ import annotations

import contextlib
import hashlib
import logging
import re
import asyncio
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
        *,
        db_path: str | None = None,
    ):
        """
        Accepts either:
          - HybridMemory(config)         — ConfigParser / dict-like
          - HybridMemory("path/to.db")   — positional path string
          - HybridMemory(db_path="path/to.db", ...)  — explicit keyword (legacy tests)
        """
        # If db_path keyword is given, it takes precedence over positional arg
        if db_path is not None:
            resolved_db = db_path
        elif _looks_like_config(config_or_db_path):
            cfg = config_or_db_path
            resolved_db = cfg.get("memory", "sqlite_file", fallback=cfg.get("memory", "db_path", fallback="memory/memory.db"))
            chroma_path = cfg.get("memory", "chroma_dir", fallback=cfg.get("memory", "chroma_path", fallback=chroma_path))
            model_name = cfg.get("memory", "embedding_model", fallback=model_name)
        else:
            resolved_db = str(config_or_db_path)

        self.db_path = resolved_db
        self.chroma_path = chroma_path
        self.model_name = model_name
        self.mode = "sqlite-only"
        self._llm: Any | None = None
        self._enable_llm_context_titles = True

        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self.semantic = SemanticMemory(chroma_path=self.chroma_path, model_name=self.model_name)

        from core.memory.sqlite_pool import SQLitePool
        self._pool = SQLitePool(self.db_path, pool_size=3)

        self._init_lock = asyncio.Lock()
        self._sqlite_initialized = False
        self._background_tasks: set[asyncio.Task[Any]] = set()

    async def initialize(self, index_path: str = "") -> dict[str, Any]:
        await self._ensure_db_initialized()
        semantic_ready = False
        try:
            semantic_ready = bool(await self.semantic.initialize())
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
            self._track_background_task(
                asyncio.create_task(
                    self.index_codebase(index_path),
                    name="hybrid_memory_code_index",
                )
            )
            result["codebase_index"] = {"status": "background_indexing_started"}

        return result

    def _track_background_task(self, task: asyncio.Task[Any]) -> None:
        self._background_tasks.add(task)

        def _finalize_background_task(done_task: asyncio.Task[Any]) -> None:
            self._background_tasks.discard(done_task)
            with contextlib.suppress(asyncio.CancelledError):
                exc = done_task.exception()
                if exc is not None:
                    logger.warning("HybridMemory background task failed: %s", exc)

        task.add_done_callback(_finalize_background_task)

    def set_llm(
        self,
        llm: Any | None,
        *,
        enable_context_titles: bool = True,
    ) -> None:
        """Attach an LLM used for optional low-latency context titling."""
        self._llm = llm
        self._enable_llm_context_titles = bool(enable_context_titles)

    async def _ensure_db_initialized(self) -> None:
        if self._sqlite_initialized:
            return
        async with self._init_lock:
            if not self._sqlite_initialized:
                await self._init_sqlite()
                self._sqlite_initialized = True

    async def _init_sqlite(self) -> None:
        async with self._pool.acquire() as conn:
            await conn.executescript(
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

                CREATE TABLE IF NOT EXISTS actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action TEXT NOT NULL,
                    result TEXT,
                    success INTEGER NOT NULL DEFAULT 1,
                    metadata TEXT,
                    timestamp TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_preferences_updated_at ON preferences(updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_episodes_timestamp ON episodes(timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_conversations_timestamp ON conversations(timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_actions_timestamp ON actions(timestamp DESC);
                """
            )

    async def store_preference(self, key: str, value: str) -> bool:
        await self._ensure_db_initialized()
        now = datetime.now().isoformat()
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO preferences (key, value, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (key, value, now),
            )

        if self.mode == "hybrid":
            try:
                await self.semantic.store_preference(key, value)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Semantic preference store failed: %s", exc)
        return True

    async def store_episode(self, event: str, category: str = "general") -> bool:
        await self._ensure_db_initialized()
        now = datetime.now().isoformat()
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO episodes (event, category, timestamp) VALUES (?, ?, ?)",
                (event, category, now),
            )

        if self.mode == "hybrid":
            try:
                await self.semantic.store_episode(event, category)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Semantic episode store failed: %s", exc)
        return True

    async def store_episodes_batch(self, events: list[str], category: str = "general") -> bool:
        await self._ensure_db_initialized()
        now = datetime.now().isoformat()
        
        async with self._pool.acquire() as conn:
            await conn.executemany(
                "INSERT INTO episodes (event, category, timestamp) VALUES (?, ?, ?)",
                [(event, category, now) for event in events],
            )

        if self.mode == "hybrid":
            try:
                if hasattr(self.semantic, "store_episodes_batch"):
                    await self.semantic.store_episodes_batch(events, category=category)
                else:
                    for event in events:
                        await self.semantic.store_episode(event, category=category)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Semantic batch store failed: %s", exc)
                
        return True


    async def store_conversation(self, user_input: str, assistant_response: str, session_id: str = "default") -> bool:
        await self._ensure_db_initialized()
        now = datetime.now().isoformat()
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO conversations (user_input, assistant_response, session_id, timestamp) VALUES (?, ?, ?, ?)",
                (user_input, assistant_response, session_id, now),
            )

        if self.mode == "hybrid":
            try:
                await self.semantic.store_conversation_turn(user_input, assistant_response, session_id)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Semantic conversation store failed: %s", exc)
        return True

    async def recall_preferences(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if self.mode == "hybrid":
            try:
                raw = await self.semantic.recall_preferences(query, top_k=top_k, threshold=0.0)
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

        await self._ensure_db_initialized()
        tokens = self._query_tokens(query)
        async with self._pool.acquire() as conn:
            async with conn.execute(
                "SELECT key, value FROM preferences ORDER BY updated_at DESC LIMIT 200"
            ) as cursor:
                rows = await cursor.fetchall()

        ranked: list[dict[str, Any]] = []
        for row in rows:
            key = str(row["key"] or "")
            val = str(row["value"] or "")
            score = self._score_text(key, tokens)
            if tokens and score < 0.70:
                continue
            ranked.append(
                {
                    "key": key,
                    "value": val,
                    "score": score if tokens else 1.0,
                }
            )

        if tokens:
            ranked.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        return ranked[: max(1, top_k)]

    async def recall_all(self, query: str, top_k: int = 5) -> dict[str, list[dict[str, Any]]]:
        if self.mode == "hybrid":
            try:
                raw = await self.semantic.recall_all(query, top_k=top_k, threshold=0.0)
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
            "preferences": await self.recall_preferences(query, top_k=top_k),
            "episodes": await self._recall_sqlite_episodes(query, top_k=top_k),
            "conversations": await self._recall_sqlite_conversations(query, top_k=top_k),
        }

    @staticmethod
    def _query_tokens(query: str) -> list[str]:
        tokens = re.findall(r"[a-z0-9]{3,}", str(query or "").lower())
        return tokens[:10]

    @staticmethod
    def _score_text(text: str, tokens: list[str]) -> float:
        if not tokens:
            return 0.5
        lowered = str(text or "").lower()
        hits = sum(1 for token in tokens if token in lowered)
        if hits <= 0:
            return 0.0
        return min(1.0, hits / max(1.0, float(len(tokens))))

    async def _recall_sqlite_episodes(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        tokens = self._query_tokens(query)
        await self._ensure_db_initialized()
        async with self._pool.acquire() as conn:
            async with conn.execute(
                "SELECT event, category, timestamp FROM episodes ORDER BY timestamp DESC LIMIT 200"
            ) as cursor:
                rows = await cursor.fetchall()

        ranked: list[dict[str, Any]] = []
        for row in rows:
            event = str(row["event"] or "")
            category = str(row["category"] or "")
            haystack = f"{event} {category}".strip()
            score = self._score_text(haystack, tokens)
            if tokens and score <= 0.0:
                continue
            ranked.append(
                {
                    "event": event,
                    "category": category,
                    "timestamp": str(row["timestamp"] or ""),
                    "score": score if tokens else 0.4,
                    "document": event,
                }
            )

        ranked.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        return ranked[: max(1, top_k)]

    async def _recall_sqlite_conversations(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        tokens = self._query_tokens(query)
        await self._ensure_db_initialized()
        async with self._pool.acquire() as conn:
            async with conn.execute(
                "SELECT user_input, assistant_response, timestamp FROM conversations ORDER BY timestamp DESC LIMIT 200"
            ) as cursor:
                rows = await cursor.fetchall()

        ranked: list[dict[str, Any]] = []
        for row in rows:
            user_text = str(row["user_input"] or "")
            assistant_text = str(row["assistant_response"] or "")
            haystack = f"{user_text} {assistant_text}".strip()
            score = self._score_text(haystack, tokens)
            if tokens and score <= 0.0:
                continue
            ranked.append(
                {
                    "user_input": user_text,
                    "assistant_response": assistant_text,
                    "timestamp": str(row["timestamp"] or ""),
                    "score": score if tokens else 0.4,
                    "document": f"User: {user_text}\nAssistant: {assistant_text}",
                }
            )

        ranked.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        return ranked[: max(1, top_k)]

    async def build_context_block(self, query: str, n_results: int = 5) -> str:
        try:
            from core.memory.context_compressor import ContextCompressor

            compressor = ContextCompressor(
                threshold=0.0,
                llm=self._llm,
                enable_llm_title=self._enable_llm_context_titles,
            )
            recalled = await self.recall_all(query, top_k=n_results)
            return await compressor.compress(query, recalled)
        except Exception:  # noqa: BLE001
            return ""

    async def recall(self, query: str, top_k: int = 5) -> str:
        """Backward-compatible text recall helper used by legacy paths."""
        hits = await self.recall_preferences(query, top_k=top_k)
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

    async def store_code_file(self, file_path: str, content: str) -> int:
        """Parse Python code and store class/function chunks for semantic retrieval."""
        from core.memory.code_indexer import extract_code_chunks
        
        chunks = extract_code_chunks(file_path, content)
        chunks_stored = 0
        
        for item in chunks:
            chunk_id = item["chunk_id"]
            chunk = item["chunk"]
            metadata = item["metadata"]

            if self.mode == "hybrid":
                await self._semantic_store_code_chunk(chunk_id, chunk, metadata)

            # Keep a structured breadcrumb in sqlite even when semantic is disabled.
            await self.store_episode(f"{chunk_id}\n{chunk[:3000]}", category="code")
            chunks_stored += 1
            
        return max(1, chunks_stored)

    async def _semantic_store_code_chunk(self, chunk_id: str, chunk: str, metadata: dict[str, Any]) -> None:
        try:
            if hasattr(self.semantic, "store_code_chunk"):
                await self.semantic.store_code_chunk(chunk_id, chunk, metadata=metadata)
            else:
                await self.semantic.store_episode(f"{chunk_id}\n{chunk}", category="code")
        except Exception as exc:  # noqa: BLE001
            logger.debug("Semantic code chunk store failed: %s", exc)

    async def index_codebase(self, root_path: str) -> dict[str, int]:
        """Index changed Python files only, based on content hash, in batches of 32 chunks."""
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

        py_files = []
        for py_file in root.rglob("*.py"):
            if any(part in exclude_dirs for part in py_file.parts):
                continue
            py_files.append(py_file)
            await asyncio.sleep(0)

        chunks_to_index = []

        for py_file in py_files:
            try:
                # Read content asynchronously via thread pool
                content = await asyncio.to_thread(py_file.read_text, encoding="utf-8", errors="replace")
                content_hash = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()
                hash_key = f"code_hash::{py_file.resolve()}"

                await self._ensure_db_initialized()
                async with self._pool.acquire() as conn:
                    async with conn.execute("SELECT value FROM preferences WHERE key=?", (hash_key,)) as cursor:
                        row = await cursor.fetchone()
                    if row and row["value"] == content_hash:
                        stats["skipped_files"] += 1
                        continue

                    await conn.execute(
                        "INSERT INTO preferences (key, value, updated_at) VALUES (?, ?, ?) "
                        "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                        (hash_key, content_hash, datetime.now().isoformat()),
                    )

                from core.memory.code_indexer import extract_code_chunks
                extracted = extract_code_chunks(str(py_file), content)
                chunks_to_index.extend(extracted)
                stats["indexed_files"] += 1

            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to index %s: %s", py_file, exc)
                stats["errors"] += 1

            while len(chunks_to_index) >= 32:
                batch = chunks_to_index[:32]
                chunks_to_index = chunks_to_index[32:]
                await self._index_chunks_batch(batch, stats)
                await asyncio.sleep(0.01)

            await asyncio.sleep(0.01)

        if chunks_to_index:
            await self._index_chunks_batch(chunks_to_index, stats)

        return stats

    async def _index_chunks_batch(self, batch: list[dict], stats: dict) -> None:
        """Index a batch of code chunks in a single operation."""
        # 1. SQLite Breadcrumbs
        try:
            await self._ensure_db_initialized()
            async with self._pool.acquire() as conn:
                for item in batch:
                    await conn.execute(
                        "INSERT INTO episodes (event, category, timestamp) VALUES (?, ?, ?)",
                        (f"{item['chunk_id']}\n{item['chunk'][:3000]}", "code", datetime.now().isoformat())
                    )
        except Exception as exc:
            logger.warning("Failed to store batch chunks to SQLite: %s", exc)

        # 2. Chroma batched write
        if self.mode == "hybrid":
            try:
                events = [f"{item['chunk_id']}\n{item['chunk']}" for item in batch]
                if hasattr(self.semantic, "store_episodes_batch"):
                    await self.semantic.store_episodes_batch(events, category="code")
                else:
                    for event in events:
                        await self.semantic.store_episode(event, category="code")
            except Exception as exc:
                logger.warning("Failed to store batch chunks to Chroma: %s", exc)

        stats["indexed_chunks"] += len(batch)

    def stats(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "sqlite": True,
            "semantic": self.mode == "hybrid",
        }

    # ── Backward-compat fact API (used by legacy tests) ───────────────────

    async def store_fact(self, key: str, value: str, source: str = "user", **_kwargs) -> None:
        """Store a key-value fact (alias for store_preference)."""
        await self.store_preference(key, value)

    async def get_fact(self, key: str):
        """Return a simple object with .key and .value, or None if not found."""
        await self._ensure_db_initialized()
        async with self._pool.acquire() as conn:
            async with conn.execute(
                "SELECT key, value FROM preferences WHERE key=?", (key,)
            ) as cursor:
                row = await cursor.fetchone()
        if row is None:
            return None

        class _Fact:
            def __init__(self, k, v):
                self.key = k
                self.value = v
            def __repr__(self):
                return f"Fact(key={self.key!r}, value={self.value!r})"

        return _Fact(row["key"], row["value"])

    async def list_facts(self, limit: int = 50) -> list:
        """Return recent facts as objects with .key and .value."""
        await self._ensure_db_initialized()
        async with self._pool.acquire() as conn:
            async with conn.execute(
                "SELECT key, value FROM preferences ORDER BY updated_at DESC LIMIT ?",
                (max(1, limit),),
            ) as cursor:
                rows = await cursor.fetchall()

        class _Fact:
            def __init__(self, k, v):
                self.key = k
                self.value = v

        return [_Fact(r["key"], r["value"]) for r in rows]

    async def count(self) -> int:
        """Return number of stored facts/preferences."""
        await self._ensure_db_initialized()
        async with self._pool.acquire() as conn:
            async with conn.execute("SELECT COUNT(*) FROM preferences") as cursor:
                row = await cursor.fetchone()
        return row[0] if row else 0

    # ── Extended action-tracking API (used by Session 4/5 tests) ─────────

    async def store_action(self, action: str, result: str = "", success: bool = True, metadata: dict | None = None) -> None:
        """Record a tool action in the actions table."""
        import json as _json
        await self._ensure_db_initialized()
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO actions (action, result, success, metadata, timestamp) VALUES (?, ?, ?, ?, ?)",
                (action, result, int(success), _json.dumps(metadata or {}), datetime.now().isoformat()),
            )

    async def store_failure(self, action: str, error: str = "", metadata: dict | None = None) -> None:
        """Record a failed tool call."""
        await self.store_action(action, result=error, success=False, metadata=metadata)

    async def recent_actions(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return recent actions, newest first."""
        import json as _json
        await self._ensure_db_initialized()
        async with self._pool.acquire() as conn:
            async with conn.execute(
                "SELECT action, result, success, metadata, timestamp FROM actions ORDER BY id DESC LIMIT ?",
                (max(1, limit),),
            ) as cursor:
                rows = await cursor.fetchall()
        return [
            {
                "action": r["action"],
                "result": r["result"],
                "success": bool(r["success"]),
                "metadata": _json.loads(r["metadata"] or "{}"),
                "timestamp": r["timestamp"],
            }
            for r in rows
        ]

    async def set_preference(self, key: str, value: str, category: str = "general", **_kwargs) -> None:
        """Store a categorised preference (category stored as prefix key)."""
        scoped_key = f"{category}::{key}" if category and category != "general" else key
        await self.store_preference(scoped_key, value)
        # Also store the bare key for lookup without category prefix
        await self.store_preference(key, value)

    async def get_preferences(self, category: str = "") -> dict[str, str]:
        """Return preferences matching the given category prefix."""
        prefix = f"{category}::" if category else ""
        await self._ensure_db_initialized()
        async with self._pool.acquire() as conn:
            if prefix:
                async with conn.execute(
                    "SELECT key, value FROM preferences WHERE key LIKE ? ORDER BY updated_at DESC",
                    (f"{prefix}%",),
                ) as cursor:
                    rows = await cursor.fetchall()
                return {r["key"].removeprefix(prefix): r["value"] for r in rows}
            async with conn.execute(
                "SELECT key, value FROM preferences ORDER BY updated_at DESC"
            ) as cursor:
                rows = await cursor.fetchall()
            return {r["key"]: r["value"] for r in rows}

    async def cleanup_stale_data(self, max_age_days: int = 30) -> dict[str, int]:
        """Remove episodes/actions older than max_age_days. Returns removal counts."""
        from datetime import timedelta
        cutoff = (datetime.now() - timedelta(days=max_age_days)).isoformat()

        removed_episodes = 0
        removed_actions = 0

        await self._ensure_db_initialized()
        async with self._pool.acquire() as conn:
            cur = await conn.execute("DELETE FROM episodes WHERE timestamp < ?", (cutoff,))
            removed_episodes = cur.rowcount

            cur = await conn.execute("DELETE FROM actions WHERE timestamp < ?", (cutoff,))
            removed_actions = cur.rowcount

        return {
            "episodes": removed_episodes,
            "actions": removed_actions,
        }
    async def close(self) -> None:
        """Close database pool and release resources."""
        background_tasks = list(self._background_tasks)
        for task in background_tasks:
            task.cancel()
        if background_tasks:
            await asyncio.gather(*background_tasks, return_exceptions=True)
        self._background_tasks.clear()

        if hasattr(self, "_pool") and self._pool is not None:
            await self._pool.close()

        if hasattr(self, "semantic") and self.semantic is not None:
            await self.semantic.close()


__all__ = ["HybridMemory"]
