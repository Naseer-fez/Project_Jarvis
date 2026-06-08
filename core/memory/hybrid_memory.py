"""Hybrid memory: SQLite for structure plus optional Chroma semantic memory."""

from __future__ import annotations

import contextlib
import asyncio
from pathlib import Path
from typing import Any
import logging

from core.memory.semantic_memory import SemanticMemory
from core.memory.sqlite_pool import SQLitePool
from core.memory.sqlite_storage import SQLiteStorage
from core.memory.retriever import MemoryRetriever
from core.memory.code_indexer_service import CodeIndexerService

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
        self._pool = SQLitePool(self.db_path, pool_size=3)
        self._storage = SQLiteStorage(self._pool)
        self._retriever = MemoryRetriever(self._pool, self.semantic)
        self._indexer = CodeIndexerService(self._pool, self.semantic, self.store_episode)

        self._init_lock = asyncio.Lock()
        self._background_tasks: set[asyncio.Task[Any]] = set()

    async def initialize(self, index_path: str = "") -> dict[str, Any]:
        await self._ensure_db_initialized()
        semantic_ready = False
        try:
            semantic_ready = bool(await self.semantic.initialize())
        except Exception as exc:
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

    def set_llm(self, llm: Any | None, *, enable_context_titles: bool = True) -> None:
        self._llm = llm
        self._enable_llm_context_titles = bool(enable_context_titles)

    async def _ensure_db_initialized(self) -> None:
        async with self._init_lock:
            await self._storage.init_schema()
            
    # Need to keep this private alias used heavily in the older file
    async def _init_sqlite(self) -> None:
        await self._ensure_db_initialized()

    async def store_preference(self, key: str, value: str) -> bool:
        await self._storage.store_preference(key, value)
        if self.mode == "hybrid":
            try:
                await self.semantic.store_preference(key, value)
            except Exception as exc:
                logger.debug("Semantic preference store failed: %s", exc)
        return True

    async def store_episode(self, event: str, category: str = "general") -> bool:
        await self._storage.store_episode(event, category)
        if self.mode == "hybrid":
            try:
                await self.semantic.store_episode(event, category)
            except Exception as exc:
                logger.debug("Semantic episode store failed: %s", exc)
        return True

    async def store_episodes_batch(self, events: list[str], category: str = "general") -> bool:
        await self._storage.store_episodes_batch(events, category)
        if self.mode == "hybrid":
            try:
                if hasattr(self.semantic, "store_episodes_batch"):
                    await self.semantic.store_episodes_batch(events, category=category)
                else:
                    for event in events:
                        await self.semantic.store_episode(event, category=category)
            except Exception as exc:
                logger.debug("Semantic batch store failed: %s", exc)
        return True

    async def store_conversation(self, user_input: str, assistant_response: str, session_id: str = "default") -> bool:
        await self._storage.store_conversation(user_input, assistant_response, session_id)
        if self.mode == "hybrid":
            try:
                await self.semantic.store_conversation_turn(user_input, assistant_response, session_id)
            except Exception as exc:
                logger.debug("Semantic conversation store failed: %s", exc)
        return True

    async def recall_preferences(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        return await self._retriever.recall_preferences(
            query, top_k, self.mode == "hybrid", self._ensure_db_initialized
        )

    async def recall_all(self, query: str, top_k: int = 5) -> dict[str, list[dict[str, Any]]]:
        return await self._retriever.recall_all(
            query, top_k, self.mode == "hybrid", self._ensure_db_initialized
        )

    @staticmethod
    def _query_tokens(query: str) -> list[str]:
        return MemoryRetriever.query_tokens(query)

    @staticmethod
    def _score_text(text: str, tokens: list[str]) -> float:
        return MemoryRetriever.score_text(text, tokens)

    async def _recall_sqlite_episodes(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        return await self._retriever._recall_sqlite_episodes(query, top_k, self._ensure_db_initialized)

    async def _recall_sqlite_conversations(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        return await self._retriever._recall_sqlite_conversations(query, top_k, self._ensure_db_initialized)

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
        except Exception:
            return ""

    async def recall(self, query: str, top_k: int = 5) -> str:
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
        return "\\n".join(lines)

    async def store_code_file(self, file_path: str, content: str) -> int:
        from core.memory.code_indexer import extract_code_chunks
        chunks = extract_code_chunks(file_path, content)
        chunks_stored = 0
        for item in chunks:
            chunk_id = item["chunk_id"]
            chunk = item["chunk"]
            metadata = item["metadata"]
            if self.mode == "hybrid":
                try:
                    if hasattr(self.semantic, "store_code_chunk"):
                        await self.semantic.store_code_chunk(chunk_id, chunk, metadata=metadata)
                    else:
                        await self.semantic.store_episode(f"{chunk_id}\n{chunk}", category="code")
                except Exception as exc:
                    logger.debug("Semantic code chunk store failed: %s", exc)
            await self.store_episode(f"{chunk_id}\n{chunk[:3000]}", category="code")
            chunks_stored += 1
        return max(1, chunks_stored)

    async def index_codebase(self, root_path: str) -> dict[str, int]:
        return await self._indexer.index_codebase(
            root_path, self.mode == "hybrid", self._ensure_db_initialized
        )

    def stats(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "sqlite": True,
            "semantic": self.mode == "hybrid",
        }

    async def store_fact(self, key: str, value: str, source: str = "user", **_kwargs) -> None:
        await self.store_preference(key, value)

    async def get_fact(self, key: str):
        await self._ensure_db_initialized()
        async with self._pool.acquire() as conn:
            async with conn.execute("SELECT key, value FROM preferences WHERE key=?", (key,)) as cursor:
                row = await cursor.fetchone()
        if row is None:
            return None
        class _Fact:
            def __init__(self, k, v):
                self.key, self.value = k, v
            def __repr__(self):
                return f"Fact(key={self.key!r}, value={self.value!r})"
        return _Fact(row["key"], row["value"])

    async def list_facts(self, limit: int = 50) -> list:
        await self._ensure_db_initialized()
        async with self._pool.acquire() as conn:
            async with conn.execute(
                "SELECT key, value FROM preferences ORDER BY updated_at DESC LIMIT ?", (max(1, limit),)
            ) as cursor:
                rows = await cursor.fetchall()
        class _Fact:
            def __init__(self, k, v):
                self.key, self.value = k, v
        return [_Fact(r["key"], r["value"]) for r in rows]

    async def count(self) -> int:
        await self._ensure_db_initialized()
        async with self._pool.acquire() as conn:
            async with conn.execute("SELECT COUNT(*) FROM preferences") as cursor:
                row = await cursor.fetchone()
        return row[0] if row else 0

    async def store_action(self, action: str, result: str = "", success: bool = True, metadata: dict | None = None) -> None:
        await self._storage.store_action(action, result, success, metadata)

    async def store_failure(self, action: str, error: str = "", metadata: dict | None = None) -> None:
        await self.store_action(action, result=error, success=False, metadata=metadata)

    async def recent_actions(self, limit: int = 20) -> list[dict[str, Any]]:
        return await self._storage.recent_actions(limit)

    async def set_preference(self, key: str, value: str, category: str = "general", **_kwargs) -> None:
        scoped_key = f"{category}::{key}" if category and category != "general" else key
        await self.store_preference(scoped_key, value)
        await self.store_preference(key, value)

    async def get_preferences(self, category: str = "") -> dict[str, str]:
        prefix = f"{category}::" if category else ""
        await self._ensure_db_initialized()
        async with self._pool.acquire() as conn:
            if prefix:
                async with conn.execute(
                    "SELECT key, value FROM preferences WHERE key LIKE ? ORDER BY updated_at DESC", (f"{prefix}%",)
                ) as cursor:
                    rows = await cursor.fetchall()
                return {r["key"].removeprefix(prefix): r["value"] for r in rows}
            async with conn.execute("SELECT key, value FROM preferences ORDER BY updated_at DESC") as cursor:
                rows = await cursor.fetchall()
            return {r["key"]: r["value"] for r in rows}

    async def close(self) -> None:
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
