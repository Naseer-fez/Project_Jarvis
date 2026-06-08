import asyncio
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

class CodeIndexerService:
    """Handles codebase indexing and chunk extraction for hybrid memory."""
    
    def __init__(self, db_pool, semantic_memory, store_episode_cb):
        self.db_pool = db_pool
        self.semantic = semantic_memory
        self.store_episode = store_episode_cb

    async def index_codebase(self, root_path: str, is_hybrid: bool, init_schema_cb: Callable) -> dict[str, int]:
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
                content = await asyncio.to_thread(py_file.read_text, encoding="utf-8", errors="replace")
                content_hash = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()
                hash_key = f"code_hash::{py_file.resolve()}"

                await init_schema_cb()
                async with self.db_pool.acquire() as conn:
                    async with conn.execute("SELECT value FROM preferences WHERE key=?", (hash_key,)) as cursor:
                        row = await cursor.fetchone()
                    if row and row["value"] == content_hash:
                        stats["skipped_files"] += 1
                        continue

                    async with conn.execute(
                        "INSERT INTO preferences (key, value, updated_at) VALUES (?, ?, ?) "
                        "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                        (hash_key, content_hash, datetime.now().isoformat()),
                    ):
                        pass

                from core.memory.code_indexer import extract_code_chunks
                extracted = extract_code_chunks(str(py_file), content)
                chunks_to_index.extend(extracted)
                stats["indexed_files"] += 1

            except Exception as exc:
                logger.warning("Failed to index %s: %s", py_file, exc)
                stats["errors"] += 1

            while len(chunks_to_index) >= 32:
                batch = chunks_to_index[:32]
                chunks_to_index = chunks_to_index[32:]
                await self._index_chunks_batch(batch, stats, is_hybrid, init_schema_cb)
                await asyncio.sleep(0.01)

            await asyncio.sleep(0.01)

        if chunks_to_index:
            await self._index_chunks_batch(chunks_to_index, stats, is_hybrid, init_schema_cb)

        return stats

    async def _index_chunks_batch(self, batch: list[dict], stats: dict, is_hybrid: bool, init_schema_cb: Callable) -> None:
        try:
            await init_schema_cb()
            async with self.db_pool.acquire() as conn:
                for item in batch:
                    async with conn.execute(
                        "INSERT INTO episodes (event, category, timestamp) VALUES (?, ?, ?)",
                        (f"{item['chunk_id']}\n{item['chunk'][:3000]}", "code", datetime.now().isoformat())
                    ):
                        pass
        except Exception as exc:
            logger.warning("Failed to store batch chunks to SQLite: %s", exc)

        if is_hybrid:
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
