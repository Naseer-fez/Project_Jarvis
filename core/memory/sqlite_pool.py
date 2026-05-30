"""SQLite connection pool for Project Jarvis."""

from __future__ import annotations

import asyncio
import contextlib
import aiosqlite


class SQLitePool:
    """A simple async-safe connection pool for SQLite."""

    def __init__(self, db_path: str, pool_size: int = 3):
        self.db_path = db_path
        self.pool_size = pool_size
        self._pool: asyncio.Queue | None = None
        self._lock = asyncio.Lock()

    async def _init_pool(self):
        if self._pool is not None:
            return
        self._pool = asyncio.Queue(maxsize=self.pool_size)
        for _ in range(self.pool_size):
            conn = await aiosqlite.connect(self.db_path, timeout=30.0)
            conn.row_factory = aiosqlite.Row
            await self._pool.put(conn)

    @contextlib.asynccontextmanager
    async def acquire(self):
        """Acquire a connection from the pool, committing on success or rolling back on error."""
        async with self._lock:
            if self._pool is None:
                await self._init_pool()

        conn = await self._pool.get()
        try:
            yield conn
            await conn.commit()
        except Exception:
            await conn.rollback()
            raise
        finally:
            await self._pool.put(conn)

    async def close(self):
        """Close all connections in the pool."""
        async with self._lock:
            if self._pool is None:
                return
            while not self._pool.empty():
                conn = await self._pool.get()
                await conn.close()
            self._pool = None
