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
        self._all_conns: set = set()
        self._in_use_conns: set = set()
        self._closed = False
        self._close_waiter: asyncio.Event | None = None
        self._lock = asyncio.Lock()

    async def _init_pool(self):
        if self._pool is not None:
            return
        pool = asyncio.Queue(maxsize=self.pool_size)
        conns = []
        try:
            for _ in range(self.pool_size):
                conn = await aiosqlite.connect(self.db_path, timeout=30.0)
                conn.row_factory = aiosqlite.Row
                # Configure Write-Ahead Logging (WAL) and performance/concurrency defaults
                await conn.execute("PRAGMA journal_mode=WAL;")
                await conn.execute("PRAGMA synchronous=NORMAL;")
                conns.append(conn)
                await pool.put(conn)
        except Exception:
            # Roll back/close all opened connections on initialization failure to prevent resource leak
            for conn in conns:
                try:
                    await conn.close()
                except Exception:
                    pass
            raise
        else:
            self._pool = pool
            self._all_conns = set(conns)
            self._closed = False

    @staticmethod
    async def _rollback_quietly(conn) -> None:
        try:
            await conn.rollback()
        except Exception:
            pass

    @staticmethod
    async def _close_connection(conn) -> None:
        try:
            await conn.close()
        except Exception:
            pass

    @contextlib.asynccontextmanager
    async def acquire(self):
        """Acquire a connection from the pool, committing on success or rolling back on error."""
        if self._closed:
            raise RuntimeError("Database pool is closed")

        async with self._lock:
            if self._pool is None:
                await self._init_pool()

        pool = self._pool
        if pool is None or self._closed:
            raise RuntimeError("Database pool is closed")

        try:
            conn = await pool.get()
        except RuntimeError as e:
            raise RuntimeError("Database pool is closed") from e

        async with self._lock:
            if self._closed or self._pool is None or conn not in self._all_conns:
                self._all_conns.discard(conn)
                should_close = True
            else:
                self._in_use_conns.add(conn)
                should_close = False

        # Double check if pool was closed while waiting.
        if should_close:
            await self._close_connection(conn)
            raise RuntimeError("Database pool was closed during acquire")

        try:
            yield conn
            await conn.commit()
        except asyncio.CancelledError:
            await self._rollback_quietly(conn)
            raise
        except Exception:
            await self._rollback_quietly(conn)
            raise
        finally:
            should_close = False
            close_waiter = None
            async with self._lock:
                self._in_use_conns.discard(conn)
                if not self._closed and self._pool is not None and conn in self._all_conns:
                    self._pool.put_nowait(conn)
                else:
                    self._all_conns.discard(conn)
                    should_close = True

                if self._closed and not self._in_use_conns and not self._all_conns:
                    close_waiter = self._close_waiter
                    self._close_waiter = None

            if should_close:
                await self._close_connection(conn)

            if close_waiter is not None and not close_waiter.is_set():
                close_waiter.set()

    async def close(self):
        """Close all connections in the pool."""
        drained_conns = []
        close_waiter = None
        async with self._lock:
            self._closed = True
            pool = self._pool
            if pool is None:
                return

            # Wake up any tasks currently blocked on get()
            if hasattr(pool, "_getters"):
                for getter in pool._getters:
                    if not getter.done():
                        getter.set_exception(RuntimeError("Database pool is closed"))

            # Drain idle connections from the queue and close them outside the lock.
            while not pool.empty():
                try:
                    conn = pool.get_nowait()
                except asyncio.QueueEmpty:
                    break
                else:
                    drained_conns.append(conn)
                    self._all_conns.discard(conn)

            self._pool = None

            if self._in_use_conns and self._all_conns:
                close_waiter = self._close_waiter or asyncio.Event()
                self._close_waiter = close_waiter
            else:
                self._all_conns.clear()
                self._close_waiter = None

        for conn in drained_conns:
            await self._close_connection(conn)

        if close_waiter is not None:
            await close_waiter.wait()
