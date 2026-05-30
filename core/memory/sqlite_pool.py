"""SQLite connection pool for Project Jarvis."""

from __future__ import annotations

import contextlib
import queue
import sqlite3


class SQLitePool:
    """A simple thread-safe connection pool for SQLite."""

    def __init__(self, db_path: str, pool_size: int = 3):
        self.db_path = db_path
        self.pool_size = pool_size
        self._pool = queue.Queue(maxsize=pool_size)
        for _ in range(pool_size):
            conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30.0)
            conn.row_factory = sqlite3.Row
            self._pool.put(conn)

    @contextlib.contextmanager
    def acquire(self):
        """Acquire a connection from the pool, committing on success or rolling back on error."""
        try:
            conn = self._pool.get(timeout=30.0)
        except queue.Empty:
            raise TimeoutError("SQLitePool: timed out waiting for a free connection") from None
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.put(conn)
