# API Analyst Report: memory\sqlite_pool.py

## Dependencies
- `from __future__ import annotations`
- `import asyncio`
- `import contextlib`
- `import aiosqlite`

## Schemas & API Contracts (Classes)

### Class `SQLitePool`
> A simple async-safe connection pool for SQLite.

**Methods:**
- `def __init__(self, db_path: str, pool_size: int=3)`
- `async def _init_pool(self)`
- @staticmethod
- `async def _rollback_quietly(conn: aiosqlite.Connection) -> None`
- @staticmethod
- `async def _close_connection(conn: aiosqlite.Connection) -> None`
- @contextlib.asynccontextmanager
- `async def acquire(self)`
  - *Acquire a connection from the pool, committing on success or rolling back on error.*
- `async def close(self)`
  - *Close all connections in the pool.*

