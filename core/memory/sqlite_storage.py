import json
import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

class SQLiteStorage:
    """Handles raw SQLite operations for memory (preferences, episodes, conversations, actions)."""
    
    def __init__(self, pool):
        self.pool = pool
        self._sqlite_initialized = False

    async def init_schema(self) -> None:
        if self._sqlite_initialized:
            return
        async with self.pool.acquire() as conn:
            async with conn.execute("PRAGMA journal_mode=WAL;"):
                pass
            
            # Handle schema evolution for legacy databases
            for table in ["episodes", "conversations", "actions"]:
                try:
                    async with conn.execute(f"ALTER TABLE {table} ADD COLUMN timestamp TEXT DEFAULT ''"):
                        pass
                except Exception:
                    pass
            try:
                async with conn.execute("ALTER TABLE preferences ADD COLUMN updated_at TEXT DEFAULT ''"):
                    pass
            except Exception:
                pass

            async with conn.executescript(
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
            ):
                pass
            self._sqlite_initialized = True

    async def store_preference(self, key: str, value: str) -> None:
        await self.init_schema()
        now = datetime.now().isoformat()
        async with self.pool.acquire() as conn:
            async with conn.execute(
                "INSERT INTO preferences (key, value, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (key, value, now),
            ):
                pass

    async def store_episode(self, event: str, category: str = "general") -> None:
        await self.init_schema()
        now = datetime.now().isoformat()
        async with self.pool.acquire() as conn:
            async with conn.execute(
                "INSERT INTO episodes (event, category, timestamp) VALUES (?, ?, ?)",
                (event, category, now),
            ):
                pass

    async def store_episodes_batch(self, events: list[str], category: str = "general") -> None:
        await self.init_schema()
        now = datetime.now().isoformat()
        async with self.pool.acquire() as conn:
            async with conn.executemany(
                "INSERT INTO episodes (event, category, timestamp) VALUES (?, ?, ?)",
                [(event, category, now) for event in events],
            ):
                pass

    async def store_conversation(self, user_input: str, assistant_response: str, session_id: str) -> None:
        await self.init_schema()
        now = datetime.now().isoformat()
        async with self.pool.acquire() as conn:
            async with conn.execute(
                "INSERT INTO conversations (user_input, assistant_response, session_id, timestamp) VALUES (?, ?, ?, ?)",
                (user_input, assistant_response, session_id, now),
            ):
                pass

    async def store_action(self, action: str, result: str, success: bool, metadata: dict | None) -> None:
        await self.init_schema()
        async with self.pool.acquire() as conn:
            async with conn.execute(
                "INSERT INTO actions (action, result, success, metadata, timestamp) VALUES (?, ?, ?, ?, ?)",
                (action, result, int(success), json.dumps(metadata or {}), datetime.now().isoformat()),
            ):
                pass

    async def recent_actions(self, limit: int = 20) -> list[dict[str, Any]]:
        await self.init_schema()
        async with self.pool.acquire() as conn:
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
                "metadata": json.loads(r["metadata"] or "{}"),
                "timestamp": r["timestamp"],
            }
            for r in rows
        ]

