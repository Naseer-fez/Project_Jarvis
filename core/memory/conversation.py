"""Legacy conversation-memory compatibility shim."""

from __future__ import annotations

from datetime import datetime, timezone

from .base import BaseMemory


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ConversationMemory(BaseMemory):
    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    def store(self, role: str, content: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO conversation_messages (role, content, created_at)
                VALUES (?, ?, ?)
                """,
                (role, content, _utcnow_iso()),
            )

    def recall(self, limit: int = 5) -> list[dict[str, str]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content, created_at
                FROM conversation_messages
                ORDER BY id DESC
                LIMIT ?
                """,
                (max(1, int(limit)),),
            ).fetchall()
        return [dict(row) for row in rows]


__all__ = ["ConversationMemory"]
