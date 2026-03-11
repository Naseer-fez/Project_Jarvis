"""Legacy preference-store compatibility shim."""

from __future__ import annotations

from datetime import datetime, timezone

from .base import BaseMemory


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class PreferenceStore(BaseMemory):
    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS preferences (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def store_preference(self, key: str, value: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO preferences (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
                """,
                (key, value, _utcnow_iso()),
            )

    def retrieve_preference(self, key: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM preferences WHERE key = ?",
                (key,),
            ).fetchone()
        return None if row is None else str(row["value"])

    def set(self, key: str, value: str) -> None:
        self.store_preference(key, value)

    def get(self, key: str) -> str | None:
        return self.retrieve_preference(key)


__all__ = ["PreferenceStore"]
