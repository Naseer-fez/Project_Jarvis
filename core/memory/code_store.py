"""Legacy code-store compatibility shim."""

from __future__ import annotations

import ast
from datetime import datetime, timezone

from .base import BaseMemory


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class CodeStore(BaseMemory):
    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS code_files (
                    path TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    parsed_ok INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def store_code_file(self, path: str, content: str) -> dict[str, object]:
        parsed_ok = True
        try:
            ast.parse(content)
        except SyntaxError:
            parsed_ok = False

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO code_files (path, content, parsed_ok, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    content = excluded.content,
                    parsed_ok = excluded.parsed_ok,
                    updated_at = excluded.updated_at
                """,
                (path, content, int(parsed_ok), _utcnow_iso()),
            )

        return {"success": True, "parsed_ok": parsed_ok}


__all__ = ["CodeStore"]
