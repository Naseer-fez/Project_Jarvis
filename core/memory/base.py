"""Small sqlite-backed base class for legacy memory shims."""

from __future__ import annotations

import sqlite3
from pathlib import Path


class BaseMemory:
    def __init__(self, db_path: str = "memory/compat.db") -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Subclasses can create tables here."""


__all__ = ["BaseMemory"]
