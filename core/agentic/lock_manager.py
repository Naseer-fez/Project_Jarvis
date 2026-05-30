"""Lock manager for multi-agent parallel coding workspace.

Coordinates file access among multiple concurrent agents working in the same folder.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path


class LockManager:
    """Manages workspace file write locks using a SQLite database."""

    def __init__(self, db_path: str | Path = "data/workspace_locks.db"):
        self.db_path = Path(db_path)
        # Ensure data directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the locks table if it does not exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS file_locks (
                    filepath TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    acquired_at REAL NOT NULL,
                    expires_at REAL NOT NULL
                )
                """
            )
            conn.commit()

    def acquire_lock(
        self,
        filepath: str | Path,
        agent_id: str,
        ttl_seconds: float = 30.0,
    ) -> bool:
        """Attempt to acquire a write lock on a file path.

        Returns True if successful, False otherwise.
        """
        filepath_str = str(Path(filepath).resolve().as_posix())
        now = time.time()
        expires_at = now + ttl_seconds

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Check if there is an active non-expired lock
            cursor.execute(
                "SELECT agent_id, expires_at FROM file_locks WHERE filepath = ?",
                (filepath_str,),
            )
            row = cursor.fetchone()

            if row:
                current_agent, lock_expires = row
                if current_agent == agent_id:
                    # Already owned by this agent, update expiration
                    cursor.execute(
                        "UPDATE file_locks SET expires_at = ?, acquired_at = ? WHERE filepath = ?",
                        (expires_at, now, filepath_str),
                    )
                    conn.commit()
                    return True
                
                if now > lock_expires:
                    # Lock has expired, force-reclaim it
                    cursor.execute(
                        "UPDATE file_locks SET agent_id = ?, acquired_at = ?, expires_at = ? WHERE filepath = ?",
                        (agent_id, now, expires_at, filepath_str),
                    )
                    conn.commit()
                    return True
                
                # Active lock exists and owned by another agent
                return False
            else:
                # No lock exists, insert new one
                try:
                    cursor.execute(
                        "INSERT INTO file_locks (filepath, agent_id, acquired_at, expires_at) VALUES (?, ?, ?, ?)",
                        (filepath_str, agent_id, now, expires_at),
                    )
                    conn.commit()
                    return True
                except sqlite3.IntegrityError:
                    # Rare race condition where insert happens concurrently
                    return False

    def release_lock(self, filepath: str | Path, agent_id: str) -> bool:
        """Release a write lock on a file path if held by this agent.

        Returns True if released successfully.
        """
        filepath_str = str(Path(filepath).resolve().as_posix())
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT agent_id FROM file_locks WHERE filepath = ?",
                (filepath_str,),
            )
            row = cursor.fetchone()
            if row and row[0] == agent_id:
                cursor.execute(
                    "DELETE FROM file_locks WHERE filepath = ?",
                    (filepath_str,),
                )
                conn.commit()
                return True
            return False

    def is_locked(self, filepath: str | Path) -> tuple[bool, str | None]:
        """Check if a file is locked and who holds the lock."""
        filepath_str = str(Path(filepath).resolve().as_posix())
        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT agent_id, expires_at FROM file_locks WHERE filepath = ?",
                (filepath_str,),
            )
            row = cursor.fetchone()
            if row:
                agent_id, expires_at = row
                if now < expires_at:
                    return True, agent_id
                else:
                    # Lock has expired, clean up
                    cursor.execute(
                        "DELETE FROM file_locks WHERE filepath = ?",
                        (filepath_str,),
                    )
                    conn.commit()
            return False, None

    def clear_all_locks(self) -> None:
        """Clear all locks in the workspace."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM file_locks")
            conn.commit()
