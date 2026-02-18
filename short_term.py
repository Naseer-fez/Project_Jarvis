"""
memory/short_term.py
═════════════════════
In-memory short-term conversation buffer.
Holds the last N messages for context window injection.
Resets on shutdown — not persisted.
"""

from collections import deque
from datetime import datetime


class ShortTermMemory:
    """
    Rolling window of recent messages.
    Default: last 20 exchanges kept in RAM.
    """

    def __init__(self, max_turns: int = 20):
        self._buffer: deque[dict] = deque(maxlen=max_turns * 2)  # user + assistant

    def add(self, role: str, content: str):
        """Add a message. role = 'user' | 'assistant'"""
        self._buffer.append({
            "role": role,
            "content": content,
            "ts": datetime.utcnow().isoformat(),
        })

    def get_messages(self) -> list[dict]:
        """Return all buffered messages as list of {role, content}."""
        return [{"role": m["role"], "content": m["content"]} for m in self._buffer]

    def get_recent(self, n: int = 6) -> list[dict]:
        """Return last N messages."""
        msgs = self.get_messages()
        return msgs[-n:] if len(msgs) > n else msgs

    def clear(self):
        self._buffer.clear()

    def __len__(self):
        return len(self._buffer)
