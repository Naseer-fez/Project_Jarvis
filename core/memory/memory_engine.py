"""
MemoryEngine — lightweight contextual memory for Jarvis.
Stores episodic events per session and provides relevance-based retrieval.
Persists to JSONL file; loaded at session start.
"""

import json
import logging
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("Jarvis.MemoryEngine")

MEMORY_FILE = Path("./outputs/memory_snapshot.jsonl")
MAX_SESSION_ENTRIES = 200


@dataclass
class MemoryEntry:
    timestamp: float
    category: str          # "episodic" | "preference" | "reflection" | "tool_result"
    content: str
    tags: list[str] = field(default_factory=list)
    session_id: Optional[str] = None

    def matches(self, query: str) -> bool:
        q = query.lower()
        return q in self.content.lower() or any(q in t.lower() for t in self.tags)


class MemoryEngine:
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or f"session_{int(time.time())}"
        self._entries: list[MemoryEntry] = []
        self._load()

    def _load(self):
        if MEMORY_FILE.exists():
            try:
                with open(MEMORY_FILE, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            data = json.loads(line)
                            self._entries.append(MemoryEntry(**data))
                logger.info(f"Loaded {len(self._entries)} memory entries.")
            except Exception as e:
                logger.warning(f"Memory load failed: {e}")

    def store(self, content: str, category: str = "episodic", tags: Optional[list[str]] = None):
        """Store a new memory entry."""
        if len(self._entries) >= MAX_SESSION_ENTRIES:
            self._entries.pop(0)  # Evict oldest

        entry = MemoryEntry(
            timestamp=time.time(),
            category=category,
            content=content,
            tags=tags or [],
            session_id=self.session_id,
        )
        self._entries.append(entry)
        self._persist(entry)

    def retrieve(self, query: str, limit: int = 5) -> list[MemoryEntry]:
        """Return most recent entries matching query."""
        matches = [e for e in reversed(self._entries) if e.matches(query)]
        return matches[:limit]

    def recent(self, n: int = 10) -> list[MemoryEntry]:
        return list(reversed(self._entries[-n:]))

    def context_summary(self, query: str = "", n: int = 5) -> str:
        """Returns a short text summary for LLM context injection."""
        entries = self.retrieve(query, n) if query else self.recent(n)
        if not entries:
            return ""
        lines = [f"[Memory — {e.category}] {e.content}" for e in entries]
        return "Recent memory:\n" + "\n".join(lines)

    def _persist(self, entry: MemoryEntry):
        MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(MEMORY_FILE, "a") as f:
            f.write(json.dumps(asdict(entry)) + "\n")

    def snapshot(self) -> dict:
        return {
            "session_id": self.session_id,
            "total_entries": len(self._entries),
            "entries": [asdict(e) for e in self._entries[-50:]],  # last 50
        }

