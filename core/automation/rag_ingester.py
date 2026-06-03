from pathlib import Path
from typing import Any
import logging

logger = logging.getLogger(__name__)

class RagIngester:
    def __init__(self, chunk_size_chars: int, chunk_overlap_chars: int, memory: Any, stats: Any):
        self.chunk_size_chars = chunk_size_chars
        self.chunk_overlap_chars = chunk_overlap_chars
        self.memory = memory
        self.stats = stats

    async def store_rag_text(self, *, source: str, path: Path, text: str) -> int:
        clean = str(text or "").strip()
        if not clean:
            return 0

        chunks = self.chunk_text(clean)
        total = len(chunks)
        payloads = []
        for index, chunk in enumerate(chunks, start=1):
            payload = (
                "[RAG Source]\n"
                f"source={source}\n"
                f"path={path}\n"
                f"chunk={index}/{total}\n"
                f"content={chunk}"
            )
            payloads.append(payload)

        stored = 0
        try:
            # Assuming store_episodes_batch accepts a list of payloads and category
            await self.memory.store_episodes_batch(payloads, category="rag")
            stored = len(payloads)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to store RAG chunks: %s", exc)
            self.stats.last_error = str(exc)

        return stored

    def chunk_text(self, text: str) -> list[str]:
        size = self.chunk_size_chars
        overlap = min(self.chunk_overlap_chars, max(0, size - 20))
        if len(text) <= size:
            return [text]

        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(len(text), start + size)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(text):
                break
            start = max(start + 1, end - overlap)
        return chunks
