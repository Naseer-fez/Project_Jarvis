from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from core.memory.code_indexer import extract_code_chunks
from core.memory.hybrid_memory import HybridMemory


def test_extract_code_chunks_plain_file_keeps_content():
    content = "x = 1\nprint(x)\n"

    chunks = extract_code_chunks("plain.py", content)

    assert len(chunks) == 1
    assert chunks[0]["chunk_id"] == "file:plain.py"
    assert chunks[0]["chunk"] == "x = 1\nprint(x)"
    assert chunks[0]["metadata"]["type"] == "File"


def test_extract_code_chunks_syntax_error_keeps_content_and_marks_error():
    content = "def broken(:\n    pass\n"

    chunks = extract_code_chunks("broken.py", content)

    assert len(chunks) == 1
    assert chunks[0]["chunk_id"] == "file:broken.py"
    assert "def broken(" in chunks[0]["chunk"]
    assert chunks[0]["metadata"]["type"] == "FileSyntaxError"
    assert chunks[0]["metadata"]["error"]


@pytest.mark.asyncio
async def test_store_code_file_indexes_plain_file_content(tmp_path):
    memory = HybridMemory(db_path=str(tmp_path / "memory.db"))
    memory.mode = "sqlite-only"
    memory.store_episode = AsyncMock()  # type: ignore[method-assign]

    count = await memory.store_code_file("plain.py", "x = 1\nprint(x)\n")

    assert count == 1
    memory.store_episode.assert_awaited_once()
    event_text = memory.store_episode.await_args.args[0]
    assert event_text.startswith("file:plain.py\n")
    assert "print(x)" in event_text
