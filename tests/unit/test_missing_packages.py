from __future__ import annotations

import sys
import importlib
import contextlib
from pathlib import Path
import tempfile
import os
import pytest
import asyncio

@contextlib.contextmanager
def mock_missing_module(module_name: str, reloaded_modules: list[str]):
    # Keep track of original sys.modules values
    original_module = sys.modules.get(module_name, None)
    
    # Store original state of reloaded modules
    original_reloaded = {}
    for mod_name in reloaded_modules:
        if mod_name in sys.modules:
            original_reloaded[mod_name] = sys.modules[mod_name]
            del sys.modules[mod_name]
            
    # Set the target module to None to trigger ImportError on import
    sys.modules[module_name] = None  # type: ignore[assignment]
    
    try:
        # Reload/re-import target modules under the mock environment
        for mod_name in reloaded_modules:
            importlib.import_module(mod_name)
        yield
    finally:
        # Restore sys.modules to original state
        if original_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = original_module
            
        # Revert the reloaded modules
        for mod_name in reloaded_modules:
            sys.modules.pop(mod_name, None)
            if mod_name in original_reloaded:
                sys.modules[mod_name] = original_reloaded[mod_name]


@pytest.mark.asyncio
async def test_chromadb_missing_fallback():
    """Verify that when chromadb is missing, SemanticMemory initialization fails gracefully

    and HybridMemory falls back to sqlite-only mode.
    """
    import core.memory.semantic_memory
    import core.memory.hybrid_memory
    from core.memory.semantic_memory import SemanticMemory
    from core.memory.hybrid_memory import HybridMemory

    orig_chromadb = core.memory.semantic_memory.chromadb
    orig_settings = core.memory.semantic_memory.Settings

    core.memory.semantic_memory.chromadb = None  # type: ignore[assignment]
    core.memory.semantic_memory.Settings = None  # type: ignore[assignment,misc]

    try:
        # Test SemanticMemory initialization
        sm = SemanticMemory()
        res = await sm.initialize()
        assert res is False
        
        # Test HybridMemory initialization fallback
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            hm = HybridMemory(db_path=db_path, chroma_path=tmpdir)
            try:
                init_res = await hm.initialize()
                
                assert hm.mode == "sqlite-only"
                assert init_res["semantic"] is False
                assert init_res["sqlite"] is True
            finally:
                await hm.close()
                await asyncio.sleep(0.2)
                import gc
                gc.collect()
    finally:
        core.memory.semantic_memory.chromadb = orig_chromadb
        core.memory.semantic_memory.Settings = orig_settings  # type: ignore[misc]


@pytest.mark.asyncio
async def test_sentence_transformers_missing_fallback():
    """Verify that when sentence-transformers is missing, EmbeddingManager falls back

    to DeterministicMockSentenceTransformer.
    """
    with mock_missing_module("sentence_transformers", []):
        from core.memory.embeddings import EmbeddingManager
        
        em = EmbeddingManager()
        res = await em.initialize()
        assert res is True
        assert em.is_ready() is True
        assert em._model.__class__.__name__ == "DeterministicMockSentenceTransformer"
        
        # Verify it can still embed text
        vec = await em.embed("test sentence")
        assert len(vec) == 384  # Default dimension of mock model
        assert all(isinstance(val, float) for val in vec)


def test_pytesseract_missing_fallback():
    """Verify that when pytesseract is missing, screen tool and payload extractor handle it gracefully."""
    # 1. Test screen tool
    with mock_missing_module("pytesseract", []):
        from core.tools.screen import read_screen_text
        res = read_screen_text("query")
        assert res.success is False
        assert "Missing dependency" in res.error

    # 2. Test payload extractor
    with mock_missing_module("pytesseract", []):
        from core.automation.payload_extractor import PayloadExtractor
        pe = PayloadExtractor(
            max_text_chars_per_item=100,
            video_frame_interval_seconds=1.0,
            max_video_samples=1,
        )
        res = pe.extract_text_from_image(Path("dummy_image.png"))
        assert "OCR dependency missing" in res
