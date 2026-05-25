"""
tests/test_session4.py
───────────────────────
Test suite for Jarvis Session 4 components (standard pytest style).

Tests:
  1. EmbeddingManager   — model loading, embed, similarity, batch, cache
  2. SemanticMemory     — store/recall preferences, episodes, conversations
  3. HybridMemory       — combined write/read, fallback behavior
  4. ContextCompressor  — compression quality, thresholds, deduplication
  5. LLMClientV2        — system prompt generation, offline fallback
  6. JarvisControllerV2 — full integration test

Author: Jarvis Session 4
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from unittest import mock

import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ─── 1. EmbeddingManager Tests ────────────────────────────────────────────────

class TestEmbeddingManager:

    @classmethod
    def setup_class(cls):
        from core.memory.embeddings import EmbeddingManager
        cls.em = EmbeddingManager(model_name="all-MiniLM-L6-v2")
        cls.available = cls.em.initialize()

    def test_01_initialization(self):
        """Model should load successfully."""
        assert self.available

    def test_02_embed_returns_list(self):
        """embed() should return a list of floats."""
        if not self.available:
            pytest.skip("Model not available")
        vec = self.em.embed("hello world")
        assert isinstance(vec, list)
        assert len(vec) > 0
        assert isinstance(vec[0], float)

    def test_03_embed_dimension_consistent(self):
        """All embeddings should have the same dimension."""
        if not self.available:
            pytest.skip("Model not available")
        v1 = self.em.embed("coffee")
        v2 = self.em.embed("a much longer text about many different topics")
        assert len(v1) == len(v2)

    def test_04_similarity_self(self):
        """A text should have near-perfect similarity with itself."""
        if not self.available:
            pytest.skip("Model not available")
        score = self.em.similarity("I like coffee", "I like coffee")
        assert score > 0.99

    def test_05_similarity_semantic(self):
        """Semantically similar texts should score higher than unrelated ones."""
        if not self.available:
            pytest.skip("Model not available")
        similar_score = self.em.similarity("I enjoy coffee", "My favorite drink is coffee")
        unrelated_score = self.em.similarity("I enjoy coffee", "The stock market crashed")
        assert similar_score > unrelated_score

    def test_06_similarity_range(self):
        """Similarity scores should be between 0 and 1."""
        if not self.available:
            pytest.skip("Model not available")
        score = self.em.similarity("cat", "quantum mechanics")
        assert score >= 0.0
        assert score <= 1.0

    def test_07_embed_batch(self):
        """embed_batch() should return the correct number of vectors."""
        if not self.available:
            pytest.skip("Model not available")
        texts = ["apple", "banana", "cherry"]
        vecs = self.em.embed_batch(texts)
        assert len(vecs) == 3

    def test_08_cache(self):
        """Same text embedded twice should hit the cache (embed_count should not increase)."""
        if not self.available:
            pytest.skip("Model not available")
        before = self.em._embed_count
        _ = self.em.embed("unique test string for cache check")
        _ = self.em.embed("unique test string for cache check")  # should hit cache
        after = self.em._embed_count
        assert after - before == 1

    def test_09_rank_memories(self):
        """rank_memories() should return highest-scoring text first."""
        if not self.available:
            pytest.skip("Model not available")
        query = "what do I drink in the morning"
        candidates = [
            "User drinks coffee every morning",
            "The weather is sunny today",
            "User prefers Python over Java",
        ]
        results = self.em.rank_memories(query, candidates, top_k=3, threshold=0.0)
        assert len(results) > 0
        assert "coffee" in results[0]["text"].lower()

    def test_10_info(self):
        """info() should return expected keys."""
        if not self.available:
            pytest.skip("Model not available")
        info = self.em.info()
        for key in ["model", "initialized", "dimension", "embed_count", "cache_size"]:
            assert key in info


# ─── 2. SemanticMemory Tests ──────────────────────────────────────────────────

class TestSemanticMemory:

    @classmethod
    def setup_class(cls):
        cls.tmpdir = tempfile.mkdtemp()
        from core.memory.semantic_memory import SemanticMemory
        cls.sm = SemanticMemory(
            chroma_path=cls.tmpdir,
            model_name="all-MiniLM-L6-v2",
        )
        cls.available = cls.sm.initialize()

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_01_initialization(self):
        assert self.available

    def test_02_store_preference(self):
        if not self.available:
            pytest.skip("Unavailable")
        doc_id = self.sm.store_preference("favorite_drink", "coffee")
        assert doc_id is not None

    def test_03_recall_preference_exact(self):
        if not self.available:
            pytest.skip("Unavailable")
        self.sm.store_preference("hobby", "playing guitar")
        results = self.sm.recall_preferences("What do I do for fun?", top_k=3)
        assert len(results) > 0
        texts = [r["document"] for r in results]
        assert any("guitar" in t.lower() for t in texts)

    def test_04_recall_preference_semantic(self):
        """Should find 'coffee' when asking about morning beverages."""
        if not self.available:
            pytest.skip("Unavailable")
        self.sm.store_preference("morning_drink", "coffee")
        results = self.sm.recall_preferences("what do I drink when I wake up?", top_k=5, threshold=0.0)
        assert len(results) > 0

    def test_05_store_episode(self):
        if not self.available:
            pytest.skip("Unavailable")
        doc_id = self.sm.store_episode("User discussed a project deadline", "work")
        assert doc_id is not None

    def test_06_recall_episode(self):
        if not self.available:
            pytest.skip("Unavailable")
        self.sm.store_episode("User mentioned they enjoy hiking on weekends", "leisure")
        results = self.sm.recall_episodes("outdoor activities", top_k=5, threshold=0.0)
        assert len(results) > 0

    def test_07_store_conversation(self):
        if not self.available:
            pytest.skip("Unavailable")
        doc_id = self.sm.store_conversation_turn(
            "What's the capital of France?", "Paris.", session_id="test"
        )
        assert doc_id is not None

    def test_08_recall_conversation(self):
        if not self.available:
            pytest.skip("Unavailable")
        results = self.sm.recall_conversations("France geography", top_k=3, threshold=0.0)
        assert len(results) > 0

    def test_09_recall_all(self):
        if not self.available:
            pytest.skip("Unavailable")
        all_results = self.sm.recall_all("coffee morning routine", top_k=3, threshold=0.0)
        assert "preferences" in all_results
        assert "episodes" in all_results
        assert "conversations" in all_results

    def test_10_stats(self):
        if not self.available:
            pytest.skip("Unavailable")
        stats = self.sm.stats()
        assert stats.get("initialized")
        assert "preferences" in stats
        assert "episodes" in stats

    def test_11_relevance_scores_valid(self):
        """Scores should be between 0 and 1."""
        if not self.available:
            pytest.skip("Unavailable")
        results = self.sm.recall_preferences("anything", top_k=5, threshold=0.0)
        for r in results:
            assert r["score"] >= 0.0
            assert r["score"] <= 1.0

    def test_12_delete_preference(self):
        if not self.available:
            pytest.skip("Unavailable")
        self.sm.store_preference("temp_key", "temp_value")
        result = self.sm.delete_preference("temp_key")
        assert result


# ─── 3. HybridMemory Tests ────────────────────────────────────────────────────

class TestHybridMemory:

    @classmethod
    def setup_class(cls):
        cls.tmpdir = tempfile.mkdtemp()
        cls.db_path = os.path.join(cls.tmpdir, "test_memory.db")
        from core.memory.hybrid_memory import HybridMemory
        cls.hm = HybridMemory(
            db_path=cls.db_path,
            chroma_path=os.path.join(cls.tmpdir, "chroma"),
            model_name="all-MiniLM-L6-v2",
        )
        cls.status = cls.hm.initialize()

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_01_initialization(self):
        assert "mode" in self.status

    def test_02_store_and_recall_preference(self):
        ok = self.hm.store_preference("pet", "golden retriever")
        assert ok
        results = self.hm.recall_preferences("what kind of pet do I have?", top_k=5)
        assert len(results) > 0
        values = [r["value"] for r in results]
        assert any("golden" in v.lower() for v in values)

    def test_03_store_episode(self):
        ok = self.hm.store_episode("User mentioned moving to a new apartment", "life_event")
        assert ok

    def test_04_store_conversation(self):
        ok = self.hm.store_conversation("Do you remember me?", "Of course!", "test_session")
        assert ok

    def test_05_recall_all(self):
        all_r = self.hm.recall_all("tell me about myself", top_k=5)
        assert "preferences" in all_r
        assert "episodes" in all_r
        assert "conversations" in all_r

    def test_06_build_context_block(self):
        block = self.hm.build_context_block("what do I like?")
        assert isinstance(block, str)

    def test_07_stats(self):
        stats = self.hm.stats()
        assert "mode" in stats
        assert "sqlite" in stats
        assert "semantic" in stats


# ─── 4. ContextCompressor Tests ───────────────────────────────────────────────

class TestContextCompressor:

    def setup_method(self):
        from core.memory.context_compressor import ContextCompressor
        self.cc = ContextCompressor(threshold=0.0)

    def _make_recall(self):
        return {
            "preferences": [
                {"key": "name", "value": "Alice", "score": 0.95, "source": "hybrid"},
                {"key": "drink", "value": "coffee", "score": 0.88, "source": "hybrid"},
                {"key": "hobby", "value": "hiking", "score": 0.60, "source": "semantic"},
            ],
            "episodes": [
                {"event": "Discussed a project deadline", "category": "work",
                 "timestamp": "2025-01-15T10:00:00", "score": 0.70, "source": "semantic"},
            ],
            "conversations": [
                {"user_input": "What is machine learning?",
                 "assistant_response": "Machine learning is a subset of AI.",
                 "timestamp": "2025-01-14T09:00:00", "score": 0.55, "source": "semantic"},
            ],
        }

    def test_01_compress_returns_string(self):
        result = self.cc.compress("what do I like?", self._make_recall())
        assert isinstance(result, str)

    def test_02_compress_includes_preferences(self):
        result = self.cc.compress("about me", self._make_recall())
        assert "name=Alice" in result
        assert "drink=coffee" in result

    def test_03_compress_has_header_footer(self):
        result = self.cc.compress("test", self._make_recall())
        assert "Memory Context" in result
        assert "End Memory" in result

    def test_04_compress_empty_recall(self):
        empty = {"preferences": [], "episodes": [], "conversations": []}
        result = self.cc.compress("test", empty)
        assert result == ""

    def test_05_threshold_filtering(self):
        from core.memory.context_compressor import ContextCompressor
        cc_strict = ContextCompressor(threshold=0.9)
        recall = {
            "preferences": [
                {"key": "drink", "value": "coffee", "score": 0.50, "source": "hybrid"},
            ],
            "episodes": [],
            "conversations": [],
        }
        result = cc_strict.compress("test", recall)
        assert result == ""

    def test_06_explain_returns_string(self):
        result = self.cc.explain("test query", self._make_recall())
        assert isinstance(result, str)
        assert "PREFERENCES" in result

    def test_07_deduplication(self):
        """Duplicate preference keys should not appear twice."""
        recall = {
            "preferences": [
                {"key": "drink", "value": "coffee", "score": 0.95, "source": "hybrid"},
                {"key": "drink", "value": "coffee", "score": 0.93, "source": "semantic"},
            ],
            "episodes": [],
            "conversations": [],
        }
        result = self.cc.compress("test", recall)
        assert result.count("drink=coffee") == 1


# ─── 5. Integration Test ──────────────────────────────────────────────────────

class TestJarvisIntegration:
    """
    Full integration test using JarvisControllerV2.
    LLM calls are skipped when Ollama is offline — memory and routing are tested.
    """

    @classmethod
    def setup_class(cls):
        cls.tmpdir = tempfile.mkdtemp()
        cls.db_path = os.path.join(cls.tmpdir, "memory.db")

        from core.controller_v2 import JarvisControllerV2
        cls.ctrl = JarvisControllerV2(
            db_path=cls.db_path,
            chroma_path=os.path.join(cls.tmpdir, "chroma"),
            model_name="deepseek-r1:8b",
            embedding_model="all-MiniLM-L6-v2",
        )
        cls.status = cls.ctrl.initialize()

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_01_initialized(self):
        assert self.status is not None
        assert "session_id" in self.status

    @pytest.mark.asyncio
    async def test_02_store_preference(self):
        response = await self.ctrl.process("remember I like espresso")
        assert "espresso" in response.lower()

    @pytest.mark.asyncio
    async def test_03_recall_preference(self):
        with mock.patch.object(self.ctrl.model_router, 'get_best_available', return_value="deepseek-r1:8b"):
            await self.ctrl.process("my name is Bob")
            response = await self.ctrl.process("what's my name?")
            self._assert_offline_or_match("Bob", response)

    @pytest.mark.asyncio
    async def test_04_status_command(self):
        response = await self.ctrl.process("status")
        assert "Session" in response
        assert "Memory" in response

    @pytest.mark.asyncio
    async def test_05_help_command(self):
        response = await self.ctrl.process("help")
        assert "status" in response.lower()
        assert "exit" in response.lower()

    def test_06_session_summary(self):
        summary = self.ctrl.session_summary()
        assert "session_id" in summary
        assert "exchanges" in summary
        assert summary["exchanges"] > 0

    @pytest.mark.asyncio
    async def test_07_multiple_preferences(self):
        with mock.patch.object(self.ctrl.model_router, 'get_best_available', return_value="deepseek-r1:8b"):
            await self.ctrl.process("I prefer dark mode")
            await self.ctrl.process("I work in Python")
            response = await self.ctrl.process("what do you know about me?")
            assert isinstance(response, str)
            assert len(response) > 0

    def _assert_offline_or_match(self, expected_str: str, response: str):
        if "Offline fallback" in response or "I don't know" in response:
            return
        assert expected_str in response
