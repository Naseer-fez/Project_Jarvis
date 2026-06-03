"""
memory/semantic_memory.py
─────────────────────────
Semantic memory layer for Jarvis using ChromaDB (local vector store)
and sentence-transformers for embedding generation.

Responsibilities:
  - Generate embeddings from text using a local sentence-transformer model
  - Store memories (preferences, episodic events, conversation turns) as vectors
  - Retrieve top-K most relevant memories for any query
  - Provide relevance scoring and threshold filtering
  - Combine with SQLite (long_term.py) for a hybrid exact + semantic recall

Collections (ChromaDB):
  - jarvis_preferences   : key/value preference entries
  - jarvis_episodes      : episodic memory events
  - jarvis_conversations : conversation history turns

Author: Jarvis Session 4
"""

import uuid
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

try:
    import chromadb
    from chromadb.config import Settings
except Exception:
    chromadb = None  # type: ignore[assignment]
    Settings = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

DEFAULT_MODEL      = "all-MiniLM-L6-v2"   # 80 MB, fast, good quality
DEFAULT_TOP_K      = 5
DEFAULT_THRESHOLD  = 0.30                  # cosine similarity (0–1); lower = more results
CHROMA_PATH        = str(Path(__file__).resolve().parent.parent.parent / "data" / "chroma")

COLLECTION_PREFS   = "jarvis_preferences"
COLLECTION_EPISODES= "jarvis_episodes"
COLLECTION_CONVOS  = "jarvis_conversations"


# ─── SemanticMemory ────────────────────────────────────────────────────────────

class SemanticMemory:
    """
    Local vector-based memory using ChromaDB + sentence-transformers.

    Usage:
        sm = SemanticMemory()
        sm.store_preference("favorite_drink", "coffee")
        results = sm.recall("What do I like to drink?", top_k=3)
    """

    def __init__(
        self,
        chroma_path: str = CHROMA_PATH,
        model_name: str = DEFAULT_MODEL,
        embedding_manager: Any = None,
    ):
        self.chroma_path = chroma_path
        self.model_name  = model_name
        self.embedding_manager = embedding_manager
        self._client: Any = None
        self._collections: Dict[str, Any] = {}
        self._initialized = False

    # ── Init ──────────────────────────────────────────────────────────────────

    async def initialize(self) -> bool:
        """
        Lazy initialization — load model and connect to ChromaDB.
        Returns True on success, False on failure.
        """
        if self._initialized:
            return True

        if chromadb is None or Settings is None:
            logger.warning("ChromaDB is not installed; semantic memory disabled.")
            return False
             
        try:
            if self.embedding_manager is None:
                from core.memory.embeddings import get_embedding_manager
                self.embedding_manager = get_embedding_manager(self.model_name)

            if not self.embedding_manager.is_ready():
                if not await self.embedding_manager.initialize():
                    logger.warning("Embedding manager failed to initialize; disabling semantic memory.")
                    return False

            logger.info(f"Connecting to ChromaDB at: {self.chroma_path}")
            self._client = chromadb.PersistentClient(
                path=self.chroma_path,
                settings=Settings(anonymized_telemetry=False),
            )

            # Create or get all collections
            for name in [COLLECTION_PREFS, COLLECTION_EPISODES, COLLECTION_CONVOS]:
                self._collections[name] = self._client.get_or_create_collection(
                    name=name,
                    metadata={"hnsw:space": "cosine"},
                )

            self._initialized = True
            logger.info("SemanticMemory initialized successfully.")
            return True

        except Exception as e:
            logger.error(f"SemanticMemory initialization failed: {e}", exc_info=True)
            return False

    async def _ensure_init(self):
        if not self._initialized:
            if not await self.initialize():
                raise RuntimeError("SemanticMemory is not initialized.")

    async def _embed(self, text: str) -> List[float]:
        """Generate a normalized embedding vector for the given text."""
        return await self.embedding_manager.embed(text)

    def _collection(self, name: str):
        return self._collections[name]

    # ── Store ─────────────────────────────────────────────────────────────────

    async def store_preference(self, key: str, value: str) -> str:
        """
        Upsert a user preference into the vector store.
        The document text is: "key: value" for rich semantic matching.
        Returns the document ID.
        """
        await self._ensure_init()
        doc_id   = f"pref_{key}"
        doc_text = f"{key}: {value}"
        embedding = await self._embed(doc_text)

        await asyncio.to_thread(
            self._collection(COLLECTION_PREFS).upsert,
            ids=[doc_id],
            embeddings=[embedding],
            documents=[doc_text],
            metadatas=[{
                "key":        key,
                "value":      value,
                "updated_at": datetime.now().isoformat(),
            }],
        )
        logger.debug(f"Stored preference: {doc_id} → {doc_text}")
        return doc_id

    async def store_episode(self, event: str, category: str = "general") -> str:
        """
        Store an episodic memory event.
        Returns the document ID.
        """
        await self._ensure_init()
        doc_id    = f"ep_{uuid.uuid4().hex[:12]}"
        embedding = await self._embed(event)

        await asyncio.to_thread(
            self._collection(COLLECTION_EPISODES).add,
            ids=[doc_id],
            embeddings=[embedding],
            documents=[event],
            metadatas=[{
                "category":  category,
                "timestamp": datetime.now().isoformat(),
            }],
        )
        logger.debug(f"Stored episode: {doc_id}")
        return doc_id

    async def store_episodes_batch(
        self,
        events: list[str],
        category: str = "general",
    ) -> list[str]:
        """
        Store a batch of episodic memory events efficiently.
        Returns the list of document IDs.
        """
        if not events:
            return []
        await self._ensure_init()
        embeddings = await self.embedding_manager.embed_batch(events)
        doc_ids = [f"ep_{uuid.uuid4().hex[:12]}" for _ in range(len(events))]

        await asyncio.to_thread(
            self._collection(COLLECTION_EPISODES).add,
            ids=doc_ids,
            embeddings=embeddings,
            documents=events,
            metadatas=[{
                "category":  category,
                "timestamp": datetime.now().isoformat(),
            } for _ in range(len(events))],
        )
        logger.debug(f"Stored {len(events)} episodes in batch")
        return doc_ids


    async def store_conversation_turn(
        self,
        user_input: str,
        assistant_response: str,
        session_id: str = "default",
    ) -> str:
        """
        Store a conversation exchange as a single vector document.
        The document combines user + assistant text for richer context matching.
        Returns the document ID.
        """
        await self._ensure_init()
        doc_id    = f"conv_{uuid.uuid4().hex[:12]}"
        doc_text  = f"User: {user_input}\nAssistant: {assistant_response}"
        embedding = await self._embed(doc_text)

        await asyncio.to_thread(
            self._collection(COLLECTION_CONVOS).add,
            ids=[doc_id],
            embeddings=[embedding],
            documents=[doc_text],
            metadatas=[{
                "user_input":          user_input,
                "assistant_response":  assistant_response,
                "session_id":          session_id,
                "timestamp":           datetime.now().isoformat(),
            }],
        )
        logger.debug(f"Stored conversation turn: {doc_id}")
        return doc_id

    # ── Recall ────────────────────────────────────────────────────────────────

    async def recall_preferences(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> List[Dict]:
        """
        Retrieve the most semantically similar preferences to the query.
        Returns a list of result dicts sorted by relevance score (descending).
        """
        return await self._query_collection(COLLECTION_PREFS, query, top_k, threshold)

    async def recall_episodes(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> List[Dict]:
        """Retrieve the most relevant episodic memories for the query."""
        return await self._query_collection(COLLECTION_EPISODES, query, top_k, threshold)

    async def recall_conversations(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> List[Dict]:
        """Retrieve the most relevant past conversation turns for the query."""
        return await self._query_collection(COLLECTION_CONVOS, query, top_k, threshold)

    async def recall_all(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> Dict[str, List[Dict]]:
        """
        Recall across ALL collections simultaneously.
        Returns a dict with keys: 'preferences', 'episodes', 'conversations'.
        Each value is a sorted list of result dicts.
        """
        results = await asyncio.gather(
            self.recall_preferences(query, top_k, threshold),
            self.recall_episodes(query, top_k, threshold),
            self.recall_conversations(query, top_k, threshold)
        )
        return {
            "preferences":    results[0],
            "episodes":       results[1],
            "conversations":  results[2],
        }

    # ── Core Query ────────────────────────────────────────────────────────────

    async def _query_collection(
        self,
        collection_name: str,
        query: str,
        top_k: int,
        threshold: float,
    ) -> List[Dict]:
        """
        Internal: query a ChromaDB collection, filter by threshold, return results.
        ChromaDB returns cosine *distance* (0=identical, 2=opposite).
        We convert: similarity = 1 - (distance / 2) → range [0, 1]
        """
        await self._ensure_init()
        collection = self._collection(collection_name)

        try:
            # Don't query empty collections — ChromaDB raises on n_results > count
            count = await asyncio.to_thread(collection.count)
            if count == 0:
                return []

            actual_k = min(top_k, count)
            embedding = await self._embed(query)

            results = await asyncio.to_thread(
                collection.query,
                query_embeddings=[embedding],
                n_results=actual_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.warning(
                "Semantic query failed for collection '%s': %s",
                collection_name,
                exc,
            )
            return []

        hits = []
        if not results["ids"]:
            return hits

        ids       = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for doc_id, doc, meta, dist in zip(ids, documents, metadatas, distances):
            # Convert cosine distance → similarity score [0, 1]
            similarity = max(0.0, 1.0 - (dist / 2.0))
            if similarity >= threshold:
                hits.append({
                    "id":         doc_id,
                    "document":   doc,
                    "metadata":   meta,
                    "score":      round(similarity, 4),
                    "collection": collection_name,
                })

        # Sort highest relevance first
        hits.sort(key=lambda x: x["score"], reverse=True)
        return hits

    # ── Delete ────────────────────────────────────────────────────────────────

    async def delete_preference(self, key: str) -> bool:
        """Delete a preference by key. Returns True if deleted."""
        await self._ensure_init()
        doc_id = f"pref_{key}"
        try:
            await asyncio.to_thread(self._collection(COLLECTION_PREFS).delete, ids=[doc_id])
            logger.debug(f"Deleted preference: {doc_id}")
            return True
        except Exception as e:
            logger.warning(f"Could not delete preference {key}: {e}")
            return False

    async def clear_collection(self, collection_name: str) -> bool:
        """Delete and recreate a collection (full wipe). Use with caution."""
        await self._ensure_init()
        try:
            await asyncio.to_thread(self._client.delete_collection, collection_name)
            self._collections[collection_name] = await asyncio.to_thread(
                self._client.get_or_create_collection,
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"Cleared collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to clear collection {collection_name}: {e}", exc_info=True)
            return False

    # ── Stats ─────────────────────────────────────────────────────────────────

    async def stats(self) -> Dict[str, Any]:
        """Return counts for all collections."""
        if not self._initialized:
            return {"initialized": False}
        prefs_count = await asyncio.to_thread(self._collection(COLLECTION_PREFS).count)
        episodes_count = await asyncio.to_thread(self._collection(COLLECTION_EPISODES).count)
        convos_count = await asyncio.to_thread(self._collection(COLLECTION_CONVOS).count)
        return {
            "initialized":    True,
            "model":          self.model_name,
            "preferences":    prefs_count,
            "episodes":       episodes_count,
            "conversations":  convos_count,
        }

    def is_ready(self) -> bool:
        return self._initialized

    async def close(self) -> None:
        """Close/stop the ChromaDB client if it has one."""
        if self._client is not None:
            try:
                if hasattr(self._client, "_system") and hasattr(self._client._system, "stop"):
                    await asyncio.to_thread(self._client._system.stop)
            except Exception:
                pass
            self._client = None
            self._initialized = False

