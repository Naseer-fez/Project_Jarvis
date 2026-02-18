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
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

DEFAULT_MODEL      = "all-MiniLM-L6-v2"   # 80 MB, fast, good quality
DEFAULT_TOP_K      = 5
DEFAULT_THRESHOLD  = 0.30                  # cosine similarity (0–1); lower = more results
CHROMA_PATH        = "D:/AI/Jarvis/data/chroma"

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
    ):
        self.chroma_path = chroma_path
        self.model_name  = model_name
        # typed as Any because SentenceTransformer is imported lazily
        self._model: Any = None
        self._client: Optional[chromadb.ClientAPI] = None
        self._collections: Dict[str, Any] = {}
        self._initialized = False

    # ── Init ──────────────────────────────────────────────────────────────────

    def initialize(self) -> bool:
        """
        Lazy initialization — load model and connect to ChromaDB.
        Returns True on success, False on failure.
        """
        if self._initialized:
            return True
            
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)

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
            logger.error(f"SemanticMemory initialization failed: {e}")
            return False

    def _ensure_init(self):
        if not self._initialized:
            if not self.initialize():
                raise RuntimeError("SemanticMemory is not initialized.")

    def _embed(self, text: str) -> List[float]:
        """Generate a normalized embedding vector for the given text."""
        return self._model.encode(text, normalize_embeddings=True).tolist()

    def _collection(self, name: str):
        return self._collections[name]

    # ── Store ─────────────────────────────────────────────────────────────────

    def store_preference(self, key: str, value: str) -> str:
        """
        Upsert a user preference into the vector store.
        The document text is: "key: value" for rich semantic matching.
        Returns the document ID.
        """
        self._ensure_init()
        doc_id   = f"pref_{key}"
        doc_text = f"{key}: {value}"
        embedding = self._embed(doc_text)

        self._collection(COLLECTION_PREFS).upsert(
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

    def store_episode(self, event: str, category: str = "general") -> str:
        """
        Store an episodic memory event.
        Returns the document ID.
        """
        self._ensure_init()
        doc_id    = f"ep_{uuid.uuid4().hex[:12]}"
        embedding = self._embed(event)

        self._collection(COLLECTION_EPISODES).add(
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

    def store_conversation_turn(
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
        self._ensure_init()
        doc_id    = f"conv_{uuid.uuid4().hex[:12]}"
        doc_text  = f"User: {user_input}\nAssistant: {assistant_response}"
        embedding = self._embed(doc_text)

        self._collection(COLLECTION_CONVOS).add(
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

    def recall_preferences(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> List[Dict]:
        """
        Retrieve the most semantically similar preferences to the query.
        Returns a list of result dicts sorted by relevance score (descending).
        """
        return self._query_collection(COLLECTION_PREFS, query, top_k, threshold)

    def recall_episodes(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> List[Dict]:
        """Retrieve the most relevant episodic memories for the query."""
        return self._query_collection(COLLECTION_EPISODES, query, top_k, threshold)

    def recall_conversations(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> List[Dict]:
        """Retrieve the most relevant past conversation turns for the query."""
        return self._query_collection(COLLECTION_CONVOS, query, top_k, threshold)

    def recall_all(
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
        return {
            "preferences":    self.recall_preferences(query, top_k, threshold),
            "episodes":       self.recall_episodes(query, top_k, threshold),
            "conversations":  self.recall_conversations(query, top_k, threshold),
        }

    # ── Core Query ────────────────────────────────────────────────────────────

    def _query_collection(
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
        self._ensure_init()
        collection = self._collection(collection_name)

        # Don't query empty collections — ChromaDB raises on n_results > count
        count = collection.count()
        if count == 0:
            return []

        actual_k  = min(top_k, count)
        embedding = self._embed(query)

        results = collection.query(
            query_embeddings=[embedding],
            n_results=actual_k,
            include=["documents", "metadatas", "distances"],
        )

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

    def delete_preference(self, key: str) -> bool:
        """Delete a preference by key. Returns True if deleted."""
        self._ensure_init()
        doc_id = f"pref_{key}"
        try:
            self._collection(COLLECTION_PREFS).delete(ids=[doc_id])
            logger.debug(f"Deleted preference: {doc_id}")
            return True
        except Exception as e:
            logger.warning(f"Could not delete preference {key}: {e}")
            return False

    def clear_collection(self, collection_name: str) -> bool:
        """Delete and recreate a collection (full wipe). Use with caution."""
        self._ensure_init()
        try:
            self._client.delete_collection(collection_name)
            self._collections[collection_name] = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"Cleared collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to clear collection {collection_name}: {e}")
            return False

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """Return counts for all collections."""
        if not self._initialized:
            return {"initialized": False}
        return {
            "initialized":    True,
            "model":          self.model_name,
            "preferences":    self._collection(COLLECTION_PREFS).count(),
            "episodes":       self._collection(COLLECTION_EPISODES).count(),
            "conversations":  self._collection(COLLECTION_CONVOS).count(),
        }

    def is_ready(self) -> bool:
        return self._initialized
