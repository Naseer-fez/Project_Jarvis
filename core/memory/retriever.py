import logging
import re
from typing import Any, Callable

logger = logging.getLogger(__name__)

class MemoryRetriever:
    """Handles hybrid search over semantic memory and SQLite storage."""
    
    def __init__(self, db_pool, semantic_memory):
        self.db_pool = db_pool
        self.semantic = semantic_memory

    @staticmethod
    def query_tokens(query: str) -> list[str]:
        tokens = re.findall(r"[a-z0-9]{3,}", str(query or "").lower())
        return tokens[:10]

    @staticmethod
    def score_text(text: str, tokens: list[str]) -> float:
        if not tokens:
            return 0.5
        lowered = str(text or "").lower()
        hits = sum(1 for token in tokens if token in lowered)
        if hits <= 0:
            return 0.0
        return min(1.0, hits / max(1.0, float(len(tokens))))

    async def recall_preferences(self, query: str, top_k: int, is_hybrid: bool, init_schema_cb: Callable) -> list[dict[str, Any]]:
        if is_hybrid:
            try:
                raw = await self.semantic.recall_preferences(query, top_k=top_k, threshold=0.0)
                return [
                    {
                        "key": item.get("metadata", {}).get("key", ""),
                        "value": item.get("metadata", {}).get("value", ""),
                        "score": item.get("score", 0.0),
                        "document": item.get("document", ""),
                    }
                    for item in raw
                ]
            except Exception as exc:
                logger.debug("Semantic preference recall failed: %s", exc)

        await init_schema_cb()
        tokens = self.query_tokens(query)
        async with self.db_pool.acquire() as conn:
            async with conn.execute(
                "SELECT key, value FROM preferences ORDER BY updated_at DESC LIMIT 200"
            ) as cursor:
                rows = await cursor.fetchall()

        ranked: list[dict[str, Any]] = []
        for row in rows:
            key = str(row["key"] or "")
            val = str(row["value"] or "")
            score = self.score_text(key, tokens)
            if tokens and score < 0.70:
                continue
            ranked.append({"key": key, "value": val, "score": score if tokens else 1.0})

        if tokens:
            ranked.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        return ranked[: max(1, top_k)]

    async def _recall_sqlite_episodes(self, query: str, top_k: int, init_schema_cb: Callable) -> list[dict[str, Any]]:
        tokens = self.query_tokens(query)
        await init_schema_cb()
        async with self.db_pool.acquire() as conn:
            async with conn.execute(
                "SELECT event, category, timestamp FROM episodes ORDER BY timestamp DESC LIMIT 200"
            ) as cursor:
                rows = await cursor.fetchall()

        ranked: list[dict[str, Any]] = []
        for row in rows:
            event = str(row["event"] or "")
            category = str(row["category"] or "")
            haystack = f"{event} {category}".strip()
            score = self.score_text(haystack, tokens)
            if tokens and score <= 0.0:
                continue
            ranked.append({
                "event": event,
                "category": category,
                "timestamp": str(row["timestamp"] or ""),
                "score": score if tokens else 0.4,
                "document": event,
            })

        ranked.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        return ranked[: max(1, top_k)]

    async def _recall_sqlite_conversations(self, query: str, top_k: int, init_schema_cb: Callable) -> list[dict[str, Any]]:
        tokens = self.query_tokens(query)
        await init_schema_cb()
        async with self.db_pool.acquire() as conn:
            async with conn.execute(
                "SELECT user_input, assistant_response, timestamp FROM conversations ORDER BY timestamp DESC LIMIT 200"
            ) as cursor:
                rows = await cursor.fetchall()

        ranked: list[dict[str, Any]] = []
        for row in rows:
            user_text = str(row["user_input"] or "")
            assistant_text = str(row["assistant_response"] or "")
            haystack = f"{user_text} {assistant_text}".strip()
            score = self.score_text(haystack, tokens)
            if tokens and score <= 0.0:
                continue
            ranked.append({
                "user_input": user_text,
                "assistant_response": assistant_text,
                "timestamp": str(row["timestamp"] or ""),
                "score": score if tokens else 0.4,
                "document": f"User: {user_text}\\nAssistant: {assistant_text}",
            })

        ranked.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        return ranked[: max(1, top_k)]

    async def recall_all(self, query: str, top_k: int, is_hybrid: bool, init_schema_cb: Callable) -> dict[str, list[dict[str, Any]]]:
        if is_hybrid:
            try:
                raw = await self.semantic.recall_all(query, top_k=top_k, threshold=0.0)
                return {
                    "preferences": [
                        {
                            "key": item.get("metadata", {}).get("key", ""),
                            "value": item.get("metadata", {}).get("value", ""),
                            "score": item.get("score", 0.0),
                            "document": item.get("document", ""),
                        }
                        for item in raw.get("preferences", [])
                    ],
                    "episodes": [
                        {
                            "event": item.get("metadata", {}).get("event", item.get("document", "")),
                            "category": item.get("metadata", {}).get("category", ""),
                            "timestamp": item.get("metadata", {}).get("timestamp", ""),
                            "score": item.get("score", 0.0),
                            "document": item.get("document", ""),
                        }
                        for item in raw.get("episodes", [])
                    ],
                    "conversations": [
                        {
                            "user_input": item.get("metadata", {}).get("user_input", ""),
                            "assistant_response": item.get("metadata", {}).get("assistant_response", ""),
                            "timestamp": item.get("metadata", {}).get("timestamp", ""),
                            "score": item.get("score", 0.0),
                            "document": item.get("document", ""),
                        }
                        for item in raw.get("conversations", [])
                    ],
                }
            except Exception as exc:
                logger.debug("Semantic recall_all failed: %s", exc)

        return {
            "preferences": await self.recall_preferences(query, top_k=top_k, is_hybrid=is_hybrid, init_schema_cb=init_schema_cb),
            "episodes": await self._recall_sqlite_episodes(query, top_k=top_k, init_schema_cb=init_schema_cb),
            "conversations": await self._recall_sqlite_conversations(query, top_k=top_k, init_schema_cb=init_schema_cb),
        }
