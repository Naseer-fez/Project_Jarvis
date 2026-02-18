"""
core/llm/client.py
═══════════════════
LLMClientV2 — async Ollama client for Jarvis V5.

Supports:
  - complete(prompt, system, temperature) → str
  - complete_json(prompt, system, temperature) → dict | None
  - chat(messages, query_for_memory, profile_summary) → str
  - chat_stream(messages, query_for_memory, profile_summary) → generator
  - is_ollama_running() → bool
"""

import asyncio
import json
import re
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
OLLAMA_CHAT_URL     = "http://localhost:11434/api/chat"
OLLAMA_BASE_URL     = "http://localhost:11434"
DEFAULT_MODEL       = "deepseek-r1:8b"
TIMEOUT_S           = 120

JARVIS_SYSTEM = """You are Jarvis, a local personal AI assistant.
You are intelligent, concise, and helpful.
You remember context about the user and adapt your responses accordingly.
You never call external APIs — you run fully offline."""


def _strip_think(text: str) -> str:
    """Remove DeepSeek R1 <think>...</think> reasoning blocks."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _strip_fences(text: str) -> str:
    text = re.sub(r"^```[a-z]*\n?", "", text.strip())
    text = re.sub(r"\n?```$", "", text)
    return text.strip()


class LLMClientV2:
    """
    Async LLM client wrapping Ollama's API.
    Accepts HybridMemory for context injection.
    """

    def __init__(self, hybrid_memory=None, model: str = DEFAULT_MODEL):
        self.memory = hybrid_memory
        self.model  = model

    # ── Core async methods ─────────────────────────────────

    async def complete(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.1,
    ) -> str:
        """Plain text completion. Returns '' on failure."""
        try:
            import aiohttp
        except ImportError:
            logger.error("aiohttp not installed: pip install aiohttp")
            return ""

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "top_p": 0.9},
        }
        if system:
            payload["system"] = system

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    OLLAMA_GENERATE_URL, json=payload,
                    timeout=aiohttp.ClientTimeout(total=TIMEOUT_S)
                ) as resp:
                    if resp.status != 200:
                        logger.error(f"Ollama HTTP {resp.status}")
                        return ""
                    data = await resp.json()
                    return _strip_think(data.get("response", ""))
        except aiohttp.ClientConnectorError:
            logger.error("Ollama not reachable. Run: ollama serve")
            return ""
        except asyncio.TimeoutError:
            logger.error(f"LLM timeout after {TIMEOUT_S}s")
            return ""
        except Exception as e:
            logger.error(f"LLM complete error: {e}")
            return ""

    async def complete_json(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.0,
    ) -> dict | None:
        """JSON completion. Returns None if output is not valid JSON."""
        raw = await self.complete(prompt, system=system, temperature=temperature)
        if not raw:
            return None
        cleaned = _strip_fences(raw)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse failed: {e} | raw: {cleaned[:200]}")
            return None

    # ── Sync wrappers (used by controller synchronously) ───

    def chat(
        self,
        messages: list[dict],
        query_for_memory: str = "",
        profile_summary: str = "",
    ) -> str:
        """
        Sync wrapper around async chat completion.
        Injects memory context and profile summary into system prompt.
        """
        system = self._build_system(query_for_memory, profile_summary)
        prompt = self._messages_to_prompt(messages)

        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(self.complete(prompt, system=system))
        loop.close()
        return result or "I'm sorry, I couldn't generate a response."

    def chat_stream(
        self,
        messages: list[dict],
        query_for_memory: str = "",
        profile_summary: str = "",
    ):
        """
        Streaming generator. Yields text chunks.
        Falls back to non-streaming if aiohttp not available.
        """
        # For V1 simplicity — run non-streaming and yield as one chunk
        response = self.chat(messages, query_for_memory, profile_summary)
        yield response

    # ── Helpers ────────────────────────────────────────────

    def _build_system(self, query: str = "", profile: str = "") -> str:
        parts = [JARVIS_SYSTEM]

        if profile:
            parts.append(f"\nUSER PROFILE:\n{profile}")

        if query and self.memory:
            try:
                context = self.memory.build_context_block(query, n_results=3)
                if context:
                    parts.append(f"\nRELEVANT MEMORY:\n{context}")
            except Exception as e:
                logger.warning(f"Memory context injection failed: {e}")

        return "\n".join(parts)

    def _messages_to_prompt(self, messages: list[dict]) -> str:
        """Convert message list to a flat prompt string for /api/generate."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        return "\n".join(parts)

    def is_ollama_running(self) -> bool:
        """Quick sync health check."""
        try:
            import aiohttp
            async def _check():
                async with aiohttp.ClientSession() as s:
                    async with s.get(OLLAMA_BASE_URL, timeout=aiohttp.ClientTimeout(total=3)) as r:
                        return r.status == 200
            loop = asyncio.new_event_loop()
            result = loop.run_until_complete(_check())
            loop.close()
            return result
        except Exception:
            return False
