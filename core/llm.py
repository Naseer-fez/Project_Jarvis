"""
core/llm_v2.py
───────────────
Updated LLM interface for Jarvis Session 4.

Changes from Session 3 (core/llm.py):
  - Now uses HybridMemory instead of raw LongTermMemory
  - Injects semantically-compressed context via ContextCompressor
  - Supports streaming responses
  - Better system prompt with dynamic memory block
  - Configurable model and temperature

Author: Jarvis Session 4
"""

import json
import logging
from typing import Optional, Generator

import requests

from memory.hybrid_memory import HybridMemory
from core.context_compressor import ContextCompressor

logger = logging.getLogger(__name__)


# ─── Config ────────────────────────────────────────────────────────────────────

OLLAMA_BASE_URL  = "http://localhost:11434"
DEFAULT_MODEL    = "deepseek-r1:8b"
DEFAULT_TIMEOUT  = 120   # seconds
DEFAULT_TEMP     = 0.7

SYSTEM_PROMPT_TEMPLATE = """\
You are Jarvis, a personal AI assistant. You are precise, helpful, and remember \
details about the user to provide personalized responses.

{memory_block}

Guidelines:
- Be concise and direct. Avoid unnecessary filler phrases.
- If you recall relevant information about the user, use it naturally.
- If you don't know something, say so honestly.
- Respond in plain text unless formatting is explicitly requested.
"""


class LLMClientV2:
    """
    Ollama-backed LLM client with semantic memory injection.

    Session 4 upgrade:
      - Pulls relevant context from HybridMemory before each response
      - Compresses context with ContextCompressor to stay within token budget
      - Supports both streaming and non-streaming responses
    """

    def __init__(
        self,
        hybrid_memory: Optional[HybridMemory] = None,
        model: str = DEFAULT_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        temperature: float = DEFAULT_TEMP,
    ):
        self.hybrid_memory  = hybrid_memory
        self.compressor     = ContextCompressor()
        self.model          = model
        self.base_url       = base_url
        self.temperature    = temperature
        self._generate_url  = f"{base_url}/api/generate"
        self._chat_url      = f"{base_url}/api/chat"

    # ── Health ────────────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Check if Ollama is running."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """Return available Ollama model names."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            data = r.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    # ── System Prompt ─────────────────────────────────────────────────────────

    def _build_system_prompt(self, query: str) -> str:
        """
        Build a system prompt with injected memory context.
        If HybridMemory is available, fetches semantically relevant context.
        """
        memory_block = ""

        if self.hybrid_memory:
            try:
                recall = self.hybrid_memory.recall_all(query, top_k=6)
                memory_block = self.compressor.compress(query, recall)
            except Exception as e:
                logger.warning(f"Memory recall failed (non-fatal): {e}")

        return SYSTEM_PROMPT_TEMPLATE.format(
            memory_block=memory_block if memory_block else "(No memory context available)"
        )

    # ── Non-Streaming Chat ────────────────────────────────────────────────────

    def chat(
        self,
        messages: list[dict],
        query_for_memory: Optional[str] = None,
    ) -> str:
        """
        Send a conversation to the LLM and return the full response string.

        Args:
            messages:          List of {"role": "user"/"assistant", "content": "..."} dicts.
            query_for_memory:  Text used for semantic memory lookup. Defaults to last user message.

        Returns:
            The assistant's response as a string.
        """
        if not self.is_available():
            return "[Error: Ollama is not running. Please start Ollama and try again.]"

        # Determine query for memory lookup
        if query_for_memory is None:
            user_msgs = [m["content"] for m in messages if m["role"] == "user"]
            query_for_memory = user_msgs[-1] if user_msgs else ""

        system_prompt = self._build_system_prompt(query_for_memory)

        payload = {
            "model":  self.model,
            "stream": False,
            "options": {"temperature": self.temperature},
            "system":  system_prompt,
            "messages": messages,
        }

        try:
            r = requests.post(self._chat_url, json=payload, timeout=DEFAULT_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            return data.get("message", {}).get("content", "[No response]")
        except requests.exceptions.Timeout:
            return "[Error: Request timed out. The model may be loading — try again.]"
        except Exception as e:
            logger.error(f"LLM chat error: {e}")
            return f"[Error communicating with LLM: {e}]"

    # ── Streaming Chat ────────────────────────────────────────────────────────

    def chat_stream(
        self,
        messages: list[dict],
        query_for_memory: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """
        Stream the LLM response token by token.
        Yields string chunks as they arrive.

        Usage:
            for chunk in llm.chat_stream(messages):
                print(chunk, end="", flush=True)
        """
        if not self.is_available():
            yield "[Error: Ollama is not running.]"
            return

        if query_for_memory is None:
            user_msgs = [m["content"] for m in messages if m["role"] == "user"]
            query_for_memory = user_msgs[-1] if user_msgs else ""

        system_prompt = self._build_system_prompt(query_for_memory)

        payload = {
            "model":  self.model,
            "stream": True,
            "options": {"temperature": self.temperature},
            "system":  system_prompt,
            "messages": messages,
        }

        try:
            with requests.post(
                self._chat_url,
                json=payload,
                stream=True,
                timeout=DEFAULT_TIMEOUT,
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)
                            content = chunk.get("message", {}).get("content", "")
                            if content:
                                yield content
                            if chunk.get("done"):
                                break
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"[Streaming error: {e}]"

    # ── Simple Prompt (non-chat) ──────────────────────────────────────────────

    def ask(self, prompt: str, context: Optional[str] = None) -> str:
        """
        Simple single-turn prompt without conversation history.
        Optionally inject raw context string.
        """
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, query_for_memory=context or prompt)

    # ── Info ──────────────────────────────────────────────────────────────────

    def info(self) -> dict:
        return {
            "model":             self.model,
            "base_url":          self.base_url,
            "temperature":       self.temperature,
            "ollama_available":  self.is_available(),
            "memory_enabled":    self.hybrid_memory is not None,
        }
