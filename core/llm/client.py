"""
core/llm/client.py
LLMClientV2 - async Ollama client for Jarvis.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "deepseek-r1:8b"
TIMEOUT_S = 120

JARVIS_SYSTEM = """You are Jarvis, a local personal AI assistant.
You are intelligent, concise, and helpful.
You remember context about the user and adapt your responses accordingly.
You never call external APIs - you run fully offline."""


def _strip_think(text: str) -> str:
    """Remove DeepSeek R1 <think>...</think> reasoning blocks."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _strip_fences(text: str) -> str:
    text = re.sub(r"^```[a-z]*\n?", "", text.strip())
    text = re.sub(r"\n?```$", "", text)
    return text.strip()


def _get_workspace_map(path: str, max_depth: int = 3, max_files: int = 50) -> str:
    """Generate a compact directory tree for LLM context."""
    root = Path(path)
    if not root.exists():
        return ""

    lines: list[str] = []
    count = 0

    for item in sorted(root.rglob("*")):
        if count >= max_files:
            lines.append("  ... (truncated)")
            break

        parts = item.parts
        if any(p in parts for p in ("__pycache__", ".git", "node_modules", ".venv", "venv")):
            continue

        depth = len(item.relative_to(root).parts)
        if depth > max_depth:
            continue

        indent = "  " * (depth - 1)
        marker = "[DIR]" if item.is_dir() else "[FILE]"
        lines.append(f"{indent}{marker} {item.name}")
        count += 1

    return "\n".join(lines)


class LLMClientV2:
    """
    Async LLM client wrapping Ollama's API.
    Accepts HybridMemory for context injection.
    """

    def __init__(self, hybrid_memory=None, model: str = DEFAULT_MODEL):
        self.memory = hybrid_memory
        self.model = model

    async def complete(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.1,
        keep_think: bool = False,
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
                    OLLAMA_GENERATE_URL,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=TIMEOUT_S),
                ) as resp:
                    if resp.status != 200:
                        logger.error("Ollama HTTP %s", resp.status)
                        return ""
                    data = await resp.json()
                    raw = data.get("response", "")
                    return raw if keep_think else _strip_think(raw)
        except aiohttp.ClientConnectorError:
            logger.error("Ollama not reachable. Run: ollama serve")
            return ""
        except asyncio.TimeoutError:
            logger.error("LLM timeout after %ss", TIMEOUT_S)
            return ""
        except Exception as exc:  # noqa: BLE001
            logger.error("LLM complete error: %s", exc)
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
        except json.JSONDecodeError as exc:
            logger.error("JSON parse failed: %s | raw: %s", exc, cleaned[:200])
            return None

    def chat(
        self,
        messages: list[dict],
        query_for_memory: str = "",
        profile_summary: str = "",
        workspace_path: str = "",
    ) -> str:
        """
        Sync wrapper around async completion.
        Injects memory context and profile summary into system prompt.
        """
        system = self._build_system(query_for_memory, profile_summary, workspace_path)
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
        workspace_path: str = "",
    ):
        """Streaming generator fallback implementation."""
        response = self.chat(messages, query_for_memory, profile_summary, workspace_path)
        yield response

    def _build_system(self, query: str = "", profile: str = "", workspace_path: str = "") -> str:
        parts = [JARVIS_SYSTEM]

        if profile:
            parts.append(f"\nUSER PROFILE:\n{profile}")

        if workspace_path:
            workspace_map = _get_workspace_map(workspace_path)
            if workspace_map:
                parts.append(f"\nWORKSPACE:\n{workspace_map}")

        if query and self.memory:
            try:
                context = self.memory.build_context_block(query, n_results=3)
                if context:
                    parts.append(f"\nRELEVANT MEMORY:\n{context}")
            except TypeError:
                # Backward compatibility with older memory adapters.
                try:
                    context = self.memory.build_context_block(query)
                    if context:
                        parts.append(f"\nRELEVANT MEMORY:\n{context}")
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Memory context injection failed: %s", exc)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Memory context injection failed: %s", exc)

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

            async def _check() -> bool:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        OLLAMA_BASE_URL,
                        timeout=aiohttp.ClientTimeout(total=3),
                    ) as response:
                        return response.status == 200

            loop = asyncio.new_event_loop()
            result = loop.run_until_complete(_check())
            loop.close()
            return result
        except Exception:
            return False
