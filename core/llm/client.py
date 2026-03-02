"""Async Ollama client with optional memory and workspace context injection."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "deepseek-r1:8b"
TIMEOUT_S = 120

JARVIS_SYSTEM = (
    "You are Jarvis, a local personal AI assistant.\n"
    "You are concise, technical, and truthful.\n"
    "You run on the user's local machine."
)


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


def _strip_fences(text: str) -> str:
    cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", (text or "").strip())
    cleaned = re.sub(r"\n?```$", "", cleaned)
    return cleaned.strip()


def _get_workspace_map(path: str, max_depth: int = 3, max_files: int = 50) -> str:
    """Build a compact directory view to ground model responses in local files."""
    root = Path(path)
    if not root.exists() or not root.is_dir():
        return ""

    lines: list[str] = []
    count = 0
    ignored = {"__pycache__", ".git", "node_modules", ".venv", "venv", "jarvis_env"}

    for item in sorted(root.rglob("*")):
        if count >= max_files:
            lines.append("... (truncated)")
            break

        relative = item.relative_to(root)
        if any(part in ignored for part in relative.parts):
            continue

        depth = len(relative.parts)
        if depth > max_depth:
            continue

        indent = "  " * (depth - 1)
        marker = "[DIR]" if item.is_dir() else "[FILE]"
        lines.append(f"{indent}{marker} {item.name}")
        count += 1

    return "\n".join(lines)


class LLMClientV2:
    def __init__(self, hybrid_memory: Any = None, model: str = DEFAULT_MODEL):
        self.memory = hybrid_memory
        self.model = model

    async def complete(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.1,
        keep_think: bool = False,
    ) -> str:
        """Text completion via Ollama /api/generate."""
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
                ) as response:
                    if response.status != 200:
                        logger.error("Ollama HTTP %s", response.status)
                        return ""
                    data = await response.json()
                    raw = str(data.get("response", ""))
                    return raw if keep_think else _strip_think(raw)
        except asyncio.TimeoutError:
            logger.error("LLM timeout after %ss", TIMEOUT_S)
            return ""
        except Exception as exc:  # noqa: BLE001
            logger.error("LLM completion failed: %s", exc)
            return ""

    async def complete_json(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.0,
    ) -> dict[str, Any] | None:
        raw = await self.complete(prompt, system=system, temperature=temperature)
        if not raw:
            return None

        try:
            return json.loads(_strip_fences(raw))
        except json.JSONDecodeError as exc:
            logger.error("JSON parse failed: %s", exc)
            return None

    def chat(
        self,
        messages: list[dict[str, Any]],
        query_for_memory: str = "",
        profile_summary: str = "",
        workspace_path: str = "",
    ) -> str:
        system = self._build_system(
            query=query_for_memory,
            profile=profile_summary,
            workspace_path=workspace_path,
        )
        prompt = self._messages_to_prompt(messages)

        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.complete(prompt, system=system)) or ""
        finally:
            loop.close()

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        query_for_memory: str = "",
        profile_summary: str = "",
        workspace_path: str = "",
    ):
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

        if query and self.memory is not None and hasattr(self.memory, "build_context_block"):
            try:
                context = self.memory.build_context_block(query, n_results=3)
            except TypeError:
                context = self.memory.build_context_block(query)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Memory context injection failed: %s", exc)
                context = ""

            if context:
                parts.append(f"\nRELEVANT MEMORY:\n{context}")

        return "\n".join(parts)

    @staticmethod
    def _messages_to_prompt(messages: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for message in messages:
            role = str(message.get("role", "user")).strip().capitalize()
            content = str(message.get("content", ""))
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def is_ollama_running(self) -> bool:
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
            try:
                return loop.run_until_complete(_check())
            finally:
                loop.close()
        except Exception:
            return False


__all__ = ["LLMClientV2"]
