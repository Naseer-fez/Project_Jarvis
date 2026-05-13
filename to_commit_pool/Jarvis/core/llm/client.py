"""Async LLM client — single entry point for all Jarvis LLM calls.

Architecture:
    LLMClientV2.complete(prompt, task_type)
        → ModelRouter.pick_model(task_type)   → model name
        → OllamaClient.complete(prompt, model) → response (or raise)
        → CloudLLMClient.complete(prompt)      → fallback response
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from core.config.defaults import OLLAMA_BASE_URL
from core.llm.ollama_client import OllamaClient
from core.llm.model_router import ModelRouter

logger = logging.getLogger(__name__)

try:
    from core.llm.cloud_client import CloudLLMClient
except Exception:  # pragma: no cover - cloud fallback is optional
    CloudLLMClient = None  # type: ignore[assignment]

DEFAULT_MODEL = "deepseek-r1:8b"

JARVIS_SYSTEM = (
    "You are Jarvis, a local personal AI assistant.\n"
    "You are concise, technical, and truthful.\n"
    "You run on the user's local machine."
)


def _strip_fences(text: str) -> str:
    cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", (text or "").strip())
    cleaned = re.sub(r"\n?```$", "", cleaned)
    return cleaned.strip()


_WORKSPACE_CACHE: dict[str, dict[str, Any]] = {}

def _get_workspace_map(path: str, max_depth: int = 3, max_files: int = 50) -> str:
    """Build a compact directory view to ground model responses in local files."""
    global _WORKSPACE_CACHE
    now = time.time()

    if path in _WORKSPACE_CACHE and now - _WORKSPACE_CACHE[path]["time"] < 60:
        return _WORKSPACE_CACHE[path]["data"]

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

    result = "\n".join(lines)
    _WORKSPACE_CACHE[path] = {"time": time.time(), "data": result}
    return result


class LLMClientV2:
    """Public interface — all LLM calls in Jarvis enter here.

    Wiring (in order):
        1. ``ModelRouter.pick_model(task_type)`` → model name
        2. ``OllamaClient.complete(prompt, model=…)`` → try local first
        3. ``CloudLLMClient.complete(prompt)`` → fallback if Ollama fails
    """

    def __init__(
        self,
        hybrid_memory: Any = None,
        model: str = DEFAULT_MODEL,
        profile: Any = None,
        base_url: str = OLLAMA_BASE_URL,
    ) -> None:
        self.memory = hybrid_memory
        self.model = model
        self.profile = profile
        self.base_url = base_url

        # ── Component wiring ─────────────────────────────────────────────
        self._ollama = OllamaClient(base_url=base_url)
        self.model_router: ModelRouter | None = None

        self._cloud_client = None
        import os
        if (
            CloudLLMClient is not None
            and str(os.environ.get("CLOUD_LLM_FALLBACK_ENABLED", "true")).lower() == "true"
        ):
            try:
                self._cloud_client = CloudLLMClient()
            except Exception as e:
                logger.warning("Could not init CloudLLMClient: %s", e)

    def set_router(self, router: ModelRouter) -> None:
        self.model_router = router

    # ── Core complete() — the single path every prompt takes ─────────────

    async def complete(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.1,
        task_type: str = "chat",
        keep_think: bool = False,
    ) -> str:
        """Text completion: ModelRouter → OllamaClient → CloudLLMClient fallback.

        Steps:
            1. Ask ModelRouter for the best model name for this task_type.
            2. Call OllamaClient.complete() with that model.
            3. If Ollama raises ANY exception, fall back to CloudLLMClient.
        """
        # Step 1 — pick model
        if self.model_router is not None:
            model_to_use = self.model_router.pick_model(task_type)
        else:
            model_to_use = self.model

        # Step 2 — try Ollama
        try:
            return await self._ollama.complete(
                prompt,
                system=system,
                temperature=temperature,
                model=model_to_use,
                keep_think=keep_think,
            )
        except Exception as exc:
            logger.warning("Ollama failed (%s). Attempting cloud fallback.", exc)

        # Step 3 — cloud fallback
        if self._cloud_client is not None:
            try:
                return await self._cloud_client.complete(
                    prompt, system=system, temperature=temperature
                )
            except Exception as cloud_exc:
                logger.error("Cloud fallback also failed: %s", cloud_exc)

        return ""

    # ── JSON completion helper ───────────────────────────────────────────

    async def complete_json(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.0,
        task_type: str = "planning",
    ) -> dict[str, Any] | None:
        raw = await self.complete(prompt, system=system, temperature=temperature, task_type=task_type)
        if not raw:
            return None

        try:
            return json.loads(_strip_fences(raw))
        except json.JSONDecodeError as exc:
            logger.error("JSON parse failed: %s", exc)
            return None

    # ── Chat interface (backward-compat for controller / agent loop) ─────

    async def chat_async(
        self,
        messages: list[dict[str, Any]],
        query_for_memory: str = "",
        profile_summary: str = "",
        workspace_path: str = "",
        trace_id: str | None = None,
    ) -> str:
        """Async version — use this inside any async context (agent loop, controller)."""
        if trace_id:
            logger.info("[trace=%s] Client chat_async starting", trace_id)

        system = self._build_system(
            query=query_for_memory,
            profile=profile_summary,
            workspace_path=workspace_path,
        )
        prompt = self._messages_to_prompt(messages)
        return await self.complete(prompt, system=system) or ""

    def chat(
        self,
        messages: list[dict[str, Any]],
        query_for_memory: str = "",
        profile_summary: str = "",
        workspace_path: str = "",
        trace_id: str | None = None,
    ) -> str:
        """Sync bridge — ONLY call from truly synchronous, non-async contexts."""
        self._build_system(
            query=query_for_memory,
            profile=profile_summary,
            workspace_path=workspace_path,
        )
        self._messages_to_prompt(messages)

        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                asyncio.run,
                self.chat_async(messages, query_for_memory, profile_summary, workspace_path, trace_id=trace_id)
            )
            return future.result()

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        query_for_memory: str = "",
        profile_summary: str = "",
        workspace_path: str = "",
    ):
        response = self.chat(messages, query_for_memory, profile_summary, workspace_path)
        yield response

    # ── Health check ─────────────────────────────────────────────────────

    def is_ollama_running(self) -> bool:
        """Sync health check — runs in its own thread to avoid event loop conflicts."""
        import concurrent.futures

        async def _check() -> bool:
            return await self._ollama.is_running()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            try:
                return pool.submit(asyncio.run, _check()).result(timeout=5)
            except Exception:
                return False

    # ── Internal helpers ─────────────────────────────────────────────────

    def _build_system(self, query: str = "", profile: str = "", workspace_path: str = "") -> str:
        parts = [JARVIS_SYSTEM]

        profile_obj = getattr(self, "profile", None)
        if profile_obj is not None:
            profile_injection = ""
            style_instruction = ""
            try:
                profile_injection = str(profile_obj.get_system_prompt_injection() or "").strip()
            except Exception as exc:  # noqa: BLE001
                logger.debug("Profile injection failed: %s", exc)
            try:
                style_instruction = str(profile_obj.get_communication_style() or "").strip()
            except Exception as exc:  # noqa: BLE001
                logger.debug("Profile style injection failed: %s", exc)

            combined = " ".join(part for part in (profile_injection, style_instruction) if part).strip()
            if combined:
                words = combined.split()
                if len(words) > 120:
                    combined = " ".join(words[:120])
                parts.append(f"\nPROFILE GUIDANCE:\n{combined}")

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


__all__ = ["LLMClientV2"]
