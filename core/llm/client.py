"""Async LLM client — single entry point for all Jarvis LLM calls.

Architecture:
    LLMClientV2.complete(prompt, task_type)
        → ModelRouter.pick_model(task_type)   → model name
        → OllamaClient.complete(prompt, model) → response (or raise)
        → CloudLLMClient.complete(prompt)      → fallback response
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from pathlib import Path
from typing import Any

from core.config.defaults import OLLAMA_BASE_URL
from core.llm.ollama_client import OllamaClient
from core.llm.model_router import ModelRouter
from core.llm.defaults import DEFAULT_MODEL

logger = logging.getLogger(__name__)

try:
    from core.llm.cloud_client import CloudLLMClient
except Exception:  # pragma: no cover - cloud fallback is optional
    CloudLLMClient = None  # type: ignore

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
        return str(_WORKSPACE_CACHE[path]["data"])

    if len(_WORKSPACE_CACHE) > 50:
        _WORKSPACE_CACHE.clear()

    root = Path(path)
    if not root.exists() or not root.is_dir():
        return ""

    lines: list[str] = []
    count = 0
    ignored = {"__pycache__", ".git", "node_modules", ".venv", "venv", "jarvis_env"}

    def _walk(current: Path, depth: int) -> None:
        nonlocal count
        if depth > max_depth or count >= max_files:
            return
        try:
            for item in sorted(current.iterdir()):
                if count >= max_files:
                    if lines and lines[-1] != "... (truncated)":
                        lines.append("... (truncated)")
                    break
                if item.name in ignored or item.name.startswith("."):
                    continue
                indent = "  " * (depth - 1) if depth > 0 else ""
                marker = "[DIR]" if item.is_dir() else "[FILE]"
                if depth > 0:
                    lines.append(f"{indent}{marker} {item.name}")
                    count += 1
                if item.is_dir():
                    _walk(item, depth + 1)
        except PermissionError:
            pass

    _walk(root, 0)
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
        max_concurrent: int = 4,
    ) -> None:
        self.memory = hybrid_memory
        self.model = model
        self.profile = profile
        self.base_url = base_url
        self._semaphore = asyncio.Semaphore(max_concurrent)

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

    def set_telemetry(self, telemetry: Any) -> None:
        """Connect execution telemetry for recording LLM call metrics."""
        self._telemetry = telemetry

    # ── Core complete() — the single path every prompt takes ─────────────

    async def complete(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.1,
        task_type: str = "chat",
        keep_think: bool = False,
        classification: dict[str, Any] | None = None,
    ) -> str:
        """Text completion: ModelRouter → OllamaClient → CloudLLMClient fallback.

        Steps:
            1. Ask ModelRouter for the best model name for this task_type.
            2. Call OllamaClient.complete() with that model.
            3. If response quality is poor, auto-escalate once.
            4. If Ollama raises ANY exception, fall back to CloudLLMClient
               with the correct tier.
            5. Record telemetry for every call.
        """
        async with self._semaphore:
            # Step 1 — pick model (adaptive if classification provided)
            routing_decision = None
            if self.model_router is not None:
                if classification is not None and hasattr(self.model_router, "route_adaptive"):
                    routing_decision = self.model_router.route_adaptive(classification)
                    model_to_use = routing_decision.model
                else:
                    model_to_use = self.model_router.pick_model(task_type)
            else:
                model_to_use = self.model

            tier = 2  # default tier for cloud fallback
            if routing_decision is not None:
                tier = routing_decision.tier
            elif self.model_router is not None and hasattr(self.model_router, "_registry"):
                tier = self.model_router._registry.get_tier(model_to_use)

            # Step 2 — try Ollama
            t0 = time.time()
            response = ""
            try:
                response = await self._ollama.complete(
                    prompt,
                    system=system,
                    temperature=temperature,
                    model=model_to_use,
                    keep_think=keep_think,
                )
                latency_ms = (time.time() - t0) * 1000
                self._record_telemetry(model_to_use, task_type, latency_ms, prompt, response, True)

                # Step 3 — auto-escalate if quality is poor
                if (
                    self.model_router is not None
                    and hasattr(self.model_router, "should_escalate")
                    and self.model_router.should_escalate(model_to_use, task_type, response)
                ):
                    escalated_model = self.model_router.escalate(model_to_use)
                    if escalated_model != model_to_use:
                        logger.info(
                            "Auto-escalating from %s to %s (poor quality detected)",
                            model_to_use, escalated_model,
                        )
                        t1 = time.time()
                        try:
                            response = await self._ollama.complete(
                                prompt,
                                system=system,
                                temperature=temperature,
                                model=escalated_model,
                                keep_think=keep_think,
                            )
                            latency_ms = (time.time() - t1) * 1000
                            self._record_telemetry(
                                escalated_model, task_type, latency_ms, prompt, response, True
                            )
                        except Exception as esc_exc:
                            logger.warning("Escalated model %s also failed: %s", escalated_model, esc_exc)

                return response

            except Exception as exc:
                latency_ms = (time.time() - t0) * 1000
                self._record_telemetry(model_to_use, task_type, latency_ms, prompt, "", False)
                logger.warning("Ollama failed (%s). Attempting cloud fallback.", exc)

            # Step 4 — cloud fallback with correct tier
            if self._cloud_client is not None:
                t0 = time.time()
                try:
                    response = await self._cloud_client.complete(
                        prompt, system=system, temperature=temperature, tier=tier
                    )
                    latency_ms = (time.time() - t0) * 1000
                    cloud_model = f"cloud_tier{tier}"
                    self._record_telemetry(cloud_model, task_type, latency_ms, prompt, response, True)
                    return response
                except Exception as cloud_exc:
                    latency_ms = (time.time() - t0) * 1000
                    self._record_telemetry(f"cloud_tier{tier}", task_type, latency_ms, prompt, "", False)
                    logger.error("Cloud fallback also failed: %s", cloud_exc, exc_info=True)

            return ""

    def _record_telemetry(
        self,
        model: str,
        task_type: str,
        latency_ms: float,
        prompt: str,
        response: str,
        success: bool,
    ) -> None:
        """Record call metrics to telemetry if available."""
        telemetry = getattr(self, "_telemetry", None)
        if telemetry is None:
            return
        try:
            # Rough token estimation: ~4 chars per token
            input_tokens = max(1, len(prompt) // 4)
            output_tokens = max(0, len(response) // 4)
            telemetry.record(
                model=model,
                task_type=task_type,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                success=success,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Telemetry recording failed: %s", exc)

    # ── JSON completion helper ───────────────────────────────────────────


    # ── Chat interface (backward-compat for controller / agent loop) ─────

    async def chat_async(
        self,
        messages: list[dict[str, Any]],
        query_for_memory: str = "",
        profile_summary: str = "",
        workspace_path: str = "",
        trace_id: str | None = None,
        task_type: str = "chat",
    ) -> str:
        """Async version — use this inside any async context (agent loop, controller)."""
        if trace_id:
            logger.info("Client chat_async starting", extra={"trace_id": trace_id})

        system = await self._build_system(
            query=query_for_memory,
            profile=profile_summary,
            workspace_path=workspace_path,
        )
        prompt = self._messages_to_prompt(messages)
        return await self.complete(prompt, system=system, task_type=task_type) or ""

    def chat(
        self,
        messages: list[dict[str, Any]],
        query_for_memory: str = "",
        profile_summary: str = "",
        workspace_path: str = "",
        trace_id: str | None = None,
        task_type: str = "chat",
    ) -> str:
        """Sync bridge — ONLY call from truly synchronous, non-async contexts."""
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                asyncio.run,
                self.chat_async(messages, query_for_memory, profile_summary, workspace_path, trace_id=trace_id, task_type=task_type)
            )
            return future.result()


    # ── Health check ─────────────────────────────────────────────────────


    # ── Internal helpers ─────────────────────────────────────────────────

    async def _build_system(self, query: str = "", profile: str = "", workspace_path: str = "") -> str:
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
                context = await self.memory.build_context_block(query, n_results=3)
            except TypeError:
                context = await self.memory.build_context_block(query)
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
