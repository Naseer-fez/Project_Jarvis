"""Handles routing the input to the appropriate LLM task type and generating the response."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class LLMDispatcher:
    """Routes classified requests to the appropriate LLM model via the adaptive router."""

    def __init__(self, llm, model_router, memory, profile):
        self.llm = llm
        self.model_router = model_router
        self.memory = memory
        self.profile = profile

    async def dispatch(self, text: str, classification: dict[str, Any], session_id: str, trace_id: str) -> str:
        complexity = classification.get("complexity", 0.5)

        profile_summary = ""

        # Selective context injection
        if complexity > 0.2:
            await self.memory.build_context_block(text)
            profile_summary = (
                self.profile.get_communication_style() if self.profile else ""
            )

        # Map class route to task_type
        task_type = classification.get("route", "chat")
        if task_type == "direct":
            task_type = "reflex"
        elif task_type == "premium":
            task_type = "deep_reasoning"
        elif task_type == "planner":
            task_type = "planning"
        elif task_type == "mid-tier":
            task_type = "chat"

        selected_model = self.model_router.get_best_available(task_type)
        self.llm.model = selected_model

        messages = [{"role": "user", "content": text}]

        logger.info("Dispatching to LLM: %r (task_type=%s)", selected_model, task_type, extra={"trace_id": trace_id})

        # Pass full classification through to LLMClientV2 for adaptive routing
        response = await self.llm.chat_async(
            messages,
            query_for_memory=text if complexity > 0.2 else "",
            profile_summary=profile_summary,
            trace_id=trace_id,
        )

        if not response or response == "LLM Offline.":
            from core.controller.request_rules import is_preference_relevant
            prefs = await self.memory.recall_preferences(text, top_k=5)
            for pref in prefs:
                val = pref.get("value")
                key = pref.get("key")
                if val and key and is_preference_relevant(key, text):
                    return f"Offline fallback from memory: {val}"
            return "I don't know while offline."

        await self.memory.store_conversation(text, response, session_id)
        return str(response)
