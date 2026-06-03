"""
Web search fast-path controller logic for Jarvis.
Handles explicit web searches directly, bypassing the full planner,
and synthesizes natural language responses.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from core.controller.request_rules import is_explicit_web_search, should_force_web_search

logger = logging.getLogger(__name__)


async def handle_web_search(
    user_input: str,
    trace_id: str,
    memory: Any,
    llm: Any,
    model_router: Any,
    profile: Any,
) -> str:
    """Perform a live web search, synthesize a natural language response, and fall back if needed."""
    try:
        from core.tools.web_tools import web_search as _web_search
        from core.tools.web_tools import _basic_query_cleanup

        query = _basic_query_cleanup(user_input)
        if not query:
            query = user_input.strip()

        logger.info("Executing explicit web search: %r", query, extra={"trace_id": trace_id})
        raw_results = await _web_search(query, max_results=5)
    except Exception as exc:
        logger.warning("Web search tool failed: %s", exc, extra={"trace_id": trace_id})
        return await _dispatch_llm_fallback(user_input, trace_id, memory, llm, model_router, profile)

    if not raw_results or raw_results.startswith("Web search is disabled"):
        logger.info("Web search disabled or empty, falling back to LLM", extra={"trace_id": trace_id})
        return await _dispatch_llm_fallback(user_input, trace_id, memory, llm, model_router, profile)

    if raw_results.startswith("Search failed"):
        logger.warning("Web search returned search failure: %s", raw_results, extra={"trace_id": trace_id})
        return raw_results

    # Synthesis Prompt
    synthesis_prompt = (
        f"The user asked: {user_input}\n\n"
        f"Here are the live web search results:\n{raw_results}\n\n"
        "Please give a concise, helpful reply based only on these results. "
        "Cite URLs where relevant. Do not invent information not present in the results."
    )

    if memory:
        try:
            await memory.build_context_block(user_input)
        except Exception as exc:
            logger.warning("Context building failed: %s", exc, extra={"trace_id": trace_id})

    selected_model = model_router.get_best_available("web_search_summary") if model_router else None
    if selected_model and llm:
        llm.model = selected_model

    messages = [{"role": "user", "content": synthesis_prompt}]
    profile_summary = profile.get_communication_style() if profile else ""

    try:
        if llm:
            response = await llm.chat_async(
                messages,
                query_for_memory=user_input,
                profile_summary=profile_summary,
                trace_id=trace_id,
                task_type="web_search_summary",
            )
            if response and response != "LLM Offline.":
                return response
    except Exception as exc:
        logger.warning("Web search synthesis LLM call failed: %s", exc, extra={"trace_id": trace_id})

    return _format_raw_fallback(raw_results)


async def _dispatch_llm_fallback(
    user_input: str,
    trace_id: str,
    memory: Any,
    llm: Any,
    model_router: Any,
    profile: Any,
) -> str:
    """Clean fallback to raw LLM completion when search tool fails or is disabled."""
    if memory:
        try:
            await memory.build_context_block(user_input)
        except Exception as exc:
            logger.warning("Context building failed during fallback: %s", exc, extra={"trace_id": trace_id})

    selected_model = model_router.get_best_available("chat") if model_router else None
    if selected_model and llm:
        llm.model = selected_model

    messages = [{"role": "user", "content": user_input}]
    profile_summary = profile.get_communication_style() if profile else ""

    try:
        if llm:
            response = await llm.chat_async(
                messages,
                query_for_memory=user_input,
                profile_summary=profile_summary,
                trace_id=trace_id,
            )
            if response and response != "LLM Offline.":
                return response
    except Exception as exc:
        logger.warning("Fallback LLM dispatch failed: %s", exc, extra={"trace_id": trace_id})

    # Recovery preference recall from memory database
    if memory:
        try:
            from core.controller.request_rules import is_preference_relevant
            prefs = await memory.recall_preferences(user_input, top_k=5)
            for pref in prefs:
                val = pref.get("value")
                key = pref.get("key")
                if val and key and is_preference_relevant(key, user_input):
                    return f"Offline fallback from memory: {val}"
        except Exception as exc:
            logger.warning("Memory preference recall failed: %s", exc, extra={"trace_id": trace_id})

    return "I don't know while offline."


def _format_raw_fallback(raw_results: str) -> str:
    """Parse and format raw search results nicely when LLM synthesis is not available."""
    lines = raw_results.splitlines()
    formatted_lines = []
    current_title = None
    current_num = None

    for line in lines:
        line_stripped = line.strip()
        match = re.match(r"^(\d+)\.\s+(.*)$", line_stripped)
        if match:
            if current_title is not None:
                formatted_lines.append(f"{current_num}. {current_title}")
            current_num = match.group(1)
            current_title = match.group(2)
            continue

        if line_stripped.startswith("URL:") and current_title is not None:
            url = line_stripped[4:].strip()
            formatted_lines.append(f"{current_num}. {current_title} ({url})")
            current_title = None
            current_num = None
            continue

        if current_title is not None:
            formatted_lines.append(f"{current_num}. {current_title}")
            current_title = None
            current_num = None

        formatted_lines.append(line)

    if current_title is not None:
        formatted_lines.append(f"{current_num}. {current_title}")

    return "\n".join(formatted_lines)


__all__ = [
    "handle_web_search",
    "is_explicit_web_search",
    "should_force_web_search",
]
