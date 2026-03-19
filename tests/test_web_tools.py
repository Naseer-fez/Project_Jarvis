from __future__ import annotations

from configparser import ConfigParser
from unittest.mock import AsyncMock

import pytest

from core.tools.web_tools import SearchResult, SearchSettings, WebToolService


class FakeQuickLLM:
    def __init__(self) -> None:
        self.task_types: list[str] = []

    async def complete(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.0,
        task_type: str = "chat",
        keep_think: bool = False,
    ) -> str:
        self.task_types.append(task_type)
        if task_type == "tool_parameter_extraction":
            return "Python 3.13 release notes"
        if task_type == "web_search_summary":
            return "Top results point to the official release notes and related Python resources."
        return ""


@pytest.mark.asyncio
async def test_web_search_refines_query_and_summarizes_results() -> None:
    cfg = ConfigParser()
    cfg["web_search"] = {
        "enabled": "true",
        "provider": "ddgs",
        "default_max_results": "5",
        "summarize_results": "true",
        "auto_extract_query": "true",
        "quick_task_timeout_s": "5",
    }
    llm = FakeQuickLLM()
    service = WebToolService(SearchSettings.from_sources(cfg), llm=llm)
    service._search = AsyncMock(  # type: ignore[method-assign]
        return_value=[
            SearchResult(
                title="Python 3.13 Release Notes",
                url="https://docs.python.org/3/whatsnew/3.13.html",
                snippet="Highlights from the Python 3.13 release.",
                provider="ddgs",
            ),
            SearchResult(
                title="Python News",
                url="https://www.python.org/",
                snippet="Official Python website updates.",
                provider="ddgs",
            ),
        ]
    )

    result = await service.web_search(
        "Please search the web for the latest Python 3.13 release notes for me",
        max_results=2,
    )

    assert "Search query used: Python 3.13 release notes" in result
    assert "Original request: Please search the web for the latest Python 3.13 release notes for me" in result
    assert "Summary: Top results point to the official release notes and related Python resources." in result
    assert "1. Python 3.13 Release Notes" in result
    assert "URL: https://docs.python.org/3/whatsnew/3.13.html" in result
    assert llm.task_types == ["tool_parameter_extraction", "web_search_summary"]


@pytest.mark.asyncio
async def test_web_search_uses_fallback_summary_without_llm() -> None:
    service = WebToolService(
        SearchSettings(
            enabled=True,
            provider="ddgs",
            summarize_results=True,
            auto_extract_query=False,
        )
    )
    service._search = AsyncMock(  # type: ignore[method-assign]
        return_value=[
            SearchResult(
                title="Result One",
                url="https://example.com/1",
                snippet="Example snippet one.",
                provider="ddgs",
            ),
            SearchResult(
                title="Result Two",
                url="https://example.com/2",
                snippet="Example snippet two.",
                provider="ddgs",
            ),
        ]
    )

    result = await service.web_search("example query", max_results=2)

    assert "Search query used: example query" in result
    assert "Summary: Top matches: Result One; Result Two" in result
