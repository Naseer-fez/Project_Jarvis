"""
Configurable web search and web scraping tools for Jarvis.
"""

from __future__ import annotations

import asyncio
import configparser
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

try:
    from ddgs import DDGS
except ImportError:  # pragma: no cover - optional dependency at runtime
    DDGS = None  # type: ignore[assignment]

try:
    import requests
except ImportError:  # pragma: no cover - optional dependency at runtime
    requests = None  # type: ignore[assignment]

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover - optional dependency at runtime
    BeautifulSoup = None  # type: ignore[assignment]

logger = logging.getLogger("Jarvis.WebTools")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "jarvis.ini"
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}
QUERY_EXTRACTION_SYSTEM = (
    "You convert conversational requests into concise web search queries. "
    "Return only the query text, no quotes, no bullets, no explanation."
)
SEARCH_SUMMARY_SYSTEM = (
    "You summarize web search results for a local AI assistant. "
    "Write 2-4 short sentences grounded only in the provided results. "
    "Do not invent facts. Mention uncertainty if results conflict."
)


class SupportsQuickLLM(Protocol):
    """Minimal protocol used by the web tools for fast internal tasks."""

    async def complete(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.0,
        task_type: str = "chat",
        keep_think: bool = False,
    ) -> str:
        """Return a plain-text completion."""


@dataclass(frozen=True)
class SearchSettings:
    """Runtime settings for the web tools."""

    enabled: bool = True
    provider: str = "auto"
    default_max_results: int = 5
    summarize_results: bool = True
    auto_extract_query: bool = True
    provider_timeout_s: float = 8.0
    scrape_timeout_s: float = 10.0
    quick_task_timeout_s: float = 4.0
    max_scrape_chars: int = 8000
    ddgs_region: str = "wt-wt"
    ddgs_safesearch: str = "moderate"
    tavily_api_key: str = ""

    @classmethod
    def from_sources(
        cls,
        config: configparser.ConfigParser | None = None,
    ) -> "SearchSettings":
        """Load settings from config and environment variables."""
        resolved_config = config if config is not None else _load_default_config()

        enabled = _get_bool(
            os.environ.get("WEB_SEARCH_ENABLED"),
            _config_get_bool(
                resolved_config,
                "web_search",
                "enabled",
                fallback=_config_get_bool(
                    resolved_config,
                    "execution",
                    "allow_web_search",
                    fallback=True,
                ),
            ),
        )
        provider = _get_str(
            os.environ.get("WEB_SEARCH_PROVIDER"),
            _config_get(resolved_config, "web_search", "provider", fallback="auto"),
        ).lower()
        default_max_results = _get_int(
            os.environ.get("WEB_SEARCH_DEFAULT_MAX_RESULTS"),
            _config_get_int(
                resolved_config,
                "web_search",
                "default_max_results",
                fallback=5,
            ),
        )
        summarize_results = _get_bool(
            os.environ.get("WEB_SEARCH_SUMMARIZE_RESULTS"),
            _config_get_bool(
                resolved_config,
                "web_search",
                "summarize_results",
                fallback=True,
            ),
        )
        auto_extract_query = _get_bool(
            os.environ.get("WEB_SEARCH_AUTO_EXTRACT_QUERY"),
            _config_get_bool(
                resolved_config,
                "web_search",
                "auto_extract_query",
                fallback=True,
            ),
        )
        provider_timeout_s = _get_float(
            os.environ.get("WEB_SEARCH_PROVIDER_TIMEOUT_S"),
            _config_get_float(
                resolved_config,
                "web_search",
                "provider_timeout_s",
                fallback=8.0,
            ),
        )
        scrape_timeout_s = _get_float(
            os.environ.get("WEB_SEARCH_SCRAPE_TIMEOUT_S"),
            _config_get_float(
                resolved_config,
                "web_search",
                "scrape_timeout_s",
                fallback=10.0,
            ),
        )
        quick_task_timeout_s = _get_float(
            os.environ.get("WEB_SEARCH_QUICK_TASK_TIMEOUT_S"),
            _config_get_float(
                resolved_config,
                "web_search",
                "quick_task_timeout_s",
                fallback=4.0,
            ),
        )
        max_scrape_chars = _get_int(
            os.environ.get("WEB_SEARCH_MAX_SCRAPE_CHARS"),
            _config_get_int(
                resolved_config,
                "web_search",
                "max_scrape_chars",
                fallback=8000,
            ),
        )
        ddgs_region = _get_str(
            os.environ.get("WEB_SEARCH_DDGS_REGION"),
            _config_get(resolved_config, "web_search", "ddgs_region", fallback="wt-wt"),
        )
        ddgs_safesearch = _get_str(
            os.environ.get("WEB_SEARCH_DDGS_SAFESEARCH"),
            _config_get(
                resolved_config,
                "web_search",
                "ddgs_safesearch",
                fallback="moderate",
            ),
        )
        tavily_api_key = _get_str(
            os.environ.get("TAVILY_API_KEY"),
            _config_get(resolved_config, "web_search", "tavily_api_key", fallback=""),
        )

        return cls(
            enabled=enabled,
            provider=provider or "auto",
            default_max_results=max(1, min(default_max_results, 10)),
            summarize_results=summarize_results,
            auto_extract_query=auto_extract_query,
            provider_timeout_s=max(1.0, provider_timeout_s),
            scrape_timeout_s=max(1.0, scrape_timeout_s),
            quick_task_timeout_s=max(1.0, quick_task_timeout_s),
            max_scrape_chars=max(500, max_scrape_chars),
            ddgs_region=ddgs_region or "wt-wt",
            ddgs_safesearch=ddgs_safesearch or "moderate",
            tavily_api_key=tavily_api_key.strip(),
        )


@dataclass(frozen=True)
class SearchResult:
    """Normalized search result returned by a search provider."""

    title: str
    url: str
    snippet: str
    provider: str


class WebToolService:
    """Configurable service backing the public web tool functions."""

    def __init__(
        self,
        settings: SearchSettings,
        llm: SupportsQuickLLM | None = None,
    ) -> None:
        self.settings = settings
        self.llm = llm

    async def web_search(self, query: str, max_results: int = 5) -> str:
        """Perform a web search and return a source-grounded summary."""
        if not self.settings.enabled:
            return "Web search is disabled by configuration."

        raw_query = _normalize_whitespace(query)
        if not raw_query:
            return "Search failed: query is empty."

        result_limit = max(1, min(int(max_results or self.settings.default_max_results), 10))
        effective_query = await self._extract_search_query(raw_query)

        try:
            results = await self._search(effective_query, result_limit)
        except Exception as exc:  # noqa: BLE001
            logger.error("Web search failed for %r: %s", effective_query, exc)
            return f"Search failed: {exc}"

        if not results:
            return f"No results found for query: {effective_query}"

        summary = await self._summarize_results(raw_query, effective_query, results)
        return self._format_search_output(
            original_query=raw_query,
            effective_query=effective_query,
            results=results,
            summary=summary,
        )

    async def web_scrape(self, url: str, max_chars: int = 8000) -> str:
        """Fetch and extract readable text from a web page."""
        if requests is None or BeautifulSoup is None:
            return "Error: requests and beautifulsoup4 must be installed for web scraping."

        target_url = str(url or "").strip()
        if not target_url:
            return "Scraping failed: url is empty."

        max_chars = max(500, int(max_chars or self.settings.max_scrape_chars))

        try:
            text = await asyncio.to_thread(self._scrape_page, target_url)
        except Exception as exc:  # noqa: BLE001
            logger.error("Web scrape failed for %s: %s", target_url, exc)
            return f"Scraping failed: {exc}"

        if not text:
            return "Failed to extract readable text from the page."
        if len(text) > max_chars:
            return text[:max_chars] + f"\n\n...[Truncated, total {len(text)} chars]..."
        return text

    async def _search(self, query: str, max_results: int) -> list[SearchResult]:
        """Search with the configured provider chain."""
        errors: list[str] = []
        for provider in self._provider_chain():
            try:
                if provider == "tavily":
                    return await asyncio.to_thread(
                        self._search_with_tavily,
                        query,
                        max_results,
                    )
                if provider == "ddgs":
                    return await asyncio.to_thread(
                        self._search_with_ddgs,
                        query,
                        max_results,
                    )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{provider}: {exc}")
                logger.warning("Search provider %s failed: %s", provider, exc)

        if errors:
            raise RuntimeError("; ".join(errors))
        return []

    def _provider_chain(self) -> list[str]:
        provider = (self.settings.provider or "auto").strip().lower()
        if provider == "tavily":
            return ["tavily", "ddgs"]
        if provider == "ddgs":
            return ["ddgs"]

        providers: list[str] = []
        if self.settings.tavily_api_key:
            providers.append("tavily")
        providers.append("ddgs")
        return providers

    def _search_with_ddgs(self, query: str, max_results: int) -> list[SearchResult]:
        if DDGS is None:
            raise RuntimeError("ddgs package is not installed.")

        with DDGS() as ddgs:
            try:
                raw_results = list(
                    ddgs.text(
                        query,
                        region=self.settings.ddgs_region,
                        safesearch=self.settings.ddgs_safesearch,
                        max_results=max_results,
                    )
                )
            except TypeError:
                raw_results = list(ddgs.text(query, max_results=max_results))

        return [
            SearchResult(
                title=_normalize_whitespace(str(item.get("title", "No Title"))),
                url=_normalize_whitespace(str(item.get("href", "No URL"))),
                snippet=_normalize_whitespace(str(item.get("body", ""))),
                provider="ddgs",
            )
            for item in raw_results
            if item
        ]

    def _search_with_tavily(self, query: str, max_results: int) -> list[SearchResult]:
        if requests is None:
            raise RuntimeError("requests package is not installed.")
        if not self.settings.tavily_api_key:
            raise RuntimeError("Tavily API key is not configured.")

        payload = {
            "api_key": self.settings.tavily_api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": max_results,
            "include_answer": False,
            "include_raw_content": False,
        }
        response = requests.post(
            "https://api.tavily.com/search",
            json=payload,
            timeout=self.settings.provider_timeout_s,
            headers=DEFAULT_HEADERS,
        )
        response.raise_for_status()
        data = response.json()
        raw_results = data.get("results", [])

        return [
            SearchResult(
                title=_normalize_whitespace(str(item.get("title", "No Title"))),
                url=_normalize_whitespace(str(item.get("url", "No URL"))),
                snippet=_normalize_whitespace(str(item.get("content", ""))),
                provider="tavily",
            )
            for item in raw_results
            if item
        ]

    def _scrape_page(self, url: str) -> str:
        if requests is None or BeautifulSoup is None:
            raise RuntimeError("requests and beautifulsoup4 must be installed.")

        response = requests.get(
            url,
            headers=DEFAULT_HEADERS,
            timeout=self.settings.scrape_timeout_s,
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")
        for element in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            element.decompose()

        text = soup.get_text(separator="\n")
        lines = (line.strip() for line in text.splitlines())
        chunks = (piece.strip() for line in lines for piece in line.split("  "))
        return "\n".join(chunk for chunk in chunks if chunk)

    async def _extract_search_query(self, raw_query: str) -> str:
        cleaned = _basic_query_cleanup(raw_query)
        if not self.settings.auto_extract_query or self.llm is None:
            return cleaned
        if not _needs_query_extraction(cleaned):
            return cleaned

        prompt = (
            "User request:\n"
            f"{raw_query}\n\n"
            "Return the single best web search query for this request."
        )
        try:
            response = await asyncio.wait_for(
                self.llm.complete(
                    prompt,
                    system=QUERY_EXTRACTION_SYSTEM,
                    temperature=0.0,
                    task_type="tool_parameter_extraction",
                ),
                timeout=self.settings.quick_task_timeout_s,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Search query extraction failed: %s", exc)
            return cleaned

        extracted = _clean_llm_line(response)
        return extracted or cleaned

    async def _summarize_results(
        self,
        original_query: str,
        effective_query: str,
        results: list[SearchResult],
    ) -> str:
        if not self.settings.summarize_results:
            return ""
        if self.llm is None:
            return _fallback_summary(results)

        serialized_results = "\n".join(
            f"{idx}. {item.title}\nURL: {item.url}\nSnippet: {item.snippet}"
            for idx, item in enumerate(results, start=1)
        )
        prompt = (
            f"Original request: {original_query}\n"
            f"Search query used: {effective_query}\n\n"
            f"Results:\n{serialized_results}"
        )
        try:
            response = await asyncio.wait_for(
                self.llm.complete(
                    prompt,
                    system=SEARCH_SUMMARY_SYSTEM,
                    temperature=0.1,
                    task_type="web_search_summary",
                ),
                timeout=self.settings.quick_task_timeout_s,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Search result summarization failed: %s", exc)
            return _fallback_summary(results)

        return _normalize_whitespace(response) or _fallback_summary(results)

    @staticmethod
    def _format_search_output(
        *,
        original_query: str,
        effective_query: str,
        results: list[SearchResult],
        summary: str,
    ) -> str:
        lines = [f"Search query used: {effective_query}"]
        if effective_query != original_query:
            lines.append(f"Original request: {original_query}")
        if summary:
            lines.append(f"Summary: {summary}")
        lines.append("Sources:")

        for idx, result in enumerate(results, start=1):
            lines.append(f"{idx}. {result.title}")
            lines.append(f"URL: {result.url}")
            if result.snippet:
                lines.append(f"Snippet: {result.snippet}")

        return "\n".join(lines)


_SERVICE: WebToolService | None = None


def configure_web_tools(
    *,
    config: configparser.ConfigParser | None = None,
    llm: SupportsQuickLLM | None = None,
) -> WebToolService:
    """Configure the process-wide web tool service used by the tool router."""
    global _SERVICE
    _SERVICE = WebToolService(SearchSettings.from_sources(config), llm=llm)
    return _SERVICE


def _get_service() -> WebToolService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = WebToolService(SearchSettings.from_sources())
    return _SERVICE


async def web_search(query: str, max_results: int = 5) -> str:
    """Perform a web search using the configured provider chain."""
    return await _get_service().web_search(query, max_results=max_results)


async def web_scrape(url: str, max_chars: int = 8000) -> str:
    """Fetch and extract readable text from a webpage."""
    return await _get_service().web_scrape(url, max_chars=max_chars)


def _load_default_config() -> configparser.ConfigParser | None:
    if not DEFAULT_CONFIG_PATH.exists():
        return None

    parser = configparser.ConfigParser()
    try:
        with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as handle:
            parser.read_file(handle)
    except OSError:
        return None
    return parser


def _config_get(
    config: configparser.ConfigParser | None,
    section: str,
    option: str,
    *,
    fallback: str,
) -> str:
    if config is None or not config.has_section(section):
        return fallback
    return config.get(section, option, fallback=fallback)


def _config_get_bool(
    config: configparser.ConfigParser | None,
    section: str,
    option: str,
    *,
    fallback: bool,
) -> bool:
    if config is None or not config.has_section(section):
        return fallback
    try:
        return config.getboolean(section, option, fallback=fallback)
    except ValueError:
        return fallback


def _config_get_int(
    config: configparser.ConfigParser | None,
    section: str,
    option: str,
    *,
    fallback: int,
) -> int:
    if config is None or not config.has_section(section):
        return fallback
    try:
        return config.getint(section, option, fallback=fallback)
    except ValueError:
        return fallback


def _config_get_float(
    config: configparser.ConfigParser | None,
    section: str,
    option: str,
    *,
    fallback: float,
) -> float:
    if config is None or not config.has_section(section):
        return fallback
    try:
        return config.getfloat(section, option, fallback=fallback)
    except ValueError:
        return fallback


def _get_bool(raw: str | None, fallback: bool) -> bool:
    if raw is None:
        return fallback
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(raw: str | None, fallback: int) -> int:
    if raw is None or not raw.strip():
        return fallback
    try:
        return int(raw)
    except ValueError:
        return fallback


def _get_float(raw: str | None, fallback: float) -> float:
    if raw is None or not raw.strip():
        return fallback
    try:
        return float(raw)
    except ValueError:
        return fallback


def _get_str(raw: str | None, fallback: str) -> str:
    return raw.strip() if raw is not None and raw.strip() else fallback


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def _basic_query_cleanup(text: str) -> str:
    cleaned = _normalize_whitespace(text)
    cleaned = re.sub(
        r"^(please\s+)?(search|look up|find|check|google|browse|search for)\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+(for me|online|on the web)$", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip(" .?!")


def _needs_query_extraction(query: str) -> bool:
    if len(query.split()) <= 6 and not re.search(r"[,:;?!]", query):
        return False
    return any(
        token in query.lower()
        for token in (
            "please",
            "latest",
            "search",
            "look up",
            "find",
            "check",
            "what is",
            "who is",
            "summarize",
        )
    )


def _clean_llm_line(text: str) -> str:
    cleaned = _normalize_whitespace(text)
    if not cleaned:
        return ""

    cleaned = re.sub(r"^['\"`]+|['\"`]+$", "", cleaned)

    if cleaned.startswith("{") and cleaned.endswith("}"):
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                for key in ("query", "search_query", "value"):
                    value = parsed.get(key)
                    if isinstance(value, str) and value.strip():
                        cleaned = value.strip()
                        break
        except json.JSONDecodeError:
            pass

    first_line = cleaned.splitlines()[0].strip()
    first_line = re.sub(r"^[\-\d\.\)\s]+", "", first_line)
    return first_line[:160].strip(" .")


def _fallback_summary(results: list[SearchResult]) -> str:
    top_titles = [result.title for result in results[:3] if result.title]
    if not top_titles:
        return ""
    return "Top matches: " + "; ".join(top_titles)


__all__ = [
    "SearchResult",
    "SearchSettings",
    "WebToolService",
    "configure_web_tools",
    "web_scrape",
    "web_search",
]
