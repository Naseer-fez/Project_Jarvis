# API Analyst Report: tools\web_tools.py

## Dependencies
- `from __future__ import annotations`
- `import asyncio`
- `import configparser`
- `import json`
- `import logging`
- `import os`
- `import re`
- `from dataclasses import dataclass`
- `from pathlib import Path`
- `from typing import Protocol`
- `from typing import Any`
- `from typing import cast`

## Configuration Variables
- `PROJECT_ROOT` = `Path(__file__).resolve().parents[2]`
- `DEFAULT_CONFIG_PATH` = `PROJECT_ROOT / 'config' / 'jarvis.ini'`
- `DEFAULT_HEADERS` = `{'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'}`
- `QUERY_EXTRACTION_SYSTEM` = `'You convert conversational requests into concise web search queries. Return only the query text, no quotes, no bullets, no explanation.'`
- `SEARCH_SUMMARY_SYSTEM` = `'You summarize web search results for a local AI assistant. Write 2-4 short sentences grounded only in the provided results. Do not invent facts. Mention uncertainty if results conflict.'`
- `_SERVICE` = `WebToolService(SearchSettings.from_sources(config), llm=llm)`
- `DDGS` = `None`
- `_SERVICE` = `WebToolService(SearchSettings.from_sources())`

## Schemas & API Contracts (Classes)

### Class `SupportsQuickLLM(Protocol)`
> Minimal protocol used by the web tools for fast internal tasks.

**Methods:**
- `async def complete(self, prompt: str, system: str='', temperature: float=0.0, task_type: str='chat', keep_think: bool=False) -> str`
  - *Return a plain-text completion.*


### Class `SearchSettings`
> Runtime settings for the web tools.

**Fields/Schema:**
  - `enabled: bool`
  - `provider: str`
  - `default_max_results: int`
  - `summarize_results: bool`
  - `auto_extract_query: bool`
  - `provider_timeout_s: float`
  - `scrape_timeout_s: float`
  - `quick_task_timeout_s: float`
  - `max_scrape_chars: int`
  - `ddgs_region: str`
  - `ddgs_safesearch: str`
  - `tavily_api_key: str`

**Methods:**
- @classmethod
- `def from_sources(cls, config: configparser.ConfigParser | None=None) -> 'SearchSettings'`
  - *Load settings from config and environment variables.*


### Class `SearchResult`
> Normalized search result returned by a search provider.

**Fields/Schema:**
  - `title: str`
  - `url: str`
  - `snippet: str`
  - `provider: str`



### Class `WebToolService`
> Configurable service backing the public web tool functions.

**Methods:**
- `def __init__(self, settings: SearchSettings, llm: SupportsQuickLLM | None=None) -> None`
- `async def web_search(self, query: str, max_results: int=5) -> str`
  - *Perform a web search and return a source-grounded summary.*
- `async def web_scrape(self, url: str, max_chars: int=8000) -> str`
  - *Fetch and extract readable text from a web page.*
- `async def _search(self, query: str, max_results: int) -> list[SearchResult]`
  - *Search with the configured provider chain.*
- `def _provider_chain(self) -> list[str]`
- `def _search_with_ddgs(self, query: str, max_results: int) -> list[SearchResult]`
- `def _search_with_tavily(self, query: str, max_results: int) -> list[SearchResult]`
- `def _scrape_page(self, url: str) -> str`
- `async def _extract_search_query(self, raw_query: str) -> str`
- `async def _summarize_results(self, original_query: str, effective_query: str, results: list[SearchResult]) -> str`
- @staticmethod
- `def _format_search_output(*, original_query: str, effective_query: str, results: list[SearchResult], summary: str) -> str`


## Functions & Endpoints

### `configure_web_tools`
`def configure_web_tools(*, config: configparser.ConfigParser | None=None, llm: SupportsQuickLLM | None=None) -> WebToolService`
> Configure the process-wide web tool service used by the tool router.

### `_get_service`
`def _get_service() -> WebToolService`
### `web_search`
`async def web_search(query: str, max_results: int=5) -> str`
> Perform a web search using the configured provider chain.

### `web_scrape`
`async def web_scrape(url: str, max_chars: int=8000) -> str`
> Fetch and extract readable text from a webpage.

### `_load_default_config`
`def _load_default_config() -> configparser.ConfigParser | None`
### `_normalize_whitespace`
`def _normalize_whitespace(text: str) -> str`
### `_basic_query_cleanup`
`def _basic_query_cleanup(text: str) -> str`
### `_needs_query_extraction`
`def _needs_query_extraction(query: str) -> bool`
### `_clean_llm_line`
`def _clean_llm_line(text: str) -> str`
### `_fallback_summary`
`def _fallback_summary(results: list[SearchResult]) -> str`