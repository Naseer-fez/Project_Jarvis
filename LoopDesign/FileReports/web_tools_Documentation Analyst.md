# Analysis Report for web_tools.py

## Dependencies
- __future__.annotations
- asyncio
- configparser
- json
- logging
- os
- re
- dataclasses.dataclass
- pathlib.Path
- typing.Protocol
- typing.Any
- typing.cast

## Schemas
- SupportsQuickLLM
- SearchSettings
- SearchSettings attribute: enabled
- SearchSettings attribute: provider
- SearchSettings attribute: default_max_results
- SearchSettings attribute: summarize_results
- SearchSettings attribute: auto_extract_query
- SearchSettings attribute: provider_timeout_s
- SearchSettings attribute: scrape_timeout_s
- SearchSettings attribute: quick_task_timeout_s
- SearchSettings attribute: max_scrape_chars
- SearchSettings attribute: ddgs_region
- SearchSettings attribute: ddgs_safesearch
- SearchSettings attribute: tavily_api_key
- SearchResult
- SearchResult attribute: title
- SearchResult attribute: url
- SearchResult attribute: snippet
- SearchResult attribute: provider
- WebToolService

## API Contracts
- SearchSettings.from_sources(cls, config)
- WebToolService.__init__(self, settings, llm)
- WebToolService._provider_chain(self)
- WebToolService._search_with_ddgs(self, query, max_results)
- WebToolService._search_with_tavily(self, query, max_results)
- WebToolService._scrape_page(self, url)
- WebToolService._format_search_output()
- configure_web_tools()
- _get_service()
- _load_default_config()
- _normalize_whitespace(text)
- _basic_query_cleanup(text)
- _needs_query_extraction(query)
- _clean_llm_line(text)
- _fallback_summary(results)

## Configuration Variables
- PROJECT_ROOT
- DEFAULT_CONFIG_PATH
- DEFAULT_HEADERS
- QUERY_EXTRACTION_SYSTEM
- SEARCH_SUMMARY_SYSTEM
- _SERVICE (typed)

## Assumptions & Notes
- Module Docstring: Configurable web search and web scraping tools for Jarvis.

