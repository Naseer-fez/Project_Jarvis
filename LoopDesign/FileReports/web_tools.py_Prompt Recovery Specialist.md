# File Report: web_tools.py
**Role**: Prompt Recovery Specialist

## Dependencies
- pathlib
- requests
- duckduckgo_search
- re
- typing
- logging
- json
- asyncio
- dataclasses
- bs4
- __future__
- configparser
- os

## Configuration Variables & Constants
- `QUERY_EXTRACTION_SYSTEM`: (Too long, 135 chars. Extracted to Prompts if applicable)
- `SEARCH_SUMMARY_SYSTEM`: (Too long, 185 chars. Extracted to Prompts if applicable)
- `prompt`: `f-string: User request:
{...}

Return the single best web search query for this request.`
- `prompt`: `f-string: Original request: {...}
Search query used: {...}

Results:
{...}`

## Schemas & API Contracts
### Class `SupportsQuickLLM`
**Assumptions/Doc**: Minimal protocol used by the web tools for fast internal tasks.
**Methods**: complete

### Class `SearchSettings`
**Assumptions/Doc**: Runtime settings for the web tools.
**Methods**: from_sources

### Class `SearchResult`
**Assumptions/Doc**: Normalized search result returned by a search provider.
**Methods**: 

### Class `WebToolService`
**Assumptions/Doc**: Configurable service backing the public web tool functions.
**Methods**: __init__, web_search, web_scrape, _search, _provider_chain, _search_with_ddgs, _search_with_tavily, _scrape_page, _extract_search_query, _summarize_results, _format_search_output

### Function `configure_web_tools`
**Args**: 
**Assumptions/Doc**: Configure the process-wide web tool service used by the tool router.

### Function `_get_service`
**Args**: 

### Function `web_search`
**Args**: query, max_results
**Assumptions/Doc**: Perform a web search using the configured provider chain.

### Function `web_scrape`
**Args**: url, max_chars
**Assumptions/Doc**: Fetch and extract readable text from a webpage.

### Function `_load_default_config`
**Args**: 

### Function `_normalize_whitespace`
**Args**: text

### Function `_basic_query_cleanup`
**Args**: text

### Function `_needs_query_extraction`
**Args**: query

### Function `_clean_llm_line`
**Args**: text

### Function `_fallback_summary`
**Args**: results

### Function `complete`
**Args**: self, prompt, system, temperature, task_type, keep_think
**Assumptions/Doc**: Return a plain-text completion.

### Function `from_sources`
**Args**: cls, config
**Assumptions/Doc**: Load settings from config and environment variables.

### Function `__init__`
**Args**: self, settings, llm

### Function `web_search`
**Args**: self, query, max_results
**Assumptions/Doc**: Perform a web search and return a source-grounded summary.

### Function `web_scrape`
**Args**: self, url, max_chars
**Assumptions/Doc**: Fetch and extract readable text from a web page.

### Function `_search`
**Args**: self, query, max_results
**Assumptions/Doc**: Search with the configured provider chain.

### Function `_provider_chain`
**Args**: self

### Function `_search_with_ddgs`
**Args**: self, query, max_results

### Function `_search_with_tavily`
**Args**: self, query, max_results

### Function `_scrape_page`
**Args**: self, url

### Function `_extract_search_query`
**Args**: self, raw_query

### Function `_summarize_results`
**Args**: self, original_query, effective_query, results

### Function `_format_search_output`
**Args**: 

### Function `_resolve`
**Args**: env_key, opt, parser_type, fallback, section

## Prompts and LLM Directives
- Extracted `QUERY_EXTRACTION_SYSTEM` to Prompts directory.
- Extracted `SEARCH_SUMMARY_SYSTEM` to Prompts directory.
- Extracted `prompt` to Prompts directory.
- Extracted `prompt` to Prompts directory.
