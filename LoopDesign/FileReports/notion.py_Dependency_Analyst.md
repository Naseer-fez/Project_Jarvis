# File Report: notion.py
## Role: Dependency Analyst

### 1. Library Requirements
- `aiohttp` (Third-party)
- `os`, `typing` (Standard Library)
- `integrations.base` (Local)

### 2. Service Dependencies
- Notion API v1 (`https://api.notion.com/v1`) using version header `"2022-06-28"`.

### 3. Hidden Execution Links
- Enforces strict block typing. Unrecognized block types default to `paragraph`.
- Only allows appending a text block. Rich text formatting or deeply nested structure isn't natively exposed via these tools.
- Page fetching limits blocks to top-level, max 20, keeping context compact.

### 4. Assumptions & API Contracts
- Validates the `parent_type` parameter (defaults to `page_id`).
- Content strings are strictly truncated: titles to 255 chars, text contents to 2000 chars, conforming to Notion API limits or preventing LLM token blowouts.
- Expects the `NOTION_API_KEY` to be a valid Integration Bearer Token with sufficient permissions assigned to the target pages/databases.

### 5. Configuration Variables
- `NOTION_API_KEY`
- Internal: `_NOTION_VERSION` ("2022-06-28")

### 6. Prompts Found
- None.
