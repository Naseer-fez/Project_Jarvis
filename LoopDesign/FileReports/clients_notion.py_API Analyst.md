# `notion.py` - API Analyst Report

## Overview
Notion integration using the official Notion API v1 and async `aiohttp`. Manages pages, databases, and blocks.

## Endpoints / Tools
1. `create_page`
   - Description: Create a new Notion page under a parent page or database.
   - Risk: confirm (write)
   - Arguments: `parent_id` (string, required), `title` (string, required), `content` (string), `parent_type` (string, "page_id" or "database_id", default "page_id").
2. `query_database`
   - Description: Query a Notion database and return matching pages.
   - Risk: low (read-only)
   - Arguments: `database_id` (string, required), `filter_property` (string), `filter_value` (string), `page_size` (integer, default 10).
3. `append_block`
   - Description: Append a text block to an existing Notion page.
   - Risk: confirm (write)
   - Arguments: `page_id` (string, required), `text` (string, required), `block_type` (string, default "paragraph").
4. `get_page`
   - Description: Retrieve metadata and top-level blocks from a Notion page.
   - Risk: low (read-only)
   - Arguments: `page_id` (string, required).

## External Contracts / Dependencies
- Requires `NOTION_API_KEY`.
- Target API base: `https://api.notion.com/v1`.
- API Version: `2022-06-28`.
- Depends on `aiohttp`.

## Assumptions
- Uses local strict schema validation (`_validate_block_type`, `_validate_parent_type`) to ensure valid enums before executing requests.
- Explicit truncation happens on text variables: titles truncate at 255 chars, block text contents truncate at 2000 chars.
- `get_page` fetches only the first 20 children blocks (`[:20]`) and strips out rich text content returning only type and ID.
- `query_database` constructs a simple `contains` rich text filter. Page size capped at 50.
