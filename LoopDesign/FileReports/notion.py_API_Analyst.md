# clients/notion.py API Analyst Report

## Overview
Notion API v1 integration using `aiohttp`. Enables reading and writing pages/databases.

## API Contracts & Methods
- `NotionIntegration(BaseIntegration)`
  - Hardcodes API version to `2022-06-28`.

## Tools Exposed
- `create_page(parent_id, title, content, parent_type="page_id")` [Risk: `confirm`]
  - Title is truncated to 255 chars. Content is truncated to 2000 chars.
- `query_database(database_id, filter_property, filter_value, page_size=10)` [Risk: `low`]
- `append_block(page_id, text, block_type="paragraph")` [Risk: `confirm`]
  - Text is truncated to 2000 chars.
  - Validates block type strictly against allowed set.
- `get_page(page_id)` [Risk: `low`]
  - Retrieves page metadata and up to 20 children blocks.

## Configuration Variables
- `NOTION_API_KEY`

## Assumptions & Constants
- `_NOTION_BASE = "https://api.notion.com/v1"`
- Strict limits enforced on payload string lengths before sending to Notion to prevent validation errors.

## Dependencies
- `aiohttp`

## Prompts
- None.
