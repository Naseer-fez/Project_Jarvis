# File Report: notion.py
**Path**: `d:\AI\Jarvis\integrations\clients\notion.py`
**Role**: Data Model Analyst

## Analysis Summary
This file has been analyzed for schemas, DTOs, state objects, config variables, and dependencies.

## Dependencies
- __future__.annotations
- os
- typing.Any
- integrations.base.BaseIntegration
- aiohttp
- aiohttp
- aiohttp
- aiohttp
- aiohttp

## Classes and State Objects
### `NotionIntegration`
**Variables**: name, description
**Methods**: is_available, _headers, get_tools, execute, _validate_block_type, _validate_parent_type, _create_page, _query_database, _append_block, _get_page

## Tool Schemas / DTOs
```python
    def get_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "create_page",
                "description": "Create a new Notion page under a parent page or database",
                "risk": "confirm",
                "args": {
                    "parent_id": {
                        "type": "string",
                        "description": "Parent page or database ID",
                    },
                    "title": {
                        "type": "string",
                        "description": "Page title",
                    },
                    "content": {
                        "type": "string",
                        "description": "Optional plain-text content for page body",
                        "default": "",
                    },
                    "parent_type": {
                        "type": "string",
                        "description": "'page_id' or 'database_id'",
                        "default": "page_id",
                    },
                },
                "required_args": ["parent_id", "title"],
            },
            {
                "name": "query_database",
                "description": "Query a Notion database and return matching pages",
                "risk": "low",
                "args": {
                    "database_id": {
                        "type": "string",
                        "description": "Notion database ID",
                    },
                    "filter_property": {
                        "type": "string",
                        "description": "Optional property name to filter on",
                        "default": "",
                    },
                    "filter_value": {
                        "type": "string",
                        "description": "Optional filter value (text contains)",
                        "default": "",
                    },
                    "page_size": {"type": "integer", "default": 10},
                },
                "required_args": ["database_id"],
            },
            {
                "name": "append_block",
                "description": "Append a text block to an existing Notion page",
                "risk": "confirm",
                "args": {
                    "page_id": {
                        "type": "string",
                        "description": "Notion page ID to append to",
                    },
                    "text": {
                        "type": "string",
                        "description": "Text content to append",
                    },
                    "block_type": {
                        "type": "string",
                        "description": "Block type: paragraph, bulleted_list_item, heading_2",
                        "default": "paragraph",
                    },
                },
                "required_args": ["page_id", "text"],
            },
            {
                "name": "get_page",
                "description": "Retrieve metadata and top-level blocks from a Notion page",
                "risk": "low",
                "args": {
                    "page_id": {
                        "type": "string",
                        "description": "Notion page ID",
                    },
                },
                "required_args": ["page_id"],
            },
        ]

```

## Assumptions & API Contracts
1. Config vars are expected in environment variables.
2. Schema validation is typically deferred to the registry or client implementation.