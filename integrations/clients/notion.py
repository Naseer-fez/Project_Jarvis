"""Notion integration via official Notion API v1 (async aiohttp).

Required env vars:
    NOTION_API_KEY  — Notion internal integration token

Rules:
- Schema validation before writing anything
- No raw LLM JSON written without validation
- All reads are low-risk; all writes are confirm-risk
"""

from __future__ import annotations

import os
from typing import Any

from integrations.base import BaseIntegration

_NOTION_BASE = "https://api.notion.com/v1"
_NOTION_VERSION = "2022-06-28"


class NotionIntegration(BaseIntegration):
    """Notion API integration — pages, databases, blocks (async aiohttp)."""

    name = "notion"
    description = "Create and query Notion pages, databases, and blocks"
    required_config: list[str] = ["NOTION_API_KEY"]

    def is_available(self) -> bool:
        try:
            import aiohttp  # noqa: F401
        except Exception:
            self.unavailable_reason = "aiohttp not installed"
            return False
        if not os.environ.get("NOTION_API_KEY"):
            self.unavailable_reason = "Missing env var: NOTION_API_KEY"
            return False
        return True

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {os.environ['NOTION_API_KEY']}",
            "Notion-Version": _NOTION_VERSION,
            "Content-Type": "application/json",
        }

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

    async def execute(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        args = args or {}
        try:
            if tool_name == "create_page":
                return await self._create_page(args)
            if tool_name == "query_database":
                return await self._query_database(args)
            if tool_name == "append_block":
                return await self._append_block(args)
            if tool_name == "get_page":
                return await self._get_page(args)
            return {"success": False, "data": None, "error": f"Unknown tool: {tool_name}"}
        except Exception as exc:  # noqa: BLE001
            return {"success": False, "data": None, "error": str(exc)}

    # ── Validation helpers ────────────────────────────────────────────────────

    @staticmethod
    def _validate_block_type(block_type: str) -> str:
        allowed = {"paragraph", "bulleted_list_item", "numbered_list_item", "heading_1", "heading_2", "heading_3", "to_do", "quote"}
        return block_type if block_type in allowed else "paragraph"

    @staticmethod
    def _validate_parent_type(parent_type: str) -> str:
        allowed = {"page_id", "database_id"}
        return parent_type if parent_type in allowed else "page_id"

    # ── Tool implementations ──────────────────────────────────────────────────

    async def _create_page(self, args: dict[str, Any]) -> dict[str, Any]:
        import aiohttp

        parent_id = str(args.get("parent_id", "")).strip()
        title = str(args.get("title", "")).strip()
        content = str(args.get("content", "") or "")
        parent_type = self._validate_parent_type(str(args.get("parent_type", "page_id") or "page_id"))

        if not parent_id:
            return {"success": False, "data": None, "error": "parent_id is required"}
        if not title:
            return {"success": False, "data": None, "error": "title is required"}

        children = []
        if content:
            children.append(
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"type": "text", "text": {"content": content[:2000]}}]
                    },
                }
            )

        payload: dict[str, Any] = {
            "parent": {parent_type: parent_id},
            "properties": {
                "title": {
                    "title": [{"type": "text", "text": {"content": title[:255]}}]
                }
            },
        }
        if children:
            payload["children"] = children

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{_NOTION_BASE}/pages",
                json=payload,
                headers=self._headers(),
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                data = await resp.json()
                if resp.status not in (200, 201):
                    return {
                        "success": False,
                        "data": None,
                        "error": data.get("message", str(resp.status)),
                    }
                return {
                    "success": True,
                    "data": {
                        "page_id": data["id"],
                        "url": data.get("url"),
                    },
                    "error": None,
                }

    async def _query_database(self, args: dict[str, Any]) -> dict[str, Any]:
        import aiohttp

        database_id = str(args.get("database_id", "")).strip()
        if not database_id:
            return {"success": False, "data": None, "error": "database_id is required"}

        page_size = min(50, int(args.get("page_size", 10) or 10))
        filter_prop = str(args.get("filter_property", "") or "")
        filter_val = str(args.get("filter_value", "") or "")

        payload: dict[str, Any] = {"page_size": page_size}
        if filter_prop and filter_val:
            payload["filter"] = {
                "property": filter_prop,
                "rich_text": {"contains": filter_val},
            }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{_NOTION_BASE}/databases/{database_id}/query",
                json=payload,
                headers=self._headers(),
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                data = await resp.json()
                if resp.status != 200:
                    return {
                        "success": False,
                        "data": None,
                        "error": data.get("message", str(resp.status)),
                    }

        results = [
            {
                "id": page["id"],
                "url": page.get("url"),
                "created_time": page.get("created_time"),
            }
            for page in data.get("results", [])
        ]
        return {"success": True, "data": {"results": results, "has_more": data.get("has_more", False)}, "error": None}

    async def _append_block(self, args: dict[str, Any]) -> dict[str, Any]:
        import aiohttp

        page_id = str(args.get("page_id", "")).strip()
        text = str(args.get("text", "")).strip()
        block_type = self._validate_block_type(str(args.get("block_type", "paragraph") or "paragraph"))

        if not page_id:
            return {"success": False, "data": None, "error": "page_id is required"}
        if not text:
            return {"success": False, "data": None, "error": "text is required"}

        block = {
            "object": "block",
            "type": block_type,
            block_type: {
                "rich_text": [{"type": "text", "text": {"content": text[:2000]}}]
            },
        }

        async with aiohttp.ClientSession() as session:
            async with session.patch(
                f"{_NOTION_BASE}/blocks/{page_id}/children",
                json={"children": [block]},
                headers=self._headers(),
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                data = await resp.json()
                if resp.status not in (200, 201):
                    return {
                        "success": False,
                        "data": None,
                        "error": data.get("message", str(resp.status)),
                    }
                return {
                    "success": True,
                    "data": {"block_id": data.get("results", [{}])[0].get("id", "")},
                    "error": None,
                }

    async def _get_page(self, args: dict[str, Any]) -> dict[str, Any]:
        import aiohttp

        page_id = str(args.get("page_id", "")).strip()
        if not page_id:
            return {"success": False, "data": None, "error": "page_id is required"}

        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{_NOTION_BASE}/pages/{page_id}",
                headers=self._headers(),
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                data = await resp.json()
                if resp.status != 200:
                    return {
                        "success": False,
                        "data": None,
                        "error": data.get("message", str(resp.status)),
                    }
            # Fetch children blocks
            async with session.get(
                f"{_NOTION_BASE}/blocks/{page_id}/children",
                headers=self._headers(),
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp2:
                blocks_data = await resp2.json()

        blocks = [
            {
                "type": b.get("type"),
                "id": b.get("id"),
            }
            for b in blocks_data.get("results", [])[:20]
        ]
        return {
            "success": True,
            "data": {
                "id": data["id"],
                "url": data.get("url"),
                "created_time": data.get("created_time"),
                "last_edited_time": data.get("last_edited_time"),
                "blocks": blocks,
            },
            "error": None,
        }


__all__ = ["NotionIntegration"]
