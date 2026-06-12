# File Report: github.py
**Path**: `d:\AI\Jarvis\integrations\clients\github.py`
**Role**: Data Model Analyst

## Analysis Summary
This file has been analyzed for schemas, DTOs, state objects, config variables, and dependencies.

## Dependencies
- __future__.annotations
- asyncio
- os
- typing.Any
- integrations.base.BaseIntegration
- github.Github
- github.InputFileContent
- github

## Classes and State Objects
### `GitHubIntegration`
**Variables**: name, description
**Methods**: is_available, get_tools, execute, _get_client, _make_input_file_content, _resolve_repo_name, _get_repo, _coerce_limit, _take, _matches_issue_filters, _list_open_issues, _create_issue, _close_issue, _list_open_prs, _truncate_patch, _get_pr_diff, _create_gist, _search_code

## Tool Schemas / DTOs
```python
    def get_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "list_open_issues",
                "description": "List open GitHub issues, optionally filtered by label, assignee, or milestone",
                "risk": "low",
                "args": {
                    "repo": {"type": "string", "description": "Repository like owner/name", "default": ""},
                    "label": {"type": "string", "description": "Optional label filter", "default": ""},
                    "assignee": {"type": "string", "description": "Optional assignee login filter", "default": ""},
                    "milestone": {
                        "type": "string",
                        "description": "Optional milestone title or number",
                        "default": "",
                    },
                    "limit": {"type": "integer", "default": 20},
                },
                "required_args": [],
            },
            {
                "name": "create_issue",
                "description": "Create a GitHub issue in the target repository",
                "risk": "confirm",
                "args": {
                    "repo": {"type": "string", "description": "Repository like owner/name", "default": ""},
                    "title": {"type": "string", "description": "Issue title"},
                    "body": {"type": "string", "description": "Issue body", "default": ""},
                    "labels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional label names",
                        "default": [],
                    },
                },
                "required_args": ["title"],
            },
            {
                "name": "close_issue",
                "description": "Close a GitHub issue by number",
                "risk": "confirm",
                "args": {
                    "repo": {"type": "string", "description": "Repository like owner/name", "default": ""},
                    "issue_number": {"type": "integer", "description": "GitHub issue number"},
                },
                "required_args": ["issue_number"],
            },
            {
                "name": "list_open_prs",
                "description": "List open pull requests in the target repository",
                "risk": "low",
                "args": {
                    "repo": {"type": "string", "description": "Repository like owner/name", "default": ""},
                    "limit": {"type": "integer", "default": 20},
                },
                "required_args": [],
            },
            {
                "name": "get_pr_diff",
                "description": "Return a summarized view of changed files and patch excerpts for a pull request",
                "risk": "low",
                "args": {
                    "repo": {"type": "string", "description": "Repository like owner/name", "default": ""},
                    "pr_number": {"type": "integer", "description": "Pull request number"},
                    "max_files": {"type": "integer", "default": 20},
                    "max_patch_chars": {"type": "integer", "default": 6000},
                },
                "required_args": ["pr_number"],
            },
            {
                "name": "create_gist",
                "description": "Create a GitHub gist from a code snippet or text file",
                "risk": "confirm",
                "args": {
                    "filename": {"type": "string", "description": "Gist filename"},
                    "content": {"type": "string", "description": "Full gist file content"},
                    "description": {"type": "string", "description": "Optional gist description", "default": ""},
                    "public": {"type": "boolean", "description": "Whether the gist is public", "default": False},
                },
                "required_args": ["filename", "content"],
            },
            {
                "name": "search_code",
                "description": "Search GitHub code, optionally scoped to a repository",
                "risk": "low",
                "args": {
                    "query": {"type": "string", "description": "Code search query"},
                    "repo": {"type": "string", "description": "Optional repository like owner/name", "default": ""},
                    "limit": {"type": "integer", "default": 10},
                },
                "required_args": ["query"],
            },
        ]

```

## Assumptions & API Contracts
1. Config vars are expected in environment variables.
2. Schema validation is typically deferred to the registry or client implementation.