"""GitHub integration via PyGithub.

Required env vars:
    GITHUB_TOKEN

Optional env vars:
    GITHUB_DEFAULT_REPO
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Iterable

from integrations.base import BaseIntegration

_DEFAULT_LIMIT = 20
_MAX_LIMIT = 100


class GitHubIntegration(BaseIntegration):
    """Read repository state and perform confirm-gated write actions on GitHub."""

    name = "github"
    description = "Inspect GitHub repositories, pull requests, issues, gists, and code search"
    required_config: list[str] = ["GITHUB_TOKEN"]

    def is_available(self) -> bool:
        try:
            import github  # noqa: F401
        except Exception:
            self.unavailable_reason = "PyGithub not installed"
            return False

        if not os.environ.get("GITHUB_TOKEN"):
            self.unavailable_reason = "Missing env var: GITHUB_TOKEN"
            return False
        return True

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

    async def execute(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        args = args or {}

        operations = {
            "list_open_issues": self._list_open_issues,
            "create_issue": self._create_issue,
            "close_issue": self._close_issue,
            "list_open_prs": self._list_open_prs,
            "get_pr_diff": self._get_pr_diff,
            "create_gist": self._create_gist,
            "search_code": self._search_code,
        }

        operation = operations.get(tool_name)
        if operation is None:
            return {"success": False, "data": None, "error": f"Unknown tool: {tool_name}"}

        try:
            loop = asyncio.get_running_loop()
            data = await loop.run_in_executor(None, lambda: operation(args))
            return {"success": True, "data": data, "error": None}
        except Exception as exc:  # noqa: BLE001
            return {"success": False, "data": None, "error": str(exc)}

    def _get_client(self):
        from github import Github

        return Github(os.environ["GITHUB_TOKEN"], per_page=_MAX_LIMIT)

    def _make_input_file_content(self, content: str):
        from github import InputFileContent

        return InputFileContent(content)

    def _resolve_repo_name(self, args: dict[str, Any]) -> str:
        repo_name = str(args.get("repo", "") or "").strip()
        if repo_name:
            return repo_name

        repo_name = str(os.environ.get("GITHUB_DEFAULT_REPO", "") or "").strip()
        if repo_name:
            return repo_name

        raise ValueError("repo is required when GITHUB_DEFAULT_REPO is not configured")

    def _get_repo(self, client, args: dict[str, Any]):
        return client.get_repo(self._resolve_repo_name(args))

    def _coerce_limit(self, raw_value: Any, *, default: int, max_value: int = _MAX_LIMIT) -> int:
        try:
            value = int(raw_value)
        except Exception:
            value = default
        return max(1, min(max_value, value))

    def _take(self, items: Iterable[Any], limit: int) -> list[Any]:
        result: list[Any] = []
        for item in items:
            result.append(item)
            if len(result) >= limit:
                break
        return result

    def _matches_issue_filters(self, issue: Any, args: dict[str, Any]) -> bool:
        label = str(args.get("label", "") or "").strip().lower()
        assignee = str(args.get("assignee", "") or "").strip().lower()
        milestone = str(args.get("milestone", "") or "").strip()

        if label:
            labels = {str(getattr(item, "name", "")).strip().lower() for item in getattr(issue, "labels", [])}
            if label not in labels:
                return False

        if assignee:
            assignees = {
                str(getattr(item, "login", "")).strip().lower()
                for item in getattr(issue, "assignees", [])
                if getattr(item, "login", None)
            }
            if assignee not in assignees:
                return False

        if milestone:
            issue_milestone = getattr(issue, "milestone", None)
            if issue_milestone is None:
                return False

            milestone_title = str(getattr(issue_milestone, "title", "") or "")
            milestone_number = str(getattr(issue_milestone, "number", "") or "")
            if milestone not in {milestone_title, milestone_number}:
                return False

        return True

    def _list_open_issues(self, args: dict[str, Any]) -> dict[str, Any]:
        client = self._get_client()
        repo = self._get_repo(client, args)
        limit = self._coerce_limit(args.get("limit", _DEFAULT_LIMIT), default=_DEFAULT_LIMIT)

        issues = self._take(repo.get_issues(state="open"), max(limit * 3, limit))
        filtered = [issue for issue in issues if self._matches_issue_filters(issue, args)][:limit]

        return {
            "repo": repo.full_name,
            "issues": [
                {
                    "number": issue.number,
                    "title": issue.title,
                    "url": issue.html_url,
                    "state": issue.state,
                    "author": getattr(getattr(issue, "user", None), "login", None),
                    "labels": [getattr(label, "name", "") for label in getattr(issue, "labels", [])],
                    "assignees": [getattr(user, "login", "") for user in getattr(issue, "assignees", [])],
                    "milestone": getattr(getattr(issue, "milestone", None), "title", None),
                }
                for issue in filtered
            ],
            "count": len(filtered),
        }

    def _create_issue(self, args: dict[str, Any]) -> dict[str, Any]:
        title = str(args.get("title", "") or "").strip()
        if not title:
            raise ValueError("title is required")

        client = self._get_client()
        repo = self._get_repo(client, args)
        labels = [str(item).strip() for item in args.get("labels", []) if str(item).strip()]

        create_kwargs: dict[str, Any] = {
            "title": title,
            "body": str(args.get("body", "") or ""),
        }
        if labels:
            create_kwargs["labels"] = labels

        issue = repo.create_issue(**create_kwargs)

        return {
            "repo": repo.full_name,
            "number": issue.number,
            "title": issue.title,
            "url": issue.html_url,
            "state": issue.state,
        }

    def _close_issue(self, args: dict[str, Any]) -> dict[str, Any]:
        if "issue_number" not in args:
            raise ValueError("issue_number is required")

        issue_number = int(args["issue_number"])
        client = self._get_client()
        repo = self._get_repo(client, args)
        issue = repo.get_issue(number=issue_number)
        issue.edit(state="closed")

        return {
            "repo": repo.full_name,
            "number": issue.number,
            "title": issue.title,
            "state": issue.state,
            "url": issue.html_url,
        }

    def _list_open_prs(self, args: dict[str, Any]) -> dict[str, Any]:
        client = self._get_client()
        repo = self._get_repo(client, args)
        limit = self._coerce_limit(args.get("limit", _DEFAULT_LIMIT), default=_DEFAULT_LIMIT)
        pulls = self._take(repo.get_pulls(state="open", sort="created", direction="desc"), limit)

        return {
            "repo": repo.full_name,
            "pull_requests": [
                {
                    "number": pull.number,
                    "title": pull.title,
                    "url": pull.html_url,
                    "state": pull.state,
                    "draft": bool(getattr(pull, "draft", False)),
                    "author": getattr(getattr(pull, "user", None), "login", None),
                    "head": getattr(getattr(pull, "head", None), "ref", None),
                    "base": getattr(getattr(pull, "base", None), "ref", None),
                }
                for pull in pulls
            ],
            "count": len(pulls),
        }

    def _truncate_patch(self, patch: str, remaining_chars: int) -> tuple[str, int]:
        if remaining_chars <= 0:
            return "", 0

        if len(patch) <= remaining_chars:
            return patch, remaining_chars - len(patch)

        marker = "\n... [truncated]"
        allowed = max(0, remaining_chars - len(marker))
        excerpt = patch[:allowed] + marker
        return excerpt, 0

    def _get_pr_diff(self, args: dict[str, Any]) -> dict[str, Any]:
        if "pr_number" not in args:
            raise ValueError("pr_number is required")

        client = self._get_client()
        repo = self._get_repo(client, args)
        pull = repo.get_pull(number=int(args["pr_number"]))

        max_files = self._coerce_limit(args.get("max_files", 20), default=20, max_value=100)
        remaining_patch_chars = self._coerce_limit(
            args.get("max_patch_chars", 6000),
            default=6000,
            max_value=20000,
        )

        files = []
        truncated = False
        for changed_file in self._take(pull.get_files(), max_files):
            patch = str(getattr(changed_file, "patch", "") or "")
            excerpt, remaining_patch_chars = self._truncate_patch(patch, remaining_patch_chars)
            if patch and excerpt != patch:
                truncated = True
            elif patch and remaining_patch_chars == 0 and excerpt:
                truncated = True
            elif patch and not excerpt:
                truncated = True

            files.append(
                {
                    "filename": changed_file.filename,
                    "status": changed_file.status,
                    "additions": changed_file.additions,
                    "deletions": changed_file.deletions,
                    "changes": changed_file.changes,
                    "patch_excerpt": excerpt,
                }
            )
            if remaining_patch_chars <= 0:
                break

        if getattr(pull, "changed_files", 0) > len(files):
            truncated = True

        return {
            "repo": repo.full_name,
            "number": pull.number,
            "title": pull.title,
            "url": pull.html_url,
            "state": pull.state,
            "author": getattr(getattr(pull, "user", None), "login", None),
            "merged": bool(getattr(pull, "merged", False)),
            "mergeable": getattr(pull, "mergeable", None),
            "commits": getattr(pull, "commits", None),
            "changed_files": getattr(pull, "changed_files", len(files)),
            "additions": getattr(pull, "additions", None),
            "deletions": getattr(pull, "deletions", None),
            "files": files,
            "truncated": truncated,
        }

    def _create_gist(self, args: dict[str, Any]) -> dict[str, Any]:
        filename = str(args.get("filename", "") or "").strip()
        if not filename:
            raise ValueError("filename is required")

        if "content" not in args or not str(args.get("content", "")).strip():
            raise ValueError("content is required")

        client = self._get_client()
        user = client.get_user()
        gist = user.create_gist(
            public=bool(args.get("public", False)),
            files={filename: self._make_input_file_content(str(args.get("content", "")))},
            description=str(args.get("description", "") or ""),
        )

        return {
            "id": gist.id,
            "url": gist.html_url,
            "public": gist.public,
            "description": gist.description,
        }

    def _search_code(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query", "") or "").strip()
        if not query:
            raise ValueError("query is required")

        limit = self._coerce_limit(args.get("limit", 10), default=10, max_value=50)
        repo_name = str(args.get("repo", "") or "").strip() or str(os.environ.get("GITHUB_DEFAULT_REPO", "") or "").strip()
        effective_query = f"{query} repo:{repo_name}" if repo_name else query

        client = self._get_client()
        results = self._take(client.search_code(effective_query), limit)

        return {
            "query": effective_query,
            "results": [
                {
                    "name": getattr(result, "name", None),
                    "path": getattr(result, "path", None),
                    "repository": getattr(getattr(result, "repository", None), "full_name", None),
                    "url": getattr(result, "html_url", None),
                    "sha": getattr(result, "sha", None),
                }
                for result in results
            ],
            "count": len(results),
        }


__all__ = ["GitHubIntegration"]
