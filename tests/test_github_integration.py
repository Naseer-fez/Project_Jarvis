"""
tests/test_github_integration.py

All GitHub SDK calls are mocked.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from integrations.clients.github import GitHubIntegration


def _env() -> dict[str, str]:
    return {
        "GITHUB_TOKEN": "ghp_testtoken",
        "GITHUB_DEFAULT_REPO": "octo-org/example-repo",
    }


class FakeIssue:
    def __init__(
        self,
        number: int,
        title: str,
        *,
        state: str = "open",
        labels: list[str] | None = None,
        assignees: list[str] | None = None,
        milestone: str | None = None,
    ) -> None:
        self.number = number
        self.title = title
        self.state = state
        self.html_url = f"https://github.com/octo-org/example-repo/issues/{number}"
        self.user = SimpleNamespace(login="fez")
        self.labels = [SimpleNamespace(name=name) for name in (labels or [])]
        self.assignees = [SimpleNamespace(login=name) for name in (assignees or [])]
        self.milestone = (
            SimpleNamespace(title=milestone, number=1)
            if milestone is not None
            else None
        )

    def edit(self, *, state: str) -> None:
        self.state = state


class FakePull:
    def __init__(self, number: int, title: str, files: list[SimpleNamespace]) -> None:
        self.number = number
        self.title = title
        self.html_url = f"https://github.com/octo-org/example-repo/pull/{number}"
        self.state = "open"
        self.draft = False
        self.user = SimpleNamespace(login="reviewer")
        self.head = SimpleNamespace(ref="feature/test")
        self.base = SimpleNamespace(ref="main")
        self.merged = False
        self.mergeable = True
        self.commits = 2
        self.changed_files = len(files)
        self.additions = sum(file.additions for file in files)
        self.deletions = sum(file.deletions for file in files)
        self._files = files

    def get_files(self):
        return list(self._files)


class FakeRepo:
    def __init__(self) -> None:
        self.full_name = "octo-org/example-repo"
        self._issues = [
            FakeIssue(1, "Bug one", labels=["bug"], assignees=["fez"], milestone="v1"),
            FakeIssue(2, "Docs cleanup", labels=["docs"], assignees=["alex"]),
        ]
        self._pulls = [
            FakePull(
                7,
                "Improve planner logging",
                [
                    SimpleNamespace(
                        filename="core/planner.py",
                        status="modified",
                        additions=10,
                        deletions=2,
                        changes=12,
                        patch="+" + ("a" * 80),
                    ),
                    SimpleNamespace(
                        filename="README.MD",
                        status="modified",
                        additions=5,
                        deletions=1,
                        changes=6,
                        patch="+" + ("b" * 80),
                    ),
                ],
            )
        ]

    def get_issues(self, *, state: str = "open"):
        return [issue for issue in self._issues if issue.state == state]

    def create_issue(self, **kwargs):
        issue = FakeIssue(3, kwargs["title"], labels=kwargs.get("labels", []))
        self._issues.append(issue)
        return issue

    def get_issue(self, *, number: int):
        for issue in self._issues:
            if issue.number == number:
                return issue
        raise KeyError(number)

    def get_pulls(self, *, state: str = "open", sort: str = "created", direction: str = "desc"):
        del sort, direction
        return [pull for pull in self._pulls if pull.state == state]

    def get_pull(self, *, number: int):
        for pull in self._pulls:
            if pull.number == number:
                return pull
        raise KeyError(number)


class FakeUser:
    def __init__(self) -> None:
        self.last_gist_kwargs: dict[str, object] | None = None

    def create_gist(self, **kwargs):
        self.last_gist_kwargs = kwargs
        return SimpleNamespace(
            id="gist123",
            html_url="https://gist.github.com/gist123",
            public=kwargs.get("public", False),
            description=kwargs.get("description", ""),
        )


class FakeGitHubClient:
    def __init__(self) -> None:
        self.repo = FakeRepo()
        self.user = FakeUser()
        self.last_search_query = ""

    def get_repo(self, name: str):
        assert name == "octo-org/example-repo"
        return self.repo

    def get_user(self):
        return self.user

    def search_code(self, query: str):
        self.last_search_query = query
        return [
            SimpleNamespace(
                name="planner.py",
                path="core/planner.py",
                repository=SimpleNamespace(full_name="octo-org/example-repo"),
                html_url="https://github.com/octo-org/example-repo/blob/main/core/planner.py",
                sha="abc123",
            )
        ]


def test_is_available_false_without_env():
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("GITHUB_TOKEN", None)
        gh = GitHubIntegration()
        assert gh.is_available() is False


def test_is_available_true_with_env_and_lib():
    with patch.dict(os.environ, _env()), patch.dict("sys.modules", {"github": MagicMock()}):
        gh = GitHubIntegration()
        assert gh.is_available() is True


def test_get_tools_structure_and_risk():
    gh = GitHubIntegration()
    tools = {tool["name"]: tool for tool in gh.get_tools()}
    assert {
        "list_open_issues",
        "create_issue",
        "close_issue",
        "list_open_prs",
        "get_pr_diff",
        "create_gist",
        "search_code",
    } == set(tools)
    assert tools["list_open_issues"]["risk"] == "low"
    assert tools["list_open_prs"]["risk"] == "low"
    assert tools["get_pr_diff"]["risk"] == "low"
    assert tools["search_code"]["risk"] == "low"
    assert tools["create_issue"]["risk"] == "confirm"
    assert tools["close_issue"]["risk"] == "confirm"
    assert tools["create_gist"]["risk"] == "confirm"


@pytest.mark.asyncio
async def test_repo_bound_tool_requires_repo_if_no_default():
    gh = GitHubIntegration()
    client = FakeGitHubClient()
    with patch.dict(os.environ, {"GITHUB_TOKEN": "ghp_testtoken"}, clear=True):
        with patch.object(gh, "_get_client", return_value=client):
            result = await gh.execute("list_open_issues", {})

    assert result["success"] is False
    assert "repo is required" in result["error"]


@pytest.mark.asyncio
async def test_list_open_issues_filters_label():
    gh = GitHubIntegration()
    client = FakeGitHubClient()
    with patch.dict(os.environ, _env()):
        with patch.object(gh, "_get_client", return_value=client):
            result = await gh.execute("list_open_issues", {"label": "bug", "limit": 5})

    assert result["success"] is True
    assert result["data"]["repo"] == "octo-org/example-repo"
    assert result["data"]["count"] == 1
    assert result["data"]["issues"][0]["number"] == 1


@pytest.mark.asyncio
async def test_create_issue_missing_title():
    gh = GitHubIntegration()
    result = await gh.execute("create_issue", {"title": ""})
    assert result["success"] is False
    assert "title is required" in result["error"]


@pytest.mark.asyncio
async def test_create_issue_mock_success():
    gh = GitHubIntegration()
    client = FakeGitHubClient()
    with patch.dict(os.environ, _env()):
        with patch.object(gh, "_get_client", return_value=client):
            result = await gh.execute(
                "create_issue",
                {
                    "title": "New bug",
                    "body": "Steps to reproduce",
                    "labels": ["bug"],
                },
            )

    assert result["success"] is True
    assert result["data"]["number"] == 3
    assert result["data"]["state"] == "open"


@pytest.mark.asyncio
async def test_close_issue_mock_success():
    gh = GitHubIntegration()
    client = FakeGitHubClient()
    with patch.dict(os.environ, _env()):
        with patch.object(gh, "_get_client", return_value=client):
            result = await gh.execute("close_issue", {"issue_number": 1})

    assert result["success"] is True
    assert result["data"]["state"] == "closed"


@pytest.mark.asyncio
async def test_list_open_prs_mock_success():
    gh = GitHubIntegration()
    client = FakeGitHubClient()
    with patch.dict(os.environ, _env()):
        with patch.object(gh, "_get_client", return_value=client):
            result = await gh.execute("list_open_prs", {"limit": 5})

    assert result["success"] is True
    assert result["data"]["count"] == 1
    assert result["data"]["pull_requests"][0]["number"] == 7


@pytest.mark.asyncio
async def test_get_pr_diff_truncates_large_patch_output():
    gh = GitHubIntegration()
    client = FakeGitHubClient()
    with patch.dict(os.environ, _env()):
        with patch.object(gh, "_get_client", return_value=client):
            result = await gh.execute("get_pr_diff", {"pr_number": 7, "max_patch_chars": 60})

    assert result["success"] is True
    assert result["data"]["number"] == 7
    assert result["data"]["truncated"] is True
    assert result["data"]["files"]
    assert "truncated" in result["data"]["files"][0]["patch_excerpt"]


@pytest.mark.asyncio
async def test_create_gist_mock_success():
    gh = GitHubIntegration()
    client = FakeGitHubClient()
    with patch.dict(os.environ, _env()):
        with patch.object(gh, "_get_client", return_value=client):
            with patch.object(gh, "_make_input_file_content", side_effect=lambda text: {"content": text}):
                result = await gh.execute(
                    "create_gist",
                    {
                        "filename": "snippet.py",
                        "content": "print('hi')",
                        "description": "test gist",
                        "public": False,
                    },
                )

    assert result["success"] is True
    assert result["data"]["id"] == "gist123"
    assert client.user.last_gist_kwargs is not None
    assert "snippet.py" in client.user.last_gist_kwargs["files"]


@pytest.mark.asyncio
async def test_search_code_scopes_to_default_repo():
    gh = GitHubIntegration()
    client = FakeGitHubClient()
    with patch.dict(os.environ, _env()):
        with patch.object(gh, "_get_client", return_value=client):
            result = await gh.execute("search_code", {"query": "TaskPlanner", "limit": 5})

    assert result["success"] is True
    assert result["data"]["count"] == 1
    assert "repo:octo-org/example-repo" in client.last_search_query


@pytest.mark.asyncio
async def test_unknown_tool_returns_error():
    gh = GitHubIntegration()
    result = await gh.execute("no_such_tool", {})
    assert result["success"] is False
    assert "Unknown" in result["error"]
