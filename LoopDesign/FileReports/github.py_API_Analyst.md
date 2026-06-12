# clients/github.py API Analyst Report

## Overview
Integration for interacting with GitHub repositories, pull requests, issues, gists, and code search using the `PyGithub` library.

## API Contracts & Methods
- `GitHubIntegration(BaseIntegration)`
  - `is_available()`: Checks for `github` library and `GITHUB_TOKEN`.
  - Configures `Github` client with `per_page=100`.

## Tools Exposed
- `list_open_issues(repo, label, assignee, milestone, limit)` [Risk: `low`]
- `create_issue(repo, title, body, labels)` [Risk: `confirm`]
- `close_issue(repo, issue_number)` [Risk: `confirm`]
- `list_open_prs(repo, limit)` [Risk: `low`]
- `get_pr_diff(repo, pr_number, max_files=20, max_patch_chars=6000)` [Risk: `low`]
  - Truncates patches heavily to avoid context limits.
- `create_gist(filename, content, description, public)` [Risk: `confirm`]
- `search_code(query, repo, limit)` [Risk: `low`]

## Assumptions & Constants
- Default limits: `_DEFAULT_LIMIT=20`, `_MAX_LIMIT=100`.
- Patch max characters limit `_MAX_PATCH_CHARS=6000` (up to 20000 configurable).
- Repositories are resolved from tool arguments or from `GITHUB_DEFAULT_REPO`.

## Configuration Variables
- `GITHUB_TOKEN` (required)
- `GITHUB_DEFAULT_REPO` (optional)

## Dependencies
- `github` (PyGithub)
- `asyncio`, `os`

## Prompts
- None.
