# `github.py` - API Analyst Report

## Overview
GitHub integration via the `PyGithub` package. Allows inspecting repositories, PRs, issues, gists, and searching code.

## Endpoints / Tools
1. `list_open_issues`
   - Description: List open GitHub issues, optionally filtered by label, assignee, or milestone.
   - Risk: low (read-only)
   - Arguments: `repo` (string), `label` (string), `assignee` (string), `milestone` (string), `limit` (integer, default 20).
2. `create_issue`
   - Description: Create a GitHub issue.
   - Risk: confirm (write)
   - Arguments: `repo` (string), `title` (string, required), `body` (string), `labels` (array of strings).
3. `close_issue`
   - Description: Close a GitHub issue by number.
   - Risk: confirm (write)
   - Arguments: `repo` (string), `issue_number` (integer, required).
4. `list_open_prs`
   - Description: List open pull requests.
   - Risk: low (read-only)
   - Arguments: `repo` (string), `limit` (integer, default 20).
5. `get_pr_diff`
   - Description: Summarized view of changed files and patch excerpts for a PR.
   - Risk: low (read-only)
   - Arguments: `repo` (string), `pr_number` (integer, required), `max_files` (integer, default 20), `max_patch_chars` (integer, default 6000).
6. `create_gist`
   - Description: Create a GitHub gist.
   - Risk: confirm (write)
   - Arguments: `filename` (string, required), `content` (string, required), `description` (string), `public` (boolean, default False).
7. `search_code`
   - Description: Search GitHub code, optionally scoped to a repository.
   - Risk: low (read-only)
   - Arguments: `query` (string, required), `repo` (string), `limit` (integer, default 10).

## External Contracts / Dependencies
- Requires `GITHUB_TOKEN`. Optional `GITHUB_DEFAULT_REPO`.
- Uses `github` (`PyGithub`) library.

## Assumptions
- For repo-specific tools, if `repo` is not provided in args, it falls back to `GITHUB_DEFAULT_REPO`. If neither is provided, throws `ValueError`.
- `list_open_issues` fetches up to `limit * 3` or `limit` (whichever is larger, actually code says `max(limit*3, limit)`) and filters them locally (due to PyGithub search limitations or to simply filter locally easily).
- `get_pr_diff` explicitly truncates diff patches that exceed `max_patch_chars` to avoid context length bloat.
- PyGithub's operations are blocking, executed via `loop.run_in_executor`.
- Enforces strict maximum limits on paginated fetches (e.g. `max_value=100` for `_coerce_limit`).
