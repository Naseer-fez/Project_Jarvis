# Documentation Report: clients/github.py

## Assumptions
- Uses `PyGithub` package.
- Operations map directly to the GitHub API.
- Pull request diff sizes are constrained by `max_files` and `max_patch_chars` to prevent context length explosion. Truncation is explicitly handled and signaled via `truncated` flag.
- Default limit values are usually 20, max limit 100 for pagination/fetching.
- Resolves target repo dynamically from args or falls back to `GITHUB_DEFAULT_REPO`.

## Schema / API Contract
- Tools: `list_open_issues`, `create_issue`, `close_issue`, `list_open_prs`, `get_pr_diff`, `create_gist`, `search_code`.
- `get_pr_diff` returns structured dict containing diff excerpts for each changed file.

## Dependencies
- `github` (PyGithub external)
- `asyncio`, `os`, `typing`

## Configuration Variables
- `GITHUB_TOKEN`
- `GITHUB_DEFAULT_REPO` (optional)

## Prompts
None.
