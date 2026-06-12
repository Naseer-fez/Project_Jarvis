# File Report: github.py
## Role: Dependency Analyst

### 1. Library Requirements
- `asyncio`, `os`, `typing` (Standard Library)
- `github` (Third-party, specifically `PyGithub` which provides `Github` and `InputFileContent`)
- `integrations.base` (Local)

### 2. Service Dependencies
- GitHub REST API.

### 3. Hidden Execution Links
- `PyGithub` makes synchronous HTTP requests, wrapped in `run_in_executor`.
- `get_pr_diff` dynamically calculates whether diff text should be truncated by limiting string length to `remaining_chars`.
- `search_code` implicitly injects `repo:{repo_name}` into the search string if scoped.

### 4. Assumptions & API Contracts
- If `GITHUB_DEFAULT_REPO` is set, `repo` argument can be omitted from most tools.
- Paginates via `PyGithub` standard iteration, taking manually limited items (`_take(items, limit)`).
- Risk schema: `create_issue`, `close_issue`, `create_gist` are `confirm`. Everything else `low`.
- Filtering for milestones/assignees pulls down issues open in the repo and processes matching locally in Python rather than relying entirely on GitHub API queries, effectively pulling up to `limit * 3` before matching.

### 5. Configuration Variables
- `GITHUB_TOKEN` (Required)
- `GITHUB_DEFAULT_REPO` (Optional)

### 6. Prompts Found
- None.
