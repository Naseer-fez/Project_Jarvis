# Folder Analysis: dashboard

## Folder Purpose
Contains components related to dashboard.

## Findings
- **DASHBOARD-001** (High): The `/gui-audit` route is mounted using FastAPI's `StaticFiles` without any authentication checks, exposing sensitive screenshots of the user's PC to anyone who can access the dashboard's network port.
- **DASHBOARD-002** (High): The `/api/convert` endpoint performs blocking file operations and synchronous conversion directly on the main event loop thread.
- **DASHBOARD-003** (Medium): Synchronous component calls (`_load_ai_os_overview`, `manager.create()`, `manager.complete()`, `_load_active_goals()`) are executed directly inside `async def` route handlers, potentially blocking the event loop with disk/database I/O.
- **DASHBOARD-004** (High): The `target_format` field in the `/api/convert` endpoint is not validated or sanitized for special characters before being passed to the converter.

## Risks & Dependencies
See full project roadmap.
