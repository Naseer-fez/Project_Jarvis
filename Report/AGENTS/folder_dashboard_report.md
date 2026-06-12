ISSUE ID: DASHBOARD-001
SEVERITY: High
CATEGORY: Security vulnerabilities
FILES: d:\AI\Jarvis\dashboard\server.py
DESCRIPTION: The `/gui-audit` route is mounted using FastAPI's `StaticFiles` without any authentication checks, exposing sensitive screenshots of the user's PC to anyone who can access the dashboard's network port.
ROOT CAUSE: `app.mount("/gui-audit", StaticFiles(...))` mounts the directory statically at the application root, bypassing the authentication enforcement (`_is_authorized`) applied to the API and page routes.
EVIDENCE: 
```python
GUI_AUDIT_DIR = PROJECT_ROOT / "outputs" / "gui_audit"
app.mount("/gui-audit", StaticFiles(directory=str(GUI_AUDIT_DIR)), name="gui-audit")
```
POTENTIAL IMPACT: Unauthorized users on the local network (or internet, if the port is exposed) can access and view screenshots of the user's desktop, leading to a severe privacy and security breach.
RECOMMENDED FIX: Remove the public `app.mount` for `/gui-audit`. Instead, create a dedicated, authenticated endpoint (similar to `/api/view-file`) that streams the requested screenshot only if `await _is_authorized(request)` is True.

ISSUE ID: DASHBOARD-002
SEVERITY: High
CATEGORY: Async issues
FILES: d:\AI\Jarvis\dashboard\server.py
DESCRIPTION: The `/api/convert` endpoint performs blocking file operations and synchronous conversion directly on the main event loop thread.
ROOT CAUSE: The `api_convert` function is defined with `async def`, causing FastAPI to schedule it on the main asyncio event loop. Inside, it uses synchronous blocking operations like `shutil.copyfileobj(file.file, tmp_src)` and `perform_conversion(...)` without delegating them to a thread pool.
EVIDENCE: 
```python
@app.post("/api/convert")
async def api_convert(
    request: Request,
    file: UploadFile = File(...),
    target_format: str = Form(...)
):
...
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_src:
        shutil.copyfileobj(file.file, tmp_src)  # BLOCKING I/O
...
    output_path = perform_conversion(tmp_src_path, target_format_clean)  # BLOCKING CALL
```
POTENTIAL IMPACT: The entire FastAPI application (including health checks, WebSockets, and other user requests) will be unresponsive while a file is being uploaded, copied, or converted.
RECOMMENDED FIX: Use `await asyncio.to_thread` for the synchronous blocking operations, or change the endpoint signature from `async def` to `def` so FastAPI automatically runs it in its background thread pool (ensuring that `file.file.read()` or similar safe mechanisms are used appropriately).

ISSUE ID: DASHBOARD-003
SEVERITY: Medium
CATEGORY: Async issues
FILES: d:\AI\Jarvis\dashboard\server.py
DESCRIPTION: Synchronous component calls (`_load_ai_os_overview`, `manager.create()`, `manager.complete()`, `_load_active_goals()`) are executed directly inside `async def` route handlers, potentially blocking the event loop with disk/database I/O.
ROOT CAUSE: Route handlers like `ai_os_page`, `api_ai_os`, `goals_add`, and `goals_complete` are declared as `async def` but call synchronous functions that perform disk reads (loading YAML blueprints) or database queries without offloading them.
EVIDENCE:
```python
@app.get("/goals")
async def goals_page(request: Request):
...
            goals = _load_active_goals(manager) # Blocking DB query
            _refresh_goal_count()
```
POTENTIAL IMPACT: Momentary unresponsiveness of the web dashboard and other async tasks when users load the AI OS page or manage goals.
RECOMMENDED FIX: Wrap the blocking operations in `await asyncio.to_thread(...)`, or convert the endpoint handlers from `async def` to synchronous `def` functions.

ISSUE ID: DASHBOARD-004
SEVERITY: High
CATEGORY: Data validation issues
FILES: d:\AI\Jarvis\dashboard\server.py
DESCRIPTION: The `target_format` field in the `/api/convert` endpoint is not validated or sanitized for special characters before being passed to the converter.
ROOT CAUSE: The endpoint accepts any string from the `Form(...)` input, only applying `.strip().lower()`. It does not restrict the string to alphanumeric characters, allowing potential path traversal payloads or command injection payloads to be passed to `perform_conversion`.
EVIDENCE:
```python
    target_format: str = Form(...)
...
    target_format_clean = target_format.strip().lower()
    try:
        output_path = perform_conversion(tmp_src_path, target_format_clean)
```
POTENTIAL IMPACT: An authenticated attacker could supply a crafted `target_format` (e.g., `../evil.sh`) to overwrite arbitrary files on the system, or potentially execute arbitrary code if the underlying converter invokes a shell command.
RECOMMENDED FIX: Validate `target_format_clean` using a regular expression (e.g., `^[a-z0-9]+$`) to ensure it only contains valid, safe file extension characters before proceeding.
