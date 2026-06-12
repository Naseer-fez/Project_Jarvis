# Evidences Collected

### Evidence for AUDIT-001
`re.compile(r"(?i)\b(password|passwd|token|api[_-]?key)\s*=\s*([^\s,;]+)")`

### Evidence for AUDIT-002
`re.compile(r"(?i)\b(password|passwd|token|api[_-]?key)\s*=\s*([^\s,;]+)")`

### Evidence for AUDIT-003
`_LONG_SECRET = re.compile(r"\b[a-zA-Z0-9]{32,}\b")`

### Evidence for AUDIT-004
`value = str(text or "")`

### Evidence for AUDIT-005
`[dashboard\nport=invalid`

### Evidence for AUDIT-006
The `memory.db` file size is exactly 36 bytes.

### Evidence for AUDIT-007
`sqlite_file=audit/edge_cases/memory/memory.db`

### Evidence for JARVIS-CONFIG-001
Line 125 in jarvis.ini: `blueprint_file = config/ai_os.json`
The directory listing of `d:\AI\Jarvis\config` only contains `jarvis.ini`, `settings.env`, and `settings.env.template`.

### Evidence for JARVIS-CONFIG-002
`settings.env.template` contains `HOME_ASSISTANT_URL`, `HOME_ASSISTANT_TOKEN`, `GITHUB_TOKEN`, `GITHUB_DEFAULT_REPO`, and various `WEB_SEARCH_*` keys (lines 13-42) which are entirely absent from `settings.env` (which only goes up to line 21).

### Evidence for JARVIS-CONFIG-003
`jarvis.ini` (Lines 67-79):
[web_search]
enabled = true
provider = auto
default_max_results = 5

`settings.env.template` (Lines 31-42):
WEB_SEARCH_ENABLED=true
WEB_SEARCH_PROVIDER=auto
WEB_SEARCH_DEFAULT_MAX_RESULTS=5

### Evidence for JARVIS-CONFIG-004
Lines 8-9 (in `[ollama]`): 
planner_model = deepseek-r1:8b
vision_model = llava
Lines 23-25 (in `[models]`): 
plan_model = deepseek-r1:8b
vision_model = llava

### Evidence for JARVIS-CONFIG-005
Lines 41-43:
forbidden_actions = execute_shell, delete_file, shell_exec, file_delete, registry_write, format_disk, wipe_disk
blocked_actions = execute_shell, delete_file, shell_exec, file_delete, registry_write, format_disk, wipe_disk
critical_actions = execute_shell, delete_file, shell_exec, file_delete, registry_write, format_disk, wipe_disk

### Evidence for JARVIS-CORE-001
`controller_v2.py` lines 234-235:
```python
        try:
            from core.controller.complexity_scorer import classify_request
            self.current_classification = classify_request(text)
```
Later in `_dispatch_llm` line 195:
```python
        classification = getattr(self, "current_classification", {})
```

### Evidence for JARVIS-CORE-002
1. `controller_v2.py` line 241 invokes `self.memory_subsystem.update_profile(user_input, response)`, which synchronously executes `profile.save()` using blocking `os.replace`.
2. `LiveAutomationEngine._run_loop()` in `live_automation.py` calls `_save_state()`, executing `self.state_file.write_text()` directly on the loop every 3 seconds.
3. `goal_runner.py` executes `persist_goal_state()` directly inside the async loop `check_due_goals()`.

### Evidence for JARVIS-CORE-003
`intent_handlers.py` lines 24 and 38:
```python
        la = getattr(ctx, "live_automation", None)
```
Whereas in `controller_v2.py` line 145:
```python
        self.automation_manager = AutomationManager(...)
```

### Evidence for JARVIS-CORE-004
`controller_v2.py` lines 304-306:
```python
            print(f"DEBUG: Before process(text='{text}')", flush=True)
            response = await self.process(text)
            print(f"DEBUG: After process, response='{response}'", flush=True)
```

### Evidence for DASHBOARD-001
```python
GUI_AUDIT_DIR = PROJECT_ROOT / "outputs" / "gui_audit"
app.mount("/gui-audit", StaticFiles(directory=str(GUI_AUDIT_DIR)), name="gui-audit")
```

### Evidence for DASHBOARD-002
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

### Evidence for DASHBOARD-003
```python
@app.get("/goals")
async def goals_page(request: Request):
...
            goals = _load_active_goals(manager) # Blocking DB query
            _refresh_goal_count()
```

### Evidence for DASHBOARD-004
```python
    target_format: str = Form(...)
...
    target_format_clean = target_format.strip().lower()
    try:
        output_path = perform_conversion(tmp_src_path, target_format_clean)
```

### Evidence for DATA-001
`2026-02-27 22:46:36,128 [ERROR] JarvisMain: Could not import Phase 4 modules: cannot import name 'Controller' from 'core.controller_v2' (D:\AI\Jarvis\core\controller_v2.py)`
`2026-02-27 22:46:36,129 [CRITICAL] JarvisMain: Initialization failed: cannot import name 'Controller' from 'core.controller_v2'`

### Evidence for DesignCHnage-001
```python
def update_ledger():
    with open(COVERAGE_LEDGER_PATH, 'r', encoding='utf-8') as f:
        content = f.read()
```

### Evidence for DesignCHnage-002
`1. [Repository_Census.md](file:///d:/AI/Jarvis/Repository_Census.md)`

### Evidence for DesignCHnage-003
```python
    for filename, content in files.items():
        with open(ARCHITECTURE_DIR / filename, "w", encoding="utf-8") as f:
            f.write(content)
```

### Evidence for DesignCHnage-004
`Batch_02: ... fpdf, textwrap, s`
`Batch_03: ... base64, teleg`
`Batch_05: ... inspect, configpar`

### Evidence for DesignCHnage-005
```python
    deps = set()
    ...
    content += f"Data exchanges primarily with: {', '.join(list(deps)[:5])}\n"
```

### Evidence for DesignCHnage-006
```python
# In generate_evidence:
content += f"- **{file}**: Verified {len(data['classes'])} classes and {len(data['functions'])} top-level functions.\n"
```

### Evidence for DesignCHnage-007
The directory itself and the root file `DesignCHnage.md` reflect this. Both scripts map to `DESIGN_CHANGE_DIR = ROOT_DIR / "DesignCHnage"`.

### Evidence for DOCS-001
`![Jarvis System Architecture Infographic](file:///C:/Users/FEZ%20NASEER/.gemini/antigravity/brain/b174c800-fc53-4e9e-9930-11744ec2b80d/system_architecture_1780132164075.png)`

### Evidence for DOCS-002
The documentation tree fails to list the `controller_v2.py` file (which is mentioned elsewhere in the same doc) and numerous active directories such as `config/`, `context/`, `logging/`, and `ops/`.

### Evidence for DOCS-003
```mermaid
    subgraph Service Pool [Core Services Bootstrapping]
...
    Controller --> ServicePool
```

### Evidence for FINAL-001
`08_Phase_2_Root_Cause_Analysis.md` lists ISSUE 1 as "Zombie Tasks & Inconsistent Teardown". Conversely, `Phase_2_3_4_Report.md` lists Hotspot 1 as "Duplicate System Architectures" and Strategy A as "Architectural De-Duplication".

### Evidence for FINAL-002
`02_Execution_Graph.md` states "Dynamically loads ControllerClass (e.g., controller_v2.py), initializes it." while `Execution_Graph.md` states "Instantiates JarvisControllerV2" explicitly.

### Evidence for FINAL-003
Line 18 contains the text: "- **Dashbaord**: Runs an internal GUI server..."

### Evidence for INTEGRATIONS-001
`path = os.path.abspath(str(args.get("path", "outputs/screenshot.png") ...)` followed directly by `await loop.run_in_executor(None, lambda: pyautogui.screenshot(path))` allows paths like `../../../../Windows/System32/somefile.png`.

### Evidence for INTEGRATIONS-002
`content = CALENDAR_PATH.read_text()` followed immediately by `content.replace("END:VCALENDAR", ...)` and `CALENDAR_PATH.write_text(updated)` without `threading.Lock` or `filelock`.

### Evidence for INTEGRATIONS-003
`dt.replace(tzinfo=timezone.utc)` followed by `{ "dateTime": self._to_rfc3339(start_str), "timeZone": tz }` in the API body.

### Evidence for INTEGRATIONS-004
`def is_available(self) -> bool: return True` paired with `from icalendar import Calendar` inside the `_list_events` method.

### Evidence for INTEGRATIONS-005
`for integration in self._integrations.values(): tools = integration.get_tools() or []` does not verify if the integration actively owns the tool in `self._tool_owner`.

### Evidence for INTEGRATIONS-006
`data = await resp.json(); header_map = {}; for h in data.get("payload", {}).get("headers", []):` will fail silently because the error response lacks a payload node.

### Evidence for LOGS-001
`2026-02-20 16:41:40,012 ERROR jarvis.voice.stt: faster-whisper not installed — STT unavailable`

### Evidence for LOGS-002
`2026-02-21 20:26:14,068 WARNING jarvis.controller: Ollama not reachable - planner will degrade gracefully` and `All cloud providers failed unconfigured. LLM or are`.

### Evidence for LOGS-003
The presence of the file `d:\AI\Jarvis\logs\audit.jsonl.corrupted` alongside `audit.jsonl.bak`.

### Evidence for MEMORY-001
Line 2 of user_profile.json contains: `"name": "hacked"`

### Evidence for REQUIREMENTS-001
`fpdf>=1.7,<2.0` is specified in `full.txt`.

### Evidence for REQUIREMENTS-002
`types-PyYAML>=6.0,<7.0` is present in `dev.txt`, yet `PyYAML` is completely absent from `base.txt`, `full.txt`, and all other files.

### Evidence for REQUIREMENTS-003
`dev.txt` contains `-r base.txt` and `types-Markdown>=3.10,<4.0`. `full.txt` contains `markdown>=3.10,<4.0`. Installing `dev.txt` will install type stubs for markdown without the library itself, breaking static type checking and making it impossible to test full stack features.

### Evidence for JARVIS-RUNTIME-001
`2026-02-18 13:13:05,152 [WARNING] huggingface_hub.utils._http: Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.`

### Evidence for JARVIS-TESTS-001
```python
# In d:\AI\Jarvis\tests\unit\test_agent_loop.py (line 92)
# Simulate user rejecting the prompt
trace = await engine.run("test goal", context, confirm_callback=lambda prompt: False)
```

### Evidence for 1
`core/runtime/dashboard_runtime.py` contains `from dashboard.server import app`. Meanwhile, `dashboard/server.py` contains multiple core imports like `from core.security.auth import AuthManager`.

### Evidence for 2
`bootstrap.py` executes `integration_registry.register_safety_rules(...)`. However, `services.py` injects `CapabilityRegistry` as the sole `tool_router`, which only processes `builtin_tools` and dynamic plugins, completely ignoring `IntegrationRegistry`.

### Evidence for 3
`DashboardRuntime.stop()` explicitly calls `self._thread.join(timeout=timeout)`. `core/runtime/entrypoint.py` calls this method directly within the `finally` block of the `async_run` coroutine.

### Evidence for 4
`JarvisControllerV2.shutdown()` successfully cancels and awaits `self._goal_check_task`, but it does not invoke `self.goal_runner.persist_goal_state()`.

### Evidence for 5
`requirements.lock` is missing packages such as `pyautogui`, `speechrecognition`, and `opencv-python`. Meanwhile, `jarvis.spec` enforces `hiddenimports=['pyautogui', 'speech_recognition', 'cv2', ...]`.

### Evidence for 6
`grep` analysis confirms zero active imports of `audit.audit_logger` in the execution paths. `core/logging/logger.py` independently implements `redact_sensitive_data` and an `AuditLog` class.

