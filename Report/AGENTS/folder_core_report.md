ISSUE ID: JARVIS-CORE-001
SEVERITY: High
CATEGORY: Race conditions
FILES: `d:\AI\Jarvis\core\controller_v2.py`
DESCRIPTION: The `current_classification` attribute is stored directly on the `JarvisControllerV2` singleton instance inside the async `process()` method without locking. Because `process()` yields execution to an intent router before the classification is consumed by `_dispatch_llm`, concurrent user requests will overwrite each other's classification state.
ROOT CAUSE: Storing request-specific execution state (`current_classification`) on the instance rather than passing it explicitly down the function call chain as a local variable.
EVIDENCE: 
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
POTENTIAL IMPACT: Under concurrent load, the controller might assign complex tasks to simple chat pathways (or vice-versa), bypassing the planner or failing to allocate sufficient context, leading to request failures.
RECOMMENDED FIX: Remove `self.current_classification`. Compute it and store it in a local variable inside `process()`, then pass it explicitly as an argument to `_dispatch_llm` and any other routes that require it.

ISSUE ID: JARVIS-CORE-002
SEVERITY: Medium
CATEGORY: Async issues
FILES: `d:\AI\Jarvis\core\controller_v2.py`, `d:\AI\Jarvis\core\profile.py`, `d:\AI\Jarvis\core\controller\goal_runner.py`, `d:\AI\Jarvis\core\automation\live_automation.py`
DESCRIPTION: Synchronous file I/O operations block the main asyncio event loop across multiple components. Specifically, updating profiles, persisting goal states, and saving background automation states write directly to disk without offloading the work to threads.
ROOT CAUSE: Calling synchronous standard library I/O (like `write_text`, `open`, `os.replace`, and `json.dump`) inside async functions or event loop scheduled callbacks instead of offloading them.
EVIDENCE: 
1. `controller_v2.py` line 241 invokes `self.memory_subsystem.update_profile(user_input, response)`, which synchronously executes `profile.save()` using blocking `os.replace`.
2. `LiveAutomationEngine._run_loop()` in `live_automation.py` calls `_save_state()`, executing `self.state_file.write_text()` directly on the loop every 3 seconds.
3. `goal_runner.py` executes `persist_goal_state()` directly inside the async loop `check_due_goals()`.
POTENTIAL IMPACT: Spikes in disk I/O will stall the event loop, causing the system to appear unresponsive, delaying automated tasks, and stuttering active integrations like voice synthesis.
RECOMMENDED FIX: Offload all disk I/O operations into background threads using `await asyncio.to_thread()` or `loop.run_in_executor()`. Alternatively, utilize an asynchronous file writing library like `aiofiles`.

ISSUE ID: JARVIS-CORE-003
SEVERITY: Medium
CATEGORY: Logic errors
FILES: `d:\AI\Jarvis\core\controller\intent_handlers.py`, `d:\AI\Jarvis\core\controller_v2.py`
DESCRIPTION: The intent handler for "automation" commands expects the `live_automation` engine to be mounted directly on the context (`ctx.live_automation`). However, `controller_v2.py` wraps it in an `automation_manager` subsystem, leaving the handler trying to extract `None`.
ROOT CAUSE: Structural desynchronization after a facade/subsystem refactor. The intent route checking logic was not updated to reflect the new nested attribute path.
EVIDENCE: 
`intent_handlers.py` lines 24 and 38:
```python
        la = getattr(ctx, "live_automation", None)
```
Whereas in `controller_v2.py` line 145:
```python
        self.automation_manager = AutomationManager(...)
```
POTENTIAL IMPACT: The intent router completely bypasses any commands starting with "automation scan", "automation status", or "rag search " because the condition evaluates to `None`, making manual RAG querying unavailable to the user.
RECOMMENDED FIX: Update `intent_handlers.py` to retrieve the engine correctly: `la = getattr(getattr(ctx, "automation_manager", None), "live_automation", None)`.

ISSUE ID: JARVIS-CORE-004
SEVERITY: Low
CATEGORY: Error handling gaps
FILES: `d:\AI\Jarvis\core\controller_v2.py`
DESCRIPTION: The CLI execution loop does not wrap the invocation of `await self.process(text)` in a `try/except` block.
ROOT CAUSE: Assuming `process()` gracefully handles all exceptions internally, which is rarely true across a wide subsystem suite (e.g. LLM request timeouts, invalid DAG plans).
EVIDENCE: 
`controller_v2.py` lines 304-306:
```python
            print(f"DEBUG: Before process(text='{text}')", flush=True)
            response = await self.process(text)
            print(f"DEBUG: After process, response='{response}'", flush=True)
```
POTENTIAL IMPACT: An unexpected network error or data payload issue will crash the entire CLI application unexpectedly.
RECOMMENDED FIX: Wrap the `await self.process(text)` call with a broad `try/except Exception` block within the `run_cli` loop, logging the failure trace and informing the user gracefully without killing the session.
