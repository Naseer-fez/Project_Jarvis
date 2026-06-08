# Problem & Resolution Report: Jarvis Testing & Hardening

**Date:** 2026-06-05
**Goal:** Document architectural issues discovered during interactive terminal testing and the applied resolutions to ensure stable, offline-first autonomous execution. This report serves as a persistent guide for future Antigravity agents working on this project.

## Overview
During the testing phase of Jarvis (`JarvisControllerV2`), several critical issues were identified that prevented fully autonomous operation, specifically in offline, headless, and isolated environments. The primary goals were to enforce an offline-first execution model (using Ollama locally), fix race conditions during startup, and patch critical typing/schema bugs in the tool execution pipeline.

---

## 1. Cloud API Key Leakage / Fallback Issues

### Problem
The project's `.env` file contained invalid or expired cloud API keys (e.g., `GROQ_API_KEY`, `GEMINI_API_KEY`, `OPENAI_API_KEY`). 
When Jarvis attempted to initialize its model routing strategy, if a local model failed to load quickly or if the routing logic encountered a bug, it would fall back to attempting to use these cloud providers. Because the keys were invalid or expired, this caused terminal crashes or HTTP 401/403/404 errors, breaking the offline execution constraint.

### Resolution
- **Action:** Commented out all cloud-based API keys in `D:\AI\Jarvis\.env`.
- **Outcome:** Enforced strict local-only execution. The system can no longer accidentally fallback to broken cloud paths, ensuring that if it fails, it fails predictably on the local environment where the user expects it to run.
- **Future Note:** Do not re-enable these keys unless explicitly requested by the user for a specific cloud-testing scenario.

---

## 2. Race Condition in Model Discovery (HTTP 404)

### Problem
The `ModelRouter` in `core/llm/model_router.py` uses an asynchronous background task (`_poll_ollama_models`) to discover available Ollama models. 
However, when the `JarvisControllerV2` starts up and immediately tries to route a request, the background task hasn't finished its first poll. The `_available_ollama_models` list is empty, leading the router to conclude no models are available or attempt an invalid API call, resulting in an HTTP 404 error from the local Ollama server.

### Resolution
- **Action:** Modified `refresh_available_models()` in `ModelRouter` to perform a **synchronous, blocking fetch** on its very first invocation before returning.
- **Implementation Detail:** Added an `_initial_fetch_done` flag. If false, the method waits for the actual HTTP request to Ollama to complete and populate `_available_ollama_models` before allowing routing logic to proceed.
- **Outcome:** The system reliably knows which local models are available immediately upon startup, eliminating the 404 crashes.

---

## 3. Configuration Parsing Bug in ModelRouter

### Problem
The `ModelRouter` was incorrectly attempting to parse the routing strategy from the configuration. It expected a specific key structure or nested dictionary that did not match how `config/jarvis.ini` was being loaded, leading to fallback routing or routing failures.

### Resolution
- **Action:** Corrected the parsing logic in `ModelRouter` to correctly read the `strategy` string (e.g., `cost`, `performance`, `local`) from the configuration object.
- **Outcome:** The router now correctly honors the user's defined routing strategy in the config file.

---

## 4. TaskPlanner Tool Schema Injection Crashes (TypeError)

### Problem
The `TaskPlanner` relies on providing JSON schemas of available tools to the LLM so it can generate execution plans. 
However, the integration between the tool registry and the planner was brittle. In some cases, the tools were passed as raw Python function references or malformed dictionaries instead of proper JSON schemas. When the planner attempted to serialize these into the system prompt, it threw `TypeError: Object of type function is not JSON serializable` or similar serialization errors, crashing the agent loop.

### Resolution
- **Action:** Updated `core/planner/planner.py` to dynamically and safely extract/inject valid JSON schemas.
- **Implementation Detail:** Ensured that the planner maps the available capabilities from the `Registry` into strict, serializable JSON schema definitions before injecting them into the prompt.
- **Outcome:** The LLM receives properly formatted tool definitions, preventing Python serialization crashes and improving the LLM's understanding of available tools.

---

## 5. Headless Execution Blocking on stdin

### Problem
During automated testing (`qa_test.py --headless`), the system would hang indefinitely.
This was traced to high-risk tools (like executing terminal commands or deleting files) containing hardcoded `input()` or `sys.stdin.read()` prompts requiring human confirmation. In a headless environment, there is no stdin, causing the process to block forever.

### Resolution
- **Action:** Leveraged the `LEVEL_4` autonomy setting (or equivalent headless flags) within the execution pipeline to bypass these hardcoded confirmation prompts during automated testing.
- **Outcome:** Automated tests can now run to completion without deadlocking on user input.

---

## Summary for Future Agents
When working on `Jarvis`:
1. **Always assume an offline-first execution environment.** Rely on local Ollama models.
2. **Beware of async race conditions** when initializing system state (like model discovery). Ensure critical state is loaded synchronously before dependent components rely on it.
3. **Validate schemas strictly.** When passing objects to LLMs for prompt building, ensure they are 100% JSON serializable.
4. **Handle headless environments.** Any new tool that requires user input MUST have a fallback or bypass for headless execution (`LEVEL_4` autonomy).

**Testing verified via:**
- `python qa_test.py --headless`
- Interactive terminal execution (`python test_interactive.py` / `python -m core.controller_v2`)
