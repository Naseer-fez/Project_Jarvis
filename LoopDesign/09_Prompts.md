# 09 Prompts Subsystem

## 1. Why does this subsystem exist?
The Prompts subsystem exists to act as the primary control interface between the deterministic software architecture of Jarvis and the non-deterministic, generative nature of Large Language Models (LLMs). It establishes the behavioral boundaries, persona constraints, and output formatting rules required for the system's autonomous reasoning loops. Without prompts, the LLM cannot discern its role, the operational context, or the strict programmatic contracts (like returning structured JSON) expected by the calling Python modules.

## 2. What responsibility does it own?
The Prompt subsystem owns:
* **Persona and System Definition**: Defining the identity ("Jarvis"), system constraints (running locally), and behavioral style (concise, technical, truthful).
* **Task-Specific Operational Directives**: Instructing specialized LLM calls on how to perform discrete sub-tasks, such as:
  * **Reflection & Remediation**: Evaluating tool execution outcomes, diagnosing root causes of failures, and formulating fixes.
  * **Data Extraction**: Converting conversational requests into concise web search queries.
  * **Summarization**: Compressing retrieved web content or historic memory blocks into compact text or short titles.
  * **Visual Reasoning**: Directing multimodal models to locate UI elements via screenshots and returning coordinates in strict JSON format.
* **Context Formatting**: Managing dynamic templates (`f-strings`) that inject runtime state (goals, plans, tool observations, query results) into the prompt string before dispatch.

## 3. How does it interact with the rest of the system?
Prompts are deeply embedded within the execution flows of the `llm_orchestrator`, `agent_loop`, and specialized capability modules:
* **LLM Clients (`client.py`, `cloud_client.py`)**: System prompts (e.g., `JARVIS_SYSTEM`) are sent as the root context for interactions.
* **Agent Loop (`agent_loop.py`)**: Uses the `user_prompt` to combine the current Goal, Plan, and Tool Observations. The `REFLECT_SYSTEM_PROMPT` is dynamically applied during the post-execution evaluation phase.
* **Capability Tools (`web_tools.py`, `gui_control.py`, `context_compressor.py`)**: Utilize distinct, zero-shot system prompts (e.g., `QUERY_EXTRACTION_SYSTEM`, `TITLE_SYSTEM`) to quickly parse inputs or format outputs without invoking the full agentic persona.
* **State Management & Memory**: Prompts format memory episodes and preference states to inject them into the active context window.

## 4. What would break if removed?
The entire autonomous system would suffer a catastrophic collapse:
* **Loss of Autonomy**: The `agent_loop` would fail to parse LLM outputs, as the model would no longer be forced into the strict output schemas (like JSON for GUI targeting or specific plan steps) required by `TaskPlanner`.
* **Behavioral Degradation**: The AI would revert to a generic conversational assistant, outputting verbose, unformatted text containing Markdown quotes, bullets, and filler phrases, completely breaking downstream text parsing.
* **Functional Failure**: Web tools would execute searches with overly conversational queries instead of optimized keywords. Memory context compression would fail, leading to rapid context window exhaustion. Multimodal GUI automation would hallucinate instead of returning precise `x, y` pixel coordinates.

## 5. How would it be rebuilt from scratch without source code?
To rebuild the prompt subsystem from scratch:
1. **Catalog Task Boundaries**: Identify every distinct LLM interaction point in the architecture (Planning, Execution/GUI, Reflection, Memory Compression, Web Search Query Extraction, Web Summary).
2. **Define Strict System Personas**: Create a root System Prompt enforcing the "Jarvis" persona, emphasizing technical brevity, local execution constraints, and absolute prohibition of filler phrasing.
3. **Build Zero-Shot Capability Prompts**:
   * *Web Extraction*: "You convert conversational requests into concise web search queries. Return only the query text, no quotes, no bullets, no explanation. If the context is empty or irrelevant, return NO_DATA."
   * *Web Summarization*: "Summarize search results. Write 2-4 short sentences grounded only in the provided results. Do not invent facts. If the context is empty or irrelevant, return NO_DATA."
   * *GUI Targeting*: "Locate the UI target. Return strict JSON with keys: found, x, y, width, height, confidence, reason. Use absolute image pixels."
   * *Reflection*: "Review executed plan and observations. If failed: state root cause and fix. If successful: summarize. Be direct. Address the user in second person."
   * *Error Correction Fallbacks*: If output parsing fails, an iterative feedback prompt must trigger: "You output {previous_output}, but I expected JSON. Fix it."
4. **Implement Context Templates**: Construct string templates with clearly delimited sections (e.g., `Goal:\n{...}\n\nPlan:\n{...}\n\nTool observations:\n{...}`) to inject runtime state securely.
5. **Strict Programmatic Schemas**:
   * **Main Agent Loop / Plan Output Schema (TaskPlanner)**:
     ```json
     {
       "intent": "user request",
       "summary": "overall plan summary",
       "confidence": 0.9,
       "steps": [
         {
           "id": 1,
           "action": "tool_name",
           "description": "why we call this tool",
           "parameters": {"<argument_name>": "<argument_value>"}
         }
       ],
       "clarification_needed": false,
       "clarification_prompt": ""
     }
     ```
   * **Memory and State Injection Schemas (SQLite)**:
     - Preferences: `key TEXT PRIMARY KEY, value TEXT, updated_at TEXT`
     - Episodes: `id INTEGER PRIMARY KEY, event TEXT, category TEXT, timestamp TEXT`
     - Conversations: `id INTEGER PRIMARY KEY, user_input TEXT, assistant_response TEXT, session_id TEXT, timestamp TEXT`
     - Actions: `id INTEGER PRIMARY KEY AUTOINCREMENT, action TEXT NOT NULL, result TEXT, success INTEGER NOT NULL DEFAULT 1, metadata TEXT, timestamp TEXT NOT NULL`
   * **Tool Exposure Formatting**: Tools are exposed to the LLM via JSON schema representation injected into the prompt:
     ```json
     {
       "tools": [
         {
           "name": "tool_name",
           "description": "Execute tool_name",
           "parameters": {
             "param_name": {
               "type": "string",
               "default": "value",
               "required": true
             }
           }
         }
       ]
     }
     ```
   * **Fallback Triggers & Directives**: Explicit `FALLBACK` and `NO_DATA` directives must be implemented for failure states, ensuring the LLM returns `NO_DATA` when context is empty or irrelevant. Iterative parser error loops must retry on structural failures.

## 6. Adversarial Considerations & Security Gaps
Based on Red Team audits, the current prompt architecture exhibits severe vulnerabilities that must be addressed during reconstruction:
* **Second-Order Prompt Injection**: The system inherently trusts its own state files. Malicious payloads injected into user-modifiable states (e.g., `user_profile.json` communication styles) will act as persistent prompt overrides, hijacking the LLM.
* **Missing Negative Guardrails**: The prompts lack `<Safety_Rules>` or destructive-action budgets. The persona is fundamentally a "cheerleader," defaulting to extreme trust and risking unbounded system damage in headless modes.
* **Context Poisoning**: Dynamic injection of un-sanitized external data (like large, fetched HTML blocks or massive source code diffs) can overwrite system instructions. Prompts must be wrapped in strict delimiters and utilize input sanitization to ensure external content cannot break out of its designated context block.
