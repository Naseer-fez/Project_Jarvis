# End-to-End Execution Flows

## Overview
The Execution flows illustrate how user commands traverse the various architectural layers of the Jarvis AI system. The primary entry point is either the CLI (via standard input) or the Dashboard UI (FastAPI `/command` endpoint). Both pathways terminate at `JarvisControllerV2.process()`, which serves as the core orchestration junction. The `IntentRouter`, `AgentLoopEngine`, `AutomationManager`, and supporting subsystems determine the exact sequence of backend execution paths.

Below are the exhaustive traces for major system behaviors.

---

## 1. Web Search & LLM Fallback Flow
Triggered by explicit web search requests (e.g., *"search Python docs"*). If the web tool fails or is disabled, the system gracefully degrades to an LLM fallback coupled with local semantic memory.

```mermaid
sequenceDiagram
    participant User
    participant Frontend as Dashboard (FastAPI) / CLI
    participant Controller as JarvisControllerV2
    participant Router as IntentRouter
    participant Web as handle_web_search
    participant SearchAPI as web_search (DuckDuckGo/Google)
    participant MemDB as HybridMemory (SQLite + Chroma)
    participant LLM as LLMClientV2

    User->>Frontend: "search Python docs"
    Frontend->>Controller: process("search Python docs", trace_id)
    Controller->>Controller: classify_request()
    Controller->>Router: route(lowered, text, context)
    Router->>Web: handle_explicit_web()
    Web->>SearchAPI: execute web_search(query)
    SearchAPI-->>Web: Return raw search results
    
    alt Search Succeeded
        Web->>MemDB: build_context_block(user_input)
        MemDB-->>Web: Relevant context string
        Web->>LLM: chat_async(Synthesis Prompt with search results)
        LLM-->>Web: Natural language summary & citations
        Web->>MemDB: store_conversation(user_input, response)
    else Search Failed / Offline
        Web->>Web: _dispatch_llm_fallback()
        Web->>LLM: chat_async(messages)
        LLM-->>Web: Fallback LLM response
        alt Offline Recovery (if LLM fails)
            Web->>MemDB: recall_preferences(user_input)
            MemDB-->>Web: Offline preference fallback match
        end
    end
    
    Web-->>Router: Response string
    Router-->>Controller: Routed response
    Controller->>Frontend: CommandResponse (JSON)
    Frontend-->>User: Display Synthesized Output
```

---

## 2. Desktop Automation Execution Flow
Triggered by OS control requests (e.g., *"open Notepad"*, *"click on the browser"*).

```mermaid
sequenceDiagram
    participant User
    participant Frontend as Dashboard (FastAPI) / CLI
    participant Controller as JarvisControllerV2
    participant Router as IntentRouter
    participant DesktopPlan as plan_desktop_command
    participant DesktopExec as handle_desktop_command
    participant OS as PyAutoGUI API
    
    User->>Frontend: "open Notepad"
    Frontend->>Controller: process(text)
    Controller->>Router: route()
    
    Router->>DesktopPlan: plan_desktop_command("open Notepad")
    DesktopPlan-->>Router: Action Plan Mapping (App/Key/Mouse)
    
    Router->>DesktopExec: handle_desktop_command(user_input)
    DesktopExec->>OS: press_keys("win")
    DesktopExec->>OS: type("notepad")
    DesktopExec->>OS: press_keys("enter")
    OS-->>DesktopExec: Execution success signal
    
    DesktopExec-->>Router: "Successfully executed: open Notepad"
    Router-->>Controller: Routed response
    Controller->>Controller: _dashboard_update(state="IDLE")
    Controller-->>Frontend: CommandResponse (JSON)
    Frontend-->>User: Display Output & UI confirmation
```

---

## 3. Agentic / Planner Flow (DAG Task Execution)
Triggered by complex or multi-step requests requiring high autonomy, reasoning, and tool use (e.g., *"Analyze my logs and generate a markdown report"*).

```mermaid
sequenceDiagram
    participant User
    participant Frontend
    participant Controller
    participant Router as IntentRouter
    participant Planner as TaskPlanner
    participant Loop as AgentLoopEngine
    participant DAG as DAG Executor
    participant Tools as ToolRegistry (System)
    participant LLM as Ollama / LLMClientV2
    
    User->>Frontend: "Analyze logs and generate report"
    Frontend->>Controller: process()
    
    Controller->>Controller: classify_request() -> complexity > 0.5, route="planner"
    Controller->>Router: route()
    Router->>Planner: handle_agentic() -> plan(user_input)
    Planner->>LLM: generate execution steps
    LLM-->>Planner: JSON DAG Execution Plan
    Planner-->>Router: Validated Execution Plan
    
    Router->>Loop: run(goal, plan, TaskExecutionContext)
    Loop->>Loop: _ensure_thinking_state() / transition(PLANNING)
    
    Loop->>Loop: RiskEvaluator.evaluate_plan(plan)
    alt Requires Confirmation (AutonomyLevel < LEVEL_4)
        Loop->>Frontend: prompt user "High-impact actions. Continue?"
        Frontend-->>Loop: Approved
    end
    
    Loop->>DAG: execute(plan)
    loop Over Steps (Parallel/Sequential based on dependencies)
        DAG->>Tools: invoke step action
        Tools-->>DAG: ToolObservation (success/failure metrics)
    end
    DAG-->>Loop: Aggregated DAG Results
    
    Loop->>Loop: transition(REFLECTING)
    Loop->>LLM: _reflect(goal, plan, observations)
    LLM-->>Loop: Final Reflection Synthesis
    Loop->>Loop: transition(COMPLETED) -> transition(IDLE)
    
    Loop-->>Router: ExecutionTrace.final_response
    Router-->>Controller: Final response
    Controller-->>Frontend: CommandResponse (JSON)
    Frontend-->>User: Agentic Completion Report
```

---

## 4. Subsystem Flow: Automation Manager (RAG & Indexing)
Triggered by directory automation intents (e.g., *"automation scan"*, *"rag search python modules"*).

```mermaid
sequenceDiagram
    participant User
    participant Frontend
    participant Controller
    participant Router as IntentRouter
    participant AM as AutomationManager
    participant LA as LiveAutomation
    participant CodeIndexer
    participant MemDB as SemanticMemory (Chroma)
    
    User->>Frontend: "rag search 'auth logic'"
    Frontend->>Controller: process()
    Controller->>Router: route()
    
    Router->>AM: handle_automation()
    AM->>LA: search_rag("auth logic")
    LA->>CodeIndexer: Delegate semantic lookups
    CodeIndexer->>MemDB: query embeddings
    MemDB-->>CodeIndexer: Return matched code chunks & metadata
    CodeIndexer-->>LA: Formatted retrieval context
    
    LA-->>AM: Result summary string
    AM-->>Router: Response string
    Router-->>Controller: Final response
    Controller-->>Frontend: Output Result
```

---

## 5. Storage Flow: Hybrid Memory (SQLite + Vector DB)
Triggered when the user implicitly or explicitly declares a preference, fact, or entity memory.

```mermaid
sequenceDiagram
    participant User
    participant Controller
    participant Router as IntentRouter
    participant Intent as handle_preference_intent
    participant HybridMem as HybridMemory
    participant SQLite as SQLite Storage (WAL)
    participant Chroma as ChromaDB
    
    User->>Controller: "remember I prefer using Python 3.12"
    Controller->>Router: route()
    Router->>Intent: handle_preference_intent()
    Intent->>HybridMem: store_preference(key, value)
    
    par Dual Write
        HybridMem->>SQLite: insert into `preferences` table
        SQLite-->>HybridMem: commit success
    and
        HybridMem->>Chroma: semantic.store_preference() (upsert embeddings)
        Chroma-->>HybridMem: embedded and stored
    end
    
    Intent-->>Router: "Preference saved."
    Router-->>Controller: Final response
    Controller->>Controller: _dashboard_update()
```

---

## 6. Scheduled Goal Mutation Flow
Triggered by tasks establishing background goals or cron jobs.

```mermaid
sequenceDiagram
    participant User
    participant Controller
    participant Router as IntentRouter
    participant GM as GoalManager
    participant Sched as Scheduler
    participant Runner as GoalRunner
    participant File as goals.json
    
    User->>Controller: "add goal to health check server at 5pm"
    Controller->>Router: route()
    Router->>GM: handle_goal_intent()
    
    GM->>GM: create Goal Object (status=active)
    GM->>Sched: add_job(cron_pattern, Task)
    Sched-->>GM: Assigned job_id
    
    GM-->>Router: mutation=true (Goal state changed)
    Router-->>Controller: Intent handled
    
    Controller->>Runner: persist_goal_state()
    Runner->>File: Write serialized goals to disk
    Controller->>Controller: _dashboard_update(active_goals=N)
```
