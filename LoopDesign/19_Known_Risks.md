# 19. Known Risks & Adversarial Vulnerabilities

## 1. System Intent & Purpose (WHY does this exist?)
The Known Risks domain exists as an architectural mandate to formally define the failure domains, operational boundaries, unmitigated technical debt, and security vulnerabilities of the system. While other layers of the architecture define the "happy path" and functional aspirations, this document acts as the reality check. It prevents the reconstruction effort from blindly implementing flawed, implicitly trusted logic that would collapse under real-world runtime conditions.

## 2. Core Responsibilities (WHAT does it own?)
This domain owns the classification and documentation of:
*   **Concurrency & State Hazards:** Identifying race conditions, lock mismatches, and data corruption vectors.
*   **Resource & Timeout Exhaustion:** Defining the bounds of memory scalability, retry logic, and rollback fragility.
*   **Adversarial & Security Vectors:** Tracking implicit trust vulnerabilities, default privilege escalations, and persistent prompt injection risks.
*   **Data Integrity Fractures:** Managing schema drifts, split-brain memory occurrences, and temporal logic breakages.

## 3. Systemic Interactions (HOW does it interact with the rest of the system?)
The Known Risks domain acts as a strict constraint layer across all primary subsystems:
*   **State Management & Persistence:** It overrides naive Read-Modify-Write JSON usage and dual-database architectural patterns, demanding a redesign toward unified, ACID-compliant, atomic operations.
*   **Control Flow & Engine:** It constraints the state machine workflows by highlighting how async/thread lock mismatches and hardcoded systemic timeouts break topological rollbacks.
*   **Prompt Architecture:** It overrides the default positive-only ("cheerleader") prompts, injecting strict negative constraints, safety rules, and structural enforcements to prevent prompt injection and persona schizophrenia.
*   **Automation & Network Integrations:** It highlights the systemic dangers of unbounded arrays and jitter-less exponential backoffs, forcing strict boundaries on external API interactions and event ingestion.

## 4. Impact of Removal (WHAT breaks if removed?)
If this domain is ignored during reconstruction, the resulting system will suffer catastrophic, systemic failure under load:
*   **Silent State Corruption:** Concurrent async agents will mutate the state machine simultaneously due to OS-thread lock mismatches, while overlapping file writes will irreparably corrupt the JSON state graphs.
*   **Memory Exhaustion (OOM):** Unbounded O(N) deduplication arrays in automation state will balloon indefinitely, permanently crashing the automation loop.
*   **Split-Brain Hallucinations:** The Retrieval-Augmented Generation (RAG) pipeline will query conflicting databases and broken lexicographical timestamps, resulting in out-of-order memory retrieval and contradictory behaviors.
*   **God-Mode Compromise:** Default administrative privileges and unsanitized state-to-prompt injections will allow bad actors (or rogue AI actions) to persistently hijack the system context.
*   **Thundering Herd DDoS:** Synchronized network failure retries without randomized jitter will continuously DDoS downstream APIs.

## 5. Reconstruction Mandate (HOW would it be rebuilt from scratch?)
To rebuild this system without source code, the engineering team must treat the following mitigations as absolute prerequisites before any functional logic is implemented:

### A. Unified Concurrency & Persistence Model
*   **Locking:** Establish a single, unified asynchronous concurrency model. All blocking OS-thread locks must be eradicated from async task flows.
*   **Atomic State:** Flat JSON file persistence must be replaced. High-frequency state must either be migrated to SQLite running in WAL (Write-Ahead Logging) mode or use strict atomic temporary file swapping to guarantee idempotency during power loss or crashes.

### B. Bounded Resource & Execution Constraints
*   **O(1) Data Structures:** Unbounded arrays for deduplication must be migrated to indexed SQL tables with explicit unique constraints to prevent OOM and disk exhaustion.
*   **Jittered Backoffs:** All network retry mechanisms and exponential backoffs must implement randomized jitter to prevent Thundering Herd scenarios.
*   **Rollback Integrity:** The control flow engine must guarantee that LIFO rollbacks execute in an isolated timeout context, ensuring that an outer-loop timeout does not orphan partial state changes or leave data transactions hanging.

### C. Unified State & Temporal Coherence
*   **Single Source of Truth:** Eradicate dual databases. Consolidate all memory and episodic storage into a single, highly normalized schema.
*   **Temporal Standardization:** Enforce strict UTC Unix Epochs or strictly validated ISO-8601 strings across all data stores to guarantee flawless chronological and lexicographical sorting.

### D. Zero-Trust Security & Prompt Sandboxing
*   **Least Privilege Default:** Ensure the authentication schema strictly defaults to non-admin roles. Implement clear destructive-action budgets and headless manual confirmation boundaries.
*   **Sanitized State Injection:** Treat all user-supplied state (e.g., interaction history, preferences) as hostile. It must be heavily sanitized and enclosed in strict delimiter boundaries before being injected into the system prompt to prevent second-order prompt injections.
*   **Unified Constraints:** Prompts must enforce strict structural schemas (e.g., JSON schemas for tool targets) and include explicit failure-state fallback instructions to handle edge cases gracefully.

### E. Literal Bounded Constraints & Extracted Schemas

**1. Bounded Limits (Extracted from Codebase):**
*   `max_seen_fingerprints`: 10000 (OOM mitigation for flat deduplication arrays)
*   `max_scrape_chars`: 8000 (Web context truncation)
*   `_OCR_MAX_TEXT_CHARS`: 4000 (Vision/Screen context limitation)
*   `max_bytes`: 32768 (File read token limit)
*   `max_results`: 1000 (Fast search limit) / 5 (Web search limit)
*   `stt_max_duration_s`: 8 (Audio input boundary)
*   `max_parallel_tasks`: 3 (Concurrency constraint)
*   `max_facts`: 10000 (Memory limits)

**2. Raw SQLite Relational Schemas (`memory.db` / `jarvis_memory.db` & `auth.db`):**
```sqlite
-- Unified Memory Schema Constraints
CREATE TABLE IF NOT EXISTS preferences (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY,
    timestamp TEXT DEFAULT '',
    content TEXT
);
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    timestamp TEXT DEFAULT '',
    role TEXT,
    content TEXT
);
CREATE TABLE IF NOT EXISTS actions (
    id TEXT PRIMARY KEY,
    timestamp TEXT DEFAULT '',
    action_type TEXT,
    status TEXT,
    details TEXT
);

-- Zero-Trust Authentication Schema
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password_hash TEXT NOT NULL,
    is_admin INTEGER NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS api_tokens (
    token_hash TEXT PRIMARY KEY,
    label TEXT,
    created_at REAL NOT NULL,
    last_used REAL
);
```

**3. Raw JSON Data Structures (State Files):**

*automation_state.json:*
```json
{
  "saved_at": "String (ISO-8601)",
  "seen_fingerprints": ["String (64-char SHA-256)"]
}
```

*goals.json:*
```json
{
  "saved_at": "String (ISO-8601)",
  "goals": ["Goal DTO"],
  "schedule": ["ScheduleItem DTO"]
}
```
