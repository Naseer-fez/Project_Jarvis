# 15. Security & Risk Management Subsystem

## 1. Architectural Intent (WHY does this subsystem exist?)
The Security subsystem exists to resolve the fundamental tension between autonomous capability and host safety. An AI agent with tool access is inherently dangerous because non-deterministic generative models are prone to hallucinations, misinterpretations of complex goals, and context-window poisoning (prompt injection) from external adversarial inputs. This subsystem acts as a rigid, deterministic firewall built around the LLM's non-deterministic reasoning. It guarantees mathematically and systematically that regardless of the LLM's generated output, no critical execution, state mutation, or unauthorized access can occur without verifiable authentication and strictly enforced authorization schemas. Its ultimate intent is to protect the integrity of the host operating system, the privacy of the user's data, and the uncorrupted state of the agent's memory.

## 2. Core Responsibilities (WHAT responsibility does it own?)
The subsystem's responsibilities are divided into three distinct pillars:

**A. Cryptographic Identity & Access Management:**
It owns the authoritative truth of "who" or "what" is invoking the system. For human operators, it enforces cryptographically secure identity verification (e.g., using PBKDF2 hashing algorithms with high computational resistance). For machine-to-machine integrations (background subagents, headless scripts, external webhooks), it manages labeled, securely hashed API tokens. It ensures the integrity of active user sessions by maintaining paired, signed artifacts (like secure cookies coupled with anti-forgery tokens) to prevent session hijacking.

**B. Autonomy Governance (Stateful Permissioning):**
It owns the macro-level constraints on the agent's operational freedom. It maintains a strict autonomy state machine (ranging from fully locked-down "Chat Only" to fully "Autonomous"). It categorizes every capability in the system as either a benign "read" operation or a state-mutating "write" operation. The governor serves as the overarching judge that categorically denies any state-mutating request if the host environment's current autonomy level has not been explicitly escalated by the human operator.

**C. Deterministic Risk Evaluation (Action Gating):**
It owns the micro-level scrutiny of the agent's intent. Before any planned action graph reaches the execution engine, it evaluates the specific tool payload against a deterministic risk ontology. It maps proposed actions to risk tiers (e.g., safe, requires confirmation, critical, forbidden) and aggregates these evaluations to construct a final, immutable verdict on whether the plan is safe to proceed, needs human intervention, or must be aborted immediately.

## 3. System Interactions (HOW does it interact with the rest of the system?)
* **With the Agent Execution Loop:** The security subsystem acts as an unavoidable tollbooth in the primary execution pipeline. After the planning layer formulates a multi-step execution graph, the engine synchronously passes the proposed actions to the risk evaluator and autonomy governor. If the verdict flags a critical risk or autonomy violation, the loop aborts the iteration, discarding the plan. If the verdict requires confirmation, the loop suspends the execution thread, pushing an asynchronous approval request to the client interface and halting further processing until explicit human clearance is received.
* **With the Capability Registry:** The subsystem interacts dynamically with the tool registration layer. As plugins and external tools are loaded into the system, they must declare their risk profiles and mutation intent (read vs. write) to the security core. This ensures the permission matrices remain synchronized with the agent's actual capabilities without relying on fragile, hardcoded lists.
* **With the API and Web Interface:** It wraps all inbound communication routes. Client endpoints must consult the access manager to validate session signatures and anti-forgery tokens before routing the payload. Background automation tasks authenticate invisibly via their dedicated hashed tokens, allowing continuous backend communication while remaining strictly gated from unauthorized network traffic.

## 4. Failure Modes (WHAT would break if removed?)
* **Catastrophic Host Compromise (Arbitrary Execution):** Without the execution gating layers, the LLM wields unchecked administrative access to the host. An adversarial payload encountered during a routine web search could successfully inject a prompt instructing the agent to delete local files, download malware, or exfiltrate environment variables, and the execution engine would blindly comply.
* **Silent Escalation & Destructive Actions:** The absence of the "confirmation" tier would allow the agent to autonomously alter system configurations, fire off sensitive emails, or publish irreversible data based entirely on hallucinated conclusions, resulting in severe privacy, reputational, or operational damage.
* **Total Interface Takeover:** Removing the identity management layer strips the API firewall. Any malicious process on the local network could construct raw HTTP requests and post them to the execution engine, hijacking the AI's capabilities entirely without ever passing through an authentication interface.
* **Cross-Site Request Forgery (CSRF) Exposure:** Without paired session and anti-forgery tokens, a logged-in user could be tricked into clicking a malicious link on an external website, which would silently forge hidden execution commands back to the local Jarvis instance, subverting human intent entirely.

## 5. Reconstruction Blueprint (HOW would it be rebuilt from scratch?)
To rebuild this subsystem purely from its architectural intent without access to the original source code, a developer must implement three strictly isolated layers:

### Step 1: The Cryptographic Ledger
Construct a dedicated database (`auth.db`) solely for authentication state, structurally segregated from the agent's semantic memory to ensure no prompt-injection attack can query credential tables. Implement an industry-standard key derivation function for storing operator passwords. Develop an HTTP middleware that mints cryptographically signed, expiring session tokens.

You must implement the following exact SQLite schema:
```sql
CREATE TABLE users (
    username TEXT PRIMARY KEY,
    password_hash TEXT NOT NULL,
    is_admin INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL
);
CREATE TABLE api_tokens (
    token_hash TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    created_at REAL NOT NULL,
    last_used_at REAL
);
```
**CRITICAL:** Note the implicit "God-Mode Default" (`is_admin INTEGER NOT NULL DEFAULT 1`). To prevent privilege escalation, explicit constraints must be added during reconstruction to restrict default admin creation.

### Step 2: The Autonomy State Machine
Implement a stateful governor enforcing a rigid escalation ladder using the following exact role matrix (`AutonomyLevel` enum):
- `CHAT_ONLY = 0` (No tool execution)
- `SUGGEST_ONLY = 1` (Describes actions but never runs them)
- `READ_ONLY = 2` (Can inspect files, web, screen automatically)
- `WRITE_WITH_CONFIRM = 3` (Can change state after approval)
- `AUTONOMOUS = 4` (Fully autonomous execution without confirmation)

Require every tool in the capability registry to self-identify its mutation intent (`read` vs. `write`) at instantiation. Enforce an unbypassable rule: if the execution loop attempts to invoke a `write` tool while the system's global autonomy state is at tier 2 or lower, throw a hard security exception and terminate the execution thread.

### Step 3: The Deterministic Risk Interceptor
Develop an evaluation engine acting as middleware between the planner and the executor. Define the exact risk ontology (`RiskLevel` enum): `LOW = 0`, `MEDIUM = 1`, `CONFIRM = 2`, `HIGH = 3`, `CRITICAL = 4` (aliased as `FORBIDDEN`).
You must recreate the external, immutable configuration matrix precisely as follows:
```ini
[risk]
# Risk levels: LOW, MEDIUM, HIGH, CRITICAL
forbidden_actions = format_disk, wipe_disk, registry_write
blocked_actions = shell_exec, file_delete
critical_actions = execute_shell, delete_file
high_risk_actions = write_file, write_file_safe, file_write, process_spawn, launch_application, app_open, click, double_click, right_click, click_text_on_screen, click_screen_target, double_click_screen_target, right_click_screen_target, type_text, hotkey, press_key, move_mouse, drag, focus_window, scroll, send_hardware_command, serial_send
medium_risk_actions = read_file, file_read, web_search, web_scrape, capture_screen, capture_region, find_text_on_screen, read_screen_text, wait_for_text_on_screen, describe_screen, get_active_window, read_sensor, sensor_read, notification
low_risk_actions = search_memory, log_event, memory_read, memory_write, speak, display, status, health_check, get_time, get_system_stats, list_directory, system_stats, vision_analyze
user_confirmed_actions = launch_application, app_open, click, double_click, right_click, click_text_on_screen, click_screen_target, double_click_screen_target, right_click_screen_target, type_text, hotkey, press_key, move_mouse, drag, focus_window, scroll, send_hardware_command, serial_send
voice_confirm_threshold = MEDIUM
```
When the agent yields an action graph, process every node through this matrix. If any node maps to `CRITICAL`, trigger a "halt and catch fire" protocol that shreds the plan. If a node maps to `CONFIRM` or `HIGH`, freeze the agent's state machine, dispatch an approval payload to the human operator, and await a cryptographically verified callback.

### Step 4: Context Isolation and State-to-Prompt Schema Boundaries
The architecture must mandate explicit context window isolation boundaries to prevent adversarial poisoning from large external payloads (e.g., Git PR diffs, scraped web pages). Furthermore, you must provide strict input sanitization and negative constraints (`<Safety_Rules>`) for state-to-prompt pipelines to prevent second-order prompt injection via state corruption.

For instance, `user_profile.json` must enforce this exact rigid schema to prevent malicious instruction overrides during state re-injection:
```json
{
  "name": "string",
  "communication_style": "string (MUST NOT contain execution directives)",
  "expertise_level": "string",
  "preferred_topics": ["string"],
  "timezone": "string",
  "language": "string",
  "interaction_count": 0,
  "first_seen": "timestamp",
  "last_seen": "timestamp"
}
```
All state objects deserialized into the LLM context must be rigorously sanitized against validation logic before being attached to the system prompt.
