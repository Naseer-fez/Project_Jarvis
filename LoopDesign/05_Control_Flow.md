# 05 Control Flow Architecture

## WHY: Purpose and Core Rationale
The Control Flow subsystem operates as the central nervous system and primary traffic director for the Jarvis architecture. It exists to bridge the gap between unpredictable, multi-modal human inputs (CLI, Voice, Dashboard, background triggers) and deterministic, multi-step system actions. Its core purpose is to impose strict, topological boundaries on agent execution—routing simple queries to bypass expensive reasoning pipelines, orchestrating heavy LLM reasoning tasks, and gating potentially dangerous autonomous actions through a rigid state machine. Without this subsystem, the agent would lack the structural scaffolding to pause for user confirmation, reflect on task failures, or maintain thread-safe execution state across asynchronous events.

## WHAT: Spheres of Responsibility
1. **Ingestion & Normalization:** Standardizes payloads across entry channels, enforces sanitization (e.g., aggressively truncating inputs to 4000 characters to prevent memory exhaustion), and initializes tracing structures (`trace_id`).
2. **Heuristic Routing & Intent Interception:** Owns the `ComplexityScorer`, evaluating structural and keyword signals to classify inputs (`Reflex`, `Chat`, `Agentic`, `Deep Reasoning`). It leverages an `IntentRouter` to bypass LLMs entirely for direct commands (e.g., "set a reminder", "remember that").
3. **Execution Pipeline Orchestration:** Manages the handoff to the `AgentLoopEngine` for complex tasks, traversing the execution pipeline from `PLANNING` through `EXECUTING` to `COMPLETED`.
4. **Concurrency & Thread-State Management:** Enforces state transitions via the `StateMachine`. It maintains process locks across UI and agent threads using a blend of `asyncio.Lock` (`_state_lock` in `controller_v2`) and OS-thread `threading.RLock`, ensuring re-entrant thread safety and protecting against async deadlocks.
5. **Autonomy Gating & Risk Enforcement:** Injects hard-stops into the execution flow. The `RiskEvaluator` and `AutonomyGovernor` intercept DAG tool plans, halting execution and requiring manual user confirmation before executing potentially destructive actions.

## HOW: Interaction and Data Flow
The control flow operates as a funnel, gradually increasing autonomy and complexity:
1. **Entry Point:** `JarvisControllerV2.process()` ingests a raw request.
2. **Intent & Complexity Bypass:** The `ComplexityScorer` evaluates the request. Hardcoded rules in `IntentRouter` immediately intercept explicitly defined goals or preferences, returning a response and bypassing the LLM.
3. **LLM Dispatching:** Unresolved queries fall to `LLMOrchestrator.dispatch()`. The `MemorySubsystem` injects relevant history and facts, while the `ModelRouter` selects the optimal local or remote model based on the complexity classification.
4. **Agentic Loop Trigger:** For tasks classified as `Agentic` (requiring tools or multi-step reasoning), `controller_services` delegates control to the `AgentLoopEngine.run()`.
5. **The Agent Loop Execution:**
   - **Plan:** The `TaskPlanner` uses an LLM to generate a sequence of steps (a DAG) based on the context.
   - **Risk Evaluation:** The `RiskEvaluator` checks the DAG against safety rules.
   - **Confirm:** The `AutonomyGovernor` halts the async event loop, shifting the state to `AWAITING_CONFIRMATION` (unless bypassed in headless or Autonomy Level 4 modes).
   - **Execute:** A worker executor runs the plan. This phase is constrained by a hardcoded 5-minute timeout (`asyncio.timeout(300)`) to prevent indefinite hanging.
   - **Reflect:** Outputs are collected as `ToolObservation`s. A reflection parser aggressively strips `<think>` tags (from deep reasoning models) via regex, allowing an LLM to analyze the execution, determine root causes of failure, and update the `ConfidenceModel`.

## WHAT BREAKS: Removal or Degradation Impact
If the Control Flow subsystem were compromised or removed:
- **Infinite Action Loops:** The absence of `StateMachine` topologies would allow the agent to blindly loop through tool execution without pausing for reflection, risking cascading failures.
- **Concurrency Deadlocks:** The architecture blends synchronous, thread-blocking libraries (e.g., PyGithub) with an `asyncio` event loop. Removing the dual-locking strategy (`StateGuard`, `RLock`, and `asyncio.Lock`) would lead to Thread-Pool exhaustion and inevitable system-wide deadlocks, leaving the agent completely unresponsive.
- **Resource Exhaustion & DOS:** Without payload truncation and the `StateMachine`'s rolling 100-event audit cap, adversarial payloads or unbounded states would trigger rapid memory ballooning and fatal OOM crashes.
- **Runaway Autonomy:** The `AutonomyGovernor` and `RiskEvaluator` intercepts would vanish, permitting the agent to execute irreversible file system or network commands entirely unchecked.

## RECONSTRUCTION: Rebuilding Without Source Code
To rebuild this subsystem purely from intent:
1. **Define State Topology Mathematically:** Start by creating a strictly enforced Enum (`IDLE`, `THINKING`, `PLANNING`, `RISK_EVALUATION`, `AWAITING_CONFIRMATION`, `EXECUTING`, `REFLECTING`, `COMPLETED`, `ERROR`). Map out and strictly enforce allowed topological transitions. An invalid transition must raise a fatal error, preventing orphaned execution states.
2. **Implement Dual-Lock Concurrency:** Design a locking system that bridges the async-sync divide. A `StateGuard` context manager must exist to map `asyncio.CancelledError` directly to an `ABORTED` state. Implement a global async lock for the UI loop, but use an OS-level `threading.RLock` to protect state mutations against re-entrant workers.
3. **Build the Heuristic Scorer:** Write a regex/keyword-based router that triages inputs into structural categories before an LLM is ever invoked, establishing an intent bypass to save compute cycles.
4. **Construct the Yielding Loop Engine:** Build an asynchronous `AgentLoopEngine` that sequentially yields to a task planner, evaluates risk, yields to the host for interactive confirmation, and finally offloads to an executor queue with a strict maximum-iteration limit.
5. **Add Reflection Parsers:** Implement regex sanitizers to strip reasoning tokens (like `<think>`) from raw LLM text streams before synthesizing final observations, guaranteeing the reflection LLM focuses purely on deterministic tool outputs.

## 6. Programmatic Schemas

### State Machine Topology (`State` Enum)
```python
from enum import Enum

class State(str, Enum):
    IDLE = 'IDLE'
    THINKING = 'THINKING'
    PLANNING = 'PLANNING'
    RISK_EVALUATION = 'RISK_EVALUATION'
    AWAITING_CONFIRMATION = 'AWAITING_CONFIRMATION'
    APPROVED = 'APPROVED'
    CANCELLED = 'CANCELLED'
    ACTING = 'ACTING'
    OBSERVING = 'OBSERVING'
    REFLECTING = 'REFLECTING'
    REVIEWING = 'REVIEWING'
    EXECUTING = 'EXECUTING'
    COMPLETED = 'COMPLETED'
    SPEAKING = 'SPEAKING'
    LISTENING = 'LISTENING'
    TRANSCRIBING = 'TRANSCRIBING'
    ERROR = 'ERROR'
    ABORTED = 'ABORTED'
    SHUTDOWN = 'SHUTDOWN'
```

### Autonomy Governor Gating (`AutonomyLevel` IntEnum)
```python
from enum import IntEnum

class AutonomyLevel(IntEnum):
    CHAT_ONLY = 0
    SUGGEST_ONLY = 1
    READ_ONLY = 2
    WRITE_WITH_CONFIRM = 3
    AUTONOMOUS = 4
```

### Risk Evaluator Schemas
```python
from enum import IntEnum
from dataclasses import dataclass

class RiskLevel(IntEnum):
    LOW = 0
    MEDIUM = 1
    CONFIRM = 2
    HIGH = 3
    CRITICAL = 4
    FORBIDDEN = 4

@dataclass
class RiskResult:
    level: RiskLevel
    blocking_actions: list[str]
    confirm_actions: list[str]
    high_risk_actions: list[str]
    reasons: list[str]
```

### Execution Trace Schema (`ExecutionTrace` dataclass)
```python
from dataclasses import dataclass
from typing import Optional, Any

@dataclass
class ExecutionTrace:
    goal: str
    iterations: int
    plan: Optional[dict[str, Any]]
    observations: list[dict[str, Any]]
    risk_scores: list[dict[str, Any]]
    think_blocks: list[str]
    reflection: Optional[str]
    final_response: str
    success: bool
    stop_reason: str
    started_at: float
    ended_at: Optional[float]
```

### Complexity Scorer Output Schema
```python
# Returned by ComplexityScorer.classify_request()
{
    "class": "Reflex | Chat | Agentic | Deep_Reasoning",
    "complexity": float,
    "route": str,
    "skip_planner": bool,
    "estimated_tokens": int,
    "needs_reasoning": bool,
    "needs_tools": bool,
    "needs_vision": bool,
    "context_weight": float
}
```

### Automation State Persistence (`automation_state.json`)
```json
{
  "saved_at": "string (ISO-8601)",
  "seen_fingerprints": ["string (64-character SHA-256)"]
}
```
