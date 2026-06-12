# 10. Agent Execution Loop & Autonomy (The Core Engine)

## 1. System Intent & Rationale (WHY it exists)
The Agent subsystem exists to transform the system from a passive, conversational LLM into an autonomous, goal-oriented executor. It acts as the cognitive engine of Jarvis, responsible for continuous iterative problem solving. Instead of blindly executing single commands, it orchestrates a persistent loop (`plan -> risk -> confirm -> execute -> reflect`), allowing the system to handle complex, multi-step asynchronous tasks, recover from failures, and dynamically adjust to new observations.

## 2. Core Responsibilities (WHAT it owns)
The subsystem is the central nervous system for autonomous task resolution:
- **DAG Generation & Orchestration:** Translating natural language goals into a structured Directed Acyclic Graph (DAG) of actionable steps (`_build_plan` and `_normalize_steps`).
- **Risk & Autonomy Governance:** Interfacing with the `RiskEvaluator` and `AutonomyGovernor` to decide whether a task is safe to execute automatically (e.g., headless `LEVEL_4` autonomy) or requires manual user confirmation.
- **Execution Tracing:** Maintaining an `ExecutionTrace` object (tracking `goal`, `iterations`, `plan`, `observations`, `risk_scores`, `think_blocks`, `reflection`, `final_response`, `success`, `stop_reason`, and timestamps) across up to a default of 10 maximum iterations.
- **Tool Routing:** Querying the `CapabilityRegistry` to map planned steps to actual tool and function signatures, gathering output via `ToolObservation`.
- **State Machine Transitions:** Managing the lifecycle inside the `TaskExecutionContext` and modifying the `StateMachine`, seamlessly shifting between Thinking, Confirming, Executing, and Reflecting states.
- **Rollback & Recovery:** Working with the execution engine to handle failures, ensuring that multi-step plans can be cleanly reverted or retried if sub-tasks fail.

## 3. System Interactions (HOW it interacts)
- **Upstream Triggers:** Receives natural language objectives and execution contexts via asynchronous task inputs or the client APIs.
- **Autonomy & Risk Pipelines:** Before execution, plans are evaluated step-by-step. Destructive tasks pause the state machine and the asynchronous event loop, awaiting a human callback (`_ask_confirmation`).
- **LLM Integration:** By default, interacts with local models (e.g., `http://localhost:11434` for DeepSeek-R1). It parses raw text to extract internal reasoning via `<think>` regex tags, and structures output using strict system prompts like `REFLECT_SYSTEM_PROMPT`.
- **Data Defense & Truncation:** Employs functions like `_truncate_obs()` and `_truncate_observation()` to aggressively limit tool outputs (e.g., capped at 800-4000 characters), preventing context-window poisoning and out-of-memory crashes.
- **Concurrent Execution:** Reads configurations like `max_concurrent_workers` to execute steps asynchronously via an asyncio event loop while attempting to manage shared state across parallel workers.

## 4. Architectural Weaknesses & Adversarial Vulnerabilities
Red Team adversarial analysis highlights severe discrepancies between the design intent and actual execution, which must be directly addressed during any production deployment or reconstruction:
- **Catastrophic State Locks:** The documentation claims robust `asyncio` multi-agent processing, but underlying state modifications either use thread-bound locks (`threading.RLock`) or lack locking entirely (e.g., flat JSON writes for profiles). This guarantees race conditions, deadlocks, and split-brain memory when handling concurrent DAGs.
- **Rollback Timeout Orphans:** The execution graph is bound by a hardcoded 5-minute limit (`asyncio.timeout(300)`). If this timeout fires during a Last-In-First-Out (LIFO) rollback, the rollback itself is cancelled, fracturing the agent's state permanently.
- **Persona Schizophrenia & Unbounded Trust:** The core reflection prompt (`REFLECT_SYSTEM_PROMPT`) demands concise, technical responses, but capability prompts force a verbose, cheerful persona. Furthermore, the absence of negative constraint directives (`<Safety_Rules>`) leaves the system highly susceptible to second-order prompt injections from compromised data.
- **Parsing Fragility:** Relying on regex to extract `<think>` blocks and silently truncating JSON strings risks ReDoS attacks and feeding malformed, unclosed payloads to downstream parsers.

## 5. Dependency Analysis & Critical Failure (WHAT breaks if removed)
If the Agent Loop (`agent_loop.py`, `agentic_goal_manager.py`) is removed, Jarvis devolves into a simple query-response chatbot. 
- The system immediately loses the capacity for multi-step logic and iterative problem solving.
- Automated failure recovery, self-reflection, and LIFO topological rollbacks cease to exist.
- There is no mechanism to guard against risky operations, as the risk evaluation pipeline is tightly coupled to the loop's DAG parser.
- The `CapabilityRegistry` becomes dormant and isolated, unable to be chained or invoked autonomously.

## 6. Clean-Room Reconstruction Directive (HOW to rebuild from scratch)
To rebuild the Agent Loop without source code, engineer a resilient, highly-concurrent state machine and event loop:
1. **The Loop Engine:** Create an `AgentLoopEngine` class that drives an `asyncio` loop bounded by `_DEFAULT_MAX_ITERATIONS` (10). Track state purely using an `ExecutionTrace` dataclass.
2. **DAG Planner:** Upon receiving a goal, prompt the LLM to output a strict, schema-bound JSON graph containing `steps` (id, action, params, dependencies). Do not use regex to parse critical control flow.
3. **Risk Gating:** Before executing any node, calculate a risk score. If the score exceeds the threshold defined by the active autonomy level, trigger an interrupt, emit a confirmation event, and `await` an external callback.
4. **Execution & Concurrency:** Execute independent DAG nodes in parallel using `asyncio.gather()`. Crucially, enforce ACID-compliant, atomic file-locking (via `asyncio.Lock` and `.tmp` atomic file swaps) for any state mutations to prevent corruption from concurrent workers.
5. **Resilient Rollbacks:** Implement a LIFO topological rollback stack that runs in a shielded context (`asyncio.shield`), ensuring that if the parent task times out (e.g., at 300s), the rollback logic still executes to completion.
6. **Reflection Phase:** After node execution, parse the `ToolObservation`s, cleanly truncate large text payloads, and feed them into a unified, conflict-free reflection prompt. Have the LLM analyze failures (root cause + fix) or summarize success, deciding if the goal is met or if a replan is required.

## 7. Literal Data Schemas & Task Parsing Keys

### DAG Plan JSON Schema
The exact programmatic JSON structure output by the LLM planner (`core/planner/planner.py`):
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

### DAG Execution Parsing Keys
The actual internal representations expected by `_normalize_steps` and the `DAGExecutor` (`core/executor/dag.py`, `core/executor/engine.py`):
- **`id`**: Parsed as integer or string ID.
- **`action`**: The tool to invoke (can also fallback to checking the key `"tool"`).
- **`description`**: Rationale for the step.
- **`params`**: The execution engine and normalizer look for the `"params"` key, or gracefully fall back to `"parameters"` or `"args"`.
- **`depends_on`**: An array of preceding step IDs this node requires before it can begin (used for topological sort).
- **`retry_count`**: Integer defining the max backoff attempts on failure.
- **`rollback`**: An optional nested object mapping to `{ "action": "rollback_tool", "params": {...} }`.

### ExecutionTrace Dataclass
The `ExecutionTrace` Python dataclass definition (`core/agent/agent_loop.py`) storing the lifecycle:
```python
@dataclass
class ExecutionTrace:
    goal: str
    iterations: int = 0
    plan: Optional[dict[str, Any]] = None
    observations: list[dict[str, Any]] = field(default_factory=list)
    risk_scores: list[dict[str, Any]] = field(default_factory=list)
    think_blocks: list[str] = field(default_factory=list)
    reflection: Optional[str] = None
    final_response: str = ""
    success: bool = False
    stop_reason: str = ""
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None
```
